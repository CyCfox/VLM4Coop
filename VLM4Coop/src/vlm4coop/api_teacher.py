"""OpenAI-compatible remote teacher; credentials stay in process memory only."""
import base64
import io
import json
import os
from pathlib import Path
import time
import threading
from random import SystemRandom
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from vlm4coop.teacher import TeacherResult
from vlm4coop.response_protocol import ProbabilityFormatError, parse_api_probabilities


API_SYSTEM_PROMPT = "You are a concise LBF multi-agent cooperative controller. Based on the attached figure, select one synchronous joint action for all {n_agents} agents, and output soft-probability JSON."


def api_system_prompt(n_agents):
    return API_SYSTEM_PROMPT.format(n_agents=n_agents)


def api_joint_prompt(n_agents):
    return f'''Blue circles A0..A{n_agents-1} denote agents, green squares F0..F{n_agents-1} denote food, and L denotes level. Rows increase downward and columns increase to the right.
Action indices: 0=NONE (wait), 1=NORTH (up), 2=SOUTH (down), 3=WEST (left), 4=EAST (right), 5=LOAD (collect).
Collection rules: An agent must be in a cell orthogonally adjacent to food (up/down/left/right); diagonal does not count as adjacent. It collects by executing LOAD in place. Within the same time step, the sum of levels of agents immediately adjacent to the same food and executing LOAD must be greater than or equal to the food level for collection to succeed; otherwise LOAD cannot be executed, and agents with sufficient total level should first be organized to approach.
Movement rules: The cell currently occupied by food cannot be entered; move to an empty cell adjacent to the food, not the food cell; avoid multiple agents simultaneously entering the same target cell; do not go out of bounds.
Phased task completion: Each time, first select one target food, find all agents that can reach the cells orthogonally adjacent to it (up/down/left/right), and choose a cooperative set whose level sum is greater than or equal to the food level; also consider distant agents. When all are adjacent and the level sum is sufficient, execute LOAD; when the level sum is insufficient, continue approaching; only non-participants or waiters choose NONE.
Key examples:
1. If food is at (2,2) and an agent is at (2,3): it is already adjacent to the food, and WEST would enter the food cell, which is an illegal action, so its probability must be 0.
2. If A0 is at (4,5) and A1 is at (5,6): A0 and A1 cannot simultaneously choose EAST and NORTH, otherwise both would enter the same cell (4,6).
Decision requirements: In the output, first provide brief reasoning: first identify positions and levels, then select the cooperative set, check movement legality and cooperative LOAD conditions, and finally assign action probabilities. The probability of illegal movement must be 0.
Finally output compact JSON: {{"agents":[{{"id":0,"probs":[six probabilities]}},...]}}. It must include IDs 0..{n_agents-1}; each agent outputs exactly six probabilities in the above action order, with values in [0,1], at most two decimal places, and each row sum strictly equal to 1.'''


class ApiRequestError(RuntimeError):
    def __init__(self, message, status_code=None, response_body=None, transient=False, retry_after=None):
        super().__init__(message)
        self.status_code, self.response_body = status_code, response_body
        self.transient, self.retry_after = transient, retry_after


class ApiCompletionFormatError(ApiRequestError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


API_CONTROL_VERSION = 'requested-off-accept-valid-json-v4'


def off_response_evidence(response):
    """Record returned reasoning honestly; this is not an acceptance gate."""
    message = response['choices'][0].get('message', {})
    tokens = response.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
    if message.get('reasoning_content') or tokens not in (None, 0):
        return 'returned_reasoning'
    return 'reported_zero' if tokens == 0 else 'not_reported'


def completion_text(response):
    """Prefer final content; reasoning-only replies may still contain valid JSON."""
    message = response['choices'][0].get('message', {})
    for source in ('content', 'reasoning_content'):
        value = message.get(source)
        if isinstance(value, str) and value.strip():
            return value, source
    raise ApiCompletionFormatError('API returned no response text; see saved usage/finish_reason', response)


def build_api_cache_identity(config):
    """Generation/input identity; transport, caching and parser labels excluded."""
    from vlm4coop.rendering import RENDERER_VERSION
    operational = {'concurrency', 'pure_vlm_use_cache', 'timeout_seconds',
                   'parser_version', 'failure_policy', 'api_control_version'}
    identity = {key: value for key, value in config.items() if key not in operational}
    identity.update(renderer=RENDERER_VERSION, system_prompt=API_SYSTEM_PROMPT,
                    system_prompt_templates={n: api_system_prompt(n) for n in (3, 4, 5)},
                    prompt_templates={n: api_joint_prompt(n) for n in (3, 4, 5)})
    return identity


class OpenCodeTeacher:
    def __init__(self, config, audit_path, cache=None, purpose='distillation'):
        self.config, self.cache = dict(config), cache
        if purpose not in ('pure','distillation'): raise ValueError(purpose)
        self.purpose = purpose
        self.format_failures = self.format_retries = self.failed_decisions = 0
        self.thinking_control_retries = self.thinking_control_failures = 0
        self.responses_with_reasoning = 0
        self._invalid_cache_keys = set()
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._key = os.environ.get('OPENCODE_API_KEY', '').strip()
        if not self._key:
            raise ValueError('OPENCODE_API_KEY is required (never put credentials in configuration)')
        self.calls = 0
        self.active_requests = self.max_active_requests = 0
        self._io_lock = threading.Lock()
        self.session = 'vlm4coop-' + uuid.uuid4().hex
        # Transport-only policy: preserve the generation/cache identity.
        self.transport_policy = dict(max_attempts=5, initial_backoff_seconds=5,
            maximum_backoff_seconds=60, retry_http_statuses=[429, 500, 502, 503, 504])
        self.audit_path.with_name('transport_policy.json').write_text(
            json.dumps(self.transport_policy, indent=2))
        self.audit_path.with_name('response_policy.json').write_text(json.dumps(dict(
            purpose=purpose,max_format_attempts=3,
            fallback='all_agents_NONE' if purpose=='pure' else 'mask_distillation_only_keep_PPO',
            numeric_policy='strict_probability_validation' if purpose=='pure' else 'clip_each_entry_to_0_1_then_normalize'),indent=2))
        self.control_mode = 'request_off_accept_returned_reasoning'
        self.audit_path.with_name('thinking_control_policy.json').write_text(json.dumps(dict(
            mode=self.control_mode,max_attempts=1,control_retries_enabled=False,
            accept_hidden_reasoning=True,require_valid_probability_JSON=True,
            control_version=API_CONTROL_VERSION),indent=2))

    def _safe(self, value):
        return json.loads(json.dumps(value, ensure_ascii=False).replace(self._key, '[REDACTED]'))

    def request(self, text, image=None, label='request', max_tokens=None, system=None, parameters=None):
        for transport_attempt in range(self.transport_policy['max_attempts']):
            try:
                return self._request_once(text, image, label, max_tokens, system, parameters,
                                          transport_attempt=transport_attempt)
            except ApiRequestError as exc:
                retryable = exc.transient or exc.status_code in self.transport_policy['retry_http_statuses']
                if not retryable or transport_attempt + 1 == self.transport_policy['max_attempts']:
                    raise
                delay = min(self.transport_policy['maximum_backoff_seconds'],
                            self.transport_policy['initial_backoff_seconds'] * 2**transport_attempt)
                # OS randomness must not perturb the student's seeded RNG stream.
                delay += SystemRandom().uniform(0, delay * .25)
                if exc.retry_after is not None:
                    delay = max(delay, exc.retry_after)
                with self._io_lock:
                    with self.audit_path.with_name('transport_retries.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(label=label, failed_attempt=transport_attempt + 1,
                            status_code=exc.status_code, sleep_seconds=delay, at_unix=time.time())) + '\n')
                time.sleep(delay)

    def _request_once(self, text, image=None, label='request', max_tokens=None, system=None,
                      parameters=None, transport_attempt=0):
        content = text
        if image is not None:
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
            content = [{'type': 'text', 'text': text},
                       {'type': 'image_url', 'image_url': {'url': uri}}]
        messages = [{'role': 'user', 'content': content}]
        if system:
            messages.insert(0, {'role': 'system', 'content': system})
        payload = {'model': self.config['model'], 'messages': messages,
                   'stream': False}
        budget = self.config.get('max_tokens') if max_tokens is None else max_tokens
        if budget is not None:
            payload['max_tokens'] = budget
        for name in ('thinking', 'reasoning_effort', 'temperature', 'top_p'):
            if name in self.config:
                payload[name] = self.config[name]
        for name, value in (parameters or {}).items():
            if value is None:
                payload.pop(name, None)
            else:
                payload[name] = value
        # No API key or image base64 enters the persisted request record.
        event = {'label': label, 'model_requested': payload['model'],
                 'transport_attempt': transport_attempt, 'started_at_unix': time.time(),
                 'endpoint': self.config['endpoint'], 'session_id': self.session, 'image_count': int(image is not None),
                 'prompt': text, 'system': system,
                 'thinking': payload.get('thinking', 'provider_default'),
                 'reasoning_effort': payload.get('reasoning_effort', 'provider_default'),
                 'temperature_requested': payload.get('temperature'),
                 'top_p_requested': payload.get('top_p'),
                 'max_tokens': payload.get('max_tokens'), 'stream': False}
        headers = {'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json',
                   'User-Agent': 'VLM4Coop-Research/0.1', 'x-opencode-session': self.session}
        started = time.monotonic()
        with self._io_lock:
            self.calls += 1
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
            event["call_index"] = self.calls
        try:
            request = Request(self.config['endpoint'], json.dumps(payload).encode(), headers=headers)
            with urlopen(request, timeout=self.config.get('timeout_seconds', 120)) as response:
                event['http_status'] = response.status
                body = response.read().decode('utf-8', errors='replace')
                event['request_id'] = response.headers.get('x-request-id')
            result = json.loads(body)
            event['response'] = result
            if result.get('error') or not result.get('choices'):
                raise ApiRequestError('API returned no completion choices')
            if off_response_evidence(result)=='returned_reasoning':
                with self._io_lock: self.responses_with_reasoning += 1
            answer,source = completion_text(result)
            event['answer_source'] = source
            event['valid_transport_completion'] = True
            return answer, result
        except HTTPError as exc:
            event.update(http_status=exc.code, error_body=exc.read().decode('utf-8', errors='replace'))
            retry_after = None
            try:
                retry_after = max(0, float(exc.headers.get('Retry-After')))
            except (TypeError, ValueError, AttributeError):
                pass
            raise ApiRequestError(f'API HTTP {exc.code}; see redacted request audit',
                                  exc.code, self._safe(event['error_body']), retry_after=retry_after) from None
        except (URLError, TimeoutError) as exc:
            event['error'] = str(exc)
            raise ApiRequestError('API transport failure; see redacted request audit', transient=True) from None
        except json.JSONDecodeError as exc:
            event['error'] = str(exc)
            raise ApiRequestError('API JSON failure; see redacted request audit') from None
        except ApiRequestError as exc:
            event['error'] = str(exc)
            raise
        finally:
            event['latency_seconds'] = time.monotonic() - started
            event['completed_at_unix'] = time.time()
            with self._io_lock:
                self.active_requests -= 1
                with self.audit_path.open('a') as stream:
                    stream.write(json.dumps(self._safe(event), ensure_ascii=False, allow_nan=False) + '\n')

    def lookup_cache(self, key, n_agents):
        cached = self.cache.get(key)
        if cached is None: return None
        try:
            # Parser/failure policy is intentionally absent from the key. Read
            # the original model text under the current numeric rule instead
            # of reusing an old, differently normalized probability array.
            cached.probs,cached.repaired = parse_api_probabilities(cached.raw,n_agents,self.purpose)
        except ProbabilityFormatError as exc:
            self.cache.hits -= 1; self.cache.misses += 1
            self._invalid_cache_keys.add(key)
            with self.audit_path.with_name('cache_rejections.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(key=key,raw=cached.raw,old_probabilities=cached.probs.tolist(),error=str(exc)))+'\n')
            return None
        cached.format_attempts = 0
        return cached

    def store_cache(self, key, result):
        if result.source=='teacher':
            self.cache.put(key,result,replace=key in self._invalid_cache_keys)
            self._invalid_cache_keys.discard(key)

    def _policy_response(self, image, n_agents, decision_id, format_attempt):
        # OFF is a request preference. Returned reasoning never spends another
        # request by itself; only the probability parser decides usability.
        raw,response=self.request(api_joint_prompt(n_agents),image=image,
            system=api_system_prompt(n_agents),
            label=f'policy_{n_agents}p_{decision_id}_format{format_attempt}_control0')
        reasoning=response.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')
        return raw,response,reasoning,0

    def predict(self, image, n_agents, use_cache=True):
        key=self.cache.key(image,n_agents) if self.cache is not None else None
        if use_cache and self.cache is not None:
            cached=self.lookup_cache(key,n_agents)
            if cached is not None: return cached
        began=time.monotonic()
        decision_id=uuid.uuid4().hex
        raw=''
        for format_attempt in range(1,4):
            if format_attempt>1:
                with self._io_lock: self.format_retries+=1
            try:
                raw,response,reasoning,control_attempt=self._policy_response(image,n_agents,decision_id,format_attempt)
                probs,repaired=parse_api_probabilities(raw,n_agents,self.purpose)
            except (ProbabilityFormatError,ApiCompletionFormatError) as exc:
                if isinstance(exc,ApiCompletionFormatError): raw=''
                with self._io_lock:
                    self.format_failures+=1
                    with self.audit_path.with_name('format_failures.jsonl').open('a') as stream:
                        stream.write(json.dumps(self._safe(dict(decision_id=decision_id,
                            format_attempt=format_attempt,purpose=self.purpose,error=str(exc),raw=raw)))+'\n')
                continue
            result=TeacherResult(probs,raw,repaired,time.monotonic()-began,
                response.get('usage',{}).get('completion_tokens',0),
                format_attempts=format_attempt,decision_id=decision_id)
            accepted=dict(response_id=response.get('id'),model_returned=response.get('model'),
                requested_thinking=self.config.get('thinking'),reasoning_tokens=reasoning,
                thinking_evidence=(off_response_evidence(response)
                    if self.config.get('thinking',{}).get('type')=='disabled' else 'off_not_requested'),
                control_attempt=control_attempt,control_version=API_CONTROL_VERSION,
                control_mode=self.control_mode,answer_source=completion_text(response)[1],
                decision_id=decision_id,format_attempts=format_attempt,purpose=self.purpose,
                key=key,probabilities=probs.tolist(),repaired=repaired)
            with self._io_lock:
                with self.audit_path.with_name('teacher_labels.jsonl').open('a') as stream:
                    stream.write(json.dumps(self._safe(accepted),allow_nan=False)+'\n')
            if use_cache and self.cache is not None: self.store_cache(key,result)
            return result
        with self._io_lock:
            self.failed_decisions+=1
            with self.audit_path.with_name('teacher_fallbacks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(decision_id=decision_id,purpose=self.purpose,
                    format_attempts=3,source='format_failure'))+'\n')
        # The caller has the experiment role and resolves this
        # explicit failure. It is not a VLM label and must never enter SQLite.
        return TeacherResult(None,raw,False,time.monotonic()-began,source='format_failure',
                             format_attempts=3,decision_id=decision_id)

    def predict_many(self, images, n_agents, use_cache=True):
        from vlm4coop.api_batch import predict_many
        return predict_many(self, images, n_agents, use_cache)
