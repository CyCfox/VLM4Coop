"""Transport contract tests use test credentials and mocked HTTP, never an API."""
import io
import json
from urllib.error import HTTPError
import pytest
from PIL import Image
from vlm4coop.api_teacher import OpenCodeTeacher, ApiRequestError


def make_teacher(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENCODE_API_KEY', 'TEST-CREDENTIAL-ONLY')
    return OpenCodeTeacher({'endpoint': 'https://example.invalid/chat/completions',
                            'model': 'test-model', 'max_tokens': 256}, tmp_path / 'audit.jsonl')


def test_image_contract_and_no_credentials_in_audit(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    def fake_open(request, timeout):
        assert request.get_header('Authorization') == 'Bearer TEST-CREDENTIAL-ONLY'
        payload = json.loads(request.data)
        assert payload['model'] == 'test-model' and not payload['stream']
        content = payload['messages'][-1]['content']
        assert len(content) == 2 and content[0] == {'type': 'text', 'text': 'read image'}
        assert content[1]['image_url']['url'].startswith('data:image/png;base64,')
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'model': 'actual-test-model', 'choices': [
                    {'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}).encode()
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    text, response = teacher.request('read image', Image.new('RGB', (5, 5)))
    assert text == 'OK' and response['model'] == 'actual-test-model'
    audit = teacher.audit_path.read_text()
    assert 'TEST-CREDENTIAL-ONLY' not in audit and 'base64' not in audit
    assert json.loads(audit)['image_count'] == 1


def test_http_failure_preserves_reason_and_redacts_echoed_secret(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    def fake_open(request, timeout):
        raise HTTPError(request.full_url, 403, 'Forbidden', {},
                        io.BytesIO(b'{"error":"FreeTierError TEST-CREDENTIAL-ONLY"}'))
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    with pytest.raises(ApiRequestError, match='403'):
        teacher.request('ping')
    audit = teacher.audit_path.read_text()
    assert 'FreeTierError' in audit and 'TEST-CREDENTIAL-ONLY' not in audit
    assert json.loads(audit)['http_status'] == 403 and teacher.calls == 1


def test_no_final_content_is_not_a_success(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    class Response:
        status, headers = 200, {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return b'{"choices":[{"message":{"content":null,"reasoning_content":"thinking"},"finish_reason":"length"}]}'
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(ApiRequestError, match='no final text'):
        teacher.request('ping')
    assert 'valid_transport_completion' not in json.loads(teacher.audit_path.read_text())


def test_native_thinking_options_are_sent_and_audited(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config.update(thinking={'type': 'enabled'}, reasoning_effort='low')
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        assert payload['thinking'] == {'type': 'enabled'} and payload['reasoning_effort'] == 'low'
        assert 'temperature' not in payload  # provider ignores it in thinking mode
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"OK"}}]}'
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    teacher.request('ping')
    event = json.loads(teacher.audit_path.read_text())
    assert event['reasoning_effort'] == 'low' and event['thinking']['type'] == 'enabled'


def test_provider_budget_omits_client_token_and_effort_limits(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config.update(max_tokens=None, thinking={'type': 'enabled'})
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        assert 'max_tokens' not in payload and 'reasoning_effort' not in payload
        assert payload['thinking'] == {'type': 'enabled'}
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"OK"}}]}'
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    teacher.request('ping')
    assert json.loads(teacher.audit_path.read_text())['max_tokens'] is None


def test_probe_returns_nonzero_and_keeps_failure_record(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    import sys
    script = Path(__file__).resolve().parents[1] / 'scripts/api_teacher_probe.py'
    spec = importlib.util.spec_from_file_location('api_probe_test_module', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'results').mkdir()
    (tmp_path / 'configs/api_probe.yaml').write_text('model: test-only\n')
    class UnavailableTeacher:
        calls = 1
        def __init__(self, *args): pass
        def request(self, *args, **kwargs): raise ApiRequestError('test network unavailable')
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    monkeypatch.setattr(module, 'OpenCodeTeacher', UnavailableTeacher)
    monkeypatch.setattr(sys, 'argv', [str(script), '--episodes', '1'])
    assert module.run() == 1
    summaries = list((tmp_path / 'results').glob('*/summary.json'))
    result = json.loads(summaries[0].read_text())
    assert result['status'] == 'failed' and result['failed_phase'] == 'connectivity'
    assert result['formal'] is False and 'episodes' not in result


def test_verified_off_falls_back_after_real_unsupported_none_error(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config.update(thinking={'type': 'disabled'}, reasoning_effort='none')
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        if payload.get('reasoning_effort') == 'none':
            raise HTTPError(request.full_url, 400, 'Bad Request', {}, io.BytesIO(
                b'{"error":{"message":"none reasoning effort is not supported by deepseek-v4.1-flash model"}}'))
        assert payload['thinking']['type'] == 'disabled' and 'reasoning_effort' not in payload
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                answer = json.dumps({'agents': [{'id': i, 'probs': [.1,.1,.1,.1,.1,.5]} for i in range(3)]})
                return json.dumps({'id': 'accepted-test-response', 'choices': [{'message': {'content': answer}}],
                    'usage': {'completion_tokens': 90, 'completion_tokens_details': {'reasoning_tokens': 0}}}).encode()
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    result = teacher.predict(Image.new('RGB', (5,5)), 3, use_cache=False)
    assert result.probs.shape == (3,6) and teacher.calls == 2
    label = json.loads((tmp_path / 'teacher_labels.jsonl').read_text())
    assert label['reasoning_tokens'] == 0 and label['control_attempt'] == 1
    assert label['response_id'] == 'accepted-test-response'


def test_off_policy_never_accepts_emitted_reasoning(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config.update(thinking={'type': 'disabled'}, reasoning_effort='none')
    class Response:
        status, headers = 200, {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return b'{"choices":[{"message":{"content":"{}","reasoning_content":"still thinking"}}],"usage":{"completion_tokens_details":{"reasoning_tokens":10}}}'
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(ApiRequestError, match='verify thinking disabled'):
        teacher.predict(Image.new('RGB', (5,5)), 3, use_cache=False)
    assert teacher.calls == 2 and not (tmp_path / 'teacher_labels.jsonl').exists()


def test_off_policy_accepts_visible_reasoning_and_expands_system_agent_count(monkeypatch, tmp_path):
    from vlm4coop.api_teacher import api_system_prompt
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config.update(thinking={'type': 'disabled'})
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        assert payload['thinking'] == {'type':'disabled'} and 'reasoning_effort' not in payload
        assert payload['messages'][0]['content'] == api_system_prompt(3)
        assert '{n_agents}' not in payload['messages'][0]['content']
        answer = 'A0 and A1 are adjacent; choose LOAD. A2 waits.\n' + json.dumps({'agents':[
            {'id':i, 'probs':[0,0,0,0,0,1]} for i in range(3)]})
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'id':'visible-reasoning-test', 'choices':[{'message':{'content':answer}}],
                    'usage':{'completion_tokens_details':{'reasoning_tokens':0}}}).encode()
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    result = teacher.predict(Image.new('RGB',(5,5)),3,use_cache=False)
    assert result.probs.argmax(-1).tolist()==[5,5,5]
    assert result.raw.startswith('A0 and A1') and not result.repaired


def test_503_retries_are_bounded_and_preserve_student_rng(monkeypatch, tmp_path):
    import random
    teacher = make_teacher(monkeypatch, tmp_path)
    calls, sleeps = [], []
    def fake_open(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError(request.full_url, 503, 'Unavailable', {}, io.BytesIO(b''))
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"OK"}}]}'
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    monkeypatch.setattr('vlm4coop.api_teacher.time.sleep', sleeps.append)
    random.seed(2026); before = random.getstate()
    assert teacher.request('ping')[0] == 'OK'
    assert random.getstate() == before
    assert len(calls) == 3 and len(sleeps) == 2
    events = [json.loads(x) for x in teacher.audit_path.read_text().splitlines()]
    assert [x['http_status'] for x in events] == [503, 503, 200]
    assert teacher.active_requests == 0
    assert len(teacher.audit_path.with_name('transport_retries.jsonl').read_text().splitlines()) == 2


def test_persistent_503_stops_after_five_attempts(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    def fail(request, timeout):
        raise HTTPError(request.full_url, 503, 'Unavailable', {}, io.BytesIO(b''))
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fail)
    monkeypatch.setattr('vlm4coop.api_teacher.time.sleep', lambda delay: None)
    with pytest.raises(ApiRequestError, match='503'):
        teacher.request('ping')
    assert teacher.calls == 5 and teacher.active_requests == 0


def test_top_p_is_sent_and_audited(monkeypatch, tmp_path):
    teacher = make_teacher(monkeypatch, tmp_path)
    teacher.config['top_p'] = .95
    def fake_open(request, timeout):
        assert json.loads(request.data)['top_p'] == .95
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"OK"}}]}'
        return Response()
    monkeypatch.setattr('vlm4coop.api_teacher.urlopen', fake_open)
    teacher.request('ping')
    assert json.loads(teacher.audit_path.read_text())['top_p_requested'] == .95


def test_api_cache_identity_ignores_operations_but_separates_top_p(tmp_path):
    from copy import deepcopy
    from vlm4coop.api_teacher import build_api_cache_identity
    from vlm4coop.teacher import ProbabilityCache
    config = dict(endpoint='https://example.invalid/chat/completions', model='test',
        top_p=.95, concurrency=100, pure_vlm_use_cache=True, timeout_seconds=600,
        parser_version='v1', failure_policy='repair', api_control_version='old')
    identity = build_api_cache_identity(config)
    excluded = ('concurrency', 'pure_vlm_use_cache', 'timeout_seconds',
                'parser_version', 'failure_policy', 'api_control_version')
    assert not set(excluded).intersection(identity)
    changed = deepcopy(config)
    changed.update(concurrency=1000, pure_vlm_use_cache=False, timeout_seconds=30,
                   parser_version='new', failure_policy='new', api_control_version='new')
    assert build_api_cache_identity(changed) == identity
    image = Image.new('RGB', (2, 2), 'red')
    a = ProbabilityCache(tmp_path/'a.sqlite', identity)
    b = ProbabilityCache(tmp_path/'b.sqlite', build_api_cache_identity(changed))
    assert a.key(image, 3) == b.key(image, 3)
    changed['top_p'] = .90
    c = ProbabilityCache(tmp_path/'c.sqlite', build_api_cache_identity(changed))
    assert a.key(image, 3) != c.key(image, 3)
    for cache in (a, b, c): cache.close()
