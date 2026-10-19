from pathlib import Path
p=Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s=p.read_text()
s=s.replace("class ApiRequestError(RuntimeError):\n    pass", "class ApiRequestError(RuntimeError):\n    def __init__(self, message, status_code=None, response_body=None):\n        super().__init__(message)\n        self.status_code, self.response_body = status_code, response_body\n\n\nAPI_CONTROL_VERSION = 'verified-off-none-native-v1'")
s=s.replace("max_tokens=None, system=None):", "max_tokens=None, system=None, parameters=None):")
s=s.replace("        # No API key or image base64 enters the persisted request record.", "        for name, value in (parameters or {}).items():\n            if value is None:\n                payload.pop(name, None)\n            else:\n                payload[name] = value\n        # No API key or image base64 enters the persisted request record.")
s=s.replace("'endpoint': self.config['endpoint'], 'image_count':", "'endpoint': self.config['endpoint'], 'session_id': self.session, 'image_count':")
s=s.replace("raise ApiRequestError(f'API HTTP {exc.code}; see redacted request audit') from None", "raise ApiRequestError(f'API HTTP {exc.code}; see redacted request audit',\n                                  exc.code, self._safe(event['error_body'])) from None")
old="""        raw, response = self.request(api_joint_prompt(n_agents), image=image,
                                     system=API_SYSTEM_PROMPT, label=f'policy_{n_agents}p')
        probs, repaired = parse_probabilities(raw, n_agents)
"""
new="""        disabled = self.config.get('thinking', {}).get('type') == 'disabled'
        alternatives = [None]
        if disabled:
            # Real gateway evidence: some routes reject effort=none, whereas
            # others ignore the native thinking field. Try at most two distinct
            # documented controls and accept only an observed zero-reasoning answer.
            alternatives.append({'reasoning_effort': None if self.config.get('reasoning_effort') == 'none' else 'none'})
        for attempt, parameters in enumerate(alternatives):
            try:
                raw, response = self.request(api_joint_prompt(n_agents), image=image,
                    system=API_SYSTEM_PROMPT, label=f'policy_{n_agents}p_control{attempt}',
                    parameters=parameters)
            except ApiRequestError as exc:
                unsupported = (exc.status_code == 400 and
                    'none reasoning effort is not supported' in (exc.response_body or '').lower())
                if disabled and attempt == 0 and unsupported:
                    continue
                raise
            message = response['choices'][0]['message']
            reasoning = response.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
            if disabled and (reasoning != 0 or message.get('reasoning_content')):
                if attempt == 0:
                    continue
                raise ApiRequestError('Could not verify thinking disabled using either gateway control')
            break
        else:
            raise ApiRequestError('No usable policy response')
        probs, repaired = parse_probabilities(raw, n_agents)
        accepted = dict(response_id=response.get('id'), model_returned=response.get('model'),
            requested_thinking=self.config.get('thinking'), reasoning_tokens=reasoning,
            control_attempt=attempt, control_version=API_CONTROL_VERSION,
            key=self.cache.key(image, n_agents) if self.cache is not None else None,
            probabilities=probs.tolist(), repaired=repaired)
        with self.audit_path.with_name('teacher_labels.jsonl').open('a') as stream:
            stream.write(json.dumps(self._safe(accepted), allow_nan=False) + '\\n')
"""
assert old in s
s=s.replace(old,new)
p.write_text(s)
p=Path('VLM4Coop/src/vlm4coop/training.py')
s=p.read_text().replace('OpenCodeTeacher, api_joint_prompt, API_SYSTEM_PROMPT', 'OpenCodeTeacher, api_joint_prompt, API_SYSTEM_PROMPT, API_CONTROL_VERSION')
s=s.replace('identity = dict(config["teacher"], renderer=RENDERER_VERSION,\n                        system_prompt=API_SYSTEM_PROMPT,', 'identity = dict(config["teacher"], renderer=RENDERER_VERSION,\n                        api_control_version=API_CONTROL_VERSION, system_prompt=API_SYSTEM_PROMPT,')
p.write_text(s)
p=Path('VLM4Coop/scripts/api_kd_smoke.py')
s=p.read_text()
s=s.replace("        events = [json.loads(row) for row in (output / 'teacher_calls.jsonl').read_text().splitlines()]\n        assert all(event['response']['usage']['completion_tokens_details']['reasoning_tokens'] == 0\n                   for event in events), 'Provider did not honor explicit none'", "        accepted = [json.loads(row) for row in (output / 'teacher_labels.jsonl').read_text().splitlines()]\n        assert len(accepted) == 8 and all(row['reasoning_tokens'] == 0 for row in accepted), 'Accepted labels did not verify thinking disabled'")
p.write_text(s)
