from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s = p.read_text()
s = s.replace("'messages': messages, 'temperature': 0,", "'messages': messages,")
s = s.replace("        # No API key or image base64 enters the persisted request record.", "        for name in ('thinking', 'reasoning_effort'):\n            if name in self.config:\n                payload[name] = self.config[name]\n        # No API key or image base64 enters the persisted request record.")
s = s.replace("'prompt': text, 'system': system, 'temperature': 0,", "'prompt': text, 'system': system,\n                 'thinking': payload.get('thinking', 'provider_default'),\n                 'reasoning_effort': payload.get('reasoning_effort', 'provider_default'),")
p.write_text(s)
