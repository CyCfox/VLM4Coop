from pathlib import Path
p=Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s=p.read_text()
s=s.replace("'max_tokens': max_tokens or self.config['max_tokens'], 'stream': False}", "'stream': False}\n        budget = self.config.get('max_tokens') if max_tokens is None else max_tokens\n        if budget is not None:\n            payload['max_tokens'] = budget")
s=s.replace("'max_tokens': payload['max_tokens'], 'stream': False}", "'max_tokens': payload.get('max_tokens'), 'stream': False}")
s=s.replace('You are a cooperative grid-game controller. Reason briefly, then output the requested JSON.', 'You are a cooperative grid-game controller. Consider the team together and return the requested final JSON when ready.')
s=s.replace('Choose ONLY the next simultaneous move for {n_agents} LBF agents in this image. Use at most 128 words of reasoning; do not search or compare complete episode plans.', 'Choose the next simultaneous move for {n_agents} LBF agents in this image. Consider how all agents can cooperate to collect the food.')
p.write_text(s)
