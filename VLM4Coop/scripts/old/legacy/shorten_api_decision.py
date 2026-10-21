from pathlib import Path
p=Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s=p.read_text()
marker='\n\nclass ApiRequestError(RuntimeError):'
insert='''

API_SYSTEM_PROMPT = "You are a cooperative grid-game controller. Reason briefly, then output the requested JSON."


def api_joint_prompt(n_agents):
    return f''' + "'''" + '''Choose ONLY the next simultaneous move for {n_agents} LBF agents in this image. Use at most 128 words of reasoning; do not search or compare complete episode plans.
Blue circles A0..A{n_agents-1} are agents. Green F squares are food. L is level. Row labels at left increase downward; column labels at top increase rightward.
Actions in order: 0=NONE, 1=NORTH, 2=SOUTH, 3=WEST, 4=EAST, 5=LOAD.
Agents cannot move outside the grid, onto food, or into occupied cells. LOAD succeeds only if agents immediately adjacent to the same food choose LOAD together with summed levels at least its level. Approach food or a needed partner when loading is currently impossible. Avoid collisions. The goal is to eventually collect all food.
Commit to useful moves now. Return only JSON {{"agents":[{{"id":0,"probs":[six numbers]}},...]}} with every ID 0..{n_agents-1}. Each row has six probabilities in [0,1] summing to 1, at most three decimals.''' + "'''" + '''
'''
assert marker in s
s=s.replace(marker,insert+marker)
s=s.replace('self.request(joint_prompt(n_agents), image=image,\n                                     system=SYSTEM_PROMPT,', 'self.request(api_joint_prompt(n_agents), image=image,\n                                     system=API_SYSTEM_PROMPT,')
p.write_text(s)
p=Path('VLM4Coop/src/vlm4coop/training.py')
s=p.read_text().replace('from vlm4coop.api_teacher import OpenCodeTeacher\n        identity', 'from vlm4coop.api_teacher import OpenCodeTeacher, api_joint_prompt, API_SYSTEM_PROMPT\n        identity')
s=s.replace('system_prompt=SYSTEM_PROMPT,\n                        prompt_templates={n: joint_prompt(n)', 'system_prompt=API_SYSTEM_PROMPT,\n                        prompt_templates={n: api_joint_prompt(n)')
p.write_text(s)
