from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/teacher.py')
s = p.read_text()
start = s.index('def joint_prompt(')
end = s.index('\n\n@dataclass', start)
s = s[:start] + '''SYSTEM_PROMPT = "You are a concise cooperative game controller. Spend reasoning on joint action selection. Do not narrate the image or repeat rules. Finish with the requested probability JSON."


def joint_prompt(n_agents):
    return f"""Choose one simultaneous joint move for all {n_agents} agents in the attached LBF grid.
Blue circles A0..A{n_agents-1} are agents; green squares F are food; L is level. Rows increase down, columns right.
Action indices: 0=NONE(wait), 1=NORTH(up), 2=SOUTH(down), 3=WEST(left), 4=EAST(right), 5=LOAD.
LOAD works only for agents immediately adjacent to the SAME food, and ONLY if the sum of levels of those adjacent agents choosing LOAD simultaneously reaches its level.
A distant agent cannot LOAD food. An agent alone cannot load food above its level. Move partners into adjacent cells before loading together. Avoid colliding moves.
The goal is to collect ALL food efficiently over future steps, not to collect it immediately this step. Choose useful approaches if no food can be collected now.
Use your brief reasoning to decide actions for the whole team, WITHOUT transcribing the grid or re-explaining these rules.
Then return ONLY one compact JSON object: {{"agents":[{{"id":0,"probs":[six decimal numbers]}},...]}}.
Include exactly IDs 0..{n_agents-1}, exactly six probabilities each in the action order above. Each value must be in [0,1], each row sum exactly 1; never output an all-zero row. Express your confidence over the available moves."""
''' + s[end:]
s = s.replace('''messages = [{"role": "user",''', '''messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user",''')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/batched_teacher.py')
s = p.read_text().replace('TeacherResult, joint_prompt, parse_probabilities', 'TeacherResult, joint_prompt, parse_probabilities, SYSTEM_PROMPT')
s = s.replace("messages = [[{'role': 'user', 'content': [", "messages = [[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': [")
p.write_text(s)
p = Path('VLM4Coop/configs/experiments.yaml')
s = p.read_text().replace('joint-lbf-v3-budget256', 'joint-lbf-v4-budget256')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/training.py')
s = p.read_text().replace('LocalQwenTeacher, joint_prompt', 'LocalQwenTeacher, joint_prompt, SYSTEM_PROMPT')
s = s.replace('identity = dict(config["teacher"], renderer=RENDERER_VERSION,', 'identity = dict(config["teacher"], renderer=RENDERER_VERSION, system_prompt=SYSTEM_PROMPT,')
p.write_text(s)
