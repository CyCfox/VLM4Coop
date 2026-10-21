"""Frozen LBF action order and verbatim local/API prompts."""
ACTIONS = ("NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD")


SYSTEM_PROMPT = "You are a concise cooperative game controller. Spend reasoning on joint action selection. Do not narrate the image or repeat rules. Finish with the requested probability JSON."


def joint_prompt(n_agents):
    return f"""Choose one simultaneous joint move for all {n_agents} agents in the attached LBF grid.
Blue circles A0..A{n_agents-1} are agents; green squares F are food; L is level. Rows increase down, columns right.
Action indices: 0=NONE(wait), 1=NORTH(up), 2=SOUTH(down), 3=WEST(left), 4=EAST(right), 5=LOAD.
LOAD works only for agents immediately adjacent to the SAME food, and ONLY if the sum of levels of those adjacent agents choosing LOAD simultaneously reaches its level.
A distant agent cannot LOAD food. An agent alone cannot load food above its level. Move partners into adjacent cells before loading together. Avoid colliding moves.
The goal is to collect ALL food efficiently over future steps, not to collect it immediately this step. Choose useful approaches if no food can be collected now.
Use your brief reasoning to decide actions for the whole team, WITHOUT transcribing the grid or re-explaining these rules.
Then return ONLY one compact JSON object: {{"agents":[{{"id":0,"probs":[six decimal numbers]}},...]}}.
Include exactly IDs 0..{n_agents-1}, exactly six probabilities each in the action order above. Each value must be in [0,1], with at most three decimal places; each row sum exactly 1, never all zero.
Example of syntax ONLY (choose your own probabilities): {{"id":0,"probs":[0.05,0.15,0.1,0.1,0.5,0.1]}}.
Express confidence over the available moves; values are not action numbers."""


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

