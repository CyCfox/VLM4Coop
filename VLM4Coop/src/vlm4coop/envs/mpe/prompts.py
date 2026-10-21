"""User-specified MPE three-agent prompts; probabilities stay in native order."""
ACTIONS=('NONE','LEFT','RIGHT','DOWN','UP')
PROMPT_VERSION='mpe-spread-3p-user-exact-cooperation-v6'
SYSTEM_TEMPLATE='You are a concise MPE multi-agent cooperative controller. Based on the attached figure, select a synchronous joint action for all 3 agents and output soft-probability JSON.'
USER_TEMPLATE='''Image description: A0..A2 denote agents (A0 red, A1 orange, A2 blue), and G0..G2 denote green stationary landmarks. World x increases to the right, and y increases upward. The thick white arrow of each agent indicates the current velocity direction; the arrow length does not represent speed magnitude; the label V indicates speed magnitude (world units/second). When velocity is zero, it is shown as a dot. Agents and landmarks may occlude each other; when a landmark overlaps with an agent, coverage should be maintained, so if a numbered landmark is not displayed, it means it has been covered. Landmark numbering does not represent a fixed agent assignment relationship.
Action indices: 0=NONE (no thrust), 1=LEFT (thrust to the left), 2=RIGHT (thrust to the right), 3=DOWN (thrust downward), 4=UP (thrust upward).
Physics rules: Actions apply thrust rather than direct movement. Time step dt=0.1, velocity damping=0.25, mass=1, thrust magnitude=5. Position is first updated according to current velocity, and the next velocity is 0.75×current velocity+0.1×total applied force (action thrust and collision force). NONE does not stop immediately; the agent will still move due to inertia. When approaching a landmark, it should decelerate in advance, and if necessary apply reverse thrust for braking to avoid overshooting the target.
Cooperation rules: Reasonably assign different landmarks to different agents, prioritize reducing the sum of distances from all landmarks to their nearest agents, and avoid multiple agents concentrating on the same landmark. Agents that have covered a landmark should try to maintain coverage, and other agents should continue toward uncovered landmarks.
Movement rules: Landmarks can be entered and are not obstacles; agents will collide with each other, so they should avoid getting too close to each other, and actions cannot be decided based only on the relative positions of targets.
Decision requirements: Prioritize reducing the sum of distances from all landmarks to their nearest agents, improve final landmark coverage, and reduce collisions. First briefly reason in the output, identify agent positions, velocities, and target assignments, then judge whether acceleration, braking, or steering is needed, and finally assign action probabilities. Finally output compact JSON: {{"agents":[{{"id":0,"probs":[five probabilities]}},...]}}. Must include ID 0..2; each agent outputs exactly five probabilities in the above action order, with values in [0,1], at most two decimal places, and each row's sum strictly equals 1.'''


def system_prompt(n_agents):
    if n_agents!=3: raise ValueError('The user-specified MPE teacher requires exactly 3 agents')
    return SYSTEM_TEMPLATE


def joint_prompt(n_agents):
    if n_agents!=3: raise ValueError('The user-specified MPE teacher requires exactly 3 agents')
    return USER_TEMPLATE.format()


def cache_identity():
    from vlm4coop.envs.mpe.rendering import RENDERER_VERSION
    return dict(environment='pettingzoo-1.24.3-simple_spread_v3',renderer=RENDERER_VERSION,
        action_order=ACTIONS,system_prompt=SYSTEM_TEMPLATE,
        system_prompt_templates={3:system_prompt(3)},prompt_templates={3:joint_prompt(3)},
        dynamics=dict(dt=.1,damping=.25,thrust=5,position_before_velocity=True),
        observation='silent_comm_removed_14d',local_ratio=.5,episode_limit=25)
