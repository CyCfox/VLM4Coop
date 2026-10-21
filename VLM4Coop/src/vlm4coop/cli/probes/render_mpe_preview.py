"""Save a genuine stepped MPE velocity screenshot, without actor training or API."""
import argparse
import json
from pathlib import Path
from PIL import Image
from vlm4coop.envs.mpe.adapter import MPEGymBridge
from vlm4coop.envs.mpe.rendering import render_geometry,RENDERER_VERSION
from vlm4coop.utils.paths import PROJECT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=19)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--actions',type=int,nargs=3,choices=range(5),default=[1,2,4],
                   help='Repeated native joint actions for A0 A1 A2 (3=DOWN, 4=UP)')
    args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(PROJECT.resolve()): raise ValueError('Output must remain in VLM4Coop')
    if not 0<=args.steps<=25: raise ValueError('Steps must be in [0,25]')
    output.mkdir(parents=True,exist_ok=False)
    env=MPEGymBridge()
    try:
        env.reset(seed=args.seed)
        for _ in range(args.steps): env.step(args.actions)
        Image.fromarray(env.render()).save(output/'mpe_velocity_rgb.png')
        metadata=dict(seed=args.seed,steps=args.steps,joint_actions=args.actions,
            renderer=RENDERER_VERSION,geometry=render_geometry(env.world),
            agents=[dict(id=i,position=a.state.p_pos.tolist(),velocity=a.state.p_vel.tolist(),
                         speed=float((a.state.p_vel**2).sum()**.5)) for i,a in enumerate(env.world.agents)],
            landmarks=[dict(id=i,position=g.state.p_pos.tolist()) for i,g in enumerate(env.world.landmarks)],
            API_requests=0,formal=False)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(dict(status='saved',image=str(output/'mpe_velocity_rgb.png'),API_requests=0)))
    finally: env.close()


if __name__=='__main__': main()
