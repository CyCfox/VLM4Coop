"""Standalone interactive replay from real completed states and unmodified RGBs."""
import argparse
import base64
import json
from pathlib import Path
import tarfile
import yaml
from PIL import Image
from vlm4coop.envs.lbf.adapter import register_lbf, CoopGymmaWrapper

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--seed', type=int, default=1200)
args = parser.parse_args()
root = args.run
validated = json.loads((root / 'replay_validation.json').read_text())
assert any(row['seed'] == args.seed for row in validated['completed_episodes_replayed'])
steps = [json.loads(x) for x in (root / 'steps.jsonl').read_text().splitlines()
         if json.loads(x)['seed'] == args.seed]
with tarfile.open(root / 'source_snapshot.tar.gz') as archive:
    config = yaml.safe_load(archive.extractfile('configs/experiments.yaml').read())
key = register_lbf(3, config['environment'])
env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                      common_reward=True, reward_scalarisation='sum')
try:
    env.reset(seed=args.seed)
    for row in steps:
        env.step(row['actions'])
    final = root / f'episode_{args.seed}_terminal.png'
    Image.fromarray(env.render()).save(final)
finally:
    env.close()
frames, total = [], 0.
for t, row in enumerate(steps):
    path = root / f'episode_{args.seed}_step_{t:03d}.png'
    frames.append({'src': 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode(),
                   'step': t, 'return': total, 'remaining': len(row['before']['food']),
                   'actions': row['actions'], 'repaired': row['repaired']})
    total += row['reward']
frames.append({'src': 'data:image/png;base64,' + base64.b64encode(final.read_bytes()).decode(),
               'step': len(steps), 'return': total, 'remaining': len(steps[-1]['after']['food']),
               'actions': None, 'repaired': False})
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LBF 真实回合回放</title>
<style>body{font:16px system-ui,sans-serif;max-width:740px;margin:30px auto;padding:0 18px;background:#f5f7fa;color:#172332}h1{font-size:26px}p{line-height:1.6}img{display:block;width:min(100%,588px);height:auto;background:white;border:1px solid #d6dce3;margin:15px 0}button{padding:8px 18px;margin-right:12px;font:inherit}input{width:65%;vertical-align:middle}#state{font-variant-numeric:tabular-nums}small{color:#526174}</style>
<h1>DeepSeek · LBF 真实回合回放</h1>
<p id="intro"></p><p>图片是实际模型输入；末帧由真实环境重放生成。逐步状态、图片、动作、奖励与终止条件均已重放核对。</p>
<img id="frame" alt="真实LBF环境状态"><p id="state"></p><p id="action"></p>
<button id="play">播放</button><input id="seek" type="range" min="0" value="0" step="1" aria-label="环境步数">
<p><small>蓝色圆形：Agent；绿色方形：Food；L：等级。此为小规模能力测试，不属于正式论文实验。</small></p>
<script>
const frames=FRAME_DATA;
const names=['NONE','NORTH','SOUTH','WEST','EAST','LOAD'];
const slider=document.getElementById('seek');slider.max=frames.length-1;
let timer=null;
document.getElementById('intro').textContent='deepseek-v4.1-flash · seed SEED_VALUE · '+(frames.length-1)+' 步 · 剩余 Food '+frames.at(-1).remaining;
function show(){const f=frames[Number(slider.value)];document.getElementById('frame').src=f.src;document.getElementById('state').textContent='步数 '+f.step+' / '+(frames.length-1)+'　累计团队回报 '+f.return.toFixed(3)+'　剩余 Food '+f.remaining;document.getElementById('action').textContent=f.actions?'本步动作：'+f.actions.map((a,i)=>'A'+i+' '+names[a]).join('； '):'回合结束';}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放';}
document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(Number(slider.value)===frames.length-1)slider.value=0;document.getElementById('play').textContent='暂停';show();timer=setInterval(()=>{slider.value=Number(slider.value)+1;show();if(Number(slider.value)===frames.length-1)stop();},850);};
slider.oninput=()=>{stop();show();};show();
</script></html>'''
page = page.replace('FRAME_DATA', json.dumps(frames)).replace('SEED_VALUE', str(args.seed))
output = root / f'episode_{args.seed}_replay.html'
output.write_text(page)
print(output)
