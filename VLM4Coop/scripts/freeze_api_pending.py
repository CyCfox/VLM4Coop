from pathlib import Path
import yaml
p=Path('VLM4Coop/configs/experiments.yaml')
c=yaml.safe_load(p.read_text())
c['protocol_status']='api_teacher_validation_pending'
api=yaml.safe_load(Path('VLM4Coop/configs/api_probe.yaml').read_text())
api.pop('profile',None)
api.update(backend='openai_api',parser_version='strict-repair-v1',
           failure_policy='repair_identifiable_maximum_else_fail')
c['teacher']=api
p.write_text('# Original plan plus latest user API decision; formal deployment is blocked until validation.\n'+yaml.safe_dump(c,sort_keys=False))
p=Path('VLM4Coop/scripts/api_teacher_probe.py')
s=p.read_text()
s=s.replace("            return\n        phase = 'closed_loop'", "            return 1\n        phase = 'closed_loop'")
s=s.replace("        print(json.dumps({'status': 'failed', 'phase': phase, 'error': str(exc)}), flush=True)\n    finally:", "        print(json.dumps({'status': 'failed', 'phase': phase, 'error': str(exc)}), flush=True)\n        return 1\n    finally:")
s=s.replace("    run()\n", "    raise SystemExit(run())\n")
p.write_text(s)
