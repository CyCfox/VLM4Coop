"""Offline replay of archived completions through the current teacher (zero HTTP)."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
from vlm4coop.api_teacher import OpenCodeTeacher, off_response_evidence
from vlm4coop.config import PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    events=[json.loads(line) for line in (args.run/'teacher_calls.jsonl').read_text().splitlines()]
    labels={row['response_id']:row for row in map(json.loads,(args.run/'teacher_labels.jsonl').read_text().splitlines())}
    frozen=json.loads((args.run/'config.json').read_text())
    counts=Counter(); results=[]
    with tempfile.TemporaryDirectory(prefix='reasoning_replay_',dir=PROJECT/'runtime') as temp:
        for index,event in enumerate(events):
            response=event.get('response')
            if not response or not response.get('choices'):
                counts['transport_errors_not_replayed']+=1
                continue
            class ArchivedResponse:
                status,headers=200,{}
                def __enter__(self): return self
                def __exit__(self,*unused): pass
                def read(self): return json.dumps(response).encode()
            # The invalid hostname and mocked transport prohibit accidental API use.
            config=dict(frozen['experiment_config']['teacher'],endpoint='https://example.invalid/offline')
            with patch.dict(os.environ,{'OPENCODE_API_KEY':'OFFLINE-REPLAY-ONLY'}),patch(
                'vlm4coop.api_teacher.urlopen',return_value=ArchivedResponse()):
                teacher=OpenCodeTeacher(config,Path(temp)/str(index)/'calls.jsonl',purpose='distillation')
                result=teacher.predict(Image.new('RGB',(2,2)),frozen['agents'],use_cache=False)
            evidence=off_response_evidence(response)
            accepted=result.source=='teacher'
            assert teacher.calls==(1 if accepted else 3)
            assert teacher.thinking_control_retries==teacher.thinking_control_failures==0
            counts['responses']+=1
            counts['accepted' if accepted else 'masked_after_three_format_failures']+=1
            if evidence=='returned_reasoning':
                counts['with_reasoning']+=1
                counts['reasoning_accepted' if accepted else 'reasoning_format_masked']+=1
            if accepted and result.repaired: counts['numeric_repairs']+=1
            old=labels.get(response.get('id'))
            if old is not None:
                assert accepted
                np.testing.assert_array_equal(result.probs,np.asarray(old['probabilities'],dtype=np.float32))
                counts['existing_labels_unchanged']+=1
            results.append(dict(call_index=event['call_index'],response_id=response.get('id'),
                thinking_evidence=evidence,source=result.source,simulated_attempts=teacher.calls,
                repaired=result.repaired))
    assert counts['existing_labels_unchanged']==len(labels)
    report=dict(status='passed',counts=dict(counts),new_api_requests=0,production_cache_writes=0,
        note='Each archived response is replayed in isolation; invalid replies are repeated synthetically three times. This is a parser/retry validation, not an experiment result.',responses=results)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='responses'},indent=2))


if __name__=='__main__': main()
