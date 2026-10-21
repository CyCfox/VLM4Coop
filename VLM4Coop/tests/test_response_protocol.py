import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from vlm4coop.teachers.api import OpenCodeTeacher
from vlm4coop.teachers.response_protocol import clip_and_normalize, parse_api_probabilities, ProbabilityFormatError
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult


def answer(values):
    return json.dumps({'agents':[{'id':i,'probs':values} for i in range(3)]})


def teacher_with_responses(monkeypatch,tmp_path,texts,purpose='distillation',cache=None):
    monkeypatch.setenv('OPENCODE_API_KEY','TEST-ONLY')
    teacher=OpenCodeTeacher(dict(model='test',endpoint='https://example.invalid',
        concurrency=2,thinking={'type':'disabled'}),tmp_path/'calls.jsonl',cache,purpose)
    texts=iter(texts)
    class Response:
        status,headers=200,{}
        def __init__(self,text): self.text=text
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return json.dumps({'id':f'test-{teacher.calls}',
                'choices':[{'message':{'content':self.text}}],
                'usage':{'completion_tokens_details':{'reasoning_tokens':0}}}).encode()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',lambda *a,**k:Response(next(texts)))
    return teacher


def test_clip_then_normalize_user_example_and_out_of_range():
    p,fixed=clip_and_normalize([1,.5,.25,.25])
    np.testing.assert_allclose(p,[.5,.25,.125,.125]); assert fixed
    p,fixed=clip_and_normalize([2,-.5,.5,.5,0,0])
    np.testing.assert_allclose(p,[.5,0,.25,.25,0,0]); assert fixed
    with pytest.raises(ProbabilityFormatError): clip_and_normalize([-1,0,0])


@pytest.mark.parametrize('raw',[answer([0,0,0,0,0,.98,.02]),'Brief analysis. {"agents":[]}',answer([0,0,0,0,0,1])[:-2]])
def test_new_schema_requires_complete_six_entry_rows(raw):
    for purpose in ('pure','distillation'):
        with pytest.raises(ProbabilityFormatError): parse_api_probabilities(raw,3,purpose)


def test_format_failures_retry_twice_then_accept_third(monkeypatch,tmp_path):
    teacher=teacher_with_responses(monkeypatch,tmp_path,
        ['bad',answer([0]*7),'Reasoning first.\n'+answer([0,0,0,0,0,1])],purpose='pure')
    result=teacher.predict(Image.new('RGB',(2,2)),3,use_cache=False)
    assert teacher.calls==3 and teacher.format_retries==2 and teacher.format_failures==2
    assert result.source=='teacher' and result.format_attempts==3
    assert result.probs.argmax(-1).tolist()==[5]*3


def test_three_format_failures_pure_mac_executes_none(monkeypatch,tmp_path):
    from vlm4coop.evaluation.pure_vlm import TeacherMAC
    teacher=teacher_with_responses(monkeypatch,tmp_path,['bad']*3,purpose='pure')
    mac=TeacherMAC(teacher,3,use_cache=False)
    mac.runner=SimpleNamespace(frames={(0,0):np.zeros((2,2,3),dtype=np.uint8)})
    actions=mac.select_actions(None,0,0,[0],True)
    assert actions.tolist()==[[0,0,0]] and mac.none_fallbacks==1 and teacher.calls==3
    assert not (tmp_path/'teacher_labels.jsonl').exists()


def test_distillation_numeric_clip_does_not_consume_format_retry(monkeypatch,tmp_path):
    teacher=teacher_with_responses(monkeypatch,tmp_path,[answer([2,-.5,.5,.5,0,0])])
    result=teacher.predict(Image.new('RGB',(2,2)),3,use_cache=False)
    np.testing.assert_allclose(result.probs[0],[.5,0,.25,.25,0,0])
    assert teacher.calls==1 and result.repaired and result.source=='teacher'


def test_failed_teacher_labels_are_not_cached(monkeypatch,tmp_path):
    cache=ProbabilityCache(tmp_path/'cache.sqlite',{'test':'fallback'})
    teacher=teacher_with_responses(monkeypatch,tmp_path,['bad']*6,cache=cache)
    image=Image.new('RGB',(2,2))
    try:
        for _ in range(2):
            result=teacher.predict_many([image],3)[0]
            assert result.source=='format_failure' and result.probs is None
        assert teacher.calls==6 and cache.writes==0
        assert cache.get(cache.key(image,3)) is None
    finally: cache.close()


def test_cached_raw_probabilities_follow_current_numeric_rule(monkeypatch,tmp_path):
    cache=ProbabilityCache(tmp_path/'cache.sqlite',{'test':'reinterpret'})
    image=Image.new('RGB',(2,2))
    cache.put(cache.key(image,3),TeacherResult(np.tile([.8,.1,.1,0,0,0],(3,1)),
        answer([.8,.2,.1,0,0,0]),True))
    teacher=teacher_with_responses(monkeypatch,tmp_path,[],cache=cache)
    try:
        result=teacher.predict_many([image],3)[0]
        np.testing.assert_allclose(result.probs[0],[.8/1.1,.2/1.1,.1/1.1,0,0,0])
        assert result.cache_hit and teacher.calls==0
    finally: cache.close()


def test_reasoning_without_json_uses_three_format_attempts(monkeypatch,tmp_path):
    teacher=teacher_with_responses(monkeypatch,tmp_path,[],purpose='pure')
    class Response:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return json.dumps({'choices':[{'message':{'content':None,'reasoning_content':'still thinking'}}],
                'usage':{'completion_tokens_details':{'reasoning_tokens':10}}}).encode()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',lambda *a,**k:Response())
    result=teacher.predict(Image.new('RGB',(2,2)),3,use_cache=False)
    assert result.source=='format_failure' and teacher.calls==3
    assert teacher.format_failures==3 and teacher.failed_decisions==1
    assert teacher.responses_with_reasoning==3 and teacher.thinking_control_retries==0


def test_verified_off_empty_final_text_uses_three_format_attempts(monkeypatch,tmp_path):
    teacher=teacher_with_responses(monkeypatch,tmp_path,[None]*3,purpose='pure')
    result=teacher.predict(Image.new('RGB',(2,2)),3,use_cache=False)
    assert result.source=='format_failure' and result.format_attempts==3
    assert teacher.calls==teacher.format_failures==3


@pytest.mark.parametrize('text',[answer([0,0,0,0,0,1]),None])
def test_absent_reasoning_usage_stays_unknown_without_control_retry(monkeypatch,tmp_path,text):
    teacher=teacher_with_responses(monkeypatch,tmp_path,[],purpose='pure')
    class Response:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return json.dumps({'id':f'test-{teacher.calls}',
                'choices':[{'message':{'content':text}}],
                'usage':{'prompt_tokens':768,'completion_tokens':211}}).encode()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',lambda *a,**k:Response())
    result=teacher.predict(Image.new('RGB',(2,2)),3,use_cache=False)
    if text is None:
        assert result.source=='format_failure' and teacher.calls==3
    else:
        assert result.source=='teacher' and teacher.calls==1
        label=json.loads((tmp_path/'teacher_labels.jsonl').read_text())
        assert label['reasoning_tokens'] is None and label['thinking_evidence']=='not_reported'


@pytest.mark.parametrize('tokens,hidden,expected',[
    (None,'hidden','returned_reasoning'),(10,'','returned_reasoning'),
    (0,'hidden','returned_reasoning'),(None,'','not_reported'),(0,'','reported_zero')])
def test_thinking_evidence_does_not_treat_missing_as_zero(tokens,hidden,expected):
    from vlm4coop.teachers.api import off_response_evidence
    response={'choices':[{'message':{'reasoning_content':hidden}}],
              'usage':{'completion_tokens_details':{'reasoning_tokens':tokens}}}
    assert off_response_evidence(response)==expected


@pytest.mark.parametrize('source',['content','reasoning_content'])
@pytest.mark.parametrize('values',[[0,0,0,0,0,1],[2,-.5,.5,.5,0,0]])
def test_returned_reasoning_accepts_valid_or_repaired_json_once_and_caches(monkeypatch,tmp_path,source,values):
    monkeypatch.setenv('OPENCODE_API_KEY','TEST-ONLY')
    cache=ProbabilityCache(tmp_path/'cache.sqlite',{'test':'reasoning'})
    teacher=OpenCodeTeacher(dict(model='test',
        endpoint='https://opencode.ai/inference/openai/v1/chat/completions',
        thinking={'type':'disabled'},concurrency=1000),tmp_path/'calls.jsonl',cache,purpose='distillation')
    payloads=[]
    class Response:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            message={'content':None,'reasoning_content':'brief joint planning'}
            message[source]='<think>joint planning\n'+answer(values)+'</think>'
            return json.dumps({'id':f'test-{teacher.calls}',
                'choices':[{'message':message}],
                'usage':{'completion_tokens_details':{'reasoning_tokens':100}}}).encode()
    def request(req,**kwargs):
        payloads.append(json.loads(req.data)); return Response()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',request)
    image=Image.new('RGB',(2,2))
    try:
        result=teacher.predict(image,3)
        assert result.source=='teacher' and teacher.calls==1
        np.testing.assert_allclose(result.probs[0],clip_and_normalize(values)[0])
        label=json.loads((tmp_path/'teacher_labels.jsonl').read_text())
        assert label['reasoning_tokens']==100 and label['control_attempt']==0
        assert label['thinking_evidence']=='returned_reasoning' and label['answer_source']==source
        assert teacher.predict(image,3).cache_hit and teacher.calls==cache.writes==1
    finally: cache.close()
    assert payloads[0]['thinking']=={'type':'disabled'} and 'reasoning_effort' not in payloads[0]
    assert teacher.format_failures==teacher.failed_decisions==0
    assert teacher.thinking_control_failures==teacher.thinking_control_retries==0
    assert teacher.responses_with_reasoning==1


def test_final_content_has_priority_over_reasoning_draft(monkeypatch,tmp_path):
    from vlm4coop.teachers.api import completion_text
    response={'choices':[{'message':{'content':answer([0,0,0,0,0,1]),
                                    'reasoning_content':answer([1,0,0,0,0,0])}}]}
    raw,source=completion_text(response)
    assert source=='content' and parse_api_probabilities(raw,3,'distillation')[0].argmax(-1).tolist()==[5]*3
    response['choices'][0]['message']['content']='Final answer: {"agents":['
    raw,_=completion_text(response)
    with pytest.raises(ProbabilityFormatError): parse_api_probabilities(raw,3,'distillation')
