"""Concurrent transport/cache/worker alignment with mocked HTTP only."""
import base64
import io
import json
import threading
from types import SimpleNamespace
import numpy as np
from PIL import Image
from vlm4coop.teachers.api import OpenCodeTeacher
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult


def test_parallel_single_image_requests_and_main_thread_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENCODE_API_KEY', 'TEST-CREDENTIAL-ONLY')
    cache = ProbabilityCache(tmp_path / 'labels.sqlite', {'test': 'parallel'})
    teacher = OpenCodeTeacher({'endpoint':'https://example.invalid/chat/completions',
        'model':'test-model','max_tokens':256,'concurrency':2,
        'thinking':{'type':'enabled'},'reasoning_effort':'low'}, tmp_path / 'requests.jsonl', cache)
    barrier = threading.Barrier(2)
    def fake_open(request, timeout):
        body = json.loads(request.data)
        parts = body['messages'][-1]['content']
        assert len(parts) == 2 and sum(x['type']=='image_url' for x in parts) == 1
        encoded = next(x['image_url']['url'] for x in parts if x['type']=='image_url').split(',',1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        action = 1 if image.getpixel((0,0))[0] else 4
        barrier.wait(timeout=5)  # proves two real transport calls overlap
        class Response:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                probs = [0.]*6; probs[action]=1.
                answer = json.dumps({'agents':[{'id':i,'probs':probs} for i in range(3)]})
                return json.dumps({'id':f'test-{action}','model':'test-model',
                    'choices':[{'message':{'content':answer,'reasoning_content':'x'*25000}}],
                    'usage':{'completion_tokens':100,'completion_tokens_details':{'reasoning_tokens':20}}}).encode()
        return Response()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen', fake_open)
    images = [Image.new('RGB',(4,4),name) for name in ('red','green','red')]
    result = teacher.predict_many(images,3)
    assert [x.probs.argmax(-1).tolist() for x in result] == [[1]*3,[4]*3,[1]*3]
    assert teacher.calls == 2 and cache.writes == 2
    events = [json.loads(x) for x in teacher.audit_path.read_text().splitlines()]
    assert sorted(x['call_index'] for x in events) == [1,2]
    assert len((tmp_path / 'teacher_labels.jsonl').read_text().splitlines()) == 2
    again = teacher.predict_many(images,3)
    assert teacher.calls == 2 and all(x.cache_hit for x in again)
    cache.close()


def test_pure_batch_duplicates_still_make_real_calls(monkeypatch, tmp_path):
    from vlm4coop.teachers.api_batch import predict_many
    image = Image.new('RGB',(4,4),'red')
    cache = ProbabilityCache(tmp_path / 'labels.sqlite', {'test':'pure'})
    p = np.zeros((3,6),np.float32); p[:,2]=1
    cache.put(cache.key(image,3),TeacherResult(p,'cached test only',False))
    called = []
    def uncached(image,n_agents,use_cache):
        assert use_cache is False
        called.append(1)
        return TeacherResult(p,'new test only',False)
    teacher = SimpleNamespace(config={'concurrency':2},cache=cache,predict=uncached)
    result = predict_many(teacher,[image,image,image],3,use_cache=False)
    assert len(called) == len(result) == 3 and cache.hits == 0
    assert cache.writes == 1  # the one preloaded test entry only
    cache.close()


def test_pure_mac_batch_keeps_worker_order_and_skips_terminal_images():
    from vlm4coop.evaluation.pure_vlm import TeacherMAC
    class Teacher:
        def predict_many(self,images,n_agents,use_cache):
            assert use_cache is True and len(images) == 2
            result=[]
            for image in images:
                action=1 if image.getpixel((0,0))[0] else 4
                p=np.zeros((3,6),np.float32);p[:,action]=1
                result.append(TeacherResult(p,'test only',False))
            return result
    mac=TeacherMAC(Teacher(),3)
    mac.runner=SimpleNamespace(frames={
        (2,7):np.asarray(Image.new('RGB',(4,4),'red')),
        (1,7):np.asarray(Image.new('RGB',(4,4),'green'))})
    actions=mac.select_actions(None,7,0,[2,0,1],True)
    assert actions.tolist() == [[1]*3,[0]*3,[4]*3]
    assert not mac.runner.frames


def test_one_hundred_concurrent_cache_misses_then_all_hits(tmp_path):
    from vlm4coop.teachers.api_batch import predict_many
    cache = ProbabilityCache(tmp_path / 'hundred.sqlite', {'test': 'hundred'})
    barrier = threading.Barrier(100)
    images = [Image.new('RGB', (2, 2), (i, 0, 0)) for i in range(100)]
    calls = []
    def predict(image, n_agents, use_cache):
        assert not use_cache
        barrier.wait(timeout=15)
        calls.append(image.getpixel((0, 0))[0])
        p = np.zeros((3, 6), np.float32)
        p[:, image.getpixel((0, 0))[0] % 6] = 1
        return TeacherResult(p, 'synthetic concurrency witness only', False)
    teacher = SimpleNamespace(config={'concurrency': 100}, cache=cache, predict=predict)
    results = predict_many(teacher, images, 3)
    assert sorted(calls) == list(range(100)) and cache.writes == 100
    assert [r.probs.argmax(-1)[0] for r in results] == [i % 6 for i in range(100)]
    assert all(r.cache_hit for r in predict_many(teacher, images, 3))
    assert len(calls) == 100
    cache.close()


def test_pure_mac_uses_persistent_cache_and_deduplicates(tmp_path):
    from vlm4coop.teachers.api_batch import predict_many
    from vlm4coop.evaluation.pure_vlm import TeacherMAC
    image = Image.new('RGB', (4, 4), 'blue')
    cache = ProbabilityCache(tmp_path / 'pure.sqlite', {'test': 'pure-cache'})
    p = np.zeros((3, 6), np.float32); p[:, 5] = 1
    cache.put(cache.key(image, 3), TeacherResult(p, 'previous genuine-label stand-in', False))
    class Teacher:
        config = {'concurrency': 100}
        def predict(self, *args, **kwargs):
            raise AssertionError('A cache hit must not make an HTTP request')
        def predict_many(self, images, n_agents, use_cache):
            assert use_cache
            return predict_many(self, images, n_agents, use_cache)
    teacher = Teacher(); teacher.cache = cache
    mac = TeacherMAC(teacher, 3)
    mac.runner = SimpleNamespace(frames={(0, 0): np.asarray(image), (1, 0): np.asarray(image)})
    assert mac.select_actions(None, 0, 0, [0, 1], True).tolist() == [[5]*3, [5]*3]
    assert cache.hits == 1  # one lookup for the duplicate state
    cache.close()


def test_pure_mac_without_cache_requests_duplicate_images_again():
    from vlm4coop.teachers.api_batch import predict_many
    from vlm4coop.evaluation.pure_vlm import TeacherMAC
    image = np.asarray(Image.new('RGB', (4, 4), 'blue'))
    class Teacher:
        config = {'concurrency': 2, 'pure_vlm_use_cache': False}
        cache = None
        def __init__(self): self.calls = 0
        def predict(self, image, n_agents, use_cache):
            assert not use_cache
            self.calls += 1
            p = np.zeros((3,6), np.float32); p[:,5] = 1
            return TeacherResult(p, 'synthetic uncached action only', False)
        def predict_many(self, images, n_agents, use_cache):
            return predict_many(self, images, n_agents, use_cache)
    teacher = Teacher()
    mac = TeacherMAC(teacher, 3)
    for t in (0,1):
        mac.runner = SimpleNamespace(frames={(0,t):image, (1,t):image})
        assert mac.select_actions(None,t,0,[0,1],True).tolist()==[[5]*3,[5]*3]
    assert teacher.calls==4 and mac.decisions==4


def test_uncached_factory_never_opens_sqlite(monkeypatch, tmp_path):
    from vlm4coop.teachers.factory import make_teacher
    class Teacher:
        def __init__(self, config, audit_path, cache, purpose, protocol):
            assert cache is None
            assert protocol.n_actions == 6
            self.cache = cache
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure VLM without cache must never open SQLite')
    monkeypatch.setattr('vlm4coop.teachers.factory.ProbabilityCache', forbidden)
    monkeypatch.setattr('vlm4coop.teachers.api.OpenCodeTeacher', Teacher)
    result = make_teacher({'teacher': {'backend':'openai_api'}, 'cache_max_bytes':1000},
                          tmp_path, use_cache=False)
    assert result.cache is None


def test_parallel_failure_still_caches_every_success(tmp_path):
    import pytest
    from vlm4coop.teachers.api_batch import predict_many
    cache = ProbabilityCache(tmp_path / 'partial.sqlite', {'test': 'partial'})
    images = [Image.new('RGB', (2, 2), (i, 0, 0)) for i in range(4)]
    def predict(image, n_agents, use_cache):
        value = image.getpixel((0, 0))[0]
        if value == 0:
            raise RuntimeError('persistent service error')
        p = np.zeros((3, 6), np.float32); p[:, value] = 1
        return TeacherResult(p, 'synthetic partial result only', False)
    teacher = SimpleNamespace(config={'concurrency': 4}, cache=cache, predict=predict)
    with pytest.raises(RuntimeError, match='persistent service error'):
        predict_many(teacher, images, 3)
    assert cache.writes == 3
    assert cache.get(cache.key(images[0], 3)) is None
    for image in images[1:]:
        assert cache.get(cache.key(image, 3)).cache_hit
    cache.close()


def test_one_thousand_worker_capacity_and_cache_persistence(tmp_path):
    from vlm4coop.teachers.api_batch import predict_many
    cache = ProbabilityCache(tmp_path/'thousand.sqlite', {'test': '1000-capacity'})
    barrier = threading.Barrier(1000)
    images = [Image.new('RGB', (2, 2), (i//256, i%256, 0)) for i in range(1000)]
    called = []
    def predict(image, n_agents, use_cache):
        assert not use_cache
        pixel = image.getpixel((0, 0)); index = pixel[0]*256+pixel[1]
        barrier.wait(timeout=30)
        called.append(index)
        p=np.zeros((3,6),np.float32); p[:,index%6]=1
        return TeacherResult(p, 'synthetic 1000-worker witness only', False)
    teacher=SimpleNamespace(config={'concurrency':1000},cache=cache,predict=predict)
    results=predict_many(teacher, images, 3)
    assert sorted(called)==list(range(1000)) and cache.writes==1000
    assert [r.probs.argmax(-1)[0] for r in results]==[i%6 for i in range(1000)]
    assert all(result.cache_hit for result in predict_many(teacher, images, 3))
    assert len(called)==1000
    cache.close()
