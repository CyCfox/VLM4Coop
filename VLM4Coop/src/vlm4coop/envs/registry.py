"""Explicit adapter registration; absent selector preserves historical LBF."""
_ADAPTERS = {}


def register_adapter(name, adapter):
    if not name or adapter.name != name:
        raise ValueError('Adapter name must match its registration')
    if name in _ADAPTERS and _ADAPTERS[name] is not adapter:
        raise ValueError('Environment adapter already registered: ' + name)
    _ADAPTERS[name] = adapter


def get_adapter(config=None):
    name = (config or {}).get('environment_adapter', 'lbf')
    if name == 'lbf' and name not in _ADAPTERS:
        from vlm4coop.envs.lbf.adapter import LBFAdapter
        register_adapter('lbf', LBFAdapter())
    if name == 'mpe_simple_spread' and name not in _ADAPTERS:
        from vlm4coop.envs.mpe.adapter import MPEAdapter
        register_adapter(name, MPEAdapter())
    if name not in _ADAPTERS:
        raise ValueError('Unregistered environment adapter: ' + str(name))
    return _ADAPTERS[name]
