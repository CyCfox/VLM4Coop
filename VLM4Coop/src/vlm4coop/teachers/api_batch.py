"""Independent HTTP requests, with SQLite confined to the caller thread.

Every remote request contains one image and all agents. Training may reuse
frozen labels. Pure VLM passes use_cache=False for fresh, non-deduplicated calls.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed


def predict_many(teacher, images, n_agents, use_cache=True):
    if not images:
        return []
    concurrency = teacher.config.get('concurrency', 1)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('API concurrency must be a positive integer')
    guard = getattr(teacher, 'reasoning_guard', None)
    if guard is not None:
        concurrency = min(concurrency, guard.max_in_flight)
    persistent = use_cache and teacher.cache is not None
    keys = [teacher.cache.key(image, n_agents) for image in images] if persistent else list(range(len(images)))
    results, missing, seen = {}, [], set()
    for index, key in enumerate(keys):
        if key in seen:
            continue
        seen.add(key)
        cached = (teacher.lookup_cache(key,n_agents) if hasattr(teacher,'lookup_cache')
                  else teacher.cache.get(key)) if persistent else None
        if cached is not None:
            results[key] = cached
        else:
            missing.append((key, index))
    def generate(entry):
        _, index = entry
        return teacher.predict(images[index], n_agents, use_cache=False)
    def collect(entry, result):
        key, _ = entry
        results[key] = result
        if persistent and result.source=='teacher':
            if hasattr(teacher,'store_cache'): teacher.store_cache(key,result)
            else: teacher.cache.put(key, result)
    # Verify OFF with real, useful labels before opening the bulk request wave.
    # Cache hits do not establish anything about the current provider route.
    while missing and guard is not None and guard.clean_startup < guard.startup:
        entry = missing.pop(0)
        collect(entry, generate(entry))
    if concurrency == 1:
        for entry in missing:
            collect(entry, generate(entry))
    elif missing:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(missing))) as executor:
            futures = {executor.submit(generate, entry): entry for entry in missing}
            first_error = None
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    if guard is not None and guard.error:
                        for pending in futures:
                            pending.cancel()
                else:
                    collect(futures[future], result)
            if first_error is not None:
                raise first_error
    return [results[key] for key in keys]
