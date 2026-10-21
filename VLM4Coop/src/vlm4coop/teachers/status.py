"""Current counters shared by normal and failed training status records."""
def teacher_status(teacher):
    """Current counters, including when a teacher batch raises before PPO."""
    if teacher is None:
        return {}
    cache = getattr(teacher, 'cache', None)
    values = dict(teacher_calls=teacher.calls,
        teacher_probability_cache_enabled=cache is not None,
        peak_active_http_requests=getattr(teacher, 'max_active_requests', None),
        format_failures=getattr(teacher, 'format_failures', 0),
        format_retries=getattr(teacher, 'format_retries', 0),
        thinking_control_retries=getattr(teacher, 'thinking_control_retries', 0),
        rejected_thinking_responses=getattr(teacher, 'thinking_control_failures', 0),
        teacher_responses_with_reasoning=getattr(teacher, 'responses_with_reasoning', 0),
        teacher_format_failed_decisions=getattr(teacher, 'failed_decisions', 0))
    if cache is not None:
        values.update(cache_hits=cache.hits, cache_misses=cache.misses, cache_writes=cache.writes)
    guard = getattr(teacher, 'reasoning_guard', None)
    if guard is not None:
        values.update(reasoning_guard_error=guard.error,
                      effective_HTTP_concurrency=min(teacher.config.get('concurrency',1),guard.max_in_flight))
    return values

