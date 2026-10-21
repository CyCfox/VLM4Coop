"""Offline audit of thinking control evidence; never invokes a teacher."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    calls = [json.loads(line) for line in (run / 'teacher_calls.jsonl').read_text().splitlines()]
    details = []
    for call in calls:
        response = call.get('response', {})
        message = (response.get('choices') or [{}])[0].get('message', {})
        usage = response.get('usage', {})
        tokens = usage.get('completion_tokens_details', {}).get('reasoning_tokens')
        details.append(dict(call_index=call['call_index'], label=call['label'],
            http_status=call.get('http_status'), thinking_requested=call.get('thinking'),
            reasoning_effort_requested=call.get('reasoning_effort'),
            reasoning_tokens=tokens, usage=usage,
            reasoning_content_characters=len(message.get('reasoning_content') or ''),
            content_characters=len(message.get('content') or ''),
            message_keys=sorted(message),
            verified_off=tokens == 0 and not message.get('reasoning_content')))
    report = dict(status='offline_diagnostics', api_requests_during_audit=0,
        total_http_attempts=len(calls),
        http_status_counts=dict(Counter(str(row['http_status']) for row in details)),
        reasoning_token_counts=dict(Counter(str(row['reasoning_tokens']) for row in details)),
        verified_off_responses=sum(row['verified_off'] for row in details),
        nonempty_reasoning_responses=sum(row['reasoning_content_characters'] > 0 for row in details),
        missing_reasoning_token_usage=sum(row['reasoning_tokens'] is None for row in details),
        requested_effort_counts=dict(Counter(str(row['reasoning_effort_requested']) for row in details)),
        details=details)
    (run / 'thinking_control_diagnostics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'details'}, indent=2))
    print(json.dumps(details[:3], indent=2))


if __name__ == '__main__':
    main()
