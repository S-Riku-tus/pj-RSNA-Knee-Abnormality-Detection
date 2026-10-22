"""本人の正式提出をGet/List/limitsで一度照合し保存。書込APIは持たない。"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def observe(refs, output):
    output = Path(output) / ('observation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle
        kaggle.api.authenticate()
    if kaggle.api.config_values.get('username') != 'rsraki':
        raise ValueError('本人認証が異なる')
    from kagglesdk.competitions.types import competition_api_service as c
    raw, receipts = {}, []
    label = ''

    def capture(response):
        if response.status_code != 200 or len(response.content) > 4 * 1024**2:
            raise ValueError('読取HTTP/サイズ異常')
        (output / (label + '.json')).write_bytes(response.content)
        raw[label] = response.json()
        receipts.append({'label': label, 'HTTP': response.status_code, 'bytes': len(response.content),
                         'SHA': hashlib.sha256(response.content).hexdigest()})

    with kaggle.api.build_kaggle_client() as client:
        client._http_client._response_processor = capture
        label = 'full-own-list'
        q = c.ApiListSubmissionsRequest()
        q.competition_name, q.page, q.page_size = 'rsna-knee-abnormality-detection', 1, 100
        client.competitions.competition_api_client.list_submissions(q)
        rows = raw[label] if isinstance(raw[label], list) else raw[label]['submissions']
        label = 'limits'
        q = c.ApiGetSubmissionLimitsRequest()
        q.competition_name = 'rsna-knee-abnormality-detection'
        client.competitions.competition_api_client.get_submission_limits(q)
        if len(rows) != raw['limits']['numTotal'] or any(r['submittedBy'] != 'rsraki' for r in rows):
            raise ValueError('本人全提出数が照合できない')
        by_ref = {int(row['ref']): row for row in rows}
        specific = []
        for ref in refs:
            label = 'owned-Get-' + str(ref)
            q = c.ApiGetSubmissionRequest()
            q.ref = ref
            client.competitions.competition_api_client.get_submission(q)
            one, listed = raw[label], by_ref[ref]
            if one['submittedBy'] != 'rsraki' or int(one['ref']) != ref or any(one.get(key) != listed.get(key)
                    for key in ('status', 'publicScore', 'errorDescription', 'description', 'fileName')):
                raise ValueError('同一refのGet/Listが異なる')
            match = re.search(r'scriptVersionId=(\d+)', listed.get('url', ''))
            specific.append({'ref': ref, 'status_raw': one.get('status'), 'Public_raw': one.get('publicScore'),
                             'error_raw': one.get('errorDescription'), 'source_URL': listed.get('url'),
                             'scriptVersionId': int(match.group(1)) if match else None})
    scored, exceptions, pending = [], [], []
    for row in rows:
        if row.get('status') == 'COMPLETE' and row.get('publicScore') is not None and not row.get('errorDescription'):
            value = float(row['publicScore'])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Public値が不正')
            scored.append(row)
        elif row.get('errorDescription') or row.get('status') in ('ERROR', 'CANCELLED'):
            exceptions.append(row)
        else:
            pending.append(row)
    best = max((float(r['publicScore']) for r in scored), default=None)
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'owned_submissions': len(rows),
              '正常採点': len(scored), '例外': len(exceptions), '未返却': len(pending),
              'own_Public_best': best, 'limits': raw['limits'], 'specific': specific,
              'goal959_verified': best is not None and best >= .959,
              'best_refs': [r['ref'] for r in scored if float(r['publicScore']) == best],
              'receipts': receipts, '外部書込': 0, 'evidence': output.as_posix()}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refs', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = observe(**vars(args))
    print(json.dumps({key: value for key, value in result.items() if key != 'receipts'}, ensure_ascii=False))
