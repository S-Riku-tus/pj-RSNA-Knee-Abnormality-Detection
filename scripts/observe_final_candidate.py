"""最終候補の本人Notebookを一度だけ読取。モデル/MRIの取得・外部書込なし。"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT_FILES = {'ours_sub.csv', 'raptor224_raw.csv', 'raptor224_sub.csv', 'raptor_raw.csv',
              'raptor_sub.csv', 'submission.csv', 'blend_receipt.json', 'submission_receipt.json',
              '_own.csv', '_public_stack.csv', 'P051-reader.stdout.txt'}
NATIVE_FILES = {'native-child.log', 'native-result.json', 'adapter-receipt.json', 'execution-receipt.json',
                'child-watchdog.json', 'parent-submission.csv', 'transport-receipts.json', 'decoder-resolution.json'}


def wanted(name):
    path = PurePosixPath(name)
    return (name in ROOT_FILES
            or name in {'P051/execution.json', 'P051/native10-parent/submission.csv'}
            or len(path.parts) == 2 and path.parts[0] == 'P032' and path.parts[1] in NATIVE_FILES
            or len(path.parts) == 3 and path.parts[:2] in
            (('P032', 'isolated-uid-projection'), ('P032', 'fixed20'))
            and path.suffix in ('.json', '.csv', '.sha256'))


def observe(slug, kernel, output, *, download=False):
    if not re.fullmatch(r'p0(?:49|50|51|52|53)-[a-z0-9-]{5,50}', slug):
        raise ValueError('本人の今回の固定候補slugが必要')
    output = Path(output) / ('poll-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle
        kaggle.api.authenticate()
    if kaggle.api.config_values.get('username') != 'rsraki':
        raise ValueError('本人認証が異なる')
    from kagglesdk.kernels.types import kernels_api_service as k
    raw, receipts, inventory, downloads = {}, [], [], []
    label = ''

    def capture(response):
        if label.startswith('download:'):
            return
        body = response.content
        if response.status_code != 200 or len(body) > 8 * 1024**2:
            raise ValueError('読取HTTP/サイズ異常')
        (output / (label + '.json')).write_bytes(body)
        raw[label] = response.json()
        receipts.append({'label': label, 'HTTP': response.status_code, 'bytes': len(body),
                         'SHA': hashlib.sha256(body).hexdigest()})

    with kaggle.api.build_kaggle_client() as client:
        client._http_client._response_processor = capture
        label = 'status'
        q = k.ApiGetKernelSessionStatusRequest()
        q.user_name, q.kernel_slug, q.version_label = 'rsraki', slug, 'v1'
        status = client.kernels.kernels_api_client.get_kernel_session_status(q).to_dict(ignore_defaults=False)['status']
        label = 'saved-source'
        q = k.ApiGetKernelRequest()
        q.user_name, q.kernel_slug, q.version_label = 'rsraki', slug, 'v1'
        client.kernels.kernels_api_client.get_kernel(q)
        metadata = raw[label]['metadata']
        if metadata['id'] != kernel or metadata['currentVersionNumber'] != 1:
            raise ValueError('本人kernel/固定V1の結合が異なる')
        if download and status in ('COMPLETE', 'ERROR'):
            token, seen = '', set()
            for page in range(1, 31):
                label = 'outputs-' + str(page)
                q = k.ApiListKernelSessionOutputRequest()
                q.user_name, q.kernel_slug, q.page_size = 'rsraki', slug, 100
                if token:
                    q.page_token = token
                client.kernels.kernels_api_client.list_kernel_session_output(q)
                data = raw[label]
                inventory.extend(data.get('files', []))
                if page == 1:
                    (output / 'session.log').write_text(str(data.get('log', '')), encoding='utf8')
                token = data.get('nextPageToken') or ''
                if not token:
                    break
                if token in seen:
                    raise ValueError('出力pagination loop')
                seen.add(token)
            else:
                raise ValueError('出力page数上限')
            names = [v['fileName'] for v in inventory if wanted(v['fileName'])]
            if len(names) != len(set(names)):
                raise ValueError('出力ファイル重複')
            total = 0
            for name in sorted(names):
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
                    raise ValueError('出力path不正')
                label = 'download:' + name
                q = k.ApiDownloadKernelOutputRequest()
                q.owner_slug, q.kernel_slug, q.file_path = 'rsraki', slug, name
                response = client.kernels.kernels_api_client.download_kernel_output(q)
                try:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_content(65536):
                        body.extend(chunk)
                        if len(body) > 16 * 1024**2 or total + len(body) > 64 * 1024**2:
                            raise ValueError('出力download上限')
                    target = output / 'files' / Path(*path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(body)
                    total += len(body)
                    downloads.append({'name': name, 'HTTP': response.status_code, 'bytes': len(body),
                                      'SHA': hashlib.sha256(body).hexdigest()})
                finally:
                    response.close()
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'kernelId': kernel, 'version': 1,
              'slug': slug, 'status_raw': raw['status'], 'status_typed': status,
              'output_count': len(inventory), 'downloaded': downloads, 'errors': [],
              'evidence': output.as_posix(), 'Save_Submit_upload': 0}
    for name, value in [('summary.json', result), ('receipts.json', receipts), ('output-inventory.json', inventory)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--kernel', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    result = observe(**vars(args))
    print(json.dumps({key: value for key, value in result.items() if key != 'downloaded'}, ensure_ascii=False))
