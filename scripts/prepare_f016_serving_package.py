"""F016の両fold固定E12合格後にのみ、推論用packageをローカル構築。通信なし。"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import check_f016_fixed12 as A
import cnn_p039_bridge as P

ROOT = Path(__file__).resolve().parents[1]


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def prepare(audit, output):
    audit, output = Path(audit).resolve(), Path(output).resolve()
    P.require(audit.is_relative_to((ROOT / 'artifacts/research').resolve()) and not output.exists()
              and output.is_relative_to((ROOT / 'artifacts').resolve()), '新規workspace出力と研究監査JSON')
    raw = audit.read_bytes()
    value = json.loads(raw)
    P.require(value['schema'] == 'F016_fixedE12_current_family_proxy_audit_v1' and value['両fold候補条件'] is True
              and value['scope_SHA'] == A.SCOPE_SHA and value['fixed_epoch'] == 12
              and value['alpha'] == .1 and value['family_weight'] == .5 and value['Gold数値利用'] is False,
              'F016事前条件に合格した両fold固定E12監査が必要')
    for fold in ('0', '1'):
        result = value['folds'][fold]
        P.require(result['候補条件'] is True and result['primary']['strictly_positive']
                  and result['family_vs_actual_F013_REG']['strictly_positive'] and result['proxy_delta'] >= 0,
                  '固定単体・実家族対照・代理の全条件')
    # 元source16/config/provenance/wheel-lockはpreviewの読取検証だけを再利用。
    # previewのexecute/prepare_packageは無効のまま変更しない。
    inspected = P.inspect_sources(ROOT, 'tiny')
    scope_raw = A.SCOPE.read_bytes()
    P.require(pin(scope_raw)['sha256'] == A.SCOPE_SHA, 'F016事前scope不変')
    scope = json.loads(scope_raw)
    files = {'audit.json': raw, 'scope-policy.json': scope_raw,
             'config.json': inspected['metadata']['config.json'], 'provenance.json': inspected['metadata']['provenance.json'],
             'cnn_f016_source16/__init__.py': b''}
    files.update({'cnn_f016_source16/' + n: body for n, body in inspected['sources'].items()})
    checkpoints = {}
    for fold in (0, 1):
        directory = ROOT / scope['runs'][fold]
        path = directory / 'epoch_012.pt'
        body = path.read_bytes()
        expected = value['folds'][str(fold)]['固定E12_checkpoint']['candidate']
        P.require(pin(body)['sha256'] == expected, '監査済み固定E12重みだけをpackage化')
        files[f'fold{fold}/epoch_012.pt'] = body
        checkpoints[str(fold)] = pin(body)
    weights = ROOT / 'artifacts/pretrained/f014-convnext-tiny-fb22k-384-20261021-v1/model.safetensors'
    body = weights.read_bytes()
    P.require(pin(body) == {'bytes': 114374272, 'sha256': inspected['spec']['generic_weights_SHA']}, '固定generic全重み')
    files['generic.safetensors'] = body
    lock = json.loads((ROOT / P.WHEEL_LOCK).read_bytes())
    for ref in lock['packages']:
        name = ref['filename']
        candidates = list((ROOT / 'artifacts/tools/cnn-Linux-wheels-20261021-v1').rglob(name))
        P.require(len(candidates) == 1, '固定Linux wheel一意')
        body = candidates[0].read_bytes()
        P.require((len(body), pin(body)['sha256']) == P.WHEELS[name], 'Linux wheel hash一致')
        files['wheels/' + name] = body
    output.mkdir(parents=True, exist_ok=False)
    archive = output / 'f016-models.bin'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for name, body in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, body)
    manifest = {'schema': 'F016_fixedE12_CNN_package_v1', 'scope_SHA': A.SCOPE_SHA, 'audit_SHA': pin(raw)['sha256'],
                'members': {n: pin(body) for n, body in files.items()}, 'checkpoints': checkpoints,
                'config': json.loads(files['config.json']), 'provenance': json.loads(files['provenance.json']),
                'source16': {n: pin(body) for n, body in inspected['sources'].items()},
                'archive': pin(archive.read_bytes()), 'alpha': .1, 'family_weight': .5,
                'serving_GPU_MRI未確認': True, '外部upload_Save_Submit': 0}
    (output / 'package-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({'archive': result['archive'], 'serving_GPU_MRI未確認': True}, ensure_ascii=False))
