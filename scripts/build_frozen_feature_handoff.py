"""Build offline code and unexecuted feature-cache/inference notebooks. No downloads/uploads."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from rsna_knee.contracts import dump_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


SETUP = """from pathlib import Path
import json, sys, zipfile
CODE_INPUT = Path('/kaggle/input/your-frozen-feature-code')  # Edit code Input ONLY; never scan the Competition tree.
if not CODE_INPUT.is_dir():
    raise RuntimeError('Set CODE_INPUT to the manually attached code Input directory')
bundles = list(CODE_INPUT.rglob('rsna-frozen-features-code.zip'))
if len(bundles) > 1:
    raise RuntimeError('Multiple code ZIPs found; select one immutable code Input')
if len(bundles) == 1:
    CODE = Path('/kaggle/working/frozen-code')
    if CODE.exists():
        raise RuntimeError('Fresh kernel/output directory required')
    CODE.mkdir()
    with zipfile.ZipFile(bundles[0]) as archive:
        for name in archive.namelist():
            dest = (CODE / name).resolve()
            if not dest.is_relative_to(CODE.resolve()):
                raise ValueError('Unsafe archive member')
        archive.extractall(CODE)
else:
    roots = [p.parents[2] for p in CODE_INPUT.rglob('feature_contracts.py') if p.parent.name == 'rsna_knee' and p.parent.parent.name == 'src']
    if len(roots) != 1:
        raise RuntimeError('Attach exactly one feature code bundle, ZIP or extracted src layout')
    CODE = roots[0]
sys.path.insert(0, str(CODE / 'src'))
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
print(torch.__version__, torch.cuda.get_device_name(0))
"""

ASSETS = """# Edit these attached Input paths. Internet OFF. No file is downloaded by this notebook.
ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not ROOT.exists():
    ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')
ENCODER_REPO = Path('/kaggle/input/your-generic-dinov2/dinov2-source')
ENCODER_WEIGHTS = Path('/kaggle/input/your-generic-dinov2/dinov2_vits14_pretrain.pth')
PROVENANCE = Path('/kaggle/input/your-generic-dinov2/encoder-provenance.json')
provenance = json.loads(PROVENANCE.read_text(encoding='utf-8-sig'))
"""


def notebooks():
    prepare = [
        markdown(
            "# 独自モデル用・高解像度DINOv2特徴キャッシュ\n未実行テンプレート。まず非goldの学習partition 32検査で品質・時間を確認し、別実行で全weakを処理します。元MRIはKaggle内でのみ読みます。汎用DINOv2と出所監査を手動Inputとして追加してください。公開競技checkpointを代用しません。詳細はdocs/frozen-features.md。"
        ),
        code(SETUP),
        code(ASSETS),
        code("""from rsna_knee.feature_contracts import load_feature_config, validate_fold_audit
from rsna_knee.contracts import ID, read_csv, write_csv, index_studies
from rsna_knee.runtime import check_split, fixed_subset
CONFIG = load_feature_config(CODE / 'configs/experiments/f001-dinov2-frozen.json')
MANIFEST = Path('/kaggle/input/your-reviewed-manifest/v1-vmohitrao-research')
FOLD_AUDIT = MANIFEST / 'fold-audit-v2.json'
PILOT = True  # Keep True for the first execution. After QC, use False in a NEW notebook run.
FOLD = 0
SHARD_COUNT = 1  # For full exports only: e.g. 4 separate runs with SHARD_INDEX=0,1,2,3.
SHARD_INDEX = 0
metadata = json.loads((MANIFEST / 'manifest.json').read_text(encoding='utf-8-sig'))
assert metadata['seed'] == CONFIG['seed'] and metadata['n_folds'] == CONFIG['n_folds']
validate_fold_audit(MANIFEST, FOLD_AUDIT, metadata)
_, weak = read_csv(MANIFEST / 'weak.csv')
_, gold = read_csv(MANIFEST / 'gold.csv')
training, validation = check_split(weak, gold, FOLD, CONFIG['n_folds'])
if PILOT:
    selected = fixed_subset(training, CONFIG['seed'], 32)
    SELECTED = Path('/kaggle/working/pilot-weak.csv')
    write_csv(SELECTED, tuple(selected[0]), selected)
else:
    SELECTED = MANIFEST / 'weak.csv'
print('pilot=', PILOT, 'studies=', len(selected) if PILOT else len(weak))
"""),
        code("""from rsna_knee.feature_imaging import extract_cache, verify_feature_cache
CACHE = Path('/kaggle/working/frozen-pilot-v1' if PILOT else f'/kaggle/working/frozen-weak-v1-shard{SHARD_INDEX}')
result = extract_cache(ROOT, 'train', CACHE, CONFIG, provenance, ENCODER_REPO, ENCODER_WEIGHTS,
                       studies_csv=SELECTED, limit=32 if PILOT else None,
                       shard_index=0 if PILOT else SHARD_INDEX, shard_count=1 if PILOT else SHARD_COUNT)
verify_feature_cache(CACHE, CONFIG, allow_limited=PILOT)
print({key: result[key] for key in ('complete', 'requested', 'elapsed_seconds', 'total_payload_bytes')})
print('Full-set linear estimate, NOT a runtime guarantee:', result['elapsed_seconds'] / result['requested'] * len(weak) / 3600, 'hours')
"""),
        code("""# Input QC only, on selected training studies; no target values or metric-based tuning.
import matplotlib.pyplot as plt
from rsna_knee.feature_imaging import read_feature_windows
for study, coverage in list(result['coverage'].items())[:3]:
    detail = coverage['series'][0]
    windows, positions, geometry = read_feature_windows(ROOT / 'train_series' / study / detail['SeriesInstanceUID'], CONFIG['preprocess'])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, index in zip(axes, [0, len(windows)//2, len(windows)-1]):
        ax.imshow(windows[index, 1], cmap='gray')
        ax.set_title(f"{detail['plane']} position={positions[index]:.2f}")
        ax.axis('off')
    plt.show()
print('Review native-source detail, series choice, slice order, aspect, borders and missing series before full export.')
"""),
        markdown(
            "pilotは学習用complete cacheとして受理されません。全件実行は計測から9時間に十分収まることを確認してから行い、Outputを手動取得後にverifyします。DINOv2実重み・実MRIによる動作、時間、精度はこのテンプレートでは未確認です。"
        ),
    ]
    infer = [
        markdown(
            "# 独自DINOv2 headのReport不要推論\n学習後のbest.ptと、学習cache作成時と同じ汎用encoder・source・provenanceをInputへ手動で追加します。単独の候補です。p003へ自動混合・代理提出はしません。"
        ),
        code(SETUP),
        code(ASSETS),
        code("""from rsna_knee.feature_runtime import load_head_checkpoint, predict_feature_cache
from rsna_knee.feature_imaging import extract_cache
from rsna_knee.feature_contracts import feature_contract, json_hash
CHECKPOINT = Path('/kaggle/input/your-feature-head/best.pt')
checkpoint = load_head_checkpoint(CHECKPOINT)
CONFIG = checkpoint['config']
assert feature_contract(CONFIG, provenance) == checkpoint['feature_contract'], 'Encoder or preprocessing changed'
CACHE = Path('/kaggle/working/frozen-test-v1')
extract_cache(ROOT, 'test', CACHE, CONFIG, provenance, ENCODER_REPO, ENCODER_WEIGHTS)
result = predict_feature_cache(ROOT / 'test.csv', CACHE, CHECKPOINT, '/kaggle/working/submission.csv',
                               sample_csv=ROOT / 'sample_submission.csv')
print(result)
"""),
        markdown(
            "submission.receipt.jsonのvalid、対象UID一致、12所見順、finite値を確認します。別途保存実行が完走した後、Kaggle上で手動提出します。Visible test完走はHidden testの9時間保証ではありません。"
        ),
    ]
    return {"06_prepare_frozen_features.ipynb": prepare, "07_predict_frozen_features.ipynb": infer}


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new handoff directory")
    output.mkdir(parents=True)
    sources = sorted((ROOT / "src/rsna_knee").glob("*.py")) + [
        ROOT / "configs/experiments/f001-dinov2-frozen.json",
        ROOT / "configs/experiments/f002-dinov2-attention.json",
        ROOT / "configs/frozen-encoder-provenance.example.json",
        ROOT / "scripts/frozen_features.py",
    ]
    bundle = output / "rsna-frozen-features-code.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            info = zipfile.ZipInfo(source.relative_to(ROOT).as_posix(), (2026, 10, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    for name, cells in notebooks().items():
        notebook = {
            "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        dump_json(output / name, notebook)
    record = {
        "bundle_sha256": sha256(bundle),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "note": "Code/templates only. No encoder, real data, training, upload or submission.",
    }
    dump_json(output / "handoff.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False, indent=2))
