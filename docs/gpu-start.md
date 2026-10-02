# 別のGPU端末で開始する手順

この資料の取得・前処理・学習コマンドはGPU端末で実行する。このPCでは実行していない。公式データの取得先とGPU環境は利用者が選ぶ。

## 環境準備

Python 3.11または3.12を推奨。Windowsでも実行できるが、大量のファイルとオフライン推論の再現にはLinuxが扱いやすい。

Linux:

```bash
git clone https://github.com/S-Riku-tus/pj-RSNA-Knee-Abnormality-Detection.git
cd pj-RSNA-Knee-Abnormality-Detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
git clone https://github.com/S-Riku-tus/pj-RSNA-Knee-Abnormality-Detection.git
cd pj-RSNA-Knee-Abnormality-Detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

次に [PyTorch公式インストール選択画面](https://pytorch.org/get-started/locally/) でOSとGPUに適した **torchとtorchvisionの対応する組** をインストールする。`torch>=2.4` のAPIを使用する。CUDA wheelは端末依存のため、このプロジェクトの通常インストールには含めていない。

```bash
python -m pip install -e ".[imaging,dev]"
python -m rsna_knee doctor
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

`cuda` がtrueであることを確認する。torchvisionのimportが失敗する場合はtorchとの版の組を直す。run.jsonへ版を保存するが、torch/CUDAを含む共通lockは未作成なので、最初に成功したGPU環境で `python -m pip freeze > artifacts/gpu-environment.txt` も保存する。

## データ準備とCSV監査

大会サイトでルール同意後、**GPU端末側で**データを取得する。Kaggle Notebookなら競技Inputを追加してmountを利用できる。別ドライブのデータを使う場合は下の `data/raw` をその絶対パスへ読み替える。

```bash
python -m rsna_knee audit --data-root data/raw --output data/audit-v1.json
```

`train.csv`、`train_series.csv`、画像ディレクトリが揃っていること、件数・欠損・断面の表記を確認する。約570GBという報告値を実測していないため、展開領域とcache領域を含めて空き容量を確認する。

## ラベルの取り込み

公開NotebookのInputsから出所が明確なラベルCSVをGPU端末で取得し、`StudyInstanceUID` と12の提出列へ合わせて `data/labels/weak_labels.csv` に保存する。

自分でラベル作成器を作る場合のみ、空欄テンプレートを生成できる:

```bash
python -m rsna_knee label-template --train-csv data/raw/train.csv --output data/labels/empty-template.csv
```

空欄を埋めずに学習へ進まない。`configs/label-provenance.example.json` を `data/labels/provenance.json` へコピーし、URL、版、ライセンス、方法、作成日時、goldへの調整の有無を確認して記入する。

```bash
python -m rsna_knee prepare --train-csv data/raw/train.csv --weak-labels data/labels/weak_labels.csv --provenance data/labels/provenance.json --output-dir data/manifests/v1 --config configs/baseline.json
```

患者や施設の対応を確認できた場合は `--groups data/labels/groups.csv` を追加する。CSVは全検査の `StudyInstanceUID,group_id` を含める。manifestのfold件数と各クラスの陽性・陰性分布を確認する。

## 小規模前処理から学習へ

```bash
python -m rsna_knee cache --data-root data/raw --split train --cache-dir data/cache/train-v1 --config configs/baseline.json --limit 10
```

`coverage.json` の失敗・fallbackを確認する。キャッシュの各チャンネルはMRIの隣接スライスでありRGBではない。GPU端末の画像viewerやNotebookで原画像と比較し、断面の選択、窓の順序、所見が残ることを目視する。

問題なければ `--limit` を外して同じcacheを再開する:

```bash
python -m rsna_knee cache --data-root data/raw --split train --cache-dir data/cache/train-v1 --config configs/baseline.json
python -m rsna_knee train --manifest-dir data/manifests/v1 --cache-dir data/cache/train-v1 --run-dir artifacts/runs/e001-fold0 --config configs/baseline.json --fold 0
```

学習はCUDA専用。runが存在すると上書きせず停止する。まず5epochの初期設定でVRAM・速度・検証lossを確認し、改善へ進む。学習再開機能はまだないため、長時間runの開始前に環境と小規模動作を確認する。

## 推論と提出物確認

GPU端末の例示testで動作を確認する:

```bash
python -m rsna_knee cache --data-root data/raw --split test --cache-dir data/cache/test-v1 --config configs/baseline.json
python -m rsna_knee predict --test-csv data/raw/test.csv --cache-dir data/cache/test-v1 --checkpoint artifacts/runs/e001-fold0/best.pt --output artifacts/submissions/submission.csv
python -m rsna_knee validate-submission --submission artifacts/submissions/submission.csv --test-csv data/raw/test.csv --sample-submission data/raw/sample_submission.csv
```

例示testの成功は本採点の成功ではない。最終的には [Kaggle提出手順](kaggle-submit.md) に沿って隠しtestの推論をKaggle内で実行する。入力データの場所は推論時に置き換わるため、GPU端末で作った例示test cacheは提出へ持ち込まない。
