# GPU端末で開始する手順

現在の主端末はWindowsデスクトップのRTX 4090。ユーザーは **Kaggleで画像キャッシュを作り、ローカルで学習する** 経路を選択した。以下は利用者が順に実行する手順であり、実データの取得・キャッシュ作成・学習はまだ実行していない。元画像を扱える別端末向けの手順も後半に残す。

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
# この端末の専用 .venv は作成済み。Activateせず実体を指定できる。
.\.venv\Scripts\python.exe -m rsna_knee doctor

# 別のWindows端末でPythonが未準備なら、インストール済みuvを使用する例。
uv --no-config --cache-dir .uv-cache python install 3.12.13 --install-dir .python --no-bin
uv --no-config --cache-dir .uv-cache venv --python .python/cpython-3.12.13-windows-x86_64-none/python.exe .venv
uv --no-config --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe -e ".[imaging,dev]"
```

このPCでの `python` はMicrosoft Storeのアプリ実行エイリアスで動かなかったため、`.python/` にPython 3.12.13、`.venv/` に専用環境を用意した。numpy、pydicom、Pillow、ruffを導入済みで、torchとtorchvisionはまだない。上記のvenv再作成は既存環境がない端末でのみ行う。uvで作った環境のパッケージ操作は `uv pip --python` を使える。

次に [PyTorch公式インストール選択画面](https://pytorch.org/get-started/locally/) でOSとGPUに適した **torchとtorchvisionの対応する組** をインストールする。`torch>=2.4` のAPIを使用する。CUDA wheelは端末依存のため、このプロジェクトの通常インストールには含めていない。

この端末では選択画面をWindows・Pip・CUDAに合わせ、表示された `pip install ...` のパッケージとindex指定を `uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe ...` に渡す。PyTorchをソースからbuildする必要はない。Windowsでの確認:

初回の環境を固定する例は、公式に配布されているtorch 2.10.0とtorchvision 0.25.0のCUDA 12.8 wheel。これは実装の最小要件を満たす導入候補で、本端末での導入・CUDA動作はまだ確認していない。[公式の版対応とインストールコマンド](https://pytorch.org/get-started/previous-versions/)

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

```powershell
.\.venv\Scripts\python.exe -m rsna_knee doctor
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linuxのpip入り・activate済みvenvでの確認:

```bash
python -m pip install -e ".[imaging,dev]"
python -m rsna_knee doctor
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

`cuda` がtrueであることを確認する。torchvisionのimportが失敗する場合はtorchとの版の組を直す。run.jsonへ版を保存するが、torch/CUDAを含む共通lockは未作成。最初に成功したGPU環境でWindowsなら `uv --cache-dir .uv-cache pip freeze --python .venv/Scripts/python.exe` の出力を新しいrunディレクトリへ保存する。pip入りのLinux環境なら `python -m pip freeze` を使える。

CUDA確認後、実学習の前に人工画像のforwardを行う:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from rsna_knee.model import KneeMIL; m=KneeMIL().eval().cuda(); x=torch.rand(1,4,3,32,32,device='cuda'); mask=torch.tensor([[True,True,False,False]],device='cuda'); y=m(x,mask); assert y.shape==(1,12) and torch.isfinite(y).all(); print(y.shape)"
```

ここまでの成功はCUDAのforwardの確認であり、AMP・backward・学習精度の検証ではない。

## Kaggleのキャッシュでローカル学習する

1. このPCでコードを梱包する: `.venv/Scripts/python.exe scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/cache-v1`。新しい版では出力先も変える。
2. Kaggle画面でコードzipをInput Datasetへ手動で追加し、[00_prepare_cache.ipynb](../notebooks/00_prepare_cache.ipynb) を読み込む。競技InputをAttachし、実際のInputツリーでパスを確認する。zipファイルがあればCODE_ZIPを指定する。展開済みのsrc/とconfigs/があれば、その親ディレクトリをCODE_ROOTへ指定する。INPUT_VERSIONも記入する。NotebookとOutputはprivateで保存する。
3. CPU環境でLIMIT=10を実行する。decoder不足はKaggleのLinux/Pythonに合う依存で解決し、coverage.jsonと原画像・縮小画像を目視する。RTX 4090をこのNotebookから使用することはない。
4. 問題なければLIMIT=Noneで全件を作り、保存版のOutputを手動でダウンロードする。`export.json` のcompleteがtrueであることを確認する。Output内に追加zipを作らない。時間切れ時は保存できたprivate OutputをInputとしてAttachし、RESUME_CACHEをそのtrain-v1へ指定して再開する。
5. ダウンロードした `rsna-cache-v1/` を `data/exports/rsna-cache-v1/` へ展開する。コード・configを変更したときは別名にし、前のexportを上書きしない。

転送されたファイルのhash、検査数、元CSV hash、前処理・ソースの一致を確認する。以下はリポジトリのルートで実行する:

```powershell
.\.venv\Scripts\python.exe scripts/verify_cache_export.py data/exports/rsna-cache-v1
```

出力は `raw/train.csv`、`raw/train_series.csv`、`train-v1/*.npz`、cache.json、coverage.json、code/、audit.json、export.jsonを含む。元DICOMは含まない。`export.json` のfilesは各ファイルのSHA-256とサイズを持つ。ソースが不一致なら最初の学習では出力時のGit commitへ合わせる。画像の形状とfingerprintは学習のpreflightでも検査される。

公開ラベルを監査し、後半の「ラベルの取り込み」に従って `data/labels/weak_labels.csv` とprovenance.jsonを用意してから、次を実行する。後半の `data/raw/train.csv` はこの経路では `data/exports/rsna-cache-v1/raw/train.csv` に読み替える。必要なgroup対応があればprepareへ `--groups` を追加する。

```powershell
.\.venv\Scripts\python.exe -m rsna_knee prepare --train-csv data/exports/rsna-cache-v1/raw/train.csv --weak-labels data/labels/weak_labels.csv --provenance data/labels/provenance.json --output-dir data/manifests/v1 --config data/exports/rsna-cache-v1/code/configs/baseline.json
.\.venv\Scripts\python.exe -m rsna_knee train --manifest-dir data/manifests/v1 --cache-dir data/exports/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e001-fold0 --config data/exports/rsna-cache-v1/code/configs/baseline.json --fold 0
```

この経路ではローカルにDICOMもtest cacheも不要。現行5epochは動作確認の出発点で、まずfold 0だけを実行する。学習再開は未対応なので、長いrunの前に実データの小規模学習を別run・別configで確認する。epoch数など学習側だけを変える場合、前処理設定は維持し、新しいconfigを保存する。

学習後はbest.ptと対応するコードzipをKaggleへ手動で追加し、[01_submit.ipynb](../notebooks/01_submit.ipynb) で隠しtestのキャッシュを新しく作る。[提出手順](kaggle-submit.md) を参照。公開CoAtNetの重みはこの経路のKneeMILへ読み込めない。

## 元画像を置ける別端末でのデータ準備とCSV監査

元画像を保存できる別端末・別ドライブを使用する場合の手順。現在のCドライブへ全取得する経路は選択していない。Kaggle Notebookなら競技Inputを追加してmountを利用できる。別ドライブのデータを使う場合は下の `data/raw` をその絶対パスへ読み替える。

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
