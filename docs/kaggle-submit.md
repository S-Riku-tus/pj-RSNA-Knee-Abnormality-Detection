# Kaggleでオフライン推論を提出する

公開モデルを使う場合は作者Notebookをそのまま再現する。以下は、このリポジトリの自作モデルで学習を終えた後の経路。提出Notebookには学習コードの呼び出しを含めない。

10月3日時点でe002のローカル梱包は完了。`artifacts/kaggle/e002-baseline-fold0/` に `best.pt`、`rsna-knee-code.zip`、`bundle-manifest.json`、`RESEARCH-ONLY.json` がある。学習時とコード・重みのhashが一致し、展開コード＋実checkpointから人工test 3検査の提出契約が成功した。Kaggleで実行した結果ではない。使用ラベルはCC BY-NC 4.0の非商用研究用で、大会利用の条件をまだ確認できていない。Rulesと利用条件を照合してから、以下の手動アップロードへ進む。既存公開モデルの実測Public 0.924とは別の資産である。

## コードを梱包する

```bash
python scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/e002-baseline-fold0
```

`artifacts/kaggle/e002-baseline-fold0/rsna-knee-code.zip` とhash一覧を生成する。e002は作成済みなので上記コマンドを同じ出力先へ再実行しない。別のrunでは新しい出力先を使い、前の版を上書きしない。学習に使ったsrcで梱包する。zipはsrcのPythonファイルとbaseline.jsonの許可リストから作り、データ・重み・認証情報を入れない。学習したbest.ptは別ファイルとして用意する。Notebookひな形は `notebooks/01_submit.ipynb`。実行済みNotebookではない。転送と学習からの一連の操作は [手順4〜9](after-cache.md) を参照する。

## GPU端末とKaggle画面での準備

1. コードzipをKaggleの入力Datasetへ追加する。
2. 学習した `best.pt` を別の入力Datasetへ追加する。必要に応じprivateで保持する。
3. 提出Notebookへ競技入力、コードDataset、checkpoint DatasetをAttachする。
4. NotebookのCHECKPOINTとコードInputを実際のmount pathへ変更する。zipが残っていればCODE_ZIP、展開済みのsrc/とconfigs/があればその親ディレクトリをCODE_ROOTに指定する。必要ならWHEELSも指定する。
5. GPUを有効にし、インターネットを無効にする。
6. 実行して `submission.csv` を検査し、Save and Run All後にSubmitする。
7. 採点成功とNotebook版を確認し、台帳へ記録する。

Dataset作成・アップロード・提出をこのプロジェクトのスクリプトは自動で行わない。競技入力は旧形式 `/kaggle/input/rsna-knee-abnormality-detection` と新形式 `/kaggle/input/competitions/rsna-knee-abnormality-detection` の両方を検出する。

## 追加パッケージが必要な場合

Kaggle標準環境のtorch、torchvision、numpy、Pillow、pydicomを先に確認する。圧縮DICOMのdecoderが不足する場合は、Kaggleと同じLinux/Pythonの準備環境で必要なwheelと推移依存を保存して入力Datasetへ追加する。

```bash
python -m pip download --only-binary=:all: --dest artifacts/wheels numpy pydicom Pillow pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
```

このコマンドは準備用GPU環境でネットワーク有効のときに実行する。Windows wheelをLinux Kaggleへ持ち込まない。NotebookのWHEELSへwheelディレクトリを指定すると `--no-index --find-links` でローカルインストールする。torch/torchvisionの大型wheelは標準環境との適合を確認してから扱う。

## 推論と時間の確認

checkpoint内のconfigを使って、その採点回のtest.csvとDICOMから新しいcacheを作る。固定件数・固定UID・GPU端末のtest cacheを使わない。列順・UID集合・行順・欠損・有限値・0〜1範囲を検査する。

Notebook全体の起動、ローカル依存インストール、モデル読み込み、DICOM decode、推論、CSV作成を時間に含める。例示3検査だけで9時間内に収まるとは判断しない。GPU端末でtrainから大きい検査も含む代表例を選び、前処理込みの時間を別途測る。現時点で隠しテスト規模に対する所要時間は未測定。

最終選択数や日次上限はKaggleのライブ画面を確認する。締め切り直前は新しい構成を増やさず、採点成功済みの候補を確保する。
