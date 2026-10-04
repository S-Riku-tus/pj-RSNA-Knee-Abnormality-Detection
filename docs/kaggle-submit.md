# Kaggleでオフライン推論を提出する

公開モデルを使う場合は作者Notebookをそのまま再現する。以下は、このリポジトリの自作モデルで学習を終えた後の経路。提出Notebookには学習コードの呼び出しを含めない。

2026年10月4日 JST。A/B/Cの対照実験が完了し、C（ImageNet事前学習）のepoch 2を次の自作比較基準に暫定採用した。weak AUCは0.766895でPublicではない。[結果と採用理由](controlled-experiments.md)。凍結コードと実checkpointの人工提出契約も成功し、Cのローカル提出資産を準備済み。新しいアップロード、自作NotebookのKaggle実行・提出はまだ行っていない。

今回使うフォルダは `artifacts/kaggle/e005-pretrained-imagenet-fold0/`。以下の3段階で進める。

1. privateコードDatasetを作り、`rsna-knee-code.zip` と `bundle-manifest.json` を追加する。提案名は `rsna-knee-code-e005`。
2. private重みDatasetを作り、`best.pt` と `asset-manifest.json` を追加する。提案名は `rsna-knee-e005`。manifestにラベル作者・版・CC URL・加工内容・初期化出所を保存してある。ImageNet初期化ファイルの追加は不要。
3. 同フォルダの未実行 `01_submit_e005.ipynb` をKaggleへImportし、競技入力と上の2 InputをAttachする。提案名とrsrakiのpathを仮設定済みなので、実mountと照合する。zipが自動展開された場合は `CODE_ROOT` を `src/` と `configs/` の親へ指定する。GPU有効・Internet OFFで実行し、完走後にSave and Run All・Submit・採点確認へ進む。

Input名はまだ作成しておらず提案値である。例示testの完走と本採点の成功は分け、Notebook/Input版・全体時間・Submission ID・実測スコアを台帳へ追記する。

e002の旧資産は `artifacts/kaggle/e002-baseline-fold0/` に `best.pt`、`rsna-knee-code.zip`、`bundle-manifest.json`、`RESEARCH-ONLY.json` として保存する。学習時とコード・重みのhashが一致し、展開コード＋実checkpointから人工test 3検査の提出契約が成功した。これはKaggle実行の結果ではない。`RESEARCH-ONLY.json` は当時の採用範囲の記録として残し、新しい候補は別のbundleへその学習に対応するsrcと重みを梱包する。既存公開モデルの実測Public 0.924も別の採点済み比較基準として保持する。

10月4日に公式SDKでRulesと [Hostの外部LLM案内](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) の本文・返信を確認した。HostはNC制限だけでデータを禁止せず、賞金があるだけで商用利用とは扱わない。vmohitrao Version 3は本大会の画像学習を想定して公開されたCC BY-NC 4.0のラベルであり、非商用の研究・学習目的と帰属・ライセンス表示等の通常条件を守って採用を進める。追加個別許可が一律必須という根拠は確認されていない。元MRI・レポートを再配布せず、資産の説明へ出所・版・ライセンスと変更内容を残す。将来の商用転用は別に判断し、入賞時の成果物公開条件との不整合はその時に確認する。[一次資料と判断範囲](research/kaggle-source-eligibility-20261004.json)、[CC BY-NC 4.0本文](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

## コードを梱包する

```bash
python scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/NEXT_RUN
```

新しいrunで使う場合に `NEXT_RUN` を未使用の名前へ変更し、code zipとhash一覧を生成する。e002とe005は作成済みなので再実行不要。前の版を上書きせず、学習に使ったsrcで梱包する。zipはsrcのPythonファイルとbaseline.jsonの許可リストから作り、データ・重み・認証情報を入れない。学習したbest.ptは別ファイルとして用意する。Notebookひな形は `notebooks/01_submit.ipynb`。転送と学習からの操作は [手順4〜9](after-cache.md) を参照する。

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

10月4日に取得したRulesでは日次提出は5回、最終選択は2件。提出時のKaggle画面でも確認する。締め切り直前は新しい構成を増やさず、採点成功済みの候補を確保する。[公式Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
