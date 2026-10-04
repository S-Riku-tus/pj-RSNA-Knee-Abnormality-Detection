# Kaggleでオフライン推論を提出する

公開モデルを使う場合は作者Notebookをそのまま再現する。以下は、このリポジトリの自作モデルで学習を終えた後の経路。提出Notebookには学習コードの呼び出しを含めない。

2026年10月4日 JST。4本の追加学習が完了し、通常BNのepoch 8と固定BNのepoch 9を組み合わせる固定50:50 rank候補 `r001` の提出資産を準備した。fold 0のweak AUCは0.830249、補助11平均は0.815727。所見別の入れ替わりを見た後の追加診断であり、別foldと自作Publicの改善は未確認。[結果と判断範囲](controlled-experiments.md#固定5050-rank候補と次の判断)。ユーザーの保存実行がrank helper参照で停止したため、実保存版と現在のInputを確認して修正版を用意した。

Version 3の保存実行はCOMPLETEで、表示用3検査の前処理・両GPU推論・最終CSV検査が成功した。GPU有効・Internet OFF、Outputにsubmission.csv（852 bytes）を公式APIで確認済み。[確認記録](../experiments/r001-kaggle-v3-review-20261004.json)。現在のNotebook/Inputを変更せず、Version 3のOutputで**最終submission.csvを選択してSubmit**する。predictions_bn_update.csv／predictions_bn_freeze.csvは単体予測なので、今回のrank候補の提出には選ばない。表示用3検査の完走は本番採点の完了ではなく、隠しtestはSubmit時に差し替わる。Publicと全体時間は提出後に記録する。

以下はVersion 2の参照先エラーを修正したときの手順。現在のコードInput全15ファイルはhashが正しく、ZIPや重みの再アップロード・再学習は不要。保存済みNotebook Version 2の最初のcode cellで、RANK_MODULEの行を次へ置き換えれば今回の参照先を修正できる。

```python
RANK_MODULE = CODE_ZIP.parent / "rank50.py"
```

CODE_ROOTは現在の`.../rsna-knee-code`設定を使う。rank50.pyはその展開フォルダの中ではなくInput直下にある。[原因・確認記録](../experiments/r001-kaggle-pathfix-20261004.json)。自分で編集しない場合は [01_submit_r001_pathfix.ipynb](../artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) をImportする。こちらは欠損/hash不一致の表示も改善し、後続推論・全期待hash・予測方式は同じ。修正後、既存InputとGPU有効/Internet OFFを確認し、新しいVersionでSave & Run Allする。

Inputファイルの元フォルダは [r001-bn-rank50-fold0](../artifacts/kaggle/r001-bn-rank50-fold0/)。以下は初めてInputを準備するときの3段階で、現在は1・2のアップロードが済んでいる。

1. privateコードDatasetを作り、`rsna-knee-code.zip`、`rank50.py`、`bundle-manifest.json` を追加する。提案名は `rsna-knee-code-r001`。共通rank関数はweak診断と同じファイルで、Notebookがzip・展開source・helperのhashを照合する。
2. private重みDatasetを作り、`best_bn_update.pt`、`best_bn_freeze.pt`、`asset-manifest.json` を追加する。提案名は `rsna-knee-r001`。config・checkpoint indexのコピーも版記録として同梱できる。manifestにラベル作者・版・CC URL・加工内容・初期化出所・2モデルのepochとhashを保存済み。推論にImageNet初期化ファイルは不要。
3. 未実行の [01_submit_r001_pathfix.ipynb](../artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) をKaggleへImportし、競技入力と上の2 InputをAttachする。初めのcellの `CODE_ZIP`、`CODE_ROOT`、`RANK_MODULE`、`CHECKPOINTS` を実mountと照合する。zipが自動展開された場合は `CODE_ROOT` を `src/` と `configs/` の親へ指定し、rank helperはInput直下のまま参照する。GPU有効・Internet OFFでSave and Run Allし、完走後にSubmit・採点確認へ進む。

コードInput `rsraki/rsna-knee-code-r001` と重みInput `rsraki/rsna-knee-r001` は作成済みでread-only確認した。再保存するNotebookに現在のInputがAttachされていることも確認する。Notebookは同じpixel fingerprintを確認してtest cacheを1個作り、2モデルを順に推論し、**その回の全test検査**を所見別に順位化して50:50で合わせる。係数・所見別weightを調整するcellはない。testを独立に順位化するbatchへ分割しない。rank scoreは未校正なので、BCEを確率校正改善の証拠にしない。

ローカルでは両実重みの人工CPU/GPU契約と、保存された人工CSVから共通helperを使うrank契約が成功した。rank専用6契約チェック、helper同一性、Notebook全code cellのcompile・空outputも確認済み。実testのDICOM decode・全体時間・Kaggle採点は別に確認する。Notebook/Input版・全体時間・Submission ID・実測スコアを [台帳](../experiments/ledger.csv) へ追記する。

固定BN単体の比較候補は [e009のv2資産](../artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) と未実行 `01_submit_e009.ipynb`。weak AUCは0.803614。2モデル候補の実行時間や動作に問題があれば、別の単体比較として実行できる。v2は元のNotebook説明文に残ったepoch誤記を直した版で、code cell・重み・凍結zipは同一。元のe009資産とe005/e008の旧候補は履歴として保持する。

e002の旧資産は `artifacts/kaggle/e002-baseline-fold0/` に `best.pt`、`rsna-knee-code.zip`、`bundle-manifest.json`、`RESEARCH-ONLY.json` として保存する。学習時とコード・重みのhashが一致し、展開コード＋実checkpointから人工test 3検査の提出契約が成功した。これはKaggle実行の結果ではない。`RESEARCH-ONLY.json` は当時の採用範囲の記録として残し、新しい候補は別のbundleへその学習に対応するsrcと重みを梱包する。既存公開モデルの実測Public 0.924も別の採点済み比較基準として保持する。

10月4日に公式SDKでRulesと [Hostの外部LLM案内](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) の本文・返信を確認した。HostはNC制限だけでデータを禁止せず、賞金があるだけで商用利用とは扱わない。vmohitrao Version 3は本大会の画像学習を想定して公開されたCC BY-NC 4.0のラベルであり、非商用の研究・学習目的と帰属・ライセンス表示等の通常条件を守って採用を進める。追加個別許可が一律必須という根拠は確認されていない。元MRI・レポートを再配布せず、資産の説明へ出所・版・ライセンスと変更内容を残す。将来の商用転用は別に判断し、入賞時の成果物公開条件との不整合はその時に確認する。[一次資料と判断範囲](research/kaggle-source-eligibility-20261004.json)、[CC BY-NC 4.0本文](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

## コードを梱包する

```bash
python scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/NEXT_RUN
```

新しいrunで使う場合に `NEXT_RUN` を未使用の名前へ変更し、code zipとhash一覧を生成する。今回のr001/e009 v2は既に学習時の凍結zipをコピー済みなので再梱包不要。凍結zip SHA-256は `9c85e2f120ba2eb9f0b3fc44b99ad328783f78026e46a52e8073b377d9bd496f`、共通helperは `02bf93907ffbf817bb4f8a2ef1f43cc207f3c121e86ab377cbdb8cf7a34939b9`。前の版を上書きせず、学習に使ったsrcで梱包する。zipはsrcのPythonファイルとbaseline.jsonの許可リストから作り、データ・重み・認証情報を入れない。r001専用Notebookは2重みとzip外のhelperを要求するため、単体ひな形へ置き換えない。転送と学習からの操作は [手順4〜9](after-cache.md) を参照する。

## GPU端末とKaggle画面での準備

1. コードzip・共通rank helperをKaggleの入力Datasetへ追加する。
2. 学習した2重みを別の入力Datasetへ追加する。privateで保持し、帰属/版記録も付ける。
3. 提出Notebookへ競技入力、コードDataset、checkpoint DatasetをAttachする。
4. NotebookのCHECKPOINTSとコード/helper Inputを実際のmount pathへ変更する。zipが残っていればCODE_ZIP、展開済みのsrc/とconfigs/があればその親ディレクトリをCODE_ROOTに指定する。必要ならWHEELSも指定する。hash guardを削除せず、Input版を照合する。
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

Notebook全体の起動、ローカル依存インストール、2モデル読み込み、DICOM decode、両推論、順位化、CSV作成を時間に含める。例示3検査だけで9時間内に収まるとは判断しない。元DICOMの時間測定はKaggleの学習側で大きい検査も含め、実testの全体時間も記録する。現時点で隠しテスト規模に対する所要時間は未測定。

10月4日に取得したRulesでは日次提出は5回、最終選択は2件。提出時のKaggle画面でも確認する。締め切り直前は新しい構成を増やさず、採点成功済みの候補を確保する。[公式Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
