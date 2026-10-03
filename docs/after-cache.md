# キャッシュ作成後の学習と提出手順

更新日 2026年10月3日 JST。先に説明した「1. 公開Notebookで採点を通す、2. ローカルCUDA環境を整える、3. Kaggleで画像キャッシュを作る」に続く作業を説明する。10月3日にCUDA利用可とKaggleの全件exportを確認し、ローカル取得・全件hash検査を完了した。実学習と自作モデルの採点は未実施。段階ごとの現状は [PROJECT.md](../PROJECT.md)、確認範囲は [validation.md](validation.md) を参照する。

元MRIの前処理と提出推論はKaggle、キャッシュからの学習はこのPCのRTX 4090を使う。ブラウザでKaggleを開いても、このPCのGPUでは計算されない。ローカルに全DICOMや例示testのキャッシュを置く必要はない。

| 段階 | 作業場所 | 作るもの | 次へ進む条件 |
|---|---|---|---|
| 4 | Kaggle → このPC | 検査済みの全件画像キャッシュ | 完全性・hash・ソース・前処理が一致 |
| 5 | このPC | 監査済みweakラベルと固定fold | 出所が確認でき、goldとgroupの混入がない |
| 6 | このPCのGPU | fold 0の学習結果とcheckpoint | loss・時間・出力を確認し、比較基準が残る |
| 7 | Kaggle | 自作モデルの採点成功 | Internet OFFで隠しtestの採点が完了 |
| 8 | このPCのGPU、Kaggle | 一要因ずつの比較実験 | 同じfold・seedで改善を説明できる |
| 9 | Kaggle | 最終候補と再現記録 | 採点成功済みの候補を期限前に選択 |

## 4. 全件キャッシュをダウンロードして検査する

### Kaggleで保存した実行のOutputを取得する

`00_prepare_cache.ipynb` の10検査の確認が済んだら、`LIMIT=None` で全件を作り、Save Version / Save and Run Allで保存した実行が完了したことを確認する。編集途中のセッションと保存版を区別し、NotebookのURL・版、競技Inputの版、コードDatasetの版を記録する。

その保存版のOutputから `rsna-cache-v1/` 全体を手動でダウンロードする。必要ならOutputからprivate Datasetを作成して保管する。画面の項目名は現在のUIで確認する。[Kaggle公式のOutput Dataset手順](https://www.kaggle.com/docs/datasets)

このPCでは、リポジトリ直下の `data/exports/rsna-cache-v1/` へフォルダ構造を維持して展開する。ダウンロードzipと展開後の両方を一時的に保存する場合、その分の空き容量も確認する。別のキャッシュ版は新しいフォルダへ保存する。

```text
data/exports/rsna-cache-v1/
  export.json
  audit.json
  raw/
    train.csv
    train_series.csv
    sample_submission.csv
  train-v1/
    cache.json
    coverage.json
    <StudyInstanceUID>.npz
  code/
    src/rsna_knee/*.py
    configs/baseline.json
```

`export.json` がこの直下にあることを確認する。zipの外側にもう一段フォルダが付いた場合は、実際に `export.json` があるフォルダを検査コマンドへ指定する。元DICOMはこの転送に含めない。

### 大きいZIPを取得できない場合のファイル単位転送

10月3日、ユーザーのprivate Notebook `rsraki/rsna-knee`、Version 1（URLのscriptVersionIdは354838181）から、CLI/APIで全件exportを取得・検査した。4,407検査、4,423ファイル、7,785,521,448 bytes。Notebook本体のダウンロードには画像キャッシュは含まれないため、Outputを別途取得する。

Kaggle公式CLIの `kernels output` はOutputをファイル単位で取得する。大きいZIPを使う必要はない。ただし、確認したCLI 2.2.4では版のsuffixを解析してもOutput要求へ渡していなかった。また、このNotebookにSDKの `version_label=1` を渡すと404になった。単に `/354838181` をCLIのslug末尾へ付ける方法にはしない。URLのscriptVersionIdとNotebookのVersion番号は別である。[公式CLI資料](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)

このリポジトリの [download_cache_output.py](../scripts/download_cache_output.py) は、取得前後に最新版が要求したVersion番号であることを確認する。最新版が異なれば停止するため、対象版の取得が完了するまでは新しいNotebook版を保存しない。すでに最新版が変わった場合は、対象版のOutputからprivate Datasetを作成するなど、対象を固定できる別の経路を検討する。

CLIは学習環境と分け、Git除外の `artifacts/tools/kaggle-venv/` に導入済み。新しい端末で準備する場合は次を実行する。

```powershell
uv --cache-dir .uv-cache venv --python .venv/Scripts/python.exe artifacts/tools/kaggle-venv
uv --cache-dir .uv-cache pip install --python artifacts/tools/kaggle-venv/Scripts/python.exe kaggle==2.2.4
.\artifacts\tools\kaggle-venv\Scripts\kaggle.exe auth login
```

ログインは利用者がブラウザで行う。CLIは通常の認証機構を使い、スクリプトは認証値を表示・コピーしない。APIキーやトークンをチャットやGitへ保存しない。[公式認証資料](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md)

まずmetadataだけを確認し、その後に全ファイルを取得する。Version番号とscriptVersionIdはこの保存版専用の値であり、別の実行では実際の値へ変える。

```powershell
.\artifacts\tools\kaggle-venv\Scripts\python.exe scripts/download_cache_output.py --kernel rsraki/rsna-knee --version 1 --script-version-id 354838181 --dest data/exports/rsraki-rsna-knee-sv354838181 --metadata-only
.\artifacts\tools\kaggle-venv\Scripts\python.exe scripts/download_cache_output.py --kernel rsraki/rsna-knee --version 1 --script-version-id 354838181 --dest data/exports/rsraki-rsna-knee-sv354838181 --workers 4
```

取得先は `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/`。転送の記録はその一つ上の `transfer.json` に保存する。対象版・export hashが同じ場合、同じコマンドを再実行してSHA-256が一致する取得済みファイルを再利用できる。不一致のファイルは再取得し、検証した `.part` を置き換える。ファイル途中からのHTTP Range再開ではなく、完成したファイルを単位とする再開である。

取得器はexportの列挙ファイルだけを4並列で取得し、全ページの一覧、サイズ・hash、保存先の範囲、空き容量を検査する。元DICOM・モデルの取得、MRIデコード、学習、外部へのアップロード・提出は行わない。全件取得後に既存のexport検査器も実行し、成功した場合のみ `transfer_complete=true` を記録する。

この取得先を使う場合、以降に記載した `data/exports/rsna-cache-v1/` はすべて `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` に読み替える。移動して既存版を上書きする必要はない。

### ローカルで転送の検査を行う

以降のPowerShellコマンドは、リポジトリのルートで実行する。

```powershell
.\.venv\Scripts\python.exe scripts/verify_cache_export.py data/exports/rsna-cache-v1
```

成功すると `valid: true`、検査数、検査したファイル数が表示される。検査器は次を確認する。

- `complete=true` の全件exportである。10検査のdebug exportは受け付けない。
- ファイル一覧・サイズ・SHA-256がexport記録と一致する。
- 元CSVと画像キャッシュの検査ID集合・件数が一致する。
- 現在の `src/rsna_knee/` のPythonソースがexportに記録したソースと一致する。
- 学習用cacheの前処理fingerprintが記録したconfigと一致する。

不一致はチェックを外して進めず、転送欠落なら再取得、ソース違いならキャッシュ生成時のコードへ合わせる。資料だけを更新したcommitではsrcのhashは変わらないが、srcを変更した場合は初回検査にも影響する。キャッシュ生成時のGit commitも手動で残しておく。

検査後に `audit.json` と `train-v1/coverage.json` を読む。CSV件数、空のレポート、公式ラベルの欠損、シリーズ失敗・fallbackを確認する。hashの一致は画像内容の正しさを保証しないため、手順3で原画像とキャッシュを目視した結果も残す。完全性・前処理・目視の確認が揃えば手順5へ進む。

## 5. weakラベルを監査し、学習と検証の分割を固定する

### 公開ラベルの採用元を決める

現在のリポジトリには、学習に使えるweakラベルを取得した記録も、ラベル生成器もない。`label-template` は空欄を作るだけなので、その出力で学習はできない。

まず [公開モデルの再現手順](public-baselines.md) に挙げた作者の学習NotebookとInputsを確認する。候補を探す入口として [Pilkwang baselineのInputs](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1/input) も使えるが、この資料では特定CSVの版・ライセンス・gold非使用を確認して採用したわけではない。

採用前に次を確認し、根拠となるURLと説明を残す。

- DatasetのURLと版、CSV名とhash、利用条件。
- ラベルの生成方法。レポートをどのように12所見へ対応させ、未言及・否定・曖昧な記述をどう扱ったか。
- 公式goldラベルをコピーしていないか。抽出ルール、LLM prompt、閾値などをgoldに合わせて調整していないか。
- 元のCSVが0/1ラベルかsoft labelか。欠損、UID重複、未知のUID、列の意味に問題がないか。

確認できたCSVを `data/labels/weak_labels.csv` へ用意する。元CSVのコピーと採用した版の説明も `data/labels/` 以下へ保存する。実際の列構造を見て変換し、別の公開ラベルのschemaを推測して流用しない。

取り込み形式は次の13列である。

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

所見の値は有限の0〜1、欠損は空欄にする。未言及を一律0にせず、採用元の定義を確認する。UIDは文字列として保持し、Excelの自動変換や行番号での結合を避ける。対応は `StudyInstanceUID` で取る。0.5のsoft labelと欠損は別の意味である。

### 出所の記録を埋める

`configs/label-provenance.example.json` を参考に `data/labels/provenance.json` を作り、`source_url`、`source_version`、`license`、`method`、`created_at`、`gold_used_for_tuning` を確認して記入する。

テンプレートの `gold_used_for_tuning=false` は確認済みの事実ではない。未確認ならそのまま使わず、作者説明・生成Notebookを調べる。現行CLIはこの項目に真偽値を要求するため、未確認をfalseとして通す運用はしない。調整にgoldを使ったことが判明した場合はtrueとし、そのgold評価は探索的な結果として区別する。独立した評価が必要ならgold非使用を確認できる別のラベルを選ぶ。

### prepareを実行する

```powershell
.\.venv\Scripts\python.exe -m rsna_knee prepare --train-csv data/exports/rsna-cache-v1/raw/train.csv --weak-labels data/labels/weak_labels.csv --provenance data/labels/provenance.json --output-dir data/manifests/v1 --config data/exports/rsna-cache-v1/code/configs/baseline.json
```

同一患者などの対応が供給・確認されている場合は、全検査を一度ずつ含む `StudyInstanceUID,group_id` のCSVを用意し、上のコマンドへ `--groups data/labels/groups.csv` を追加する。対応情報のない検査を勝手に同一患者と推定しない。

出力は `data/manifests/v1/weak.csv`、`gold.csv`、`manifest.json`。現行処理は公式画像由来ラベルが一つでもある検査をgoldへ予約し、それとgroupでつながる検査もweak学習から除外する。同一レポートは空白・大小文字を正規化してまとめ、供給groupとの連結も保つ。患者対応がない場合、これだけで患者独立を保証できない。

`manifest.json` でfold別件数と除外理由を確認する。欠損ラベルしかない行やgoldとつながる検査の除外により、weak件数は元CSVの件数より少なくなる。

現行foldはgroupとseedからのhash分割で、ラベルの層化分割ではない。各fold・各所見の観測数、欠損率、0/1ラベルの陽性・陰性数、soft labelの分布を別途集計して偏りを確認する。fold件数以外の分布集計は現行CLIにはない。偏りが大きければ学習前に分割方法を改善し、新しいmanifest版にする。全foldを埋めるために欠損を0へ変換しない。

出所、分割、gold除外、各クラスの観測数を確認したら、このmanifestを比較実験の間は固定する。ラベルやgroupを変えた場合は `data/manifests/v2/` など新しい版を作り、旧版を残す。

## 6. fold 0で学習の動作を確認し、比較基準を作る

### 最初は1 epochの別runにする

先に [CUDAと人工画像forwardの確認](gpu-start.md#環境準備) を終える。現行モデルはランダム初期化ResNet18と所見別Attentionで、公開CoAtNet/DINOの再現モデルではない。最初の目的は、画像・ラベル・分割・学習・推論が接続されていることと、時間・VRAM・lossを確認すること。

キャッシュに保存したconfigをコピーし、学習epochだけ1へ変更する。次の例は新しいconfigへ保存し、既存ファイルがあれば止める。

```powershell
New-Item -ItemType Directory -Force configs/experiments | Out-Null
$smokeConfigPath = "configs/experiments/e001-smoke.json"
if (Test-Path -LiteralPath $smokeConfigPath) { throw "新しい実験IDのconfigを指定してください" }
$smokeConfig = Get-Content -Raw -Encoding UTF8 data/exports/rsna-cache-v1/code/configs/baseline.json | ConvertFrom-Json
$smokeConfig.train.epochs = 1
$smokeConfig | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -LiteralPath $smokeConfigPath

.\.venv\Scripts\python.exe -m rsna_knee train --manifest-dir data/manifests/v1 --cache-dir data/exports/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e001-smoke-fold0 --config configs/experiments/e001-smoke.json --fold 0
```

この1 epochもfold 0以外の全weak学習検査とfold 0の全検証検査を処理する。学習CLIには少数検査だけの `--limit` はない。数検査のbackwardだけで済ませたい場合は、groupとgold除外を保った人工データ検証や専用の動作確認処理を別途用意する必要がある。Kaggleの10検査debug exportをこの学習へ渡す手順にはしない。

確認する点は、CUDA上でbackwardとoptimizer更新が進むこと、lossが有限であること、VRAM・時間に無理がないこと、checkpointと予測が保存されること。長時間処理の所要時間を判断するときは、`history.json` のepoch時間に加え、cacheの事前検査や最後のgold評価を含むコマンド全体の時間も測る。

OOMなどで失敗した場合はログを残し、新しいrun名で直した条件を試す。batch size、AMP、画像サイズなどを一度に変更しない。画像サイズを変えるとcache再作成が必要なので、最初は学習側の設定と原因を確認する。初期batch sizeは1、gradient accumulationは4。

### 問題なければ5 epochの比較基準を作る

```powershell
.\.venv\Scripts\python.exe -m rsna_knee train --manifest-dir data/manifests/v1 --cache-dir data/exports/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e002-baseline-fold0 --config data/exports/rsna-cache-v1/code/configs/baseline.json --fold 0
```

これは1 epochのrunを再開するコマンドではなく、同じseed・分割で最初から行う5 epochのrun。学習再開は未実装であり、既存runを上書きしない。途中停止したrunの `best.pt` だけからoptimizer状態を復元することもできない。

| ファイル | 確認する内容 |
|---|---|
| `config.json`, `run.json` | 設定、fold、ラベル由来、入力・ソースhash、GPU・依存の版 |
| `history.json` | epochごとのtrain loss、weak検証loss、経過時間 |
| `best.pt` | weak検証masked BCEが最小のepochの重み |
| `weak_valid_predictions.csv` | best checkpointでのfold 0予測 |
| `gold_predictions.csv`, `gold_metrics.json` | goldが存在する場合の学習後の評価 |

checkpointの選択はweak検証masked BCEで行い、goldは学習終了後に選ばれたbest checkpointを評価する。train lossとweak検証lossは集計方法も異なるため、値の差だけで過学習を断定しない。1 epochで良い精度を期待したり、公開モデルの報告スコアと直接比較して実装の良否を決めたりしない。

`experiments/TEMPLATE.md` を使ってデータを含まない実験メモを作り、`experiments/ledger.csv` に1 runごとの結果を記録する。Git commitは `run.json` に自動記録されないため、`git rev-parse HEAD` で確認してメモへ記入する。成功した環境の `uv --cache-dir .uv-cache pip freeze --python .venv/Scripts/python.exe` の出力もrunへ保存する。

### 三つの評価を混ぜない

| 評価 | 主な目的 | 注意する点 |
|---|---|---|
| weak検証masked BCE | 同一ラベル版・foldでの比較、checkpoint選択 | レポート由来ラベルへの適合で、公式AUCとは異なる |
| goldの所見別AUC | 画像由来ラベルとの整合の監査 | 陽性・陰性数が少ないクラス、公開資産のgold利用を確認 |
| 自分のPublic LB | 隠しtestでの外部評価 | 作者の報告値とは別に記録し、連続した微調整へ使い過ぎない |

goldで陽性または陰性がない所見のAUCは未定義。`macro_auc_12` は12所見すべて定義できる場合のみ出る。`available_class_macro_auc` は対象クラス数が異なる補助値なので、公式の12所見平均へ読み替えない。soft labelを0.5で二値化したAUCも公式評価と同じ意味にはならない。

## 7. 自作モデルをKaggleで採点する

fold 0の動作確認後、長い改善実験の前に、少なくとも一つの自作checkpointで提出経路を確認する。公開Notebookでの採点成功と、自作 `01_submit.ipynb` の採点成功は別の確認である。

### コードと重みを準備する

学習に使ったsrcと一致する状態で、コードzipを新しい出力先へ作る。

```powershell
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/e002-baseline-fold0
```

手動でKaggleのprivate Input Datasetへ次を追加し、版とhashを記録する。

- `artifacts/kaggle/e002-baseline-fold0/rsna-knee-code.zip`
- `artifacts/runs/e002-baseline-fold0/best.pt`

checkpointにはその学習で使ったconfigとラベル順が入っている。梱包されるbaseline.jsonを後から書き換えて実験条件を置き換える運用にはしない。

### 提出Notebookを設定する

1. [01_submit.ipynb](../notebooks/01_submit.ipynb) をKaggleへ読み込む。
2. 競技Input、コードDataset、重みDatasetをAttachする。
3. `CHECKPOINT` を実際の `best.pt` のmount pathに変更する。
4. zipが残るInputなら `CODE_ZIP`、展開済みなら `src/` と `configs/` を含む親フォルダを `CODE_ROOT` に指定する。
5. GPUを有効、InternetをOFFにして実行する。圧縮DICOMのdecoderが不足する場合は、[オフライン依存の準備](kaggle-submit.md#追加パッケージが必要な場合) に従い、Kaggleと同じLinux・PythonのwheelをInputに追加する。Windows向けwheelを持ち込まない。
6. `submission.csv` の検査成功を確認し、Save and Run Allで保存した版を手動Submitする。
7. My Submissionsで採点の完了・成功を確認する。スコア、Notebook版、Input版、全体時間、提出日時を台帳に残す。

提出Notebookはその採点回の `test.csv` とDICOMからtest cacheを作る。例示testの固定UIDやローカルtrain cacheを隠しtestの代わりに使わない。推論にReportは要求しない。

公式条件はInternet OFF、CPU/GPU Notebookとも9時間以内、出力名 `submission.csv`。Notebook内では学習を行わず、起動・依存導入・DICOM decode・予測・CSV作成の全体を時間に含める。例示の少数検査だけで全隠しtestの時間を判断せず、大きな検査も含めて計測し、実際の採点成功を確認する。[公式Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)

このNotebookは自作KneeMIL専用であり、公開CoAtNet/DINOのcheckpointをそのまま読めない。公開モデルは作者Notebookで再現し、学習コードを移植する場合は別実験として実装と資産の由来を監査する。

## 8. 同じ条件で一要因ずつ改善する

最初から5foldと大量の条件を回す必要はない。fold 0の比較基準、自作提出の成功、1 epochと提出の実測時間が揃ったら、残り日数で実行できる候補数を決める。比較元のconfig・manifest・seed・foldを固定し、変更理由と採用条件を実行前に一行書く。

次の順序は提案であり、実画像の監査と計測結果に応じて変える。

| 優先 | 試す内容 | 理由と条件 | 現行実装 |
|---|---|---|---|
| 1 | 入力画像・シリーズ失敗・ラベル対応の修正 | 所見が消える入力や誤ラベルがあるとモデル比較が成立しない | 監査・cacheはある。修正内容によって追加実装 |
| 2 | 汎用事前学習encoder | ランダム初期化だけでの少量データ学習を改善する候補 | 未実装。重みの由来・学習対象を確認して追加 |
| 3 | epoch数またはlearning rate | 同じ入力で学習曲線に応じて調整 | configで変更可能。変更は一つずつ |
| 4 | シリーズ数、窓数、解像度、crop | 見せる情報と推論時間の交換条件を比較 | 数・解像度はconfig、cropは追加実装。新しいcacheが必要 |
| 5 | ラベル版・抽出方法 | 弱教師の誤りや未言及の扱いを改善する候補 | ラベル生成器は未実装。新manifestで旧版と区別 |
| 6 | 少数モデルのensemble | 誤りが異なる候補を組み合わせる | 未実装。予測の整合・速度・採用根拠を別途検証 |

同じcacheを再利用できるのは、epoch数・learning rateなど前処理を変えない条件。解像度・series・windows・前処理ソースを変える場合は、新しいcache版と対応する提出コードをKaggleで作る。公開作者のcacheは形状と前処理契約が異なる可能性があるため、このcacheとしてそのまま扱わない。

公開競技checkpointがすでに全weak検査を学習していたら、自分のfoldの検証画像にも露出している可能性がある。その重みからfine-tuneして自分のfoldを評価しても独立したCVとは呼べない。汎用事前学習からfoldごとに学習する経路と、公開競技重みの提出再現を記録上区別する。

有望な一つか二つの条件だけを追加foldで確認する。たとえばfold 1を調べるなら、比較元も変更後も `--fold 1` で別runとして学習する。fold 0で選んだモデルをfold 1の独立検証として評価する運用にはしない。foldを追加しても現行提出Notebookに自動ensembleされるわけではない。

採用理由はweak検証の同条件比較と速度を中心に記録し、gold・Public LBは整合と外部評価として見る。小さなgold集合やPublic LBを何度も見てprompt・閾値・重みを合わせ込むと独立評価を失うため、微差だけで次々に条件を選ばない。候補ごとに全所見の結果、未定義クラス、失敗率、前処理込みの時間を残す。

## 9. 採点成功済みの候補を固定して締め切りに備える

2026年10月2日の公式確認では、参加・チーム統合期限は **10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。UTCでそれぞれ10月15日・22日の23:59をJSTへ換算している。変更があり得るため、競技画面でも確認する。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

提案する進行は次のとおり。遅れた場合は実験数を減らし、提出確認の時間を確保する。

- 10月5〜8日：転送検査、ラベル監査、固定分割、fold 0学習、自作モデルの初回採点。
- 10月9〜15日：一要因ずつ少数の改善。未確認の入力や分割の問題があれば先に解決する。
- 10月16〜19日：有望条件を追加foldで確認し、必要な場合だけ少数ensembleを実装・計測する。
- 10月20〜22日：重み・コード・依存の版を固定し、Internet OFFで再実行・採点。10月22日中に最終選択を終える。

最終選択数、日次提出回数、残りGPU枠はアカウントの現在の画面で確認する。採点成功済みの予備候補を残し、未採点の変更を最後の候補にしない。提出ボタンを押したことと採点完了は別である。

最終メモには、成功したSubmission ID、Notebook/Inputの版、Git commit、config、seed・fold、ラベル由来、checkpoint hash、実測スコア、全体時間、採用理由を残す。データ・レポート・UID一覧・cache・重みはGitへ入れず、共有できるコード・設定・集計・手順だけをmainに残す。

現時点で未確認なのは、実ラベルの採用元、実MRIのdecoderと画像品質、CUDAの学習動作、実測精度・時間、Kaggle採点結果である。各段階の完了条件を実際の結果で埋めてから次へ進む。
