# 正規化と事前学習の対照実験

2026年10月4日 JST。既存192pxキャッシュから少数train診断とA/B/C比較を行う実行手順。データ・gold・公開重みの独立性に関する方針は [実験計画](research/next-experiments-20261004.md) を継承する。全DICOM取得やKaggleへの自動アップロード・提出は行わない。

## 完了した結果と判断

| 条件 | BCEで選択したepoch | weak検証BCE ↓ | weak 12所見macro AUC ↑ | 実行時間 |
|---|---:|---:|---:|---:|
| A：random＋legacy | 1 | 0.430481 | 0.557023 | 18.40分 |
| B：random＋ImageNet正規化 | 1 | 0.411992 | 0.644652 | 17.39分 |
| C：ImageNet事前学習＋同じ正規化 | 2 | **0.371395** | **0.766895** | 17.32分 |

各条件5 epochを新規runで完了。Aの全epoch BCEと採用予測CSVは旧e002と完全一致した。BはAより9/12所見、CはBより12/12所見のAUCが改善。Synovitisを除く補助11所見平均もA 0.548570→B 0.626893→C 0.747522であり、改善全体は陰性1件のSynovitisだけによるものではない。**Cを次のローカル比較基準として暫定採用**する。追加foldの再現性とKaggleの実採点は未確認。

CのAUC最大はepoch 5の0.774608だが、BCEは0.415741へ悪化した。保存する候補は事前のBCE規則で選んだepoch 2のまま。AUCの最大値を提出スコアの予測や保存checkpointの性能として記録しない。gold監査・選択は無効、今回のPublicは未測定。既存Public 0.924は公開モデルによる別の提出。

入力・source・head初期値・分割・欠損・epoch CSV・checkpoint選択の照合は [A/B/C監査集計](../experiments/e003-e005-controlled-summary-20261004.json)、所見別差分と採用理由は [判断記録](../experiments/e005-decision-20261004.json)。所要時間はcache事前検査を含み、Python importを除く。学習中の追加train-evalは各run約18秒で別記録している。

凍結code zipと実A/B/C checkpointを使った、人工ノイズ3検査・Reportなし12所見のオフライン提出契約も成功した。Cの初期化ファイルを存在しないpathに変えたQAコピーでも予測CSVが一致し、提出にImageNet初期化ファイルを追加する必要はない。[提出契約の集計](../experiments/controlled-submit-contract-20261004.json)

`artifacts/kaggle/e005-pretrained-imagenet-fold0/` にCの `best.pt`、`rsna-knee-code.zip`、hash/帰属記録、未実行 `01_submit_e005.ipynb` を準備済み。NotebookのInput名は作成候補 `rsna-knee-code-e005` / `rsna-knee-e005` に合わせてあり、手動追加後に実mount pathを確認する。新しいアップロード・Kaggle実行・提出は行っていない。操作は [提出手順](kaggle-submit.md)。

## 実装した内容

- `StudyDataset`は学習・推論共通のtensor正規化を使う。未指定は従来のlegacy、ImageNet modeは隣接3sliceのchannelごとにmean/stdを適用する。cache内のuint8画像と画像fingerprintは変えない。
- `KneeMIL`の構築と推論は常にオフライン。学習のImageNet初期化だけ明示したローカル重みをfull SHA-256と公式prefixで検査し、1000クラスfcを含めてstrict loadする。MIL headと乱数列は保持する。
- `diagnostics.enabled=true`は既存valid推論からepoch別CSV・所見別AUC・BCE・観測数・予測分布を保存する。soft labelはAUCで二値化しない。`train_eval_studies`は学習側の固定部分集合を同じevalモードで診断する。
- 追加train-evalは専用Generatorを使用し、global RNG、loader/sampler Generator、moduleごとのtrain/eval状態を復元する。人工データでログ有無のsample順、optimizer、model state、BN buffers、RNGの完全一致を確認した。
- `diagnostics.audit_gold=false`はgold cacheのpreflightと最後のgold評価を行わない。gold CSVのUID/groupは漏洩検査に必要なため読むが、モデル・checkpoint選択・prompt・ensemble調整には使わない。
- BN統計を固定するmodeも実装済み。affine parameterの学習は維持し、毎epochの`model.train()`でも固定が保たれる。A/B/Cでは全てupdateを使う。

## 1. キャッシュの完全性と互換性を確認する

```powershell
.\.venv\Scripts\python.exe scripts/verify_cache_export.py data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1 --config configs/experiments/e003-control-random.json
```

保存exportの全ファイル・保存コード・設定・CSV・UIDを検査し、その画像前処理が現行configと互換か確認する。`valid=true`、`export_integrity=true`、`preprocess_compatible=true`が必要。モデルやログだけを変えた場合の`source_matches_repository=false`は差分が明示される。保存export自体は変更しない。全コード一致まで要求する場合は`--require-source-match`を付ける。

## 2. 人工データでCUDAと提出契約を確認する

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_synthetic_gpu.py --output-dir artifacts/qa/controlled-pipeline-next --legacy-checkpoint artifacts/runs/e002-baseline-fold0/best.pt
```

出力先は新しい名前にする。生成ノイズだけでAMP/backward/optimizer/checkpoint、両正規化のReportなし提出、旧checkpoint互換、欠損ラベル・window mask・gold非読込みを確認する。これは医用画像の精度や実test時間の検査ではない。

## 3. 少数の学習側検査への適合を診断する

```powershell
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/q001-train-fit --config configs/experiments/q001-train-fit.json --fold 0 --diagnostic-studies 16 --reason "Training-partition memorization diagnostic; not holdout evaluation"
```

q001は完了済みで再実行不要。初期eval BCE 0.694520→最良0.124106（epoch 32）、最終train-mode BCE 0.000188／eval BCE 0.179478、146.78秒だった。[q001記録](../experiments/q001-train-fit.json)。再検証が必要なら新しいrun名を割り当てる。原manifestを変更せず、fold 0学習側だけからseed付きhash順で16検査を選ぶ。dropout=0、100 epoch、拡張なしで初期と各epochのeval BCEを記録する。選択UID一覧とhashはprivate runに残す。診断モードはgoldを評価せず、学習側自身のBCEでcheckpointを保存する。結果を独立CVやPublicへ読み替えない。

十分に適合しなければ、UID/cache/label/mask/gradient/updateを確認する。train-mode lossだけ低い場合は、同じ重みのeval lossとの差をBN統計等の診断へ使う。この段階の設定を通常のA/B/Cへ混ぜない。

## 4. A/B/Cを一条件ずつ比較する

| 条件 / run | 初期化 | 正規化 | 比較元 |
|---|---|---|---|
| A / e003-control-random | random | legacy | e002の再現と追加ログ |
| B / e004-random-imagenet | random | ImageNet | Aから正規化だけ変更 |
| C / e005-pretrained-imagenet | ImageNet1K V1 | ImageNet | Bからencoder初期化だけ変更 |

全てfold 0、seed 20261002、5 epoch、192px、3系列×8窓、batch 1、蓄積4、lr=1e-4、dropout=0.2、masked BCEで固定。train-evalは同じ64検査を使い、gold監査は無効。checkpoint選択はweak BCEのまま。

```powershell
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e003-control-random --config configs/experiments/e003-control-random.json --fold 0 --reason "A: reproduce random legacy baseline with neutral epoch diagnostics"
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e004-random-imagenet --config configs/experiments/e004-random-imagenet.json --fold 0 --reason "B: change input normalization only relative to A"
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e005-pretrained-imagenet --config configs/experiments/e005-pretrained-imagenet.json --fold 0 --reason "C: change encoder initialization only relative to B"
```

上記A/B/Cは完了済みで、再実行する必要はない。新しい条件は順に一つずつ実行する。runを上書きせず、比較元を見てから新しいrunを始める。Aで既存e002のbest予測が一致するか、B/CのMIL head初期hashが同じかを確認する。sourceを実行中に編集しない。

Cの重みは `artifacts/pretrained/resnet18-imagenet1k-v1-20261004/` に取得済み。公式URLのhash prefixを確認し、full SHA-256をconfigとprovenanceへ保存した。別端末で明示取得する場合は以下を使用する。

```powershell
.\.venv\Scripts\python.exe scripts/fetch_resnet18_weights.py --output-dir artifacts/pretrained/resnet18-imagenet1k-v1-20261004
```

取得済みなら再実行しない。初期化ファイルのpathだけ端末に合わせ、SHA-256は保存値と一致させる。コードのBSD licenseと学習dataset/重みの条件は区別する。[公式重みの説明](https://github.com/pytorch/vision/blob/v0.25.0/README.md#pre-trained-model-license)

## 保存物と結果の読み方

各runの`execution.json`と`run.json`へconfig・入力hash・source hash・環境・初期化・fold・所要時間を保存する。configの元ファイルhashと保存JSONのhashは区別する。コンソールログと開始/失敗記録は兄弟の`launches/`にあり、run作成前の失敗も残る。C終了後にwrapperの中断記録を修正し、今後はKeyboardInterrupt/SystemExitを`interrupted`、終了source不一致を`invalid_source_changed`として保存する。旧A/B/Cは当時の記録を保持し、集計で開始・run・現sourceの一致を検査した。

完成した3 runのCSV/JSONだけを再集計する場合は次を使う。今回の共有集計は作成済みなので出力先を新しくする。src/config/manifestを変えた後は当時の凍結コードと入力で監査し、現在sourceに合う記録として扱わない。

```powershell
.\.venv\Scripts\python.exe scripts/summarize_controlled_runs.py --runs artifacts/runs/e003-control-random artifacts/runs/e004-random-imagenet artifacts/runs/e005-pretrained-imagenet --output artifacts/qa/controlled-summary-next.json
```

`epochs/NNN/predictions.csv`と`metrics.json`は各epochのweak診断、`train_eval_metrics.json`は学習側診断。予測やUIDはGitへ入れず、集計・hash・採用理由だけ共有実験記録へ残す。`best.pt`と`weak_valid_predictions.csv`はweak BCEが最小のepochを指し、AUC最大epochを後から採用し直さない。

`elapsed_seconds`はepoch履歴の累積で、`extra_train_eval_seconds`を別記録する。`execution.json`の全体時間はcache事前検査を含み、Python importを除く。gold無効の今回と過去のgold評価込み18分は計測範囲が違う。

採用は同条件のweak指標・所見別結果・速度・追加foldの再現性で判断する。goldやPublicを期待値に変換せず、利用条件が確認できた候補でKaggleの実test decode・オフライン・全体時間・採点を別途確認する。

## 次の比較をどう進めるか

Cが多数所見で改善した場合は、解像度やencoderを増やす前にB/Cをfold 1で比較する。fold 1はSynovitisの陰性が25件あり、陰性1件のfold 0だけで判断する偏りを減らせる。分割を作り替えず、両モデルを初期化から学習する。fold 0のcheckpointをfold 1へ流用しない。患者独立性を証明するものではない。

以下は次段階の未実行コマンド。foldだけを変更し、学習・選択条件はB/Cのまま。run名が既に使われていれば新しい名前を選ぶ。

```powershell
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e006-random-imagenet-fold1 --config configs/experiments/e004-random-imagenet.json --fold 1 --reason "B fold1: confirm normalization control on a second fixed partition"
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e007-pretrained-imagenet-fold1 --config configs/experiments/e005-pretrained-imagenet.json --fold 1 --reason "C fold1: paired encoder-initialization comparison against B fold1"
```

所見別AUC、観測数、BCE、epoch推移を両foldで比べる。Synovitisを除く11所見平均は補助診断に限り、公式12所見平均の代用にしない。別foldでも改善が保たれればCを次の基準とする。train-evalも大きく揺れる場合はBN統計固定だけを変更する。BCE-bestとAUC-bestがずれる場合は、選択規則を実行前に決めた新規runで比較し、既存checkpointを後付けで選び直さない。

自作提出の動作確認はこの比較と並行して進める。[利用条件監査](research/kaggle-source-eligibility-20261004.json) により、現在の研究・学習目的とCC帰属等を維持して採用を進める判断は済んでいる。手動でInputを追加し、[提出手順](kaggle-submit.md) のInternet OFF実行から始める。少数人工入力の成功、Kaggle例示testの完走、本採点の成功・全体時間は別々に記録する。
