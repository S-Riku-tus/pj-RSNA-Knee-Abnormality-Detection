# 正規化と事前学習の対照実験

2026年10月4日 JST。既存192pxキャッシュから少数train診断とA/B/C比較を行う実行手順。データ・gold・公開重みの独立性に関する方針は [実験計画](research/next-experiments-20261004.md) を継承する。全DICOM取得やKaggleへの自動アップロード・提出は行わない。

先行A/B/Cと「次の比較」「追加分析」の節は、10月4日の続行前の記録として残している。「未実行」「提案」「epoch 5の重み未保存」はその時点の状態。続行後の最新状態は末尾の「続行後の実行結果」を参照する。

## 完了した結果と判断

以下は先行A/B/Cについて、10月4日の続行前に記録した結果と判断。

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

`epochs/NNN/predictions.csv`と`metrics.json`は各epochのweak診断、`train_eval_metrics.json`は学習側診断。予測やUIDはGitへ入れず、集計・hash・採用理由だけ共有実験記録へ残す。先行A/B/Cの`best.pt`と`weak_valid_predictions.csv`はweak BCEが最小のepochを指し、AUC最大epochを後から採用し直さない。新規e008/e009の事前AUC規則は末尾に記す。

`elapsed_seconds`はepoch履歴の累積で、`extra_train_eval_seconds`を別記録する。`execution.json`の全体時間はcache事前検査を含み、Python importを除く。gold無効の今回と過去のgold評価込み18分は計測範囲が違う。

採用は同条件のweak指標・所見別結果・速度・追加foldの再現性で判断する。goldやPublicを期待値に変換せず、利用条件が確認できた候補でKaggleの実test decode・オフライン・全体時間・採点を別途確認する。

## 次の比較をどう進めるか

以下は10月4日の続行前の計画。実行済み部分の現状は末尾へ追記している。

Cが多数所見で改善したため、B/Cをfold 1でも比較する。fold 1はSynovitisの陰性が25件あり、陰性1件のfold 0だけで判断する偏りを減らせる。分割を作り替えず、両モデルを初期化から学習する。fold 0のcheckpointをfold 1へ流用しない。患者独立性を証明するものではない。以下は旧BCE規則で事前学習を再確認する比較で、追加の長期学習と公開方式の調査は後述する。

以下は続行前時点の未実行コマンドとして残す。e006/e007は後述のとおり完了しているので既存runへ再実行しない。foldだけを変更し、学習・選択条件はB/Cのまま。新たに実行する場合は未使用のrun名を選ぶ。

```powershell
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e006-random-imagenet-fold1 --config configs/experiments/e004-random-imagenet.json --fold 1 --reason "B fold1: confirm normalization control on a second fixed partition"
.\.venv\Scripts\python.exe scripts/run_local_experiment.py --manifest-dir data/manifests/v1-vmohitrao-research --cache-dir data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/train-v1 --run-dir artifacts/runs/e007-pretrained-imagenet-fold1 --config configs/experiments/e005-pretrained-imagenet.json --fold 1 --reason "C fold1: paired encoder-initialization comparison against B fold1"
```

所見別AUC、観測数、BCE、epoch推移を両foldで比べる。Synovitisを除く11所見平均は補助診断に限り、公式12所見平均の代用にしない。別foldでも改善が保たれればCを次の基準とする。train-evalも大きく揺れる場合はBN統計固定だけを変更する。BCE-bestとAUC-bestがずれる場合は、選択規則を実行前に決めた新規runで比較し、既存checkpointを後付けで選び直さない。

自作提出の動作確認はこの比較と並行して進める。[利用条件監査](research/kaggle-source-eligibility-20261004.json) により、現在の研究・学習目的とCC帰属等を維持して採用を進める判断は済んでいる。手動でInputを追加し、[提出手順](kaggle-submit.md) のInternet OFF実行から始める。少数人工入力の成功、Kaggle例示testの完走、本採点の成功・全体時間は別々に記録する。

## 追加分析に基づく優先順位（10月4日）

以下は同日の続行前の調査と提案であり、「未実行」等は当時の状態を表す。

今回の添付分析をHEAD `05b0710`、保存済みepoch評価、実装と一次資料に照合した。調査記録は [strategy-audit-20261004.json](research/strategy-audit-20261004.json)、今回の原文は [user-analysis-review-20261004.txt](research/user-analysis-review-20261004.txt)。新規の学習・実MRIデコード・データ/重み取得・Kaggle実行・提出は行っていない。

| Cのepoch | weak BCE | weak 12所見macro AUC | 読み方 |
|---|---:|---:|---|
| 1 | 0.402065 | 0.649943 | 初期の学習 |
| 2 | 0.371395 | 0.766895 | 当初のBCE選択で保存したモデル |
| 3 | 0.390265 | 0.771903 | BCEとAUCが乖離 |
| 4 | 0.389531 | 0.771094 | AUCも単調増加ではない |
| 5 | 0.415741 | 0.774608 | 診断上のAUC最大。重み未保存 |

epoch 2→5のSynovitis AUCは0.98→0.96と低下した。補助11所見平均は0.747522→0.757754（+0.010232）なので、12平均+0.007713はSynovitisだけの改善ではない。ただし7所見が改善、5所見が悪化し、Medial Meniscusは0.695485→0.647308（−0.048177）。全観測セルBCE悪化の約77.8%も当該所見の悪化で説明される。全体の確率校正だけの問題と決めつけない。

固定済みモデルの予測を同じ812 report/group単位で3,000回paired bootstrapした。補助11所見のC2−B1は+0.120629、percentile 95%区間[+0.078707,+0.162513]。C5−C2は+0.010232、区間[−0.005521,+0.025498]であり、長期化の改善は確証ではない。12平均はSynovitis陰性1groupが再標本から欠ける場合に未定義となり、有効な標本だけの区間は条件付きになる。患者独立性・教師の正確さ・同じfoldでの設定/epoch選択の偏りもこの区間では保証できない。

次のローカル実験は以下の順序を勧める。公開方式の版/Input監査と手動のKaggle動作確認は並行して進める。

| 順序 | 作業・一要因比較 | 判断材料 |
|---|---|---|
| 1 | 新規runへbest_bce・best_auc・lastモデルと実更新数の保存を追加 | AUCが良いepochの重みを利用できる |
| 2 | B/Cをfold 1で各5 epoch、旧BCE選択のまま再確認 | 事前学習の効果が別の固定分割でも出る |
| 3 | Cを初期化から20 epoch。epoch数以外を維持 | 5/10/15/20時点の順位付け・所見別の変化 |
| 4 | 同じ20 epochからBN running statistics固定だけを変更 | 学習/推論統計への感度。有望ならfold 1でも対照比較 |
| 5 | 同解像度の全体像と物理crop、次に解像度/系列密度等を個別比較 | 特にMedial Meniscus等の情報が入力に残るか |

長期runはweak 12所見macro AUC最大、同値ならBCE最小、さらに同値なら早いepochを選ぶ規則を開始前に記録する。12所見すべてが定義されなければ12平均を作らない。補助11所見平均・所見別変化・別foldを併記し、goldを選択に使わない。旧A/B/Cの採用epochを後付けで変更しない。

この調査時点のCのbest.ptはepoch 2だけで、epoch 5の完全な学習状態は残っていなかった。20 epochは同じ初期化からの新規runとし、最初の5 epochが旧Cの診断値/予測hashを再現するか確認する計画とした。保存や追加評価でRNG・sample順・学習状態を変えない。完全再開は別機能とし、必要時にepoch境界のmodel/optimizer/scaler/RNG/Generator/選択履歴/hashを保存して人工データで連続実行との一致を確認する。lastモデルだけを正確な再開状態と呼ばない。現行設定にschedulerはない。

5 epochが約17分という既存実測から、fold 1の2本と20 epochの2本は合計約3時間の粗い見積りになる。所要時間の保証ではないが、長期化とBNの疑問を調べる費用は小さい。20 epochで改善が弱い/所見の悪化が再現する場合は、50/100 epochやschedulerの広い探索に進む前に入力設計を優先する。有望な条件だけ追加foldへ広げる。固定分割で学習seedだけを変える前にconfig/manifestのseed分離が必要。

[公式Host回答](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)は、正解をMRIから独立に付け、Reportと食い違えば画像ラベルを正とすることを確認している。weak検証はReportからの教師への適合を測り、Publicを代替しない。既存Public 0.924を保持し、自作候補はInternet OFF・実test前処理・全体時間・採点を早期に確認する。公開競技重みを自作foldへ流用した値も、学習検査との重複が不明なら独立CVと呼ばない。

公開CoAtNetの前処理は140mm crop→336px cache→384pxモデル入力、5系列slot・64枚・6〜94%の範囲。一方、既存監査で保存した公開Training V8は44枚・15〜85%の入力corpus、未配布softラベルに依存し、gold AUCでbest/top-kを選ぶ。推論再現、学習手順の移植、配布重みの完全学習再現を区別する。現在の方針で学習手順を参考にする場合、汎用事前学習から始め、weak選択へ置き換える。[配布カード](https://www.kaggle.com/dreaddevelopment/raptor-knee-widedense)、[学習Notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)。

[DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4/output)の今回取得した索引はBest 0.937 V32、表示最新V35。V32の厳密ソース/Input版・重みのgold利用履歴は未取得なので、完全再現可能と断定しない。V35を使えばV35の新規再現として記録し、表示0.937を自分の実測欄に入れない。

[HARD/SOFTの一次報告](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)ではSOFTの公式58件での改善区間が0を跨ぎ、独立430件の二値lexicon教師では逆にHARDが優位。レポート教師と画像正解の差を示す例で、soft化の成功保証ではない。ラベル実験は検証の正解・対象・maskを固定し、学習側だけを変える。U/M欠損の一律0化や0.5化はしない。

10月4〜6日は保存と短い対照/20 epoch比較、7〜12日は入力試作・公開方式、13〜19日は有望な条件の追加foldと少数ensembleを目安にする。10月20〜22日は採点済み候補の再実行と最終選択に充てる。最終期限は10月23日08:59 JSTで、実行前に公式日程を再確認する。現在順位やメダル確率はこの監査から推定しない。

## 続行後の実行結果（e006〜e009、実行完了）

2026年10月4日 JST、続行の依頼後の追記。保存・実更新数の実装を終え、RTX 4090でe006/e007のfold 1対照とe008/e009の20 epochを新規runで完了した。全4本の保存内容が [完成集計](../experiments/e006-e009-extended-summary-20261004.json) でvalid=true。設定・seed・入力/分割/source hashと開始前の選択規則は [e006〜e009の計画と凍結記録](../experiments/e006-e009-plan-20261004.json) に保存している。BN固定の選択epoch 9は単体の対照候補とし、追加の固定rank ensemble候補は末尾に記す。別foldのBN比較と自作Publicは未確認である。

新規runは`best_bce.pt`・定義可能な場合の`best_auc.pt`・`last.pt`・5/10/15/20 epochのモデルを保存し、`checkpoint_index.json`でepoch・weak指標・予測/metrics/モデルのhash・実更新数を結び付ける。`best.pt`と`weak_valid_predictions.csv`は**そのrunの事前選択規則**のコピー。e006/e007は従来のBCE最小、e008/e009は12所見すべてが定義されたmacro AUC最大、同値ならBCE最小、さらに同値なら早いepochである。旧e003〜e005のBCE選択は変更していない。モデルだけの`last.pt`はoptimizer/scaler/RNG/Generatorを含まず、正確な学習再開には使えない。

| run・状態 | fold | 事前選択 | 選択epoch | 選択時weak BCE ↓ | 選択時12所見macro AUC ↑ | 補助11所見平均 ↑ |
|---|---:|---|---:|---:|---:|---:|
| e006：B、5 epoch完了 | 1 | BCE最小 | 5 | 0.4149572354584899 | 0.6385377399704473 | 0.6155369534447991 |
| e007：C、5 epoch完了 | 1 | BCE最小 | 3 | 0.37693114768344715 | 0.7320502783820059 | 0.7240361050015509 |
| e008：C、20 epoch完了 | 0 | AUC最大 | 8 | 0.630828931344588 | 0.7844098454841361 | 0.7666289223463302 |
| e009：C＋BN統計固定、20 epoch完了 | 0 | AUC最大 | 9 | 0.37304746405136474 | 0.8036140422002819 | 0.7866698642184893 |

全4本は [追加実験の監査集計器](../scripts/summarize_extended_runs.py) の監査が成功した。開始/終了/現行source、元/保存config、manifest/input hash、全epochの予測集合・所見順・観測数・AUC・BCE整合、固定train subset評価、checkpoint/index hashと選択、実更新累計を照合した。実checkpoint内のconfig・epoch・fold・12所見順・正規化・fingerprintもCPUで確認した。gold画像・gold評価を使わず、今回の自作Publicは未測定。既存Public 0.924は別の公開モデル提出として保持する。

### fold 1での事前学習の再確認

同じ891検査・6,207観測セルでB5→C3のBCEは−0.03802608777504275、12平均AUCは+0.09351253841155856、補助11平均は+0.1084991515567518だった。10所見のAUCが改善し、Synovitisは−0.07134020618556702、Contusionは−0.008981555733761026。全所見改善とはせず、事前学習による平均改善が別の固定分割でも出たと解釈する。fold 0とfold 1は異なる検証集合なので、その間の値をpaired差として扱わない。

[fold 1のpaired bootstrap記録](../experiments/e006-e007-fold1-bootstrap-20261004.json) は891検査・861供給group、seed 20261004、3,000反復。入力CSV・manifest・観測maskのhash、所見別の陽性/陰性/欠損数と区間も保存した。

| C3−B5 | 差 | percentile 95%区間 | 定義可能な反復 |
|---|---:|---|---:|
| 12所見macro AUC | +0.09351253841155856 | [+0.06072692855516013, +0.127134581921647] | 3,000/3,000 |
| Synovitisを除く補助11平均 | +0.1084991515567518 | [+0.07455422585495476, +0.14321740768815874] | 3,000/3,000 |

Synovitisは陽性97／陰性25、MCLは陽性16／陰性700であり、所見別の不確実性は同じではない。MCL等の所見別差の区間は0を跨ぎ、平均の改善から各所見の改善を一律に保証しない。

### Cの20 epochと最初5 epochの完全再現

e008の最初5 epochは旧e005と**全epochのBCE・12平均AUC・予測CSV SHA-256が完全一致**した。head初期値・入力/分割・環境も照合し、保存・実更新数追加でその学習軌跡が変わらないことを確認した。旧→新source差は`checkpoints.py`・`contracts.py`・`runtime.py`として記録する。旧runに現行source一致を要求せず、当時の記録を履歴として保持し、旧checkpointを後付けで採用し直していない。

| e008で保存された候補・終点 | epoch | weak BCE | 12所見macro AUC | 補助11平均 |
|---|---:|---:|---:|---:|
| 全20 epochのBCE最小 | 2 | 0.3713948995737489 | 0.7668950458344722 | 0.7475218681830605 |
| 最初5 epochのAUC最大 | 5 | 0.4157413752569746 | 0.7746078613087306 | 0.7577540305186151 |
| 全20 epochのAUC最大・事前規則による選択 | 8 | 0.630828931344588 | 0.7844098454841361 | 0.7666289223463302 |
| last、選択候補とは別の最終epoch | 20 | 1.091544485024309 | 0.7616139714998359 | 0.7626697870907301 |

学習時間の比較はBCE選択とAUC選択を混ぜず、**最初5 epochのAUC最大epoch 5→全20 epochのAUC最大epoch 8**で行った。[時間延長のpaired bootstrap記録](../experiments/e008-duration-bootstrap-20261004.json) はfold 0の821検査・812供給group・5,727観測セルを用いる。同じseedと3,000反復で、結果は次のとおり。

| epoch 8−epoch 5 | 差 | percentile 95%区間 | 定義可能な反復 |
|---|---:|---|---:|
| 12所見macro AUC | +0.009801984175405435 | [−0.004061022122451632, +0.024397682157697376] | 1,876/3,000 |
| Synovitisを除く補助11平均 | +0.008874891827715019 | [−0.006410513650803365, +0.023781166447565085] | 3,000/3,000 |
| Medial Meniscus AUC | −0.020547593363127392 | [−0.050121131611952585, +0.009545581449307866] | 3,000/3,000 |

6所見改善・6所見悪化だった。Medial MeniscusはAUC 0.6473078983481203→0.6267603049849929、BCE 0.8985950401739052→1.56104091421178。全セルBCEも+0.21508755608761337と悪化した。20 epochを実行して保存可能なAUC候補は増えたが、**学習時間を延ばした平均改善は両区間が0を跨ぎ、確定していない**。epoch 20の値も選択epochより低く、「20 epochまで一様に改善した」とは扱わない。

### 実更新数・実行範囲と読み方

| run | optimizer更新機会 | 実更新 | AMP skip | 全体時間（秒、表示は小数6桁） |
|---|---:|---:|---:|---:|
| e006 | 4,145 | 4,142 | 3 | 1,057.386118 |
| e007 | 4,145 | 4,143 | 2 | 1,051.637065 |
| e008 | 16,940 | 16,932 | 8 | 4,026.711290 |
| e009 | 16,940 | 16,930 | 10 | 3,889.968551 |

実更新はoptimizer post-step hookの呼出しを数え、GradScalerによるskipを除く。機会を実更新と同一視しない。e008の最初5 epoch累積は機会4,235／実更新4,233／skip 2である。全体時間の元値はe006 `1057.3861175000056`、e007 `1051.6370650000026`、e008 `4026.7112901000073`、e009 `3889.9685513000004`秒で、各private runの`execution.json`を監査して転記した。4本の合計は10,025.703秒（約2時間47分）。cache事前検査・最終source照合を含み、Python importを除く。実行時間は今回の測定であり次のrunの保証ではない。

両bootstrapは [CSV比較CLI](../scripts/bootstrap_weak_comparison.py) のNumPy 2.5.3・PCG64で、groupをソートし、1反復ごとに`multinomial(K, full(K, 1/K))`を引く。group内の全検査へ同じ重みを付け、左右の予測でも同じgroup重みを使い、AUCの同値scoreは0.5寄与とする。95%区間は定義された差のpercentileを線形補間したもの。12／11の分母を未定義クラスの除外で縮めず、未定義反復数を明示した。bootstrapはgoldを開かず、出力に検査・group識別子やReportを含めない。

fold 0のSynovitis陰性は1groupだけなので、時間比較ではそのgroupが欠けた1,124反復で12平均が未定義となった。**12平均の区間は残った1,876反復への条件付きであり、全反復の改善を保証する区間ではない**。いずれも既に同じweak foldで選択したepoch同士の記述比較で、epoch/config選択の偏り、患者独立性、レポート教師の正確さは区間の保証対象外。所見別区間にも多重比較の補正はしていない。

ImageNet事前学習を基準にする根拠は別foldでも増えた。一方、通常BNで学習時間だけを増やした改善は未確定である。以下のBN比較も平均値と所見別を分けて判断し、weak AUCをPublicや画像由来goldの期待値へ変換しない。

### BN固定の効果と所見別の入れ替わり

[通常BN8→固定BN9](../experiments/e008-e009-bn-bootstrap-20261004.json) の12平均差は+0.019204、95%区間[−0.006118,+0.041423]、補助11差は+0.020041、区間[−0.005828,+0.044933]だった。8所見改善・4所見悪化。**平均の優位性は区間が0を跨ぎ、別foldでの確認を要する**。固定側のBCEは0.373047で通常側0.630829より低いが、全所見を改善した条件ではない。

| 所見 | 通常BN8 AUC | 固定BN9 AUC | 差の95%区間 |
|---|---:|---:|---|
| Medial Meniscus | 0.626760 | 0.726981 | [+0.049938,+0.150287] |
| ACL | 0.884258 | 0.716538 | [−0.231302,−0.108359] |
| Lateral Meniscus | 0.726879 | 0.640717 | [−0.148982,−0.022711] |
| PF OA | 0.849206 | 0.744198 | [−0.161622,−0.050915] |

Medial MeniscusのBCEは1.561041→0.621693へ改善した。MCL・Lateral OA・Baker's等も改善する一方、ACL等の悪化は大きい。平均値だけからBN固定へ全面的に切り替える判断はしない。

[固定BNの5→9 epoch比較](../experiments/e009-duration-bootstrap-20261004.json) は12差+0.024807、区間[+0.008788,+0.042459]、補助11差+0.022516、区間[+0.004783,+0.041191]。固定側では5 epoch以降の改善を観測した。ただし同foldで選択したepochの記述比較で、選択の偏りを含む区間ではない。BCE最良はepoch 7（0.340113、AUC 0.799820）、最後のepoch 20はBCE 0.824827／AUC 0.766349。両20 epochとも、終点が最良ではなかった。

`artifacts/kaggle/e008-pretrained20-auc-fold0/` と `artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0/` に凍結zip・選択重み・帰属/hash記録・未実行Notebookを新規保存した。人工192px・24窓・3検査×12所見でCPU/GPU提出契約が成功し、Report・初期ImageNetファイルの推論利用は不要。固定BNの全60 running bufferは初期状態と完全一致し、学習可能なparameterは変化していた。Kaggleの実test decode・全体時間・自作採点は未実施である。

次の入力比較は [物理cropの契約](research/physical-crop-contract-20261004.json) に整理した。既存cacheにはPixelSpacingがないためKaggleの元DICOMから別cacheを作る。同じ192px・系列/window・正規化等を固定してcropだけを変更し、gold連結groupを除いた24〜32件の学習側QCから始める。imaging.py全文hashの互換性を維持する設計を先に決め、旧cacheのhashガードを緩めない。

### 固定50:50 rank候補と次の判断

BNの所見別の入れ替わりを見た後、通常BN8と固定BN9の**固定50:50**を一度だけ追加診断した。学習前の事前登録にはない後付け候補であり、係数・所見別weightの探索やgold利用はしていない。[診断記録](../experiments/e008-e009-rank50-review-20261004.json)、[固定BNとの差のbootstrap](../experiments/e009-rank50-bootstrap-20261004.json)。

| fold 0候補 | weak 12所見macro AUC | 補助11平均 |
|---|---:|---:|
| 通常BN epoch 8 | 0.7844098454841361 | 0.7666289223463302 |
| 固定BN epoch 9 | 0.8036140422002819 | 0.7866698642184893 |
| 固定50:50 rank平均 | **0.8302493068964057** | **0.8157265166142609** |

全821検査を、所見のラベル欠損に関係なくモデル別・所見別に順位化する。同値scoreは平均順位とし、正規化順位`(rank−0.5)/N`を等重みで平均する。整数の二倍平均順位を使い、`(d通常+d固定−2)/(4N)`を一度だけfloatへ変換して同順位の丸め分裂を防ぐ。推論でも、その回の全test検査を一緒に順位化する。分割batch毎の順位化は別方式になる。共通helperと確定予測CSVの9,852値は完全一致した。

固定BN比の12平均差+0.026635、95%区間[+0.013988,+0.040100]は1,876定義反復への条件付き。補助11差+0.029057、区間[+0.015129,+0.043443]は全3,000反復で定義された。7所見改善・4悪化・1同値。ACL・Lateral Meniscus・PF OAを回復する一方、Lateral OA・Baker'sは固定BN単体より悪化する。rank値のBCE 0.625756は未校正順位scoreの計算で、確率校正の改善とは解釈しない。

この区間は既に選んだ予測CSVの比較であり、同foldでのepoch選択と後付けensemble候補の選択、不確かなweak教師、患者独立性、test集団への順位変換の移行を検証する区間ではない。したがって全面採用の根拠をfold 0だけで完結させず、**Kaggleの実採点→通常/固定BNをfold 1で対照実行して固定比率を確認→192px物理cropの少数QC**を次の順序とする。固定BN単体も時間や動作の比較候補として残す。

提出準備・Input版・時間の記録は [Kaggle提出手順](kaggle-submit.md)。現在の自作値はいずれもweak評価であり、既存公開モデルの実測Public 0.924と直接の改善差を計算しない。提出資産の準備や人工契約成功も、実test decode・9時間内完走・採点成功とは別に記録する。

今回の候補・採用範囲・次の対照設計は [判断記録](../experiments/e006-e009-decision-20261004.json)、入力/資産/文書の最終確認は [検証記録](../experiments/e006-e009-verification-20261004.json) に保存した。r001の提出資産と固定BN単体のv2を用意し、旧run・旧bundleを保持している。
