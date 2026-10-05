# 公開ベースラインを最初の比較基準にする

10月5日から、採点済みp001 Public 0.924を正式な比較基準にし、公開学習済みモデルの推論比較を最優先にする。r001 Public 0.749も公式APIで確認済み。[提出監査](../experiments/submission-audit-20261005.json)、[判断・段階別計画](research/public-model-strategy-20261005.md)。このリポジトリの自作ResNetへ別アーキテクチャの重みを読み込むことはできない。

## 10月5日の次候補p002

**第1段階の準備を完了し、第2段階のKaggle保存実行へ渡す。** DINOsaur V4のV32を基に、現在使うInputの版を5件へ固定した。これは新しい組合せの候補p002であり、作者Best 0.937の厳密再現や自己スコアの確認ではない。[選定根拠](research/public-candidate-review-20261005.json)、[引渡し・検査記録](../experiments/p002-handoff-20261005.json)。

使用するファイルは [02_submit_p002.ipynb](../artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)。元V32の10セルの推論ソースを文字列として保持し、同じ実行環境の変数を共有して順番に実行する。追加した前後検査がInputの内容・各モデルの完走・最終CSVを確認する。元の予測処理、前処理、固定係数は変更していない。旧source-only候補とhandoff-v1は準備履歴として残し、今回の操作には使わない。

### ユーザーが行う操作

1. **新しいprivate Notebookを作る。** [競技ページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)からNotebookを作り、Import Notebookで上記`.ipynb`を読み込む。名前は例えば`rsna-knee-p002-dinosaur-v32`とする。
2. **Inputを次表の5件にする。** 競技が既に追加されていれば、追加する公開Inputは4件。Add Inputで表のURLまたは所有者・名前を検索し、指定Versionを選ぶ。重みのローカル取得や自作コードZIPのアップロードは不要。別のRaptor Datasetや元Notebookから継承した19参照を追加しない。
3. **SettingsでAcceleratorをGPU T4 x2、InternetをOFFにする。** この元方式と専門モデルruntimeは2台のT4を要求する。設定はImport後に画面で確認する。
4. **Save Version → Save & Run Allを選ぶ。** 新しいセッションで最初から最後まで実行する。保存前に同じ全処理を手動で一度実行する必要はない。保存実行と`/kaggle/working`の出力保存は[Kaggle公式Notebookガイド](https://www.kaggle.com/docs/notebooks)を参照。
5. **完了後にログとOutputを確認する。** 最初に`P002 PREFLIGHT PASSED`、最後に`P002 READY FOR MANUAL SUBMISSION`が表示され、Outputに`P002_READY.json`と`submission.csv`があることを確認する。Notebookの実行も成功していることが条件。
6. **その保存Versionの最終`submission.csv`を選んで手動Submitする。** `submission_dinosaur_v4_0937_control.csv`や`submission_parent_exact.csv`は選ばない。採点完了後、Notebook URL・Version・Public score・実行時間・提出IDを残す。

| Add Inputで選ぶもの | 固定版 | 役割 |
|---|---:|---|
| [公式競技 rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) | 競技入力 | test画像・CSV |
| [tonylica / rsna-knee-bend-dinov3-0917-repro-assets](https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets/versions/5) | Dataset 5 | 主モデル群・校正器 |
| [renta0426 / rsna-knee-public0033-meniscus-bag-v1](https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1/versions/2) | Dataset 2 | 半月板の専門モデル・runtime |
| [metaresearch / dinov2 / PyTorch / small](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1) | Model 1 | DINOv2モデル定義・基礎重み |
| [mattiaangeli / knee-mri-fold-weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights/versions/2) | Dataset 2 | オフライン用timm 1.0.22 wheel |

**途中でエラーになった場合は提出しない。** エラーの末尾とNotebook URLを保存して原因を確認する。Inputの版やGPUを直した後は、新しいセッションでSave & Run Allする。途中セルだけの再実行や、検査を消しての提出はしない。検査開始後の失敗では、この実行で生成したCSVを`.disabled`へ移し、提出用ファイルとして残さない。既存CSVがあるセッションは開始時に拒否する。

### 準備で確認したことと残る実行確認

元19参照から有効ファイルを追跡し、上記5 Inputsに集約した。[対応表](research/p002-effective-inputs-20261005.json)には公開資産47ファイルの期待サイズ・SHA-256と、除外した参照の理由がある。重みpayloadはローカルで取得しておらず、配布manifest等に基づく期待hashとの照合はKaggleの開始セルで行う。[環境調査](research/p002-environment-20261005.json)、[出力契約](research/p002-output-contract-20261005.json)。

NotebookはDINO 20 member、DINOv3 5 fold、Raptor 4 armの予測形状・値域と完走、専門モデルのreceipt、固定校正器の適用、test UIDと12列、V6候補CSVとreceiptのhash一致を検査する。timmはhash確認済みの添付wheelからネット接続なしで固定版を導入し、他パッケージの版は実行時に記録する。全モデルの実MRI実行、Kaggle環境との互換性、本番の所要時間・Publicはまだ未確認。表示用testの成功だけで本番採点の完了とはしない。

最終V6の外側係数はACL=0.80、Lateral OA/PF OA/Synovitis=0.35、Baker's=0.375に固定している。`P002_READY.json`は保存実行の検査記録であり、0.937や0.924超えの保証ではない。

採用条件は0.924を上回る自己実測、意図した全分岐の完走、時間余裕。未達ならp001を維持し、係数探索を連続して行わない。公開重みのgold利用は別途監査し、ここでの結果を独立gold/OOF評価とは呼ばない。未配布の作者softラベルは推論比較の前提条件にしない。

V32と固定したRenta runtimeを静的に確認した範囲では、実行時のgold AUC計算・係数fit・checkpoint選択・学習処理はなく、calibratorとV6係数は固定値の適用だった。ただし容量計画の件数取得に`pd.read_csv(train.csv)`で全列を一度読み込むため、「gold/Reportを一切読み込まない」とは言わない。Report列への依存はないが、競技の`train.csv`ファイルは必要である。`test_series`不存在時のtrain側fallbackを防ぐため、追加した開始検査は正しいtest mountを要求する。公開資産の帰属と利用条件は元Notebookと環境調査へ残し、作者Inputをそのまま参照する。

[NTejasの公開実装](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/tree/1c386ac71385ba7683f550b87a465386ba6c36a8) も確認した。作者0.940/0.941は自己再現ではなく、参照Notebookのスコア版/Input固定が未解決で、実混合係数とreceipt説明にも不一致がある。今回は開発構造の参考に留め、追加腕やgold由来係数を先に移植しない。

## 保持する単体モデルp001

[Knee MRI twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) と [重みの配布ページ](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) が第一候補。2026年10月2日のWeb調査では、作者は単体・TTAなしのPublic 0.924を報告していた。

配布説明では、`raptor_ft_coatnet_v4_full.pt` がその成績を出したcheckpoint。CoAtNet、384px、隣接3スライス、所見別Attention、レポート由来soft labelsの構成で、提供ラベル付き58件を学習から除外したと説明する。SWA版もあるが、作者は元のcheckpointを使用した。[作者のモデル説明](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

追加調査で取得できた配布ページは **Version 2、585.66MB、CC0** と表示していた。これは索引が持つ表示版で、現在のライブ版や再現するNotebookのInput版と一致する保証はない。Notebookの厳密な版は未確定のため、Kaggleで再現前に記録する。

ここまでの配布説明は作者の報告。10月3日に既存提出の実測Public 0.924を公式API確認し、10月5日に提出ref 56766978・scriptVersionId 354569007を再確認した。V1を指定して成功版Pythonソースを保存し、版別Output一覧の識別子との一致も確認した。ソースhashは`3ec3a18fd97746e4ee44bf424775e902362ea45a25483861bbb78767c9d4b68e`。当該ソースでも140mm/336px/64枚/42窓と12所見順を確認したが、384px等のfallback値をcheckpoint内部の実測値へ読み替えない。歴史的Input版、重みhash、隠しtest時間は未確認。[監査](../experiments/submission-audit-20261005.json)。

成功版のコメントにはgoldを使った過去のarm選択や比較の記述があり、「goldを学習から除いた」という説明だけで開発全体がgold非使用とは扱えない。これは作者の沿革自己申告で、重み生成・選択ログの独立監査ではない。今回の作業で公開重み取得・再実行・新規提出は行っていない。

## 再現の手順

1. 公式競技へ参加後、上記NotebookをKaggleのCopy and Editで開く。
2. Best Scoreを出した版と現在の版を区別し、再現するNotebook版を決める。
3. 競技入力、checkpoint、必要な追加パッケージの入力を確認し、各Inputの版を記録する。
4. 元の前処理とモデル定義を保ち、GPUを選択して実行する。推論時のインターネットは無効にする。
5. 例示テストでの完走後、NotebookをSave and Run AllしてSubmitする。
6. 採点成功・実測Public LB・実行時間・Notebook/Inputの版を `experiments/ledger.csv` に記録する。
7. 複製したNotebookのソースをGPU端末で保存し、変更の比較元として固定する。データや重みはGitへ入れない。

作者版で動かない場合はInputs、環境の版、checkpointとラベル順から確認する。先にモデルや前処理を変更すると再現の原因を切り分けられない。

## その他の候補

- [Pilkwang RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) と [Inputs](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1/input)：DINO系と公開ラベルをたどる入口。添付分析には0.891とあるが、今回の本文取得ではその版のスコアを再確認できなかった。
- [単体モデルの学習Notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)：再現後、画像の見せ方と学習条件を調べる。
- [DINOsaur train](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v3-train)：別系統の学習設計の比較対象。
- [Full 4 arm ensemble](https://www.kaggle.com/code/nishantkharga/rsna-knee-full-4-arm-ensemble-v55)：単体モデルの理解後に比較。添付の成績や版は未再確認。
- [CPU pixel cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache)：デコードの再利用を考える資料。画質・スライス削減は情報損失を伴う。

公開重みの利用条件、学習対象、goldの使用、ラベル作成器のgoldへの調整も記録する。公開重みを後からランダムに分割したデータで評価して、独立した検証と呼ばない。

## 公開ラベルと学習コードの採用監査（10月3日）

公式Kaggle CLI/SDKから公開metadata、ファイル一覧、版を取得し、作者の説明と確認できたCSVの事実を分けた。記録は [label-sources-20261003.json](research/label-sources-20261003.json)。以下の版はDataset版であり、CSV名に含まれるv2/v4とは別である。

| 候補 | 確認した版・ライセンス | gold利用と今回の判断 |
|---|---|---|
| [Steven](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) | Dataset 6、CC0 | v2はgold比較からSynovitisをEffusionで補正。v4はそのラベルをblend。独立gold評価には使わない。元v1の開発履歴も未確認 |
| [Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels/versions/1) | Dataset 1、CC0 | YES/NO/UNKとconfidenceを持つ。配布生成器は未配布のllm_labelerをimportし、prompt・scoreの完全な監査ができないため未採用 |
| [lixin73](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56/versions/1) | Dataset 1、CC0 | APIのdescriptionが空で、生成方法・未言及・goldの扱いは未確認 |
| [Yehezkiel](https://www.kaggle.com/datasets/yehezkielhaganta/psuedo-labeling-knee/versions/7) | Dataset 7、MIT | 1/0/-1の定義を確認。-1は未言及であり欠損として扱う必要がある。goldへの調整は未確認 |
| [Laymond](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/versions/2) | Dataset 2、CC0 | gold結果を踏まえたprompt改訂と共起補正を説明。raw版でもpromptの独立性を保証しない |
| [nartaa](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets/versions/9) | Dataset 9、CC0 | perlabel版はgoldで所見ごとに選択。v4系は調整済みラベルを継承。gemlow単独版の開発履歴は未確認 |
| [vmohitrao](https://www.kaggle.com/datasets/vmohitrao/rsna-knee-report-labels/versions/3) | Dataset 3、CC BY-NC 4.0 | 作者READMEはgold 58件を抽出・prompt開発から除外したと明記。元train hashと4つのCSVの公開hashが一致。非商用の研究用候補として取り込み |

vmohitraoのgold非使用は作者の宣言であり、非公開の開発ログまで監査した事実ではない。生成ラベルは臨床的な正解ラベルではない。4,349行のラベルは公式goldを含まず、P=1、N/B=0、U/M=空欄、weightは観測maskである。全セルのstate・value・mask一致を検査した。

全欠損138件とgoldの同一レポート/groupにつながる4件を除外して、研究用manifestはweak 4,207件・gold 58件。供給されたreport_groupを保ち、fold 0〜4は821・891・888・778・829件。患者独立性は未確認である。

fold 0のSynovitisは観測101件のうち陽性100・陰性1、欠損率87.7%。欠損を陰性へ変えず、この偏りを実験の制約として記録する。weak BCEの良さだけで全12所見の画像性能が良いと判断しない。

10月3日時点ではCC BY-NC 4.0の利用条件を記録し、採用範囲を非商用のローカル研究としていた。10月4日に公式Rules・Host回答と照合し、研究・学習目的と帰属等を維持して大会用途へ採用を進める判断に更新した。根拠は [利用条件監査](research/kaggle-source-eligibility-20261004.json)。入賞時の権利処理や商用化は、その目的が発生した段階で確認する。ラベル・レポート・重みはGitへ入れない。

### 公開学習コードをそのまま移植しない理由

取得したDreaddevelopmentの学習コードはNotebook Version 8。goldを学習から除外する一方、各epochのgold AUCでbest checkpointとtop-kを選ぶ。配布checkpointの厳密な生成履歴との一致は未確認だが、この選択方法を本リポジトリへ移植しない。既存A/B/Cはweak検証masked BCEで選択した。以後もrun開始前にweak検証の選択規則を固定し、goldをcheckpoint選択に使わない。

Pilkwang baseline Version 15の学習分岐にも公式ラベルを高いweightで混ぜる処理がある。公開Notebookや重みのスコアを、自分の独立したholdout成績として記録しない。

## DINOsaurの版・依存・再現契約の追加監査（10月4日）

公式SDKの通常認証で、公開Notebookのソース5件、Datasetのmetadataとファイル一覧15件、小さいソース・帰属・実行契約文書13件を取得した。実MRI・ラベル表・学習済み重み・予測配列は取得せず、公開コードの実行・アップロード・提出も行っていない。出典URL、版、ソースhash、確認できた事実と未確認事項は [public-reproduction-contract-20261004.json](research/public-reproduction-contract-20261004.json) に保存した。

[DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4/output) の表示Bestは作者報告0.937のV32で、最初に取得した最新ソースはV35だった。追加調査でSDKの `version_label="v32"` を指定し、応答の版番号32とソースhashを固定できた。[取得経路と追補](research/public-reproduction-v32-followup-20261004.json)。保存InputのDataset／Notebook版、提出されたCSVと採点の対応、自分の再現スコアは未確認である。取得できたV35は、0.937 controlを別CSVへ保存した後、PublicDual・Synovitis・CoAt残差を追加する。最終 `submission.csv` はPublicCoAtのmain（内部alpha 0.16）であり、0.937 controlとは別候補になる。最新をコピーしただけでV32を再現したことにしない。

V32の最終 `submission.csv` も保存controlとは異なる。controlへV6 five-targetの外側weight（ACL 0.80、Lateral OA／PFOA／Synovitis 0.35、Baker's Cyst 0.375）を重ねる。coreのcode cell 2〜10はV35とbyte一致するが、その後の分岐が異なる。作者がどの出力CSVを採点したかはソースだけでは確定しない。V32指定のOutput一覧9ファイルを確認したが、予測payloadは取得していない。

V35のlive metadataには14 Dataset、3 Notebook Output、DINOv2-smallのModel V1と競技入力がある。Notebookに埋め込まれたInput IDには継承されたmetadataがあり、Notebook Outputの数もlive metadataと一致しない。IDを配列の順番でInput名へ対応させず、保存版のInput画面と有効ファイルのhashから固定する。

今回観測したInputの最新Dataset版は次のとおり。**この表はV32の保存Input版ではない。**

| Input | 今回観測したDataset版 | 契約上の注意 |
|---|---:|---|
| Dread maxspan／native384／native384dense | 各1 | v5 SWA／v8 SWA／v10の3重み。前処理が異なる |
| Mattia knee-mri-fold-weights | 2 | DINOv3の5 fold重みとtimm wheel |
| Tonylica consolidated assets | 5 | 現在は別公開方式の41 checkpointと校正表をまとめた資産。元V4のrecipeと同一とは限らない |
| Renta public0033 Meniscus bag | 2 | safetensors、runtime、manifest、run contractを持つ |
| Antoine E9／E11 Rad heads | 3／1 | RadImageNet由来。gold学習・係数選択の履歴を持つ |
| Mattia residual CoAt | 3 | e4/e6/e8はepoch ensemble。5-foldとは別 |
| Dread widedense | 2 | 既存0.924のv4重み候補。V35の3 Raptorとは別 |

V35はDINO／DINOv3／Rad／Raptorの推論とrank変換を重ね、途中CSVを複数回書く。DINO側は全public-frontier票の完成を要求する一方、CoAt分岐には失敗時に先行CSVを維持する処理がある。提出CSVの形式検査だけでは、意図した全分岐の完走を確認できない。member数・fingerprint・Input hash gate・分岐のreceipt・最終CSVの選択をログで確認する。表示される数分の実行時間は例示testの記録であり、hidden test全体の見積もりに使わない。

### gold履歴と利用条件

- Renta Dataset V2の配布run contractは、weak 4,349件、公式ラベル付き58件を除外、gold値・gold metricsを読まず、seed 2027で30 epochの最終重みだけを選んだと記録する。文書とsidecarのhashは確認したが、非公開実行ログや教師Pilkwang `report_labels_v2` の生成独立性まで確認したわけではない。`fold_safe_oof=false` が明記されている。
- Antoine E9 Dataset V3のREADMEは公式goldでweakを上書きして学習すると明記し、E10のblend係数選択もgold 58件を使うと説明する。配布学習ソースにもgold代入と係数探索がある。latest DINOsaur V5 train V15にもgoldを学習targetへ入れ、gold検証AUCでepochを選ぶ処理がある。historical V32重みとの厳密な生成対応は未確認で、これらを本リポジトリのgold除外学習へ移植しない。
- Tonylica V5の `SOURCE_AND_LICENSES.md` は一括CC0ではなく、CC0／Apache／CC BY-NC-SA／DINOv3の元条件等が混在すると説明する。E13の再配布licenseは過去監査で別途確定していない。公開Datasetの表示が「Other」であることを、条件なしの再配布許可と解釈しない。公開NotebookをKaggle上で参照する経路と、第三者重みを自作bundleへ再配布する経路を区別する。

### 10月4日時点の候補準備の履歴

当時は自作モデルの比較結果から確定したcheckpointを、新しいbundleへ保存して手動採点する方針だった。この経路はr001 Public 0.749の採点まで完了した。現在の次工程は冒頭のp002であり、同じr001の再提出は不要。

公開方式の次候補は、取得済みV32ソースと作者の保存Input画面を照合し、その版をCopy and Editして入力版と提出CSVを固定する経路を優先する。V32のscriptVersionId、Dataset／Notebook／Modelの版、runtime、実測Publicを揃えるまでは「0.937再現済み」としない。SDKの版指定metadataには14 Dataset／3 Notebookの参照があるが、それらの版番号がない。embedded metadataの少ないInput ID一覧から補完しない。V35を実行する場合は独立した新しい候補として記録し、保存0.937 controlと最新PublicCoAt mainのどちらを採点するかを先に固定する。今回、DINOsaurの採点可能な新規bundleを作ったとは扱わない。

CoAtNet推論の最新V7はソースとhashを固定でき、依存はnumpy・torch・timm・pandas・pydicom・cv2と `raptor_ft_coatnet_v4_full.pt`。汎用初期重みの取得をせず `pretrained=False` で構築する。140mm crop・336px画像・64スライス・6〜94% span・384px入力・42 windowがソースの契約である。作者が0.928と報告するmaxspan v5は2〜98% span・62 windowを要求するため、ファイル名だけ差し替える比較をしない。Training V8の未配布softラベル・gold選択は、この推論再現とは別の課題として残る。
