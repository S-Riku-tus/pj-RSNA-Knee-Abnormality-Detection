# RSNA Knee プロジェクトの現在地

**2026-10-21 13:25 JST・公式bestは0.958、目標0.959は未達です。P048 V4（正式ref57464683）は同一IDのGetで採点待ち（status/Publicともnull）を確認し、本日の残りは4枠です。ConvNeXt TinyとSmallは両方fold0の固定E12学習中です。** 以下の過去時刻の記述は履歴です。

Tinyはlauncher PID20048／実Python PID26668／exec82461、Smallはlauncher PID7792／実Python PID26468／exec48503で生存を確認しました。既存ユーザー処理PID19472の自然終了後に開始し、停止・優先度変更はしていません。[Smallの開始記録](experiments/F015-ConvNeXtSmall-artificial-qualified-fold0-running-20261021.json)では公式generic重みのSHA確認、実RTX4090の人工12項目PASS（全144window forward/backward・厳密復元・prefix不変・使用量1,108MiB）、元F006/Tinyと同一の3,386学習/821検証・入力hashを確認しました。Tinyの保護source16も不変です。開始前の空きはVRAM21,564MiB、RAM16.88GiBでした。Tinyの256件間隔が3回連続210秒超、fold見込み12時間超、または空きRAM4GiB未満なら、SmallだけをSTOP_REQUESTEDで停止する条件を事前固定しました。精度を見て並行候補を選んだわけではありません。

[固定E12のCSV監査](experiments/CNN-fixedE12-CSV-auditors-qualified-source-only-20261021.json)はTiny17試験・Small22試験とRuffを通過しました。旧F006のE12/全12epoch保持物、source16/18、group・欠損mask・UID・環境・実更新/skip・固定10所見AUCを照合し、BCE best aliasは候補に使いません。実CNNの精度診断は12epoch完了後まで未実行で、fold1もまだ開始していません。両foldが正でも、tensor有限性・厳密再構成・Kaggle実GPU/MRI・CSV検算・正式採点は別に確認します。Linux推論用の依存wheel3本をhash固定して保存しましたが、稼働環境への導入や外部アップロードはしていません。

**2026-10-21 13:03 JST・P048 V4を正式提出ref57464683として受理し、採点待ちです。公式最高0.958・目標0.959未達、本日1/5件使用・残4枠です。** [正式受付](artifacts/research/20261021-P048-v4-root-formal-submit-v2/summary.json)は本人Getと全44件Listの同一refを結合し、kernel139941816／Kaggle V1・source05b8b300…を確認しました。受付時のstatus/Publicはnullで、結果はまだ返っていません。以下の未提出・監査未完の記述は、それぞれの時点の履歴です。

提出前に[P048 V4の実出力監査](artifacts/research/20261021-P048-v4-root-visible-execution-v1/root-independent-actual-audit.json)の71項目を通過しました。現重み・CPU厳密復元・2T4人工確認、可視3/3検査成功・失敗0、native36値/親36値のFP32別計算と最終36値の文字列一致を確認しています。別agentのFraction順位計算・UID別executor確認と、追加取得したreceipt.sha256の一致も[提出前記録](artifacts/research/20261021-P048-v4-root-formal-submit-v2/root-preSubmit-review.json)へ結合しました。固定20%や元P033/REG等は維持し、V3 ERRORの復元親は提出していません。可視成功や順位変化をPublic改善と解釈せず、正式結果を待ちます。

F014 fold0は[13:03 JST時点の進捗](artifacts/runs/f014-convnext-tiny-fold0-20261021-v1/progress.json)でepoch1の768/3386検査、更新48/機会48・AMP skip0です。途中の精度で選択せず、固定E12と事前条件を維持して学習を継続します。fold1は未実施です。

**2026-10-21 12:59 JST・公式最高0.958、目標0.959は未達です。P045は公式0.956で不採用、R6の事前4案も採用条件を満たしませんでした。P048の修正版V4は可視実行COMPLETEですが、独立監査・正式Submitはまだです。F014のfold0学習をRTX 4090で開始しました。** 以下の10月20日以前の状態・残枠・待機記録は履歴です。

[P045正式結果](experiments/P045-formal-COMPLETE0956-DINO-rejected-20261021.json)はref57430729のGetと全43件の同一ref Listを結合し、scriptVersionId360274333・Public0.956を確認しました。DINO固定10%追加は採用せず、事前条件に従いP046の20%案もSave・正式提出なしで棄却します。[R6の確定判断](experiments/R6-four-prespecified-variants-nonpositive-no-P047-20261021.json)は共通成功1,664/1,664件、bank0/1=809/855件で、TTAなし・TTA1/4・補正なし・crop320のいずれも両bankの固定10所見差が正になりませんでした。元R6と公式bestを保持し、P047の実候補生成・Save・正式提出は行いません。追加の係数・所見別救済でこの判定を置き換えません。

[F013のローカルCUDA資格](experiments/F013-local-RTX4090-CUDA-qualified-not-serving-20261021.json)は固定E12の両foldについて人工native392入力の12出力有限、padding不変、全画像欠測拒否、重み・prefix不変、eval gammaとCPU/CUDA乱数状態不変等19項目を通過しました。FP32 CPU参照との差は0.0001274／0.0004928未満で、事前上限0.005以内です。これはRTX 4090での人工確認であり、元F009のAMP-HOLDは維持します。Kaggleの現8モデル・2T4・実MRI・融合は別に確認します。

P048はP039のCLS固定E12の2本だけをF013へ置き換え、REG・前処理・fold/家族50%平均・native20%・P033の8セルを固定します。[V3の実出力](artifacts/research/20261021-P048-root-visible-execution-v1/poll-20261021T032305413266Z/summary.json)はERRORで、Torchの先行importが既存のcuBLAS初期化前提に違反しました。元guardを保持してimport順だけを直したV4（kernel139941816／Kaggle V1、source `05b8b300cc397ae61de643b7788601f3963da489fb366cb7530ddf6946b0ccca`）は[12:57 JST取得の可視出力](artifacts/research/20261021-P048-v4-root-visible-execution-v1/poll-20261021T035729991215Z/summary.json)でCOMPLETE、native3検査成功・構造/運用失敗0・子exit0です。両版とも正式提出はまだ0。V4のモデル・CPU/GPU証跡とCSV数式を別計算で照合してから正式提出の可否を決め、可視成功をPublic改善とは扱いません。

F014はDINOv2からConvNeXt Tinyへの独立したencoder選択の比較です。構造・一般事前学習重み・特徴幅768・学習suffixが連動するため、構造だけの効果とは主張しません。人工GPUの12項目を通過後、既存処理を停止せずGPU解放を確認し、[手動起動記録](artifacts/research/20261021-F014-fold0-root-prelaunch-v1/launch.json)のPID20048／exec82461でfold0を開始しました。[run記録](artifacts/runs/f014-convnext-tiny-fold0-20261021-v1/run.json)はrunning、学習3,386・検証821、seed20261002、既存native392キャッシュ・教師/mask/group分割を固定、Goldの学習・選択・評価なしです。固定E12でF006対照と比較し、条件を満たすまでfold1は開始しません。fold1・精度評価・採用・Kaggle提出は未実施です。

**2026-10-20 21:06 JST・P048は固定REGとの50%平均でも両weak foldが正でした。公式最高0.958・目標0.959未達、P045採点待ち、R6の2件は実行中、残り2提出です。** [追加検証](experiments/P048-native50-paired-weak-positive-sourceV2-20261020.json)：fold0は0.888923→0.889624（+0.000701）、fold1は0.888657→0.889027（+0.000370）。P039の実際のCLS E12を対照とし、REGは既存E8/E10、FP32平均50%を固定。6run/予測/重みのハッシュ、元writer指標、UID/欠損mask/groupを照合しました。同じfoldの除外モデルだけを使った家族平均の診断であり、実提出の2fold平均やP033全体のOOFではありません。反復したweak検証上の小差で、Public改善は未確認です。source V2では両fold正を必須にし、7試験とruffを通過しましたが、人工CUDA未完のため実行は無効のままです。

**2026-10-20 20:55 JST・公式最高0.958、目標0.959未達。P045は採点待ち、R6の2検証は実行中、提出枠は残り2回です。** F013の人工CUDA確認のSaveはKaggleのGPUバッチ同時2件制限で未受理です。新kernel/versionはなく、推論失敗ではありません。[未受理の正本](experiments/F013-artificial-CUDA-source-ready-GPU-concurrency-wait-20261020.json)。R6が自然終了してから、最新の目標・枠・保存版を照合して手動で再試行します。既存ローカルGPU処理PID19472は維持します。

[P048の別経路](experiments/P048-F013-bridge-source-only-20261020.json)を実行無効のsourceとして準備しました。P039のCLS固定E12を2本だけF013へ替え、REG・画像処理・fold/家族平均・native20%・P033の8セルを固定します。6件の標準ライブラリ試験とruffは通過。元のbootstrap/run・残り7ファイルは保持し、routerの変更は3つの構築/証跡境界に限定しています。旧F009のHOLDとconsumerは無変更です。実際の人工CUDA証跡なしでは開始を拒否し、実行時も現在の8モデルの人工確認をMRIより先に要求します。実モデル結合・MRI・Save・正式提出はまだ0です。

**2026-10-20 20:30 JST：F013の2fold固定E12はCPUで有限・厳密復元と人工入力を通過しました。自己最高0.958・目標0.959未達、P045採点待ち、R6検証2件はRUNNING・残り2枠です。** [CPU結果正本](experiments/F013-fixed-E12-CPU-qualified-not-release-20261020.json)。各foldのsuffix28/head12 tensorが有限・元checkpointとbit一致、frozen prefix67a05dff…を復元前後で保持、人工392px encoder/headの12出力有限・padding汚染不変・全画像欠測拒否・eval gamma draw0を確認しました。CPUログ/記録7ファイル5,676B・取得エラー0、14項目を照合しました。これらのE12でAMP skipが非有限重みを生んだ証拠はありませんが、元HOLDは維持し、旧consumerも無変更です。新しいF013経路での完全なnative CUDA・実MRI・旧REG/固定20%融合との結合が必要で、提出資格はまだありません。

**2026-10-20 20:28 JST：自己最高0.958・目標0.959未達、P045採点待ち、R6検証2件はRUNNING、本日残り2枠です。** 元352はfold0=396/396・fold1=400/400完了、fold2/3処理を公式ログで確認しました。P047の条件付き生成器は一要因変換・厳密採否・実行順・4仮想候補の輸送を11試験とruffで確認しましたが、実際の候補は未選択です。[P047準備](experiments/P047-selected-R6-conditional-tools-only-20261020.json)。

[F013再分析](experiments/F013-F009-matched-counter-fixed12-exploratory-positive-20261020.json)では、旧F009の固定E12について対照/候補ともfold1の更新2495・機会2496・E10スキップ1で一致し、gamma候補の固定10平均AUC差はfold0 +0.001693・fold1 +0.000902でした。fold1の95%区間は[−0.000298,+0.002114]でゼロを含み、後付けの探索分析です。旧F009のHOLDと旧consumerは変更せず、別scopeの厳密復元・有限性を確認します。自作重み2個・元source14等29.35MBを非公開Dataset12730408/V1へ手動PUT2回/Create1回で受理し、[CPU検証139809994/V1](experiments/F013-fixed-E12-private-CPU-running-20261020.json)がRUNNINGです。入力は自分のprivate2件のみ、競技/MRI/Report/Goldなし、正式提出0。CPU結果・今後のCUDA/実MRI資格は未完、既存GPU19472維持を継続します。

**2026-10-20 19:38 JST：R6の352/320両検証が人工GPU/direct ON・OFF対replayを通過し、実MRIを処理中です。自己最高0.958・目標0.959未達、P045採点待ち、本日残り2枠です。** [crop320のGPU確認](experiments/R6-crop320-artificial-GPU-pass-MRI-running-20261020.json)。crop320はCLI変更行・誤差2.646e-10/5.537e-10・fold0/1開始を公式ログで確認しました。全foldの成功率と弱ラベル評価はまだです。読み取り専用の監視は最初の終了・エラーで停止し、rootが実出力の資格を確認します。監視はSave/Submit/モデル取得/学習を行いません。正式予測CSVの公式取得APIはP044に403を返したため、取得なし・別経路再試行なしと記録しました。既存ローカルGPU19472維持を継続します。

**2026-10-20 19:24 JST：元R6 probeの人工GPU/direct ON・OFF対replayを通過し、実MRIのfold0/1処理へ進みました。公式最高0.958・目標0.959未達、P045採点待ち・残り2枠です。** [GPU確認の正本](experiments/R6-probe-artificial-GPU-pass-MRI-running-20261020.json)。公式ログストリームで誤差3.256e-10/3.973e-10とfold0=396・fold1=400の開始を確認しました。8秒のログ待受idle timeoutはクライアントの読み取り終了であり、worker失敗とは扱いません。実MRIの成功率・弱ラベル精度はまだ未確認です。crop320は受理・実行待ち。既存GPU19472は19:18にも使用中で維持し、新しいローカルGPU学習は開始していません。

**2026-10-20 19:16 JST：自己最高0.958・目標0.959未達、P045採点待ち、本日残り2枠です。元R6検証139794653/V1はRUNNING、切り出し320検証139798784/V1は受理・QUEUEDです。** [切り出し比較](experiments/R6-crop320-paired-weak-probe-private-queued-20261020.json)・[判断メモ](docs/goal959-analysis-20261020.md)。切り出し以外の元3セルとruntime/worker/重み/入力を保持し、CLIのcrop引数352→320だけを変更しました。元の3候補とcrop320の計4候補は、両検証の共通成功集団で比較し、両bankの固定10平均AUC差が正・最小差が最大の1案だけを選ぶ方針を、結果解析前に固定しています。cropの複合TTA/後処理案は評価しません。3引数/輸送試験・3採否試験とruffを通過し、人工GPU/実画像/実精度の確認はまだです。R6からの正式提出は最大1案。P046はP045公式0.958の場合だけ検討するsourceのみ、既存GPU19472維持を継続します。

**2026-10-20 19:01 JST：自己最高0.958・目標0.959未達。P045採点待ち、R6の非提出用検証V1は実行待ちです。** [R6検証の正本](experiments/R6-two-bank-paired-weak-probe-private-running-20261020.json)：kernel139794653/V1の全ソース・3入力・private/offline/T4/Dockerを照合し、特定V1とSDK標準の現在版でQUEUEDを確認しました。検証結果はまだありません。F012の学習側だけの教師確率補正はC=0.832982→D=0.835253へ改善しましたが、旧教師A=0.837830には届かず、事前基準に従い追加fold・採用・提出を見送ります。[F012結果](experiments/F012-CPU-train-calibration-nonpositive-20261020.json)。[P046](experiments/P046-DINO20-source-only-conditional-20261020.json)はDINO比率だけ20%へ変えたsourceのみを準備しました。P045が公式0.958だった場合に限って検討し、0.957以下なら棄却、0.959以上なら追加提出不要とします。P046のSave/実画像推論/正式提出は0です。残り2枠・既存GPU処理維持を継続します。

**2026-10-20 18:41 JST：P044も公式0.958でした。自己最高0.958・目標0.959未達、P045 ref57430729は採点待ち、本日残り2枠です。** [P044結果](experiments/P044-formal-COMPLETE0958-no-adoption-20261020.json)。既存GPU処理PID19472は維持します。固定特徴量のCPU教師比較F011は、旧教師A=0.837830、JEVの値だけを置換したB=0.825155、JEVの有限値へ教師範囲を広げたC=0.832982（同じweak fold0・固定10所見平均AUC）で、改善の根拠にならずfold1・採用・提出へ進めません。[F011記録](experiments/F011-CPU-teacher-screen-nonpositive-20261020.json)。次はR6公開ソースのfold規則に従い、各検査の除外foldだけで2検証集団を推論する非提出用probeを準備しました。1664検査・group分離・自分のGold非利用・人工GPU確認を先に行い、TTAと既存後処理を事前固定の基準で比較します。7試験とruffは通過、実GPU/MRI検証はまだです。公開重みの学習集団・上流Gold/補助学習履歴の独立証明と患者独立性は未確認です。

**2026-10-20 18:00 JST：P043は公式0.958で同点、P044・新たに提出したP045は採点待ちです。自己最高0.958・目標0.959未達、本日3件使用・残り2枠です。** [P043結果正本](experiments/P043-formal-COMPLETE0958-no-adoption-20261020.json)・[P045提出正本](experiments/P045-correctedV2-formal57430729-accepted-goal959-20261020.json)。P043のGaussian融合は新bestとして採用せずP033を維持します。P045修正版は親36FP32＋最終36FP64の別実装検算と実出力26項目を通過し、全43件の公式Listでref57430729＝kernel139737264/Kaggle V1＝scriptVersionId360274333を照合しました。DINO5重みの上流Gold利用履歴は不明として記録し、自分のGold fit/selectは0です。旧P045 V1は提出していません。13:20に審査サービス上限で未実行となった採点再照会は17:54に通常経路で復旧し、審査回避・重複Submitは0です。ローカルGPUは17:55にも既存PID19472が使用中で維持、GPU画像学習は空き待ちです。固定特徴量によるCPUの教師比較F011はpolicy準備済み・まだfit前です。

**2026-10-20 13:10 JST：P045は整数順位合計へ直した別artifact V2を非公開実行中です。旧V1は正式提出しません。** [修正版の記録](experiments/P045-corrected-V2-private-running-goal959-20261020.json)。追加人工例で、同じ5fold順位合計が浮動小数点の加算順により0.56と0.5599999999999999へ分かれることを確認しました。結果の精度を見る前に修正方針を記録し、各foldの平均順位を2倍した整数の和で同順位を保つ形へ変更しました。5重み・10%・8親セル・前処理・DINO producer・8入力は同じです。8人工試験とruff、別実装Fraction oracleの人工60FP64照合/誤った同点分割の拒否がPASSです。新kernel139737264のKaggle保存版1は全セル/入力/設定10項目一致・RUNNING。元kernel139736510は保存して互換性監査にのみ使用し、両方とも正式提出0です。P043/P044は13:08照会でも採点待ち、最高0.958・目標0.959未達・残り3枠・既存GPU維持を継続します。

**2026-10-20 13:04 JST：P045の非公開Kaggle実行がRUNNINGです。公式0.959未達、P043/P044採点待ち・自己最高0.958・本日残り3枠を維持します。** [実行正本](experiments/P045-private-V1-running-goal959-20261020.json)。元P033の8科学セルをbyte保持し、公開DINOv3 5-foldの順位成分を固定10%だけ加える別仮説です。まずCPU専用監査kernel139734838/V1で5重み各94,000,039Bと全ハッシュ、各180 tensor keyと12出力を確認し、同梱timm1.0.22 wheelは公式PyPIのSHAと一致しました。新candidate kernel139736510/V1の全セル/8入力/非公開offlineT4設定を10項目で照合しました。7人工試験とruffはPASSですが、実モデルのlabel順/全key厳密ロード・人工CPU/CUDA・実MRI/CSV検算はこれからです。上流DINOの教師とGold選択履歴は不明で、clean OOFとは呼びません。P045は正式提出0件、既存ローカルGPU処理は維持します。

**2026-10-20 12:45 JST：P044を正式提出57420634として受理しました。P043 ref57420128と合わせて2件を採点追跡中、公式0.959未達・自己最高0.958です。** [P044提出正本](experiments/P044-formal57420634-accepted-goal959-20261020.json)。人工CPU/CUDA反転・4種人工decoder・R6成功3/失敗0・Reader3完走・親36値FP32/最終36値FP64の別実装検算など実出力23項目を通過し、全42件の公式Listでkernel139732287/V1・scriptVersionId360266199・受付番号の対応を照合しました。P044の可視3件の最終CSVは以前のP041と同じですが、隠れた全件や精度の同一性は示しません。本日は2件使用・残り3枠です。既存GPU処理を維持し、採点と別系統モデルの調査を続けます。

**2026-10-20 12:38 JST：F010の新scope数値診断は末尾4block案を支持せず、旧engineering HOLDを維持します。** [別記録](experiments/F010-newscope-fixed12-exploratory-negative-20261020.json)。旧evaluator・policy・HOLDを変更せず、既知のE10更新スキップ1回を記録した探索診断だけを固定E12で行いました。同じ821 UID/group/mask・固定10所見のweak平均AUCはF006対照0.883128、F010 0.878995、差−0.004134、group bootstrap95%区間[−0.013539, +0.005372]です。過去から反復したweak holdoutであり、真の悪化やPublicへの差を断定しません。新診断3試験がPASS、Gold・他fold教師・GPU・重みロード・再学習は0です。F010の採用・fold1・提出へ進まず、教師品質や別系統の多様性へ調査を移します。P043採点待ち・P044可視実行中・公式目標0.959未達は継続中です。

**2026-10-20 12:33 JST：P044の非公開Kaggle GPU実行がRUNNINGです。公式0.959は未達、自己最高0.958を維持しています。** [実行記録](experiments/P044-private-V1-running-goal959-20261020.json)。P041に対して反転224の1セルだけを変更し、残り8/9セルとReader・7入力・Dockerを保持しました。kernel139732287/V1の保存ソースと設定10項目を照合済みです。人工CPU/CUDA QAと今回の実出力の検証はこれからで、P044の正式提出はまだ0件です。P043正式提出57420128は12:31照会でも採点待ち、本日は1件使用・残り4枠です。既存のGPU処理PID19472を維持しています。

**2026-10-20 12:26 JST：P043を正式提出57420128として受理し、採点を追跡しています。自己最高0.958・目標0.959未達です。** [提出正本](experiments/P043-formal57420128-accepted-goal959-20261020.json)。可視COMPLETE・現行V1の全ソース/設定/入力・全40件の重複なしを照合し、親36値＋新36値のFP32検算と実行記録15項目を通過して手動提出しました。実可視3件は失敗0、Gaussian変換で数値36個は変わりましたが、この3件の最終順位は不変であり精度推定には使いません。本日は1件使用・残4枠。次のP044はP041へP036の鏡像224セルだけを組み合わせる別仮説として準備済みですが、まだ可視推論・正式提出を行っていません。

**2026-10-20 12:02 JST：新候補P043のKaggle可視推論がRUNNINGです。公式0.959の目標は未達で、自己最高0.958を保持しています。** [実行正本](experiments/P043-private-V1-running-goal959-20261020.json)。kernel139726098/V1・非公開offline T4・同じ4入力で受理し、保存ソースと設定を照合しました。元P033の8コードセルと出典表記を保持し、3枝のHazen normal-score融合だけを追加しています。標準unittest 11件・ruff・以前の可視3件を用いた別計算の検算を通過しましたが、新runの出力資格や精度の確認はこれからです。手動Saveは1回、正式提出は0件で、本日は残5枠。ユーザー指示で既存GPU処理を維持し、ローカル新学習は始めていません。

**2026-10-20 11:40 JST：新しい依頼として、公式Public 0.959以上の達成まで継続します。現在の自己最高は0.958で、目標は未達です。** [新scopeと事前方針](experiments/goal959-resumed-scope-and-P043-prospective-20261020.json)。昨日の4件はすべて採点済みです。本日の公式枠は11:34照会で0件使用・残5件、既存のGPU処理はユーザー指示で維持し、ローカル新学習は空き次第とします。新しい比較P043では、モデル・係数・入力を固定し、3枝の順位の融合方法だけをnormal-score poolingへ変える仮説を事前記録しました。まだ可視推論・正式提出・改善確認はありません。公開モデル調査と学習範囲の監査も続けます。以下の「追加4件のみ・承認残0」は前回scopeの履歴です。

**2026-10-20 07:18 JSTに最後の未採点だったP041を確認し、承認された追加4件はすべて公式COMPLETEとなりました。自己最高は0.958のままで、0.959以上・0.960超の目標は未達です。** [公式結果の独立正本](experiments/remaining-four-all-official-terminal-independent-20261020.json)・[最終採否と次の検証方針](experiments/remaining-four-final-decision-20261020.json)。採点待ちは0、追加提出の承認残は0です。監視も終了しました。

| 提出版 | 比較した変更 | 自分の公式Public | 採否 |
| --- | --- | --- | --- |
| P038 / ref57391696 | 作者V6の全体レシピを既存環境へ適用 | 0.957 | 不採用 |
| P040 / ref57402883 | P033の融合比率ベクトルだけを変更 | 0.957 | 不採用 |
| P041 / ref57402914 | P033へReaderだけを追加 | 0.958 | 同点として保存、基準は変更しない |
| P042 / ref57403458 | 融合前のRaptor順位だけを再正規化 | 0.957 | 不採用 |

既存のP033などの0.958を保持します。P041は構成を増やしても表示スコアの改善が確認できなかったため、基準の置き換えは行いません。同点表示は未表示桁の同一性やReaderの改善・悪化を証明しません。また、上位のスコアが0.001刻みに見えることだけでは、同じ幅ずつ改善する特定の手順を使っているとは判断できません。

次の優先事項は、追加のPublic比率探索より先に、学習範囲を監査した予測を共通のgroup分割・欠損ラベルmaskで比較できる検証基盤を整えることです。公開重みの学習範囲が不明な予測をOOFとは呼ばず、weak・gold・Publicを分け、相補的な候補の改善が別foldでも続くことを確認してから新しい提出候補を選びます。今回調べた公開Notebookの範囲では、自分の0.959以上に結び付く再現可能な候補は確認できませんでした。公開候補全体に存在しないという結論ではありません。

並行したF010は固定12 epochを正常終了しましたが、2,543/2,544更新・AMPスキップ1回により[事前の工程条件で採用保留](experiments/F010-fixed12-final-engineering-HOLD-20261020.json)です。条件を緩めたり途中epochを採用したりせず、主AUC比較・fold1・提出推論への展開には進んでいません。以下は各時点の履歴です。

**2026-10-20 07:11 JSTの公式結果：P042も0.957でした。4件中3件が0.957で確定し、P041だけ採点待ちです。自己最高0.958・目標0.959未達を維持しています。** [判断正本](experiments/remaining-four-after-P042957-P041-pending-20261020.json)・[P042の提出版照合](experiments/P042-formal-COMPLETE0957-independent-binding-20261020.json)。中間Raptor順位の再正規化は今回のPublicを改善せず、この案は採用しません。最後に提出したP042の採点は終わりましたが、先に提出したP041の公式結果まで確認を継続します。追加提出の承認残は0です。F010は固定12 epoch完了・工程条件による採用保留のままです。以下は各時点の履歴です。

**2026-10-20 06:30 JSTの公式結果：P040は0.957でした。自己最高0.958を保持し、提出済みP041・P042の採点を続けて確認します。** [今回の判断](experiments/remaining-four-after-P040957-two-pending-20261020.json)・[提出版と公式結果の照合](experiments/P040-formal-COMPLETE0957-independent-binding-20261020.json)。P040は融合比率だけを変更した比較で、この比率ベクトルは採用しません。P038の0.957と同点でもReaderの効果や個別の比率の良否は判定できません。追加4件はすべて受付済み・承認残0、0.959以上は未達です。F010は固定12 epoch完了・工程条件による採用保留を維持しています。以下は各時点の履歴です。

**2026-10-20 04:49 JST：追加4件は提出済みで、P040・P041・P042の公式採点を待っています。自己最高は0.958、0.959以上は未達です。** [提出正本](experiments/remaining-four-all-accepted-awaiting-results-20261020.json)。F010は12 epochを完了し、学習プロセスの終了コード0を確認しました。更新2,543/2,544回・E10のAMPスキップ1回により、[事前の工程条件に従って採用保留](experiments/F010-fixed12-final-engineering-HOLD-20261020.json)です。固定evaluatorを変更せず一度実行し、工程条件で終了コード1となることを独立監査22項目で確認しました。これは学習の異常終了や精度悪化の判定ではありません。ラベル・予測・checkpointの内容や主AUCの比較には進まず、条件の緩和・別epochの救済・fold1への展開は行っていません。残る3件の公式結果を引き続き追跡します。以下は各時点の履歴です。

**2026-10-20 03:54 JST：提出済みP040/P041/P042は引き続き採点待ちです。追加4件の受付は完了、承認残0・自己最高0.958を維持しています。** [提出正本](experiments/remaining-four-all-accepted-awaiting-results-20261020.json)。F010は[E11・2,331/2,332更新・AMPスキップ累計1](artifacts/research/20261020-f010-eleven-epoch-AMP-HOLD-counter-observation-v1/summary.json)。E10でのengineering HOLDを維持し、最後のE12まで学習を継続しています。完了後は[固定検証の起動プラン](artifacts/research/20261020-f010-fixed12-final-invocation-review-v1/plan.json)どおり、変更していないevaluatorで工程条件を確認して記録を閉じます。AMP倍率成長は[原因未確定の検討](artifacts/research/20261020-amp-scale-growth-readonly-hypothesis-v1/memo.json)にとどめ、新しい実験や条件変更は行っていません。以下は各時点の履歴です。

**2026-10-20 03:06 JST：追加4件は提出済み、P040/P041/P042の公式採点待ちです。自己最高0.958、0.959以上はまだ未確認です。** [提出正本](experiments/remaining-four-all-accepted-awaiting-results-20261020.json)。F010は[E10のAMPスキップ1によるengineering HOLD](experiments/F010-E10-AMP-HOLD-fixed12-continue-20261020.json)を記録しました。累計2,119/2,120更新で、事前のzero AMP条件を満たさないため採用・公開・fold1・推論consumerへ進めません。学習自体は予定した固定E12まで継続し、AUC比較や別epochへの変更は行っていません。原因は未確定です。正式3件の結果追跡を継続しています。以下は各時点の履歴です。

**2026-10-20 02:17 JST：追加4件は提出済み、P040/P041/P042の採点待ちを継続しています。自己最高0.958、0.959以上はまだ未確認です。** [提出の正本](experiments/remaining-four-all-accepted-awaiting-results-20261020.json)。F010は[E9・1,908/1,908更新・AMPスキップ0](artifacts/research/20261020-f010-nine-epoch-counter-observation-v1/summary.json)で固定E12を継続中です。新しい3件のスコアを見る前に[採用・解釈方針](artifacts/research/20261020-root-prescore-adoption-and-f010-E9-v1/decision.json)を記録し、0.958超の資格版がなければ元の基準を維持します。次目標「0.960超」は0.960そのものでは未達です。F010のfold1対照の検討は事前監査だけで、新しい学習や数値条件の変更は始めていません。以下は各時点の履歴です。

**2026-10-20 02:06 JST：許可された追加4件はすべて正式受理されました。残る3件の採点結果まで継続中です。自己最高0.958、0.959以上はまだ未確認です。** [正本](experiments/remaining-four-all-accepted-awaiting-results-20261020.json)。P038 ref57391696は公式0.957、P040 ref57402883・P041 ref57402914・P042 ref57403458は採点待ちです。制限APIは本日5件・累計40件・残り0枠、ユーザー承認範囲の残りも0件で、日付が変わっても追加提出しません。未採点を0点とは扱いません。
P042は原GPU kernel139595447/V1のまま、実親36値・変更後36値のFP32 bit一致とprovenance 33項目を確認して最後の提出を行いました。監査ファイルの配信502は解消し、V4の一式11ファイルだけで検算しています。推論再実行・データ推定補完・予備版の採用はありません。最後の残枠0がraw応答で省略されたためローカル確認処理が一度停止しましたが、正式受理と所有Get、標準SDKの0、全40件の照合を確認済みで再送していません。F010は固定E12の学習を継続しています。以下は各時点の履歴です。

**2026-10-20 01:43 JST：追加4件のうち3件を正式受理、残り1件を進行中です。自己最高0.958、0.959以上は未達です。** [今回の正本](experiments/remaining-four-three-accepted-last-running-20261020.json)。P038は公式0.957、P040（配合比のみ・ref57402883）とP041（Readerのみ・ref57402914）は採点待ちです。両候補は可視COMPLETE後、独立数値検証と保存ソース・実行記録の照合を通して手動提出しました。制限APIは本日4件・累計39件・残り1枠で、ユーザー承認範囲の残りも1件です。未採点値を0とは扱いません。
最後のP042はGPU枠解放後の手動Saveでkernel139595447/V1として受理され、全ソースと設定を照合し、すでに公式RUNNINGです。元の科学条件は変更していません。実出力の検証後に最後の提出を行い、全結果まで継続します。F010はE8/12・1,696更新・AMPスキップ0で固定E12を継続しています。以下は各時点の履歴です。

**2026-10-20 01:26 JST：P040・P041が両方とも公式RUNNINGへ変わりました。正式比較は残り3件、自己最高0.958・目標0.959以上は未達です。** [状態変化の正本](experiments/P040P041-official-RUNNING-change-20261020.json)。可視実出力はまだ未確認で、独立数値確認後に提出します。P042は既存ジョブの終了とGPU枠の解放を確認してから進めます。F010は[E8・1,696/1,696更新・AMPスキップ0](experiments/F010-eight-epoch-counters-continue-20261020.json)で継続中です。途中AUCを使わず固定E12の条件を維持しています。以下は各時点の履歴です。

**2026-10-20 01:08 JST：追加4件のうちP038は0.957で完了、残り3件を継続中です。自己最高0.958、0.959以上は未達です。** P040・P041は保存済みですが実行状態の応答は引き続き省略形`{}`（SDK解釈QUEUED）、P042は同時GPU枠の解放待ちです。待機原因は確定していません。[独立経路調査](artifacts/research/20261020-legitimate-low-cost-submission-options-independent-v1/review.json)では、保存済み可視3件のCSVだけでhidden全件の比較を作る経路は確認できませんでした。
別実行モードの最小CPU診断[D003](experiments/D003-private-CPU-interactive-unqualified-20261020.json)は手動Save1回がHTTP例外となり、直後の所有一覧0件と既存非公開診断の読取成功を照合しました。受理版・実行・出力は未確認、正式提出枠とGPU消費は0、再試行しません。例外本文を保持できなかったためHTTP状態・原因は不明です。既存GPUジョブとF010学習は維持し、実出力の独立検証後に提出する条件を継続しています。以下は各時点の履歴です。

**2026-10-20 00:35 JST：P038（57391696）は公式COMPLETE・Public 0.957でした。残3比較を最後の正式結果まで継続中です。** [判断正本](experiments/remaining-four-after-P038957-20261019.json)、[独立binding確認](experiments/P038-terminal-binding-independent-20261019.json)。自己最高0.958は保持し、0.959以上は未達。作者報告0.958と自分の再現0.957を区別します。

合成比率とReaderを同時に変えたP038の結果だけでは、どちらが悪化要因かは判定できません。準備済みのP040（比率のみ）・P041（Readerのみ）を独立した比較として続けます。両件はGPU実行待ち、P042は同時2GPU枠の解放待ちで、正式な追加提出は残3回です。F010は[E7・1,484/1,484更新・AMPスキップ0](artifacts/research/20261020-f010-seven-epoch-counter-observation-v1/summary.json)、固定E12・両foldの採用条件を維持しています。自分の提出済み予測CSVを調べる任意診断は正規APIがHTTP403を返したため取得不可で終了し、再試行・別経路は使っていません。

長時間待機へのP040限定の運用例外として、ソース・人工データ・保存版を独立確認後に正規APIへ1回正式要求しましたが、出力ファイル未生成のHTTP400で受付されませんでした。[独立照合正本](experiments/P040-pre-visible-formal-rejected-no-quota-20261020.json)は受付0・枠消費0、累計37件・残3回を確認しています。この経路は終了し、実可視出力と独立数値確認後の提出条件へ戻しました。GPUジョブ・科学要因・F010の採用条件は変更していません。作者V6の0.958は[版別リンクを確認済み](artifacts/research/20261020-p038-author-version-score-binding-readonly-v1/review.json)ですが、P038はR6配置・デコーダ・guardへの適応を含み、全Input・実行が完全同一とは断定しません。以下は各時点の履歴です。

**2026-10-19 23:44 JST：残4比較を最後の正式結果まで継続しています。** P038（57391696）は採点待ち、P040/P041はGPU実行待ち、P042は同時GPU2枠の解放待ちです。追加4件のうち正式受付1件・残3件、自己最高0.958、0.959以上は未達です。元のGPUジョブは保持しています。22:29の公式再照会でも出力0・GPU使用時間不変、22:52の公式監視も変化なしです。

切り分け用の非公開CPU診断[D001](experiments/D001-private-cpu-input-diagnostic-complete-20261019.json)と[D002 v2](experiments/D002-v2-private-cpu-stage-diagnostic-complete-20261019.json)は公式COMPLETEでした。D002 kernel139616178/V1は元P040/P041と同じ3058Bのguardを実IPythonで1回実行し、既存R6の9ファイル計886,813,460Bのサイズ・SHA、bundle、4foldを8.85秒で確認しました。成功JSONだけを取得し、エラー出力はありません。モデル復元・MRI/Report/ラベル読取・学習・GPU呼出し・正式提出は0です。確認できた範囲は共通R6のbyte整合性とCPU起動経路であり、GPU待機原因・全Inputの実マウント版ID・モデル推論・精度は未確認です。

[F010のE6観測](artifacts/research/20261019-f010-six-epoch-counter-observation-v1/summary.json)は1,272/1,272更新・AMPスキップ0、学習継続中です。途中AUCやcheckpoint選択を使わず、固定E12を待ちます。事前の推論互換性分析ではCLS last4/REG last2のprefix指紋の定義が異なるため新しいfamily別復元が必要と判明しましたが、既存方針の両fold合格条件を維持し、推論実装は始めていません。[22:04の公開上位100件](artifacts/research/20261019-public-frontier-once-130441Z/summary.json)も作者最高0.958、0.958超0件で、既知上位のsource/Input/scoreに変更はありませんでした。以下は各時点の履歴です。

**2026-10-19 21:47 JST：残4比較を継続しています。P038は採点待ち、P040・P041は実行待ち、P042はGPU2枠の解放待ちです。自己最高0.958、0.959以上は未達、正式提出は残3枠です。** 長時間待機の切り分けとして、同じInput4参照・Dockerで、MRI・Report・重みを読まない軽量CPU診断[D001](experiments/D001-private-cpu-input-diagnostic-started-20261019.json)をrootが1回手動SaveAndRunしました。非公開kernel139613277/V1・GPU/TPU/通信OFFを保存Getで確認し、診断marker待ちです。元のGPUジョブとF010学習は保持しています。この診断の成功だけではGPU推論や精度の資格にはしません。

21:00の公式Get/ListはP038の採点未返却で一致し、本日2件・累計37件・残3枠でした。21:20の[公開一次調査](experiments/public959-primary-followup-20261019-122009Z.json)も最高作者0.958で、実行可能な0.959以上の新候補はありません。0.960のDiscussionには公開source/重みがなく、0.959予想のNotebookは公式ERRORでした。作者の4教師データ・約60variantという説明は来歴の主張として記録し、Gold選択監査や既存ラベルは変更していません。F010の最終保存観測はE3・636/636更新・AMPスキップ0です。以下は各時点の履歴です。

**2026-10-19 19:54 JST：残4比較を継続中。P038は正式受付後の採点待ち、P040・P041は実行待ち、P042はGPU同時2枠の制限で実行受付待ちです。自己最高0.958、0.959以上は未達です。** 最後の候補を早めに待機列へ入れるため、P042の固定sourceをrootが一度だけ手動SaveAndRun要求しましたが、Kaggleは`Maximum batch GPU session count of 2 reached.`を返し、kernel/versionの受付値は返しませんでした。[独立照合の正本](experiments/P042-private-admission-held-GPU2-20261019.json)。その前の存在確認Get403は送信前停止であり、真正Save試行は1回・受付0・正式Submit0です。先行2件のいずれかの完了を確認してから所有slug/versionを再確認し、新しいrootレビューで手動実行します。ソース・比較要因は変更していません。

残3候補の正式申請用JSONは[独立照合済み](artifacts/research/20261019-remaining-three-inert-request-independent-v1/review.json)ですが、実CSV・数値の資格は全件未完了です。Kaggle状態照会は明示V1と公式CLI相当latestの両方でQUEUEDを確認しました。公開上位100件も19:09〜19:31の3回確認で作者最高0.958のままです。F010は[保存済みのE1観測](artifacts/research/20261019-f010-first-epoch-counter-observation-v1/summary.json)で212/212更新・AMPスキップ0、同じGPU学習を継続中。途中指標を採用には使わず、固定E12を待ちます。以下は各時点の履歴です。

**2026-10-19 19:00 JST：P039は正式0.958で、自己最高は変わりません。ユーザー承認の残4提出を最後まで進めます。** [今回の4比較](experiments/remaining-four-staged-plan-20261019.json)。P038を18:44に1件提出（57391696）して採点待ち、本日2／累計37／残3枠です。P040（P033の係数4値だけ）・P041（P033へReader段だけ）はsource／人工試験と独立レビューを通過し、private Kaggle V1へ保存・可視実行の待機中です。P042（Raptor2本の合成後の中間順位正規化だけ）もsource資格を通過し、実可視検証はこれからです。新しいPublic改善は未確認。元0.958を保全し、各候補の実推論と独立数値確認後にrootが1件ずつ手動提出します。

F009 retryは全12epochを完了しましたがE10に候補AMPスキップ1回（2495/2496更新）があり、固定採用条件で[HOLD](experiments/F009-retry-fixed12-candidate-AMP-HOLD-v2-20261019.json)です。固定v3は数学呼出し前にexit1、追加のpaired比較・例外追加・別checkpoint探索・モデル輸送は行っていません。学習時のweak成績は記録済みですが今回の選択には参照せず、モデルの精度が悪い／NaNという意味にはしません。

F010の旧待機ジョブもF009の工程条件を検出してdispatchせず停止しました。旧拒否を保全し、gamma採用とは独立したF006対照の末尾2→4block比較として[別手動起動](artifacts/research/20261019-f010-root-independent-fit-v1/manual-launch.json)を宣言し、18:45:46からfold0を学習中です。元185pinとCUDA preflightを再確認し、実PID23168／launcher24024・GPU稼働・初期history0を確認しました。元policy／QA／固定E12評価器・入力・教師・seed・foldを維持します。この学習の完了や採用を今回4枠の前提にはせず、独立のモデル強化として続けます。以下は各時点の履歴です。

**2026-10-19 12:42 JST：独自モデルの本体強化へ進み、新F010をGPU検証済み・学習待機にしました。自己最高0.958で、新しい精度改善はまだ未確認です。** [実装・実行の正本](experiments/F010-last4-qualified-queued-20261019.json)。現行独自モデルの追加によるPublic改善は未実証です。旧0.957親に対するP032／P034／P035は全て0.957でした。0.958親のP039（57378810）は12:40の公式確認でも採点未返却・エラーなし。本日1件／累計36件・残4枠です。weakとPublicの数字の大小を原因の証明とせず、単体性能と誤りの補完性を分けて改善します。

[同じfoldでの誤り分析](experiments/native-model-diversity-Public-evidence-20261019.json)では、CLS＋REGの固定50%平均はweak固定10所見で両fold改善しましたが、Public改善は示していません。F006とgamma変更F009の順位相関中央値は0.99787、50%平均はF009単体より低く、gamma版を別の多様性モデルとして加える根拠は弱い結果です。公開0.958親の同じholdout予測は未取得で、その親との誤り相関は測定できていません。

[公開学習コードの調査](experiments/public-training-recipes-and-F010-priority-20261019.json)で、データ作成・教師生成・学習設定まで確認できるS6 ConvNeXtレシピを見つけ、10件のtextを取得しました。解剖位置に基づく局所crop、3slice入力、空間／検査内attention、soft教師などが参考になります。作者0.950は読取失敗時のCoAtNet fallbackを含むpipeline値です。[Gold利用範囲の補足](experiments/public-JEV-Gold-exposure-scope-correction-20261019.json)では、JEVの上流Gold-informed選択とGold実値の直接混入を区別しました。上流曝露だけで公開教師を全面禁止にはせず、別の教師実験は直接混入・58件／共有group除外・mask・上流曝露を監査して検討します。今回F010へ教師変更は持ち込みません。

F010は既存F006の画像392px・教師4207件・head・seed・group/fold・損失・LR・12epochを維持し、DINOv2の学習対象だけ末尾2→4blockへ増やします。encoder学習パラメータは3,550,464→7,100,928、head83,352は同じです。CPU人工21試験・独立レビュー、新sourceのGPU人工144窓と既存processed115窓の各6確認、CUDA preflightを通過しました。元F006/F009のsource・runは変更していません。実精度の検証とは区別します。

12:34:27に手動起動したローカル待機PID10452（launcher18024）は、F009 retryの完全12epoch・208更新×12・skip0を待っています。新モデルの学習はまだ始まっていません。依存学習が正常完了し固定185pinとCUDA preflightが通った場合だけ、F010 fold0を別runへ1回起動します。待機開始期限は本日22:00 JST、F010学習の所要7〜10時間は目安。自動評価・fold1起動・外部upload／Save／Submitはありません。固定E12を821 UID・同mask/groupで旧F006 E12と比較し、正の差と工程条件を満たす場合に別foldを検討します。[評価器](artifacts/research/20261019-f010-last4-root-v1/evaluator-verification.json)は人工10＋独立6試験とRuffを通過し、実比較は未実施です。以下は各時点の履歴です。

**2026-10-19 11:47 JST：自己最高0.958。第一目標は公式0.959以上、その次は0.960超です。P039を1件だけ正式提出し、採点待ちです。** [今回の判断と根拠](experiments/strategy-0959-0960-P039-20261019.json)。提出57378810は11:40に受理され、11:47の公式確認でもstatus／Publicは未返却・エラーなし。失敗や0点とは扱いません。本日1件／累計36件、残り4枠です。直近5件はP033・P036・P037が0.958、P034・P035が0.957。強いP033親へ既存独自4モデルを固定20%で加えるP039を比較し、残る枠は結果と追加検証を踏まえて使います。

[公式LB調査](experiments/official-LB958-plateau-20261019.json)では1038位／6196チーム、表示0.958は178〜1344位の1167チーム、0.959以上は177チームでした。[公開Notebook上位100件](experiments/public-frontier-primary-static-review-20261019.json)の作者最高は0.958で、0.959以上は0件。新0.958 fork2件のコードは同一でした。特定上位者の0.966→0.967の理由は未確認ですが、公式Discussionには段階的改善とTTAの本人報告があります。既存0.958を固定し、一要因の改善と異なる予測の追加を検証します。公開版の表示スコア・最新版・採点版・Input版を区別します。

**F009 fold1の旧「学習中」記録は現在の状態ではありません。** 10月17日19:12のE5・1040更新・AMPスキップ0で止まり、元PIDとE12成果物はありません。原因は未確定です。推論用checkpointにoptimizer／scaler／RNGがないため継ぎ足さず、[別runの同条件12epoch](experiments/F009-fold1-full-restart-live-20261019.json)を11:45:31に手動起動しました。実学習PID22332／launcher4148、4090の稼働を確認。元runを保全し、同じseed20261002・fold1・config・8ソース・教師・cache・環境を維持しています。所要約6時間は目安です。途中の成績を採用に使わず固定E12を待ちます。

新run用の[評価手順](artifacts/research/20261019-f009-fold1-recovery-evaluator-v3/manual-v3.json)は起動前に固定しました。人工9試験＋独立5試験・Ruffが通過し、保存epoch一覧の検査不足も独立負例で発見・修正済みです。旧数学・旧HOLD・旧対照E10の既知スキップ1回だけの例外を保持します。候補全12epochスキップ0・両foldそれぞれの固定10所見AUC差が正であることが必要です。最終独立レビューは起動後・初epoch完了前に通過。実E12評価は未実施で、既存CPU復元／輸送は新v3結果に未対応です。共同条件を通った場合に別版consumer、実CPU、固定192件診断・可視推論・独立計算を準備します。Goldを自分の学習・選択には使わず、公開重みの既知Gold58選択曝露と患者独立性未確認は残します。以下は各時点の履歴です。

**2026-10-17 18:48 JST：自己最高0.958、目標0.960は未達です。独自F009用の起動処理・入力検証・Notebookのソース準備が、人工試験18件と独立レビューを通過しました。** [今回の正本](experiments/F009-Kaggle-serving-source-ready-fold1-E4-20261017.json)。旧モデル専用の起動処理は実行せず、新CLS2本と固定REG2本を扱う別経路へ接続しました。ソース108ファイル・全体116ファイルを実際に数え、P033の科学計算8セルと既存89ソースの一致を確認しました。新Notebookは実行無効・Save用JSONなしです。実モデル・実GPU・提出の成功とは区別します。

同じfold1は[4/12epoch](artifacts/research/20261017-f009-fold1-progress-v1/epoch4-counter-observation.json)・832更新／832機会・AMPスキップ0で継続中です。固定E12後に凍結済み評価を1回行い、両foldの条件を通った場合だけ実CPU復元・モデル束・輸送へ進みます。途中の成績による採用条件の変更はありません。18:39の表示処理で文字コードエラーが生じ、ツールのエラー出力へrun本文が含まれた件も記録しました。以後はPythonでカウンタだけを表示し、設定・評価方針・親の計画は変更していません。

**次の未完作業は、保存済みの同じ192件を使う診断入口の実装です。** 現在の`diagnostic-count=192`は提出用CLIの互換確認で、192件診断の対応・通過を意味しません。その後に実192件診断、可視推論、別計算によるFP32・失敗マスク確認が必要です。既存P039優先・P038次候補という提出順は維持し、提出枠を新しく確認して進めます。最後の公式枠確認は18:11、本日5件／累計35件・追加枠0扱いです。公開調査は18:24の公式上位100件で作者最高0.958のまま。台帳324行・15列・ID重複なし。以下は各時点の履歴です。

**2026-10-17 18:28 JST：F009のオフライン輸送と、新CLS2本＋既存REG2本の復元・平均計算のコードが、人工試験と独立レビューを通過しました。自己最高0.958・目標0.960未達は変わりません。** [今回の正本](experiments/F009-transport-mixed-source-ready-public958-frontier-20261017.json)。[輸送処理](src/rsna_knee/f009_transport.py)は版8c613…、[混在モデルの復元処理](src/rsna_knee/f009_mixed_runtime.py)は版736958…を固定しました。実F009の重み・モデル束・輸送・GPU推論はまだ行っていません。

輸送は人工試験6件と独立した途中失敗の試験を通過しました。全ファイルを一時ディレクトリで完成させてからまとめて保存し、途中失敗時に完了記録が残らないことを確認しました。復元処理は人工CPU試験5件＋独立4件を通過し、各foldの実ソース・クラス、旧REGの正本、4モデルの独立性、元と同じFP32平均、推論時gamma無効と乱数状態を確認しました。人工encoder・人工checkpointでの確認であり、実モデルの成功とは区別します。

次は新しいrouter／native child／Notebookへ接続します。GPU用モデル作成後のCPU containerと、終了時のGPU containerを明示的にcloseする必要があります。現在のfold1は固定12epochを続け、最後に凍結済み評価を1回実行します。両foldの条件通過後にだけ実CPU復元・モデル束作成・Kaggleの192検査と可視推論へ進みます。18:24の匿名公式上位100件は作者最高0.958が6件、0.959以上0件、既知6件のsource／Input版／score変更0でした。Heliosのslug変更は同じkernel／sourceであり、新しい改善とは扱いません。P039優先・P038次候補、提出枠0扱い・自己0.960未満という現状を維持しています。台帳323行・15列・ID重複なし。以下は各時点の履歴です。

**2026-10-17 18:15 JST：P037（57293138）も公式COMPLETE・Public 0.958になりました。提出済み5件は全件採点済みで、自己最高0.958・目標0.960未達です。** [今回の正本](experiments/P037-official958-five-results-complete-F009-live-20261017.json)。P033・P036・P037が0.958、P034・P035が0.957です。P037の受理応答・保存版139143442/V1・source07bea17a…・所有者・正式結果を照合しました。全体レシピの自己再現値であり、隠れた採点データでの各armの参加や、R7／Reader単独の効果は未確認です。既存の提出説明にあるR7offは誤記で、実コードのR7比率は0.4です。

独自F009 fold1は同じrunで[完了E3](artifacts/research/20261017-f009-fold1-progress-v1/epoch3-counter-observation.json)まで進み、624更新／624機会・AMPスキップ0、累計経過5,150.82秒です。固定E12完了前の成績値は投影・選択していません。現在は、条件通過後に必要となるオフライン輸送と新旧4モデルの復元処理を、人工CPU試験の範囲で準備しています。実F009モデルの復元・輸送・Kaggle推論は未実施です。

18:11のlimitsは本日5／累計35で、追加枠0扱い・回復時刻未確認です。結果未返却は0件になったため、この5件を繰り返し照会する工程は終了します。枠は必要時に新しく確認し、目標未達なら検証済みP039を優先、P038を次候補として保持します。F009は両foldの条件と新しい実推論の検証を通過した場合だけ提出候補にします。台帳322行・15列・ID重複なし。以下は各時点の履歴です。

**2026-10-17 18:02 JST：F009が両foldの条件を満たした場合の提出比較を、P039を対照とする別計画として固定しました。自己最高0.958・目標0.960未達は変わりません。** [条件付き比較の正本](experiments/f009-P039-parent-fixed20-CLS-pair-conditional-20261017.json)。自己0.958のP033由来の親、全所見の係数、native20%、対象10所見、REG0=E8・REG1=E10、失敗union・UID180秒・全体失敗時の扱いをP039と揃え、CLS0／1の固定E12重み2本だけをF009へ置換する計画です。旧P029M親の計画は保全しました。

宣言時点でfold1はE2の成績を既に記録していますが、値は表示・参照していません。最終E12比較前の輸送・親構成の宣言であり、学習前や最初の成績記録前の登録とは扱いません。凍結済み評価・両foldの正の差・候補スキップ0・旧HOLDは変更していません。実CPU復元、新モデル束・輸送、実192検査と可視推論、独立数値確認を経る必要があります。既存P039優先・P038次候補という順序も維持し、F009は条件付きです。台帳は321行で、今回の行に未取得のモデル成績は記入していません。以下は各時点の履歴です。

**2026-10-17 17:49 JST：P034（57291440）・P035（57292346）は公式COMPLETE・Public 0.957でした。自己最高はP033・P036の0.958、目標0.960は未達です。** [今回の正本](experiments/P034P035-official957-F009fold1-E2-runtime-audit-20261017.json)。17:41の公式応答をref・所有者・HTTP・bytes・SHAまで再照合しました。結果未返却はP037（57293138）だけで、status／Publicの省略を失敗や0点とは扱いません。本日5／累計35件、追加枠0扱い・新Submit0です。枠の回復時刻は未確認です。

同じF009 fold1は[完了E2の記録](artifacts/research/20261017-f009-fold1-progress-v1/epoch2-counter-observation.json)で2/12 epoch・416更新／416機会・AMPスキップ0、経過時間は累計3,430.05秒です。同一PID2772・起動時刻16:49:06 JST・CPU使用時間の増加も確認しました。途中のAUC・BCEを選択に使わず、固定E12完了後に凍結済み評価を1回実行します。旧v1のHOLDを残し、別v2で両foldの正の差と候補スキップ0を満たした後にだけ、準備済みCPU復元・モデル束作成へ進みます。

P034・P035は表示上の改善がありません。ただし旧版では独自部分全体の失敗時に親へ戻るため、このスコアだけで採点用データでの独自モデルの参加や処理件数は断定できません。[通常APIでのログ参照範囲](artifacts/research/20261017-p035-hidden-log-access-audit-v1/review.json)も確認し、隠れた採点実行IDとの正規の紐付けは未取得です。[実行時間の監査](artifacts/research/20261017-p039-runtime-priority-readonly-v1/review.json)では既存192検査の独自処理869.22秒・失敗0を再確認しました。提出からスコア表示までの約5時間は、待ち時間を含み得るため推論時間としては扱いません。この監査内の17:26時点のP035未返却記述は、今回の正式結果で更新されています。

次の正式比較は、枠・最新結果を再確認し、自己0.960未満なら既存の可視推論検証済みP039を優先、P038を次候補として保持します。P039は自己0.958の親へ独自4モデルを20%加える版です。[提出内容の確認用JSONと版の根拠](artifacts/research/20261017-p039-manual-submit-binding-v1/binding-review.json)も保存しました。標準APIのowner／slug／明示的な保存版1／submission.csvを固定した準備で、まだ提出していません。可視3検査の成功は隠れた全件の成功保証ではありません。現在の学習後は準備済みF006 fold2と封印済みC=0.01の外側評価を進めます。公開調査の最新確認は17:24の公式上位100件で作者最高0.958のままです。以下は各時点の履歴です。

**2026-10-17 17:33 JST：自己最高0.958、目標0.960は未達です。同じF009 fold1が1/12 epoch・208更新・AMPスキップ0まで完了し、学習を続けています。** [今回の正本](experiments/F009fold1-E1-amended-consumers-ready-public958-duplicate-20261017.json)、[epoch1のカウンタ確認](artifacts/research/20261017-f009-fold1-progress-v1/epoch1-counter-observation.json)。OS上の同一PID2772と起動時刻16:49:06 JST、CPU使用時間の継続増加も確認しました。途中のAUC・BCEは選択に使わず、固定E12を待ちます。

新v2評価結果を扱う[CPU復元処理v2](src/rsna_knee/f009_cpu_proof_v2.py)と[ローカル保存処理v3](src/rsna_knee/f009_pair_stage_v3.py)を別版で準備しました。人工CPU試験7件・保存処理の人工試験5件・Ruff・独立レビューが通過し、rootも復元のreturn前の全ASTと40tensor検査ループが旧版と同一であることを確認しました。[CPU最終証跡](artifacts/research/20261017-f009-cpu-proof-v2-preparation-v1/verification-final.json)、[保存処理最終証跡](artifacts/research/20261017-f009-pair-stage-v3-preparation-v1/verification-final.json)。旧方針・旧HOLD・旧ソース・旧registryは変更していません。実fold1比較・実CPUモデル復元・実モデル束作成・Kaggle輸送はまだです。両foldの実評価が条件を満たした後にだけ実行します。

17:24の公式スコア順上位100件は作者最高0.958、0.959以上0件でした。[新たな0.958 forkの調査](artifacts/research/20261017-lavinwins-top1gold-scored-review-v1/review.json)では、`lavinwins/top-1-gold`の採点版V4／359191499／提出57291942を明示的に結合しました。全9セル・7コードセルが取得済みLavin V5・Helios V1と一致し、P033の科学計算も同じです。[rootの別比較](artifacts/research/20261017-lavinwins-top1gold-root-equivalence-v1/review.json)でもセルsourceのUTF8 bytesが一致しました。新しい改善要因ではないため追加実行は見送ります。表題のGoldから新たなラベル使用は断定せず、既知の上流Raptor Gold58選択歴と、自分のGold不使用は区別します。

17:21の公式順位は837位／6,126チーム、締切は10月23日08:59 JSTです。[公式状態](artifacts/research/20261017-official-account-observation-082143972204Z/summary-v2.json)。17:26の正式結果はP033・P036が0.958、P034・P035・P037は未返却。本日5／累計35、追加枠0扱い・新Submit0です。GPU残量の計算値は約27時間で、提出回数の枠とは別です。枠と結果を再確認し、目標未達なら既存の検証済みP039を先行、P038を次候補として保持します。GPUでは同じfold1を完了させ、その後に準備済みF006 fold2と封印済みC=0.01の外側評価を進めます。以下は各時点の履歴です。

**2026-10-17 17:07 JST：F009 fold0の固定E12比較が完了し、同設定のfold1を4090で学習中です。自己最高0.958、目標0.960は未達です。** [今回の正本](experiments/F009E12-positive-fold1-running-P036958-20261017.json)、[fold0実比較](artifacts/research/20261017-f009-fixed12-guarded-result-v1/comparison-v2.json)、[独立確認](artifacts/research/20261017-f009-fold0-E12-independent-gate-v1/verification.json)。固定10所見のweak平均AUCは対照0.883128から0.884821、差+0.001693。全12epoch・2,544更新・AMPスキップ0と入力・コード・checkpointのSHA照合を確認しました。CI・BCEは報告専用で、Publicの増分を意味しません。

fold1は16:49:06 JSTに新runで起動し、元seed・gamma範囲・固定12epoch・前処理・ラベル欠損処理を維持しています。開始記録スクリプトの失敗後も起動を続けた不備は[別記録](artifacts/research/20261017-f009-fold1-execution-v1/start-record-check-failure-and-observed-launch.json)へ保存しました。失敗原因は旧F006 fold1対照のE10に既知のAMPスキップ1回があったことで、設定・入力・ソースの不一致ではありません。[起動後の照合](artifacts/research/20261017-f009-fold1-execution-v1/observed-launch-review-after-start.json)は通過しています。

旧「両側スキップ0」方針と評価コードは変更せず、旧方針での共同採用判定はHOLDです。[別版v2方針](experiments/f009-fold1-fixed12-joint-screen-premetric-amendment-v2-20261017.json)を16:58:35 JST、fold1のhistoryが空の時点で宣言し、[独立監査](artifacts/research/20261017-f009-fold1-amendment-v2-independent-v1/review.json)を通過しました。例外はSHA固定の旧対照E10の1回だけで、新候補は全12epochでスキップ0、両foldそれぞれの正のAUC差が必要です。これはfold0結果・fold1起動後の修正で、学習開始前の登録とは扱いません。旧対照2,495更新と候補に要求する2,496更新の差、旧consumerとの非互換も残します。

新v2評価コードは17:04:17 JSTにSHA固定し、その後17:04:58の実runもhistory=[]でした。[根拠と実行条件](experiments/f009-fold1-v2-wrapper-and-next-submissions-20261017.json)、[人工試験と独立監査の最終証跡](artifacts/research/20261017-f009-fold1-amended-evaluator-v2/verification-final.json)。人工5試験・別担当の11負例／9判定例を通過し、rootもコードと通常CLIの入力未読動作を確認しました。実fold1の比較はまだ実行していません。完了後に旧数学を1回呼び、旧HOLDと新v2の結果を分けて保存します。

16:59の公式照会ではP033（57290868）とP036（57291580）がCOMPLETE／0.958、P034・P035・P037はstatus／Public未返却でした。本日5／累計35・追加枠0扱い、再提出0です。枠が確認できて目標未達なら既存の検証済みP039を先行し、P038を次候補として保持します。P035に共通する実運用上の失敗が判明した場合は原因を確認しますが、単なる採点未返却を失敗とは扱いません。次は現在のfold1の固定E12評価、その後に準備済みF006 fold2と封印済みC=0.01モデルの外側評価を進めます。

既提出P037の説明文にある「R7off」は誤記で、実コード・実receiptはR7比率0.4です。[照合と提出順の根拠](artifacts/research/20261017-pending-results-submission-priority-v1/proposal.json)へ訂正を保存しました。R7の有無だけの比較とは解釈しません。P037／P038の可視3行の最終SHAが同じでも、hidden予測の一致は未確認です。以下は各時点の履歴です。

**2026-10-17 16:35 JST：P039は実可視推論・独立数値確認まで完了しました。自己最高0.958、目標0.960は未達です。** [最新の正本](experiments/p039-visible-qualified-own958-preserved-20261017.json)、[root監査](artifacts/research/20261017-p039-execution-v1/root-actual-output-audit-v1.json)、[別計算による確認](artifacts/research/20261017-p039-actual-independent-v1/review.json)。kernel139182563/V1は公式COMPLETE。全体489.58秒、独自モデル部分42.79秒で、可視3検査すべて成功・失敗0、4モデルの計12 forwardを確認しました。

10所見30セルに独自予測が参加し、親から26値・4順位セルが変わりました。P033親3系統の合成36値、独自4モデルの平均36値、最終20%合成36値は独立した順位・FP32計算と全一致。MCL／Synovitisの6セルは親の文字列を保持しています。正常実行と参加の確認であり、hiddenやPublic改善の証拠ではありません。実失敗行は今回なく、失敗処理の人工確認とは区別します。

16:32の公式再照会でもP033がCOMPLETE／自己0.958、残る4提出はstatus／Public未返却。本日5／累計35・追加枠0扱いのため新しい正式Submitは0です。P039とP038の成功版を保持し、実際の枠と残る採点を確認して提出順を決めます。並行するF009は直近11/12 epoch・2,332更新・skip0。同じrunの固定E12を既定のguard付き評価で比較し、正なら同条件fold1、非正ならF006 fold2を先行します。以下は各時点の履歴です。

16:24の追加確認：P039は公式RUNNINGで、出力はまだ返っていません。公開Codeスコア順上位100件の作者最高は引き続き0.958・既知5件、0.959以上と既知版のsource／Input参照変更は範囲内0でした。自己最高0.958を保持して検証を継続します。[追加観測](artifacts/research/20261017-p039-running-record-v1/public-frontier-and-first-status-supplement.json)。

**2026-10-17 16:21 JST：新候補P039をKaggle GPUで起動しました。自己最高0.958、目標0.960は未達です。** [最新の実行記録](experiments/p039-P033958-native20-GPU-running-20261017.json)。kernel139182563/V1・`rsraki/p039-p033-native20-20261017-v2`は公式RUNNING。全4コードセル、private／T4／GPU／TPU OFF／通信OFF／Docker2757、9 Inputの正規化参照を保存Getで照合しました。Input版は要求参照を固定していますが、Getは版を省略するため実runtimeのハッシュ・fingerprint確認が必要です。

科学変更は成功したP033親の固定クラス係数のみです。既存native4モデル・89ソース・合成20%・成功UID内のFP32順位合成・失敗セル保持を固定しています。新しい親照合と記録を整え、追加全体の失敗や寄与0は例外で停止させます。人工5試験・Ruff・独立レビュー・root確認を通過し、別計算で正常／部分失敗の72セルが一致しました。実可視推論、hidden、Public改善はまだ未確認です。旧P035の監視実装を保持しており、別のinnerC POSIX試験の資格を転用していません。

16:19の正式結果はP033がCOMPLETE／0.958、残る4件はstatus／Public未返却。本日5／累計35・追加枠0扱いで、新しい正式Submitは0です。P038も保持し、実検証・残る結果・公式枠を踏まえて提出順を決めます。F009は11/12 epoch・2,332更新・skip0の同一runを継続し、固定E12比較を待っています。最新の公式順位は818位／6,123チーム、締切は10月23日08:59 JSTです。以下は各時点の履歴です。

**2026-10-17 16:15 JST：自己最高0.958・目標0.960未達です。F009は11/12 epoch・2,332更新・AMP skip0まで進みました。** [最新の正本](experiments/F009E11-POSIX-actualCPU-PASS-P039-preparing-20261017.json)、[E11観測](artifacts/research/20261017-f009-gamma-preparation-v1/epoch11-observation/observation.json)。最終行の累積elapsedは18,974.21秒。固定E12の比較と次の学習順は変更せず、途中精度で選択していません。

非公開Kaggle CPU・Input0のkernel139180926/V1は公式COMPLETEでした。人工プロセス6ケース・直接の子5つで、未回収状態の検査、TERM無視後のKILL、タイムアウト、自己／既存グループへの操作拒否、全子の回収と一時ディレクトリ除去を実環境で確認しました。[実監査](artifacts/research/20261017-innerC-supervisor-POSIX-CPU-v1/root-actual-runtime-audit.json)、[独立確認](artifacts/research/20261017-innerC-supervisor-POSIX-actual-peer-v1/review.json)。約1.04秒のOS境界確認であり、実MRI・モデル・feature/head全体・親との9時間枠統合・Public改善の資格は含みません。

P039は、成功したP033親0.958とP035の既存native20%を接続する新候補として別領域で準備しています。科学変更は親の固定クラス係数のみ。旧親専用の照合でnativeが無効になる問題を避けるため、P033の識別情報とクラス別の記録を更新し、追加処理全体の失敗やnative寄与0を実行失敗とします。個別UID失敗時の親セル保持・4モデル・合成割合・数値演算は維持します。P035の正式成績は未返却で、P039の実推論・精度も未確認です。P038も保持し、正式提出の順序は実検証・残る採点・実際の枠に基づいて決めます。以下は各時点の履歴です。

**2026-10-17 16:05 JST：P033・提出57290868が公式COMPLETE／自己Public 0.958となりました。自己最高を0.001更新し、目標0.960まであと0.002です。** [結果・根拠・残作業](experiments/p033-official958-ownbest-20261017.json)、[公式raw](artifacts/research/20261017-five-formal-observer-v1/poll-20261017T070535948730Z/57290868.json)。成功したkernel139000655/V1と以前の0.957版を保持します。これは公開の固定クラス別係数を再現した改善であり、独自学習による増分とは区別します。表示以上の未丸め精度は推測しません。

残るP034／P036／P035／P037はraw status／Publicがまだ省略されています。本日5／累計35、追加枠0扱いのため再送しません。P038は可視検証済みのまま保持し、公式枠の回復と直前のsource／Input確認後に正式提出します。

独自F009は同じ4090 runを継続し、直近の確定snapshotは10/12 epoch・2,120更新・AMP skip0です。固定E12の主比較が正かつ工程条件を満たせば同条件fold1、非正ならF006 fold2を先行します。途中成績で方針を変更しません。新Cモデルの外側fold2評価は、固定済みモデル・予測を保持したまま新しいF006 fold2対照の完成後に実施します。

動的test特徴抽出controllerは人工7試験、F009のCPU復元producerと別sealer v2は人工5試験、専有子プロセス監視は人工5試験を通過し、各sourceの独立レビューとroot確認を終えました。中断時も子を回収し、拡張失敗を従来版だけの成功提出として隠さない契約です。現在はKaggle CPU上の人工プロセスによる実POSIX終了・回収確認を準備中。実モデル復元・MRI推論・親との統合・採用資格はこれからです。以下は各時点の履歴です。

**2026-10-17 15:50 JST：F009は10/12 epoch・2,120更新・AMP skip0で同じrunを継続中です。自己最高0.957、目標0.960は未達です。** [今回の正本](experiments/F009E10-publicCSV-math-ready-20261017.json)、[E10観測](artifacts/research/20261017-f009-gamma-preparation-v1/epoch10-observation/observation.json)。累積elapsed17,246.19秒。固定E12比較の基準・学習条件は変更していません。

15:52 JSTの公式照会でも新5提出のstatus／Publicはraw省略、本日5／累計35・追加枠0扱いです。[追跡記録](artifacts/research/20261017-F009E10-publicCSV-record-v1/post-record-observation-1552.json)。同じ学習PIDの稼働も確認し、再送・学習再起動は行っていません。

新C単一モデル用の[固定20% CSV算術](src/rsna_knee/orthodiff_inner_c_public_blend.py)は人工5試験・Ruff・独立レビューを通過しました。成功UIDのhead確率だけをFP32で順位化し、親の元予測と別々のFP32積・和で合成します。失敗UID・MCL／Synovitis・未対応・成功2件未満・FP32定数列は元のCSVセルを保持。親の選択・実合成・正式提出はまだです。実CSVのヘッダー順は、将来controllerが正規13列と完全一致を確認してからこの算術へ渡します。

test用の特徴抽出は、研究用4,207件やjournalに依存しない動的UID経路の設計・helper閉包確認を終え、隔離実装中です。F009のCLS2本とsource14を封印するツールも人工ファイルで3試験を通過しましたが、実CPU復元proofのproducerが残り、実sealingは拒否する状態です。これらの準備をモデルの精度・実推論資格とは扱いません。次は予定どおり固定E12の主比較を行い、条件が正なら同条件fold1、非正ならF006 fold2を先行します。以下は各時点の履歴です。

**2026-10-17 15:37 JST：最終比較の実行準備と、封印モデルの実Kaggle CPU数値検証を完了しました。自己最高0.957、目標0.960は未達です。** [今回の正本](experiments/F009E9-guards-ready-innerC-Kaggle-numeric-PASS-20261017.json)。F009は9/12 epoch・1,908更新・AMP skip0で、同じRTX4090 runが稼働中です。途中成績による変更はせず、固定E12を同じF006 E12と比較します。

学習側inner4で選んだC=0.01の単一モデルについて、人工32入力・384予測をNumPy1.22.3とKaggleのNumPy2.1.3で比較し、FP32 bitsが全一致しました。[実数値監査](artifacts/research/20261017-orthodiff-innerC-numpy-kaggle-cpu-v1/root-actual-runtime-audit.json)。非公開・offline・CPU・Input0のkernel139175515/V1は公式COMPLETE、別の内積・sigmoid計算でも一致。実MRIの推論・外側fold2 AUC・Publicの改善を確認した資格ではありません。

F009の最終評価は、現在の入力・ソース・両E12 checkpointの実SHAを確認してから既存計算を一度呼ぶwrapperを用意しました。改善時のfold1評価、報告専用の固定REG50副診断、新Cの外側fold2評価も人工試験と別担当のsource確認まで完了。F009の新ソース14本を隔離する仕組みを整え、提出環境で旧model classへ戻る問題を防ぎます。実モデルの組込み資格は今後の比較と推論検証で判断します。

実行順は既定どおり、F009 fold0の固定E12差が正で工程条件を満たせば同条件のfold1、次にF006 fold2。非正ならF006 fold2を先行します。gamma候補は両foldそれぞれの差が正であることが必要です。新Cの外側評価は、固定済みモデル・予測を保持して新F006 fold2対照の完成後に一度だけ行います。

15:32の公式Code上位100件では作者最高0.958・0.959以上0、新しい高得点のsource／Input変更0でした。[調査結果](artifacts/research/20261017-public-frontier-refresh-v6/summary-v2.json)。15:36の公式再照会でも既提出5件のstatus／Publicはraw省略、本日5／累計35で追加枠0扱い。自己スコアの上昇はまだ確認できていません。可視資格済みP038を保持し、正式提出は公式枠と結果を確認して進めます。以下は各時点の履歴です。

**2026-10-17 15:04 JST：学習側だけのC選択を実行し、C=0.01の1モデルと評価用予測を封印・数値確認しました。自己最高0.957、目標0.960は未達です。** [今回の実行記録](experiments/orthodiff-innerC-selected-sealed-F009E8-20261017.json)。元fold0／1／3／4だけのinner4で、固定10所見×4分割のstandalone AUC平均を比較しました。全4候補が工程条件を満たし、事前の最大値規則で0.01を選択しています。

| 共通C | 学習側inner4の固定10平均AUC |
| ---: | ---: |
| 0.001 | 0.713808 |
| 0.01 | 0.728854 |
| 0.1 | 0.716118 |
| 1 | 0.698955 |

選択した設定だけで成功2,409件を学習し、12分類器が警告なしで収束。外側fold2は全888件のmetadataと成功654件の予測を先に固定しました。[独立監査](artifacts/research/20261017-orthodiff-innerC-target-free-actual-audit-v1/verification.json)で、学習側だけの標準化と全7,848予測のFP32一致を確認。外側ラベル・AUCは未解封、Public候補の採用資格もまだです。元のC1両fold案は保留を維持します。

次は新しいF006 fold2固定E12対照が揃ってから、この1モデルを全888件で一度だけ比較します。F009は8/12 epoch・1,696更新・skip0で継続中。まず固定E12を評価し、既定条件で正なら同じgammaのfold1を先行、その後F006 fold2、非正ならF006 fold2を先行する方針です。P038は可視資格済みの提出候補として保持。14:52の公式List35件でも新5提出のスコアは未返却で、同じ候補を再送しません。rawのstatus省略とSDK既定PENDINGを区別して記録しました。以下は各時点の履歴です。

15:06 JSTの公式Get再照会でも新5件の状態・Publicはraw省略、本日5／累計35、追加枠0扱いです。[追跡と将来設計の記録](artifacts/research/20261017-innerC-sealed-F009E8-record-v1/post-record-observation-1506.json)。採用時のF009は4モデル中のNativeTripletMIL 2本だけを置換し、RegionalTripletMIL 2本を維持する設計。新Cモデルは単一モデルの別経路で、提出用の親は未選択です。いずれも条件付きの設計までで、実装・新GPU・stage・Submitは未実施です。

**2026-10-17 14:52 JST：次の一要因は、学習側inner4だけで共通の正則化Cを選ぶ別レシピです。自己最高0.957、目標0.960は未達です。** [事前方針](experiments/orthodiff-regional-innerC-prospective-policy-20261017.json)を固定し、rootと別担当の設計確認を終えました。元のregional1024・元group/fold・対象10所見・20%合成は維持し、C候補0.001／0.01／0.1／1を学習側の4分割だけで比較します。勝者1モデルの予測を先に封印し、新しいF006 fold2固定E12対照の完成後に全888検査を一度だけ評価する計画です。旧C1両foldの保留は維持します。現在は隔離実装・人工CPU確認までを進行し、新実データfit・fold2主評価は未実施です。

F009は同じrunで8/12 epoch・1,696更新・AMP skip0へ到達しました。[E8観測](artifacts/research/20261017-f009-gamma-preparation-v1/epoch8-observation/observation.json)。途中指標で条件を変えず固定E12まで継続します。P038は可視資格済みの候補として保持し、既提出5件の新スコアと追加枠は引き続き追跡します。以下は各時点の履歴です。

**2026-10-17 14:46 JST：2×2特徴の全件監査と両fold評価を終え、この固定20%レシピも採用保留にしました。自己最高0.957、目標0.960は未達です。** [今回の正本](experiments/orthodiff-regional-twofold-HOLD-F006fold2-CPU-ready-20261017.json)。kernel139160976/V1は公式COMPLETE。成功3,063件すべてで元の入力・mid_2・global256計算との一致、元失敗1,144行のbyte一致、追加3,136,512個のFP32値とSHAを確認しました。元の検査・group・fold・coverageを変えていません。

以下は弱ラベルの同じ検証検査・固定10所見平均AUCの差です。Publicの上昇幅ではありません。

| 特徴・固定20%レシピ | fold0の差 | fold1の差 | 判断 |
| --- | ---: | ---: | --- |
| 元のglobal256 | +0.000431 | −0.004510 | 保留 |
| Fluid_Sensitive追加 | −0.000612 | 未実行 | 事前規則で保留 |
| 固定2×2／1024次元 | +0.002622 | −0.002341 | 両fold条件未達で保留 |

2×2の両runは12分類器が正常収束し、独立再計算で予測・全rank合成セル・各AUC・1000回group CIが一致しました。両CIは0を跨ぎ、統計的な改善・悪化の確定とは扱いません。fold0だけの採用、係数や所見の選び直し、条件不成立の2head合成は行いません。[fold0監査](artifacts/research/20261017-orthodiff-regional-head0-actual-independent-v1/verification.json)、[fold1監査](artifacts/research/20261017-orthodiff-regional-head1-actual-independent-v1/verification.json)。

F006を元fold2で学習できる別namespaceのモデル・trainer・CLIも準備しました。旧モデルのbytesと数値処理を保ち、fold／schema／source結合の境界だけを変更。CPU人工5件・Ruff・独立2分担のコード監査がPASSです。[実装証拠](artifacts/research/20261017-f006-fold2-implementation-v1/verification.json)。旧F006／F009と共有6moduleは不変。新resource gate・GPU確認・実学習はまだで、現在のF009固定E12評価後に実行順を判断します。学習側だけのinnerCVで正則化を選び、新fold2対照で評価する別案も設計中です。既存の負のレシピは変更しません。

F009は同じPID22392で稼働し、E7／1,484更新／skip0。P038は可視資格済みの提出候補として保持。正式5件は14:44の公式照会でも状態・新Public未返却、本日の追加枠0扱いです。14:34の公開Code上位100件にも新0.959以上はなく、既知作者0.958の5件が最高でした。以下は各時点の履歴です。

**2026-10-17 14:30 JST：公開作者0.958の別レシピP038は可視推論と2系統の数値監査を通過しました。自己最高0.957、目標0.960は未達です。** [正本](experiments/p038-visible-qualified-F009-E7-regional-running-20261017.json)。kernel139163240/V1は公式COMPLETE。可視3検査のRaptor384／224／R6はfallback0、Reader3モデル参加を確認し、中間36値FP32・最終36値FP64が別実装2系統で一致しました。全43小出力99,719Bを照合済み、最終CSVは830B。作者8セルの計算は約469秒、事前確認を含む終端ログは約490秒です。

P038は提出候補として準備できましたが、本日5件を提出済みで追加枠0扱いのため正式Submitはまだ0です。既提出5件の結果と枠を再照会して提出順を決めます。作者0.958を自己値とは呼びません。可視3件の最終CSVはP037と同一ですが、hidden全体やスコアの同一性を示しません。Readerの内部slot読込失敗数も独立には取得できていません。

F009は同じrunでE7／1,484更新／AMP skip0を保存し、固定E12へ継続。[E7観測の正本](artifacts/research/20261017-f009-gamma-preparation-v1/epoch7-observation/observation-v2.json)。2×2分割特徴のrunも継続し、14:31の最終確認は3,232件／成功2,362／元失敗継承870。[継続観測](artifacts/research/20261017-P038-visible-E7-milestone-record-v1/continuing-jobs-observation-1432.json)。元global256固定20%とFD固定20%は保留です。既モデルの追加fitなし診断では学習・検証AUCに差があり、表現の弱さと汎化の問題を分けて検討します。[診断](artifacts/research/20261017-orthodiff-fixed-head-failure-diagnosis-v1/report-with-train-gap.txt)。

公式大会情報の14:28確認では968位／6,120チーム、締切は2026-10-23 08:59 JST、1日最大5提出です。F006 fold2用の将来準備も進めますが、現trainerはfold0／1限定のため、別実装・別gateの資格が必要です。既存F009のsourceや学習条件は変更していません。以下は各時点の履歴です。

**2026-10-17 14:19 JST：追加画像1,080件の実監査は成功しましたが、固定20%合成のhead0は採用条件を満たしませんでした。自己最高0.957、目標0.960は未達です。** [結果と実行記録](experiments/orthodiff-fluid-complete-head0-hold-P038-running-20261017.json)。Fluid_Sensitive runは公式COMPLETE・1,055成功／25失敗。失敗は斜位・方向条件22件とmetadata候補なし3件です。元成功3,063件のFP32値を維持したまま、全4,207件中4,118成功／89なしの特徴を封印・再読込検証しました。画像・モデル本体のローカル取得はありません。

元fold0の学習3,386件中3,310件、検証821件中808件に特徴があり、12分類器とも正常収束しました。固定10所見平均AUCは0.8831284→0.8825163、差−0.00061206、95% group bootstrap区間は−0.004581〜+0.003117。[独立数値監査](artifacts/research/20261017-orthodiff-fluid-head0-actual-independent-v1/verification.json)でtrain-only標準化・9,696予測・baseline／合成各9,852セル・各AUCとCIが一致。統計的悪化とは断定せず、事前の負delta規則に従ってこの固定レシピは保留し、fold1へ進めません。共通成功594件の副次改善で全821件の主評価を置き換えません。

公開作者0.958のP038はCPU実確認・独立監査を通過し、[可視GPU推論](https://www.kaggle.com/code/rsraki/p038-helios-v6-fixed-20261017-v3)をkernel139163240/V1で開始。全保存セル・Input7・private／offline／T4／Dockerを確認しました。作者V6の全体レシピで、自己0.958の再現やReader単独効果は未確認です。同Dockerでも一部ライブラリ版が作者と違う点を記録しています。2×2分割特徴のkernel139160976/V1とF009固定E12への学習も継続。正式5件は14:18の公式照会でも状態・新Public未返却、本日追加枠0扱いです。以下は各時点の履歴です。

**2026-10-17 14:08 JST：2×2分割特徴の実GPU抽出を開始しました。自己最高0.957、目標0.960は未達です。** [実行記録](experiments/orthodiff-regional-GPU-running-FD704-20261017.json)。新runはkernel139160976/V1、非公開・offline・T4・Docker2757。保存後の全セル・Input4を照合済みです。元の3,063成功で入力・系列・連続16枚・mid_2・global256の全件一致を必須にし、1,144失敗は画像を再読込せず元の行を継承します。追加するのは固定TL/TR/BL/BRの平均を連結した1,024次元特徴だけです。最初の160件は122成功／38失敗継承で進行中。これは全件資格や精度改善の証拠ではありません。

実RTX4090の人工pool試験は、独立計算・CPU・CUDAの2,048個のFP32値が完全一致し、入力と乱数状態も保持しました。[試験記録](artifacts/research/20261017-orthodiff-regional2x2-actual-artificial-GPU-v1/verification.json)。別案のFluid_Sensitive追加推論も同じkernel139158713/V1で継続し、704件中692成功／12失敗。元3,063成功は上書きしません。両案それぞれに事前方針と検証器を用意し、全件の実出力監査後に新head0、改善時だけ同条件のhead1へ進みます。元global256／固定20%群はfold1低下により保留のままです。

新たに公開作者の0.958をV6／source359117069／提出57280318と結合しました。最新未結合版と分け、P038として全体レシピの再現を準備中です。自己再現値ではありません。正式5提出は14:03の公式照会でも状態・新Public未返却、本日の追加枠0扱い。F009は同じrunのE6／1,272更新／skip0から固定E12へ継続しています。以下は各時点の履歴です。

**2026-10-17 13:53 JST：OrthoDiffusionの元の20%合成は別foldで低下し、採用を保留しました。自己最高0.957、目標0.960は未達です。** [今回の記録](experiments/orthodiff-head1-hold-fluid1080-running-F009-E6-20261017.json)。元の全件run・kernel139150600/V1は公式COMPLETE、特徴3,063成功／1,144なし。全journal・UID順・784,128個のFP32特徴・先行32件との一致、元4,102 group／5 fold／Gold UID・group重複0を確認済みです。元成功特徴を維持し、名前不明1,080件だけの追加推論を別runで開始しました。MRI本体はローカルへ取得していません。

先行する[固定32件試験](experiments/orthodiff-MRI32-S24-F8-20261017.json)はkernel139148870/V1で公式COMPLETE・独立監査PASS。PD-FS18件／非FS PD6件の24成功、残り8件はPD表記不明として除外し、検査・系列の補充やゼロ特徴の捏造はありません。ラベル・Report・学習0で、MRIはKaggle内のみで処理しました。

重み1回の厳密復元、24回の特徴推論、連続16枚×24の384 SOP、6144個のFP32特徴値とSHA、乱数復元を確認。全体64.18秒、前処理中央値0.816秒／特徴抽出中央値0.183秒、CUDA予約最大約1.22 GiBでした。[独立監査](artifacts/research/20261017-orthodiff-mri-pilot-independent-audit-v1/verification.json)。画像本体・raw中間特徴を出力していないため、画像変換やpool平均の独立再計算は未実施です。24件は全てMONO2・面内transpose／flipなしで、他枝の実MRI資格や全件対応率・分類精度はまだ主張しません。

同32件の撮影header68系列もCPUで確認済みです。別表記DPと保守的TR／TE案は未知1件にだけ該当し、残る7件を埋めるための閾値変更は行いません。[照合結果](artifacts/research/20261017-orthodiff-header32-rule-audit-v1/report.md)。全件baselineのPD規則は保持しました。運搬はJSONL追記・小status・32件ごとのstdout・終端全体SHAで、8時間は検査間の協調上限でありデコーダの強制中断ではありません。

[journal検証CLI](scripts/validate_orthodiffusion_fullweak.py)が実出力の資格を発行した後、[分類器CLI](scripts/fit_orthodiffusion_weak_head.py)を実行しました。元fold0の学習3,386件のうち成功2,469件だけで標準化と12分類器を学習。12所見とも収束し警告0、検証821件のうち特徴成功594件・なし227件です。欠損を0に変えず、検証側を標準化の学習に使っていません。

独立監査でtrain-only標準化、実予測7,128値、baselineと合成各9,852セル、各AUCとbootstrap1000回を再現しました。[fold0監査](artifacts/research/20261017-orthodiff-head-fold0-actual-independent-v1/review.json)。[NumPy推論adapter](src/rsna_knee/orthodiff_numpy_head.py)によるfold0実モデルの7,128値も、既存NumPy 1.22.3／2.5.3でFP32一致。Kaggle実環境の資格は今後の確認です。

[事前head方針v2](artifacts/research/20261017-orthodiff-fullweak-preparation-v1/weak-head-prospective-policy-v2.json)の固定20%合成では、fold0の固定10所見平均AUCは0.8831284→0.8835598、差+0.00043144。続く元fold1・891件は0.8829163→0.8784068、差−0.00450951でした。fold1の95%区間も−0.008226〜−0.001295で、実予測7644値・baseline／合成各10692セル・各AUC／CIが独立再計算で一致。[fold1監査](artifacts/research/20261017-orthodiff-head-fold1-actual-independent-v1/review.json)。新指標より前に固定した両fold差>0の条件を満たさないため、元のexplicit-PD／global256／20%群は候補化を保留。係数・対象所見は選び直していません。これはPublic親の再順位化とは別の弱ラベル診断です。

次の別runは[Fluid_Sensitive追加推論](https://www.kaggle.com/code/rsraki/orthodiff-sag-fluid1080-20261017-v3)、kernel139158713/V1。固定metadata subset1080件のうち候補系列1077／候補なし3で、元成功3063件との交差0。非公開・offline・T4・同Docker・Input3と全2セルをreadback確認し、13:52の公式状態RUNNING／stdout64成功・失敗0まで確認しました。原readerからPD名の分類・check2文だけを別routeで外し、幾何・中央連続16・画素処理・重み・特徴計算・乱数を維持。T2等を含む可能性があり、PD確認済とは呼びません。全件資格・新head評価・採用はまだです。

局所病変をglobal平均が薄める仮説に対し、固定2×2分割の1024次元特徴も別の一要因としてoffline準備中です。元3063成功で同じ入力・mid_2・global値との一致を必須にする設計で、追加availability案とは混ぜません。人工CPU試験までで、実GPU・新headは未実行です。

F009は同じrunでE6／1,272更新／skip0を13:51に保存し、固定E12へ継続。[E6の観測記録](artifacts/research/20261017-f009-gamma-preparation-v1/epoch6-observation/observation.json)。13:16の公式自己提出一覧は全35件を返し、旧P029M／P032のCOMPLETE・0.957と、新正式5件の状態／スコア未返却を確認。13:50の各Submit再照会でも新スコアは未返却でした。本日の追加枠0扱い、公開Code上位100件の12:38確認でも新0.959以上はありません。以下は各時点の履歴です。

**2026-10-17 12:36 JST：OrthoDiffusionの人工GPU特徴抽出試験が成功し、独立監査を通過しました。自己最高0.957・目標0.960未達は変わりません。** [現在の正本](experiments/orthodiff-artificial-GPU-pass-F009-E3-five-pending-20261017.json)。kernel139147175/V1で固定資産9件を再検査し、`weights_only`によるEMA全257 key・34524545 parameters・13 schedule bufferの厳密一致を確認。T4の選択GPUで人工入力1×1×16×256×256から`t=100 / mid_2`の256次元特徴を抽出し、同seed反復一致・別seed変化・CPUと可視2GPUの乱数状態復元を確認しました。

3回のdenoiser呼出に対しdecoder／最終出力層の呼出は0。CUDA予約peak約1.22 GiB、RSS peak約1.46 GiB、人工試験全体34.41秒で事前資源条件を通過しました。[実receiptの独立監査](artifacts/research/20261017-orthodiff-artificial-independent-audit-v1/verification.json)。768個のpool FP32値と記録SHAは独立照合しましたが、raw mid_2を出力していないため平均pool自体の別計算は0、実処理のsource結合による確認です。これは人工特徴・資源の資格で、MRIデータ、分類精度、Public改善、hidden処理量の資格ではありません。

独自F009はE3まで完了し、累計636更新／AMP skip0で同じRTX4090 runを継続中。[E3観測](artifacts/research/20261017-f009-gamma-preparation-v1/epoch3-observation/observation.json)。固定E12・評価方針v2を維持し、途中epochの採用は行いません。P033／P034／P036／P035／P037の正式5提出は12:29 JSTの公式照会でも状態・Public未返却。本日5／累計35、追加枠0扱い（rawの`numAllowedNow`省略とSDK既定0を区別）で再提出しません。

次のOrthoDiffusion 32検査pilotは事前仕様とoffline実装の準備中で、実MRIデコードはまだ0です。連続16slice・向き・系列選択・失敗規則を固定し、既存sampled16 cacheと同一視しません。以下は各時点の履歴です。


**2026-10-17 12:23 JST：P037を正式提出57293138として受理しました。自己最高0.957を保持し、目標0.960は未達です。** [現在の正本](experiments/p037-five-formals-orthodiff-assets-F009-E2-20261017.json)。12:18 JSTの公式照会では新5件の状態・Publicは全件未返却。本日5件／累計35件、追加提出は0枠扱いです。`numAllowedNow`はrawで省略され、SDK既定0であることを区別します。既存提出を再送せず、次の枠回復時刻は未確認です。

| 候補 | 比較の範囲 | 受理済み提出 |
| --- | --- | --- |
| P033 | P029Mの公開済み所見別係数vectorだけを変更 | 57290868 |
| P034 | P032のnative部分失敗時に成功セルを残す | 57291440 |
| P036 | P033のRaptor224に解剖学的鏡映だけを追加 | 57291580 |
| P035 | P034の数値契約を保持しUIDごとの所有プロセスと期限付き回収へ変更 | 57292346 |
| P037 | 作者Goodpjw V26のRaptor／R6／R7／Readerを含む固定全体レシピ | 57293138 |

P037は可視3件で各armの成功ログ、R6／R7失敗0、21資産・6人工fixtureのSHA、全15科学セル・6保存時点を確認。中間36値のFP32と最終36値のFP64を別実装でも完全一致と検証し、Reader段の順位vector変化はEffusionの1所見でした。[独立再検算](artifacts/research/20261017-p037-visible-second-independent-v1/audit.json)。作者0.958は自己値ではなく、この全体比較からR7／Reader単独の改善やhidden完走を断定しません。

OrthoDiffusionのCPU資産stage（kernel139145710/V1）も公式COMPLETE・独立監査PASSです。公開genericのsagittal重み1本と固定source8の計9件を検証し、source8の実本文SHA／Git blobも独立照合しました。[資産監査](artifacts/research/20261017-orthodiff-sagittal-stage-independent-audit-v1/verification-v2.json)。重み552919566B／SHAはKaggleの固定stream guardとreceiptに基づき、ローカル重み取得は0、SDK listingの独立サイズ欄はありません。これは資産準備の資格で、checkpoint復元・科学依存closure・人工GPU資源・連続16sliceの新pilot・精度は未資格です。

独自F009は12:22:40 JSTのroot監視で同じRTX4090 runが継続中、最終確認は2/12 epoch・累計424更新／AMP skip0です。train-onlyのstudy共通gammaだけを変え、[事前方針v2](experiments/f009-evaluation-policy-v2-prescore-20261017.json)の固定E12同士で評価します。BCEは副評価で、途中epoch・Gold・Publicによるcheckpoint選択は行いません。以下は各時点の履歴です。


**2026-10-17 12:03 JST：P037修正版のCPU事前試験を通過し、固定公開レシピのGPU推論を開始しました。** [現在の正本](experiments/p037-CPU-pass-GPU-running-F009-E2-20261017.json)。CPU kernel139143030/V1は21モデル資産・6人工fixtureのSHA／サイズ、4圧縮形式すべての形状・型・画素値一致を確認。Reader提供のcp312対応3 wheelで成功し、独立監査も通過しました。作者環境Docker37c64／Python3.12.13を実確認済みです。旧CPU v1の未合格記録は保持しています。

GPU版kernel139143442/V1は公式RUNNING。全コード・Input7・非公開T4・Internet OFF・CPUと同じDockerをreadback照合しました。Markdownのlist／string表現の違いは内容SHA一致で確認し、保存し直していません。作者Goodpjw V26の15科学セルはv1と同じで、元版からはR6 stage探索の1行だけを変更。Raptor／R6／R7／Readerを含む全体レシピの再現で、各要因単独の効果とは扱いません。各armの実参加・最終CSVの独立再計算を終えてから正式提出を判断します。

自己最高0.957・目標0.960未達を維持し、正式提出済み4件は11:57の照会でも新スコア・状態が未返却。本日4件／追加可能1件です。独自F009は2/12 epoch・累計424更新／skip0を確認し、同じrunを継続中。追加の限定Web一次調査でも、0.959以上と再現recipeを結合できる新候補は確認できませんでした。[調査範囲](artifacts/research/20261017-discussion-github-limited-v1/report.md)。以下は各時点の履歴です。

**2026-10-17 11:50 JST：P035を正式提出57292346として受理しました。自己最高0.957、目標0.960は未達です。** [今回の正本](experiments/p035-formal-Submit57292346-20261017.json)。P033＝57290868、P034＝57291440、P036＝57291580、P035＝57292346の4件を採点追跡中で、11:48の公式照会では全件スコア・状態が未返却。本日4件／累計34件／追加可能1件です。既存3件は再提出していません。

P035は可視3件・診断192件とも構造失敗0／運用失敗0で完走。可視の親・最終36値、4モデルの144値、192件のnative2304値を独立検証し、旧方式と同じ配列・予測を確認しました。全子プロセスの回収・一時ファイル削除も完了。[監査の集約](artifacts/research/20261017-p035-diagnostic-independent-audit-v1/summary.json)。一方、192件は464.0→869.2秒と遅く、UID最大21.53秒に対し期限180秒。今回の正常時の計算保持と終了の資格であり、hidden全体の時間・追加モデルの実寄与・精度改善はまだ確認できません。

F009は同じ4090学習runを継続中。11:56にE2完了・累計424更新／skip0を確認し、採用判断は事前方針v2の固定E12比較を待ちます。[E2観測](artifacts/research/20261017-f009-gamma-preparation-v1/epoch2-observation/observation.json)。公開作者0.958の別レシピGoodpjw V26は、R7／Readerを含む全体再現候補P037として準備しました。11:51のCPU事前確認は、公開Readerに試験用fixture自体が含まれず、科学推論前に停止。作者Docker37c64の保存は確認でき、既存の人工DICOM4形式を追加Inputで検証する修正版を準備しています。[停止の正本](experiments/p037-CPU-preflight-fixture-stop-20261017.json)、[再現調査](artifacts/research/20261017-goodpjw-v26-reproduction-audit-v1/report.md)。R7／Reader単独の効果や自己0.958再現を主張するものではありません。以下は各時点の履歴です。

**2026-10-17 11:31 JST：P035の可視推論・192検査診断を開始し、F009のE1完了を確認しました。** 自己最高0.957、目標0.960未達、新3件の正式採点は未返却です。[今回の正本](experiments/p035-running-F009-epoch1-20261017.json)。P035はP034のモデル・前処理・固定20%を保ち、UIDごとの所有CPUプロセスと期限付き回収だけを変更。Linux人工14件、実pack結合6ケース、証拠改変21負例を検証後、非公開V1のkernel139138353／139138405を起動しました。保存後の全コード・Input9・T4・Docker・Internet OFFも照合済みです。実MRIでの180秒期限・GPU予測・hidden完走はこれから確認し、正式Submitはまだ0です。

F009のE1は1721秒、optimizer更新212／skip0、gamma抽選3386を記録。同じ学習runを継続し、途中のweak指標では採用せず、事前方針v2の固定E12比較を待ちます。11:26の公開Code上位100件は引き続き作者最高0.958で、新たな0.959以上は確認できませんでした。以下は各時点の履歴です。

**2026-10-17 11:22 JST：3候補の正式提出を受理し、独自F009の学習を継続しています。自己最高0.957、目標0.960は未達です。** [現在の正本](experiments/continuation-three-submitted-F009-running-20261017.json)。新3件の自己スコア・状態は公式未返却で、重複提出はしていません。公式残枠は本日3件・累計33件・追加可能2件です。

| 候補 | 一要因の比較 | 正式提出 | 可視検証 |
| --- | --- | --- | --- |
| P033 | 作者0.958の固定所見別係数をP029Mへ適用 | 57290868 | 3件・失敗0・36値独立一致 |
| P034 | P032の部分失敗時に成功nativeセルを残す | 57291440 | 3件・失敗0・native30セル・36値独立一致 |
| P036 | P033の224モデルへ解剖学的鏡映だけを追加 | 57291580 | 3件・失敗0・36値独立一致 |

F009は同じfold0・seed20261002・12epochでtrain-only gammaだけを追加して学習中です。後続foldの判断は[評価方針v2](experiments/f009-evaluation-policy-v2-prescore-20261017.json)を使用します。新指標が出る前の11:18:13 JSTにhistory0・epochファイルなしを記録し、固定E12・固定10所見AUC改善・工程条件を維持したまま、校正に敏感なBCEを副評価へ変更。旧方針・コード・設定・gate・起動記録は保持しています。

子プロセス異常への新executorは、Kaggle CPUで**Linux人工14試験すべて成功・skip0**を確認しました。66MBの配列転送、UID対応、hard exit／timeout、SIGKILL、同process group・全終了／cleanupを検証。[CPU資格](experiments/isolated-uid-Kaggle-CPU14PASS-20261017.json)。実MRI・GPU推論・180秒期限の妥当性や精度の資格ではありません。P035のopt-in統合を準備中です。以下は各時点の履歴です。

**2026-10-17 11:11 JST：P034も正式提出し、提出57291440の受理を確認しました。** 公式COMPLETE、可視3件・失敗0・native10列／30セル有効。全36最終値・native36値・現在raw3armの親36値を独立FP32再計算で一致確認し、MCL／Synovitis保持、4モデル、source／Input、spawn、queue／process／temp終了も照合しました。[正式提出の正本](experiments/p034-formal-Submit57291440-20261017.json)。全件成功した今回の可視CSVはP032と同じSHAで、新しい部分失敗時の処理は人工例での検証です。P033（57290868）とP034の自己スコアは未返却、最高0.957・目標0.960未達を保持します。

P036（kernel139135844/V1）も公式RUNNINGです。公開別レシピから224モデルの解剖学的鏡映だけを追加し、他7セル・係数・R6／Input4を固定。CPU10件・独立600画素／平均12値・人工CUDAの軸／境界／forwardを確認後、保存内容を照合しました。[P036の記録](experiments/p036-224-mirror-visible-running-20261017.json)。可視完走と自己採点は未検証。F009の実学習は同じrunで継続中です。以下は各時点の履歴です。

**2026-10-17 11:03 JST：P033の採点を追跡しながら、P034の可視推論と独自モデルF009の学習を開始しました。** [実行の正本](experiments/p034-running-f009-training-started-20261017.json)。P034（kernel139135231/V1）は公式RUNNINGで、保存後のコード・Input9・環境も一致。P032から変えるのはnativeの部分失敗処理だけです。成功検査の`N_valid`で順位化し、失敗セルは親文字列を保持。部分列の成功数不足・FP32定数は混合せず、全件成功時は従来と一致します。限定62テスト、GPUの12498値、独立監査54876セルを通過。子プロセス全体の異常・timeoutで全親へ戻す処理は残っており、hiddenでのnative寄与や精度は未確認です。正式Submitはまだ0です。

F009は既存キャッシュとRTX4090でfold0の実学習中です。元F006とseed20261002・group/fold・12epoch・学習率を固定し、train-onlyのstudy共通gamma拡張0.85〜1.15だけを追加。CPU32件・GPU19＋6＋6チェック、新source-bound gate、当方の独立preflight（学習3386／検証821）を通過して起動しました。E12同士のweak比較を事前固定し、改善とBCE非悪化・正常完了を満たす場合にfold1へ進みます。Goldで学習・選択・評価せず、元DICOMの一括取得もありません。約30分の旧E1実測に対し、起動数分時点はhistory0ですがOSのCPU・I/O増加を確認しています。

最新の公式公開Code上位100件は作者0.958が4件、0.959以上は確認0。追加の同点版はコピー／private本体／複数変更を区別しました。[調査証拠](artifacts/research/20261017-public-frontier-v1/frontier-summary.json)。新基盤候補OrthoDiffusionは重み公開を確認しましたが、連続16sliceという入力契約が既存キャッシュと異なります。MRI-COREもcheckpointの充足監査が必要で、どちらも新重み取得・採用はしていません。[基盤調査](artifacts/research/20261017-independent-model-research-v1/static-audit.json)。自己最高は0.957、目標0.960は未達です。以下は各時点の履歴です。

**2026-10-17 10:49 JST：P032の正式結果は自己Public 0.957でした。目標0.960は未達で、P033を提出57290868として正式受理しました。** P032（57262934）は公式Get/ListでCOMPLETE・0.957・247204B。成功版P029M（57225898、0.957・247222B）を保持します。P033は公開作者0.958の固定所見別係数だけを適用した親対照で、可視3件・失敗0・36値の独立FP32照合済み。提出直前にも8コードセル・版V1・Input4・private／Internet OFF／T4を再確認しました。P033の自己スコアと状態はまだ公式未返却で、作者0.958とは区別します。[今回の正本](experiments/p032-result957-p033-submit57290868-20261017.json)。

P032の同点から「F006を混ぜても効果がない」とはまだ判断しません。可視ではnative有効時645→897Bになりましたが、本番は親単独とほぼ同じサイズで、全native無効化または親fallbackの疑いがあります。別runのサイズ差と丸めスコアだけでは確定できず、hiddenの失敗数・実効重み・CSVは取得できていません。成功検査を残す失敗処理の契約を検討し、最新公開上位の採点版も調査中です。ユーザーの最新指示に従い、0.960へ向けた実行と提出を継続します。以前の「原則1候補」は旧計画の仮定です。以下は各時点の履歴です。

**2026-10-16 18:11 JST：P032-v6を正式提出し、提出57262934の受理を公式確認しました。** 192件診断は全成功・構造失敗0・768モデルforward、最終2304値を別FP32計算で全bit一致。提出候補の可視3件も全成功で、対象10列にnative20%が実際に寄与し、親から数値26セル／順位6セルが変化しました。親・native平均・最終の全36値を独立再計算し、MCL／Synovitis保持と正常終了を確認。新Public・hidden完走はまだ公式未返却、自己最高0.957を保持しています。[提出と検証の正本](experiments/p032-v6-formal-Submit57262934-score-unreturned-20261016.json)。公式残枠は本日1件・累計30件・追加可能4件です。以下は各時点の履歴です。

採点待ちの間に、[作者Public 0.958の採点版](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters?scriptVersionId=358751289)を追加確認しました。R6やTTAを変えず、Synovitis／Lateral OA／PF OA／Baker’sのRaptor係数をまとめて変更した版です。この公開済み固定vectorだけを自己P029Mへ反映したP033（kernel139000655/V1）は可視3件・失敗0で完走し、全36値を独立FP32再計算で一致確認。R6の9ファイルSHA、埋込7モジュール、両GPU worker終了も確認しました。[P033の記録](experiments/p033-visible-pass-parent-weight-vector-unsubmitted-20261016.json)。作者0.958と自己スコアは区別し、P033は**未提出**です。

18:28 JSTの公式Get/Listでも、57262934の状態・Publicは未返却です。正式提出は本日1件・追加可能4件。次はこの既存提出の公式結果を確認して判断します。0.960以上ならその達成を記録し、未達なら新たな係数探索より、この資格確認済みP033の親対照を優先できます。hiddenでnativeが無効化された可能性はPublic平均だけでは判定できません。現時点で、新たな自己精度改善や0.960達成は主張しません。新学習・ローカルMRI／モデル取得は行っていません。

## 2026-10-16継続：spawn衝突を修正、Kaggle CPU起動確認に成功、GPU枠の解放待ち

自己最高は公式確認済みの**Public 0.957（提出57225898）**を保持しています。目標0.960は未達で、今回の正式Submitはまだ0です。[現在の実行チェックポイント](experiments/p032-v6-ready-CPU-startup-pass-GPU-capacity-wait-20261016.json)を正本とします。

新P032の子プロセスで、`trusted_imports`後の検索順により、親が保持する`serving/serving_router.py`と別の`common/serving_router.py`をspawn子が読み込む不具合を発見しました。子のタスク復元時の終了が結果待ちの長期化につながり得ます。正本ディレクトリを優先し、親のキャッシュに頼らず新規解決先とSHAを確認するv6へ修正。人工例の実spawnで修正前後を再現し、元実ソースのfresh spawnも確認、限定54テスト・ruffを通過しました。親の科学コード・4モデル・元89ソース・FP32集約と固定20%は保持しています。旧RUNNINGジョブの実トレースは未取得なので、停止原因そのものの確定とは区別します。

GPUなしの非公開`p032-cpu-startup-20261016-735352c148` V1／kernel138990364は公式COMPLETE。元bootstrap→4モデルbytesの検証・展開→メタデータbind→検索順の検証を**9.866秒**で完了し、Kaggleでも修正前の誤った解決先と修正後の正本を確認しました。モデルの復元・forward・GPU初期化・MRIデコードは0で、一時モデル・ソース・メタデータを削除し、Outputは小さなreceiptだけです。これは起動確認であり、F006予測や精度の検証ではありません。

旧診断`p032d-f006192-20261016-a0130053d5`と旧候補`p032-f00620-20261016-f217cd973c`は最終照会時RUNNING。後者は親の可視3件PASS・fallback0まで確認できましたが、追加nativeと最終混合は未確認です。観測版v5の追加Saveは`Maximum batch GPU session count of 2 reached.`で拒否され、kernel IDは発行されていません。取消APIに必要なsession IDは公式SDKから取得できず、公式MCPの取得も権限拒否のため、ユーザーへ**この旧2本だけのStop／Cancel**を依頼しました。成功済みP029Mには触れません。

v6の192件診断・提出候補、Input固定、実行payloadは準備済みです。枠が空いたら修正版で実推論を行い、失敗mask・native実効寄与・最終CSVを確認して正式提出と採点確認へ進みます。v6の新Saveや新Public改善はまだありません。過去の「原則1候補」は計画上の仮定で、今回のユーザーによる実行・提出の明示許可を制限するものではありません。

## 2026-10-16実行再開：自己0.957を公式確認、F00620%の診断と可視候補を起動

ユーザーの「提出して精度を上げる」という明示依頼で、調査のみから実行へ移りました。公式APIで提出57225898のCOMPLETE・Public **0.957**・出力247222B・エラー欄なしを確認。成功版を変更せず、新しいP032を準備しました。[再開の記録](experiments/p032-diagnostic-and-serving-running-20261016.json)。

新統合は親を再順位化せず`0.8*s+0.2*rank_N(native)`をFP32で計算し、MCL／Synovitisを保持。列単位fallbackとnative成功数・実効重み・親からの差・全無効を記録します。48限定テストと人工GPU数値照合を通過し、実F006集約headの人工forwardも成功しました。実encoder・checkpointのローカル読込ではありません。

192件の新構造層別MRI診断は非公開`p032d-f006192-20261016-a0130053d5` V1／kernel138983547で実行中。中断後に既存RUNNINGを読み取りで確認し、重複起動していません。別の新非公開`p032-f00620-20261016-f217cd973c` V1／kernel138987949で、元P029M→独立native→最終CSVの可視3件を保存実行しました。どちらもInternet OFF・T4×2・既存Input9。完走・新native寄与・hidden精度はまだ未確認、正式Submitは0です。ローカルモデル／MRI取得・実学習は0。以下は各時点の履歴です。

## 2026-10-16追記：最後に近い提出へ向けて一候補へ絞る計画

ユーザーの「これが最後くらいの提出になる」という意向を受け、[総合計画](docs/research/20261016-final-submission-plan-v1/report.md)と[判断の前提](docs/research/20261016-final-submission-plan-v1/decision.json)を新しい文書として保存しました。既存調査は保持。自己0.957（Submit57225898・ユーザー確認）を保護し、追加正式提出は原則1候補という計画上の仮定へ絞ります。実際の残枠や厳密なユーザー指定上限は未確認です。

台帳全280行を走査し、0.924→0.937→0.950→0.951→0.954→0.956→0.957の進展、旧窓数・mirror・別枝追加の同点、FOV変更・Reader混合の低下、F006〜F008計96epoch、旧hidden失敗を総合。F006固定20%は準備の第一対象ですが、採用候補は未確定。部分失敗を含むrank契約・入力条件・運用と資源の根拠が整えば一候補へ固定し、整わなければ新規Submitを省いて0.957を保持します。P030M→R7→F006の連続正式採点や追加学習は今回の標準計画から外します。

今回は文書整理とローカル記録の読取のみ。実MRI／モデル取得・読込、学習、Kaggle保存実行・提出、ライブAPI/Web再照会は行っていません。以下は前回までの方針・観測の履歴です。

## 2026-10-16：自己0.957をユーザー確認、0.960へ向けた調査を更新

**ユーザーが提出57225898（P029M）の自己Public 0.957をKaggleで確認済みと明示しました。現在の基準は0.957、次の目標は0.960以上で未達です。** 今回の当方による自己公式API再照会ではなく、[ユーザー確認記録](experiments/user-confirmed-public957-research-20261016.json)として保存しました。10月15日19:33時点の採点未返却・旧自己0.956は当時の証拠として保持します。題名0.9576を未丸め自己値とは扱いません。

[調査レポートと比較計画](docs/research/20261016-goal096-review-v1/report.md)では、P030MのR6 TTA-only、R7 OFFのbundle-onlyを少数比較し、並行してF006のhidden失敗と順位統合契約を診断する方針を推奨しています。F006はCLS両fold E12＋既存REG両fold・固定20%の追加価値を採用した親へ直接測定。添付のfallback尺度混在に加え、部分失敗時のrank母集団、親再順位化による式の変更、2fold平均をOOFと誤認する危険を確認しました。新学習は必要性を見て強度拡張一種類から比較します。

今回は調査と文書更新のみで、実MRI／モデルの取得・読込、学習、Kaggle保存・提出は0。匿名公式metadataで作者V1/script358332966の表示0.957を再確認しましたが、最新Code上位100件の試行は401のため全体を再取得していません。0.958〜0.960以上の再現可能一式は今回確認できず、不存在とは断定しません。以下の各更新はそれぞれの観測時点の履歴です。

## 2026-10-15 19:21 JSTの採点確認後：P029M提出済・P030Mの可視検証も完了

19:33:56 JSTの追加Getも同じ377バイト・同じSHAで、提出57225898の状態／エラー／Publicは未返却です。[最終チェックポイント](experiments/public957-final-checkpoint-20261015-1934.json)。19:21の自己best照合と提出枠、下記の可視資格は維持し、新しい採点成功は主張しません。

**自己最良の確認値は0.956、目標0.960超は未達です。** 作者公式表示0.957の固定R6 recipeを使った[P029M V1](https://www.kaggle.com/code/rsraki/p029m-r6-957-20261015-a91d20e5dd62)は通常sessionでCOMPLETEとなり、19:01:51 JSTに正式提出`57225898`を1回受理されました。19:21の公式Get/Listも状態・エラー・Publicフィールドを返していません。SDK既定値を公式状態へ置き換えず、自己0.957やhidden成功は未確認とします。[最新正本](experiments/p030m-visible-pass-P029M-submit-unreturned-20261015.json)。本日1件・残4・累計28は最新の公式値です。

可視3検査について、元test.csvとsampleのID／12列順／有限値／値域を確認。raw384・raw224・R6の各成功集合の平均同順位rank/Nから、元recipeの式で全36個を独立再計算し、最終CSVのfloat32ビットが一致しました。別agentも36/36一致を確認。実stdoutで全9memberのサイズ/SHA guard成功、R6の4fold・TTA ON・T4×2、各armの検査単位failure 0、worker終了0を確認しました。453.7秒は可視recipe全体の所要時間で、固定setup時間ではありません。隠しデータの精度や9時間内完走を保証しません。

提出直前の新しいGetではDockerが`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`と返り、要求／作者実runとdigestは一致しました。registry名は異なり、以前のGetでキーが欠落していたことも保持します。実ログはPython3.13／torch2.11.0+cu128です。

[P030M V1](https://www.kaggle.com/code/rsraki/p030m-r6-off-20261015-c7e91ba5032f)（kernel138832933）も同じ固定stage-b2と全guardを使い、科学的変更をTTA True→Falseだけに限定して可視COMPLETE。Rootと独立agentが3UID／正規12列／全36値のfloat32再計算一致・全arm failure 0・TTA OFF経路を確認しました。可視全体453.7→405.2秒、R6 GPU平均20.1→9.0秒／検査。ただし未変更のRaptor時間やdecode時間も動いた1回比較で、壁時計差すべてをTTAへ帰属しません。R6確率は31/36値が変わりましたが3件内の各クラス順位は同じで、最終CSVはバイト同一です。hidden予測・スコアの同一性を意味しません。正式提出は0回、P029Mのhidden結果を確認してから判断します。R6 OFF→R7 OFFは次のbundle比較として分け、係数探索をしません。

19:04 JSTの匿名公式Code先頭100件は18:24時点からID・スコア・版が不変、最高表示0.957、0.960以上0件でした。R7 V2のlinked提出も依然スコア欠落で、V1 best0.957をV2へ転記しません。公式Data説明の索引はhiddenが約1300検査と説明しています。可視3件と並行処理の平均から本番時間を単純比例で断定せず、8.3hの協調停止も全体9hのhard timeoutとは扱いません。[更新調査](experiments/public-frontier-hidden-runtime-refresh-20261015-1910.json)。ローカル実モデル取得・MRIデコード・追加学習は0。限定テストは引き続き53件中52成功・1skip、広域299件の未解決は保持します。以下は各時点の履歴です。

## 2026-10-15 18:11 JST：R6 V1固定入力の実作成完了・P029M推論を開始

固定入力用b2は非公開・CPU・Internet ON・InputなしのV1でCOMPLETE。Kaggle上だけで取得したR6 V1の9ファイル（計886,813,460バイト）について、全サイズ・全SHA一致を出力した検証JSONを照合しました。ローカルへ取得したのは2213バイトのprovenanceとログだけで、重み本文のローカル再ハッシュではありません。[追跡記録](experiments/p029m-frozen-input-private-run-20261015.json)。このstageはV1のまま更新しません。

その前のb1は `/kaggle/temp` の親が存在せず取得前に停止しました。失敗したV1を保持し、親の境界を検査して明示作成する修正を新しいb2へ適用しています。[失敗記録](experiments/p029-stage-b1-temp-parent-failure-20261015.json)。再利用できる[オフライン準備CLI](scripts/prepare_p029_stage_fallback.py)も追加しました。新runとstage slugを必須にし、出力上書きを拒否、元ソース・メタデータのSHAを束縛し、古い作者Input metadataを除去します。自動upload/Submitコードは追加していません。

18:11:16 JSTに[P029Mの非公開V1](https://www.kaggle.com/code/rsraki/p029m-r6-957-20261015-a91d20e5dd62)（kernel138824627）を1回保存・実行しました。元のR6/4fold/TTA ON・全数値処理・埋込7モジュールを保持し、固定stageの探索候補1行と、推論前の全9ファイルSHA検査だけを追加しています。Ortho/SWA/competitionと固定stage V1を接続し、旧R6 Datasetは接続しません。Internet OFF/T4/作者実runのDockerを要求。実mount・forward・CSV・hidden・自己Publicはまだ未確認で、正式Submitは0です。

P030Mも同じ固定入力とguardを保持しTTA booleanだけをOFFにした未実行対照として準備し、独立静的レビューを通過しました。自己P029Mの再現結果を確認してから使用します。最終の限定unittestは53件・52成功・1スキップ。広域299件は失敗9/エラー50/skip66で未解決、baseline比較と完全ログもなく全体PASSは主張しません。[検証範囲](docs/validation.md)。以下は各時点の履歴です。

## 2026-10-15 17:47 JST観測：P029は最初のmanifest検査で停止、Kaggle側V1 stagingを準備

**自己最良Publicは0.956を維持し、目標0.960超は未達です。P029 V2は最初のメタデータ検査だけでERRORとなり、元の科学コード・MRI処理・モデル読込へ進んでいません。** 正式Submitは0です。[停止の正本](experiments/p029-first-guard-failure-20261015.json)はkernel `138818102`／V2／17:47:40 JSTの観測です。期待するR6 V1 manifestが一意に見つからず、実際のbundleや版は観測できていません。R7が接続された、重みが壊れている、モデル精度が低いとは判断しません。

ログを文字列検索した旧pollの `R6_V1_guard_pass_line_present=true` は、tracebackが未実行のprintソースを表示したことによる誤検出でした。実stdoutの成功eventは0です。guardの成功、可視推論、hidden完走、自己0.957の証拠として使いません。未実行のV1、失敗したV2、元P029準備runは保持します。

[認証後の独立監査](experiments/authenticated-progress-refresh-20261015.json)では、17:21 JSTの公式Get/Listが自己`rsraki`のV22R `57164195`＝COMPLETE・0.956・報告エラーなしで一致しました。旧P028/P024/F006両foldはCOMPLETEに加えてhidden再実行エラー・Public欠落・APIの`totalBytes=0`があり、採点待ちではありません。詳細原因はこの応答だけでは分かりません。提出可能数5・全27件もrawで確認し、旧3件を再送しません。

同時刻のGPU quota rawでは108000秒中11361.972秒使用・予約0、残96638.028秒（26時間50分38.028秒）、更新予定は10月17日09:00 JSTです。typedでは108000秒が21600秒へ変化し、使用量文字列も破損しているため予算計算に使いません。残量は17:21時点の値であり、その後の実行や9時間以内完走を保証する値ではありません。

次は別のCPU Notebookで、**公開R6 V1だけをKaggle上で取得し、manifest SHA6508…と全member SHAを検査して出力へstageする案を準備・査読中**です。ローカルPCへ重みやDICOMを取得せず、学習もしません。stageはInternet ONの取得専用、提出推論はInternet OFFを維持し、明示したNotebook出力mountを読む最小の接続変更を別候補として検証します。stageの保存・実行・完了、実mount、GPU推論、正式提出はそれぞれ別の資格です。以下のRUNNINGや未認証という記述は各時点の履歴として保持します。

## 2026-10-15 17:32 JST：ログイン完了・旧3提出のhiddenエラー確定・P029実行中

公式CLIのブラウザ認証が成功し、`rsraki` を確認しました。17:21 JSTの認証済み公式Get/Listでは、自己V22R `57164195` はCOMPLETE・Public 0.956・エラーなし。P028 `57163437`、P024 `57166365`、F006両fold `57169291` は**COMPLETEだがhidden再実行のunhandled errorあり・Publicなし・出力0バイト**でした。待機ではなく失敗として扱い、再送しません。可視成功をhidden資格へ転用できないことが実際にも確認されました。[取得証拠](artifacts/research/20261015-authenticated-submission-status-v1/receipts.json)。現在枠は公式 `numAllowedNow=5`、累計27。GPUは30時間中約3.156時間使用・予約0でした。

P029のコードのみを新しい[非公開Notebook](https://www.kaggle.com/code/rsraki/p029-r6-957-20261015-42fbc0e9ae14)へ保存しました。V1はQUICK_SAVEで実行なし。作者出力・実行番号を除き、元の科学コード7セルとCreditsを保持し、先頭にR6 V1 manifestのハッシュ検査だけを追加しています。保存後Getで全10セルの本文一致・private・GPU・Internet OFFを確認しました。版付きGetは403で、版なしGetの返却currentVersionNumber=1を採用しています。Notebook内の作者metadataを実Input版の証拠には使いません。

V1の未実行Docker指定にはregistry名の誤りがあったため、V2では作者実runの完全名 `gcr.io/kaggle-gpu-images/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f` に修正。1回だけSAVE_AND_RUN_ALLを行い、17:32:44 JSTの公式状態はRUNNING・ログ未返却です。[操作前レビュー](artifacts/kaggle/p029-r6-957-20261015-42fbc0e9ae14/batch-review-v2.json)、[受理結果](artifacts/kaggle/p029-r6-957-20261015-42fbc0e9ae14/batch-result-v2.json)。Inputの `/1` は実版固定の保証にはせず、冒頭の実manifest照合が不一致なら元コードへ進む前に停止します。実mount、T4の実数、forward、CSV、hidden、自己Publicはまだ未確認。正式Submitは0です。

標準ライブラリの最終オフライン検証は45件実行・44成功・1スキップ・失敗/エラー0。[結果](artifacts/research/20261015-public957-followup-validation-v1/validation.json)。実symlink作成権限不足だけをスキップし、外部参照の模擬拒否は成功。以下の未認証・待機という記述は、それぞれの過去時点の履歴です。

## 2026-10-15 夕方：公式公開順位表と0.957公開候補を直接確認

**今回の匿名公式API取得で、自己 `rsraki` はPublic表示0.956・1029位／6025ランク付きチームと確認できました。自己0.956を超える作者Public表示0.957の公開候補を発見し、P029として元コード・Input版を固定した実行準備まで完了しました。自己0.957の再現・提出はまだ実施していません。** ユーザーの今回の到達目標は0.960超です。表示値による運用目標は0.961以上とし、従来の「0.960以上」と区別します。

### 順位とメダルについての新しい根拠

[匿名公式公開Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)を全6025行取得し、rank集合・チームID・自己のuserName/teamLeader/既知Submit57164195を照合しました。[監査記録](experiments/medal-frontier-audit-20261015.json)。保存時刻は16:40:59 JSTで、HTTP取得の正確な時刻は未記録です。Private結果は返っていません。

- 表示0.956は926チーム、287〜1212位に集中。自己1029位は現Publicの銅帯外です。同じ表示値でも順位には大差があります。
- 表示0.957の既存161チームは126〜286位。表示0.960は47〜66位、表示0.961は35〜46位です。これは現在の分布で、自己再現後の順位予測・保証ではありません。
- APIが返す現Publicの帯は金22位まで、銀301位まで、銅602位まで。0.960超は現Publicの上位銀相当を狙う合理的な目標ですが、最終メダル安全圏の保証にはできません。
- [公式LB説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)は約30%がPublic、残り約70%が最終評価と説明しています。終盤の改善・チーム数・Private順位の変化も未確定です。スコア表示が同じことと完全同点も区別します。

自己の公開bestは新しく照合できましたが、P028/P024/F006両fold各提出のCOMPLETE/エラー/個別スコアや、現在の日次残枠は公開LBからは判断しません。

### 0.956を超える公開候補の発見

検索索引だけでなく、Kaggle公式公開JavaScriptで確認した読み取り専用APIから、競技ID154281・SCORE_DESCENDING・EVERYONEのCode一覧先頭100件を取得しました。実配列の降順、競技タグ、linked有無、ID順を照合済みです。0.957が3件、0.956が38件、取得範囲で0.960以上は0件でした。`totalCount=0`なのに100件返ったため全件数・全件取得を主張しません。[一覧監査](experiments/public-code-score-frontier-20261015.json)。

- [Heliosli 0.9576という題名の公開Notebook](https://www.kaggle.com/code/heliosli/0-9576-latest-public-dataset-score-in-one-click?scriptVersionId=358332966)：**公式表示0.957・V1・script358332966・Submit57183099**。題名の0.9576は検証済みの精密スコアではありません。固定3外部Inputは公開メタデータで確認でき、これをP029の優先候補としました。
- [Heliosli別公開版](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-957-new-one-you-can-use)：best V1は0.957ですが、最新V2/script358425369のlinked提出にはスコアが返っていません。bestを最新V2の実績へ付け替えません。
- [R7＋Raptor＋knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext?scriptVersionId=358269940)：V1のlinked0.957を確認しましたが、追加Inputに非公開フラグがあるため、そのまま再現できる候補にはしません。

検索索引25候補と直接APIを別記録にしました。題名0.957でも実際は0.939の例を排除し、古い索引で0.950だったMatterhornが直接確認ではV9/0.954になったことも記録しました。旧「0.956超未発見」は検索時点の履歴で、今回の直接取得が新しい結論です。

### P029：学習せずに試せる具体的な実行準備

[P029 bundle manifest](artifacts/kaggle/p029-public957-r6-v1-20261015/manifest.json)、[元Notebookの完全コピー](artifacts/kaggle/p029-public957-r6-v1-20261015/source.ipynb)、[準備builder](scripts/prepare_p029_public957.py)、[独立静的監査](experiments/p029-public957-static-audit-20261015.json)を保存しました。638,918バイト・全9セルを保持し、7コードセルと埋込7モジュールをコンパイルしただけで実行していません。作者の旧出力は今回の自己結果ではありません。

最も重要なのは、R6 Dataset **Version 1／内部版20611930／bundle fe055406fad6a12f／manifest SHA6508ce69…**の固定です。同じ `rsna-knee-r6-frozen-fe055406` の最新版V2は、同じファイル名のままR7へ置換されています。単に最新Inputを選ぶと採点時のrecipeになりません。OrthoとSWAもV1固定。別の `verify_frozen_input.py` はmanifestのSHAとbundleを確認し、V2/R7誤接続を拒否します。その後、公式CLIの匿名取得で**4120バイトのmanifestメタデータだけ**を取得し、期待SHA・bundle・歴史ソースSHA・verifier成功を確認しました。[追加証拠](experiments/p029-r6-v1-manifest-metadata-verification-20261015.json)。重み本体・Kaggle上のmountedファイル・GPU推論の照合は未実施です。組立時のP029 bundleは上書きせず保持しました。

実行は次の順です。

1. 自己rsrakiのログイン済み接続と現在の提出枠・既知提出結果を読み取り確認する。
2. 新しい非公開候補として元コードと固定3Inputを保存する。Internet OFF、T4×2、採点時Docker `2757e0c7…` と実保存の設定・Input内部版を照合する。元Notebook内のGPU=falseと作者実runのGPU=trueは不一致があるため、実行設定を別途確認する。
3. 新しいbatch環境でR6 V1 manifestの別preflightを実行し、可視推論の12列・UID・finite確率・失敗/補完・実重みハッシュ・終了処理を確認する。元コードの8.3h定数は全Notebookのhard timeout保証ではなく、可視3件の時間をhiddenへ単純外挿しない。
4. 検証後に正式提出を独立した手動操作で1回実施し、完了・エラーなし・自己Publicを確認する。作者0.957と自己値を分け、結果不明のSave/Submitは再送しない。
5. 自己値が0.956以下ならInput/Docker/ソース/失敗経路を点検し、改善として採用しない。上回れば新しい対照候補とする。係数探索へ直行しない。

SWAの公開説明にはOAI由来の外部教師とGold58のcheckpoint選択再利用が明記され、競技を含む非商用の利用・改変・再配布について追記があります。ただし上流の全適格性が独立に監査完了したという意味ではありません。Orthoも上流ライセンスを保持します。自分のGoldによる学習・checkpoint/係数選択は0のまま、元Credits/OAI acknowledgementと継承履歴を保持します。

### 0.960超へ進むための比較設計

P029はR6＋flip TTAと、Raptor384/224の65:35混合を、Compactと50:50で合わせます。V22から複数要因が変わるため、差をR6単体・224単体へ帰属させません。最終rerankをしない式で、既存の固定20%プレビューツールとは別recipeです。

次の公開作者V2もコードを取得して静的比較したところ、埋込7モジュールは同一ですが、**R6→R7と同時にFLIP_TTA=True→False**が変わっていました。そのまま「重みだけ変更」とは扱えません。P029再現後は、R6のままTTAだけOFFにする[P030対照](artifacts/kaggle/p030-r6-tta-off-control-v1-20261015/manifest.json)を先に比較し、その対照と作者R7/TTA OFF版を比較すれば、二要因を分けて判断できます。

[P030 builder](scripts/prepare_p030_tta_control.py)は元ソースSHAを束縛し、cell6のboolean一箇所だけを変更したことをASTで確認します。2つのRaptor・埋込7モジュール・全係数は同じで、古い作者出力と実行番号を除き、未実行対照の説明を加えています。R6 V1の実manifestメタデータではflip_ttaが欠損するため、元コードの `manifest.get('flip_tta') OR FLIP_TTA` による勝手なTTA有効化も、固定版を使う条件で静的に排除しています。将来のKaggle mounted版と実際のTTA経路は別検証です。

この3段比較は追加学習・モデルのローカル取得を要しません。作者R7 V2の自己/作者スコアは未確認で、0.960超への改善は保証しません。既存F006両foldの固定20%混合は、重み・元証拠を復旧し、新しい親との補完性を確認してからの別候補とします。

[上位チームの公開手法調査](experiments/above960-public-methods-20261015.json)でも、現在0.963〜0.967の完全recipeは確認できませんでした。固定foldの弱検証で小差を積む、小型CNNや前処理・TTAを一要因ずつ比較するという根拠はあります。Gold58/Publicで微小差を選び続けず、report由来評価と画像由来評価の差を保持します。

### 提出接続の現在状態

プロジェクト内 `.venv` に公式Kaggle CLI2.2.4/SDK0.1.37を導入しました。標準認証の確認は未認証・SystemExitで終了し、権限エラーではありません。認証値の手動読込・表示・コピーはしていません。ブラウザ接続0件、Kaggle連携plugin検索も該当なしでした。ユーザーはログインを完了する意向を返信済みですが、完了連絡と認証成功の確認前に保存/実行/提出は進めません。

```powershell
.\.venv\Scripts\kaggle.exe auth login
```

認証情報や認証コードはチャットへ貼りません。ログイン後は同CLI/SDKでアカウントと枠を確認して進めます。今回追加したbuilderに自動アップロード・自動提出はありません。CUDA/PyTorchは引き続き利用できず、新規画像データ・重み本体取得、MRIデコード、学習は開始していません。

---

## 2026-10-15：GPUを利用できない今回の作業環境からの再監査

**確認済みの最終記録は自己Public macro ROC AUC 0.956、目標0.960以上は未達。表示値の差は0.004です。今回Kaggleの自己提出結果を再取得できていないため、これを10月15日の現在値とは扱いません。** 過去の結果と今回の調査を分け、以下より後の更新ブロックはそれぞれ記載時点の履歴として読みます。

### 前回までに到達した段階

- 自己V22R、Submit `57164195` は10月14日14:33 JSTの公式Get/ListでCOMPLETE・0.956・エラーなしと記録されています。作者V22の報告0.956とは別の自己実測で、従来自己0.954から+0.002です。原典は [V22R採用記録](experiments/V22R-official-Public956-Root-adoption-20261014.json)。
- P028 `57163437`、P024 `57166365`、S6＋F006両fold固定20% `57169291` は、[14:59の最終記録](experiments/F006-bothfold-formal-Submit57169291-Public956-checkpoint-20261014.json)ではPublic未返却です。現在も待機中なのか、採点済みなのか、失敗したのかは未確認です。結果不明の提出を再送しません。
- 当日の4件・残1・累計27は、前回公式値に受理1件を足した派生値です。今回の日次残枠には転用できません。
- F006/F007の各12epoch×2fold、F008の24epoch×2fold、計96epochは[前回のcheckpoint/CSV監査](experiments/goal-state-audit-20261014.json)で完了しています。古い「学習中」「待機」は現在の進行状態ではありません。
- F006両foldはCPU復元、Kaggle上の64 MRI/256 forward、可視3検査/20出力を通過した記録があります。可視成功はhidden全件完走・9時間制限・Public改善の証明にはなりません。F007も可視20出力は成功、正式Submitは0です。

### 今回の環境で確認できること

作業開始時のcheckoutの `data/` と `artifacts/` はREADMEのみで、前回のモデル・予測・raw APIログ・V22親切替設計の実体はありません。追跡済みJSONの記録とハッシュ参照は読めますが、参照先の内容まで再計算したとは主張しません。既存成果物が別端末でも失われたことを意味しません。今回生成したCPU監査結果は新しいrunへ保存しています。

今回 `nvidia-smi` とPyTorchは利用できませんでした。既定のPython shimはバージョン未選択ですが、`C:\Users\shiba\.pyenv\pyenv-win\versions\3.11.9\python.exe` を明示して標準ライブラリのテストは実行できます。GPUの物理的不存在は断定せず、学習可能なCUDA環境を今回確認できていない、と記録します。学習CLIのCUDAチェックは維持します。

指定の[自己提出ページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions)と[プロフィール](https://www.kaggle.com/rsraki)を照会しましたが、Web取得は本文なし/アクセスエラーで、接続済みブラウザも0件、in-app browserも利用不可でした。認証情報を探したり他プロジェクトからコピーしたりせず、自己スコア・日次枠・順位の最新値を未確認としています。

### 0.960への見込みと優先順位

0.956から0.960への差は、12所見のAUC改善の合計で概ね0.048です。仮に3所見だけが改善するなら平均+0.016が必要で、混合比20%から改善幅を計算することはできません。表示の丸めと評価標本の不確かさもあるため、達成率や成功確率を根拠なく数値化しません。

[F006の固定10所見弱比較](experiments/f006-completed-twofold-selection-comparison-20261014.json)はfold0 +0.004010、fold1 +0.002311で、供給group条件付き区間も正です。ただしcheckpoint選択後の同じweak holdoutを再利用し、fold0の副選択方針は7epoch観測後に定義しています。MCL/Synovitisを除いた固定10と提出の全12は異なり、患者独立性も未確認です。fold0の全12 bootstrapは351/1000反復が未定義でした。これらの差を自己Publicへ加算できません。BCE主選択の混合は両foldで改善根拠を示していません。

[F008の12→24epoch比較](experiments/f008-twofold-completed-weak-comparison-20261014.json)はfold1の固定10区間が0を跨ぎ、全12区間は両foldで0を跨ぎます。まず学習を48epochへ延長する根拠は弱く、完了済みモデルの正式結果を確定する方を優先します。

1. 最新のGet/Listで既知の提出3件、自己最良、現在の日次枠を読み取り確認する。過去のPENDING既定値やraw欠落を公式の待機状態に置き換えない。
2. S6＋F006両fold固定20%を対照として評価し、同じF006重み・同じ係数で親だけV22へ変える候補を比較する。旧P024から直接比較するとfold1と親の二要因が変わります。
3. 次候補は全Nでの同順位平均rank、UID対応、MCL/Synovitisの親値保持、構造失敗UIDの全12元文字列復元をCPU人工データで確認する。これは本Notebookの組立・実重み復元・Kaggle資格とは別です。
4. 元のV22/F006ソース・入力版・内容SHAを復旧した後、実組立容量、25出力、64検査、可視CSV、hidden完走/9時間、公式Publicの順に確認する。投影936,573Bは実組立の検証値ではありません。
5. ここまでで不足が明確になった場合に限り、訓練partition内のラベル品質、局所crop/スライス密度等を一要因ずつ追加比較する。Goldを学習・checkpoint選択に使いません。

### Webで再調査した事項

[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)の検索索引本文では、参加/統合期限は10月16日08:59 JST、最終提出期限は**10月23日08:59 JST**です。12所見macro ROC AUC、Notebook9時間以内、Internet OFF、`submission.csv`という条件を確認しました。ただしページを直接openした本文は0行で、詳細Rules/Hostコメント全文の今回の再確認はできていません。日次5回/最終選択2件等の詳細は[10月4日の既存調査](docs/research/kaggle-source-eligibility-20261004.json)と区別します。検索結果の相対的な「残り日数」は現在の残日数に使用しません。

[公開SWA作者の説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)は、58件のGoldを訓練から除外しながらcheckpoint選択・開発には再利用したと明記しています。Gold指標改善でもPublicが低下した例があり、公開重みの継承履歴をGold cleanへ書き換えません。これは参加者本人の報告で、Hostによる適格性裁定ではありません。V22の要求Dockerと実保存Dockerの差も保持します。

[MRI-CORE公式コード](https://github.com/mazurowski-lab/mri_foundation)はSAM ViT-B型2D/1024px/画像[0,1]を想定し、既存キャッシュへ単純差し替えできません。[論文v1](https://arxiv.org/html/2506.12186v1)の主評価はsegmentation、linear probeは部位・系列・施設分類で、膝12病変AUCではありません。[大会スレッド](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950)では適格性に関する質問があり、取得本文内でHostの確定許可/禁止を確認できません。[KneePreM v1](https://arxiv.org/abs/2609.31461v1)も新規調査しましたが、RSNA12所見の自己再現・公開重み/コード・OAI由来の適格性は未確認で、締切前の主候補へ昇格しません。

今回検証できた新しい公開0.960以上の再現可能recipeはありません。これはそのようなrecipeが存在しないという結論ではありません。現実的な短期経路は、完了済み候補の採点確認とV22親切替の比較です。実行準備はGPUなしでも進められますが、精度の到達実証にはKaggle推論と公式採点が必要です。

### 今回追加した実行準備

- [記録と証拠の監査CLI](scripts/audit_recorded_progress.py)：最終checkpointの記録スコア・観測時刻と現在値を分離し、参照先のサイズ/SHAを照合します。実行結果は[新規audit v2](artifacts/research/20261015-recorded-progress-evidence-audit-v2/audit.json)。記述子30件中、2件一致・28件欠落でした。同じファイルを指す記述子も別計数で、学習成果の達成率ではありません。認証ファイル・root外参照・モデルを読みません。
- [固定20%候補のCSVプレビュー](scripts/preview_fixed20_candidate.py)：親とnative平均の全N/12列CSVをUIDで照合し、二段階rankと失敗UID/未対応所見の親値保持を実装しました。Python floatのオフライン参照であり、実FP32推論との等価は未確認です。native平均自体の算出・モデルの同一性・失敗検出もこのツールの資格対象ではありません。
- [今回の環境監査](docs/research/environment-20261015.json)と[Web調査の出典・取得限界](experiments/web-research-096-20261015.json)を新規保存しました。

標準ライブラリでの監査例（出力先は未使用の新しい名前に変える）：

```powershell
& 'C:\Users\shiba\.pyenv\pyenv-win\versions\3.11.9\python.exe' scripts/audit_recorded_progress.py --output-dir artifacts/research/20261015-recorded-progress-evidence-audit-v3
$env:PYTHONPATH=(Resolve-Path src).Path
& 'C:\Users\shiba\.pyenv\pyenv-win\versions\3.11.9\python.exe' scripts/preview_fixed20_candidate.py --parent-csv 'path/to/parent.csv' --native-mean-csv 'path/to/native_mean.csv' --output-dir 'artifacts/research/fixed20-preview-new-run'
```

CSV例の入力パスは実ファイルへ、出力先は未使用のrun名へ置換します。実予測CSVが今回のコピーにはないため、ここで実モデルの候補CSVを生成したとは扱いません。新規成果物の上書き拒否・入力ハッシュ・数値契約と資格境界を記録します。

最終検証は新監査8件＋新CSVプレビュー14件＋既存契約7件の**29件成功**、実symlink作成権限がない1件のみスキップ（外部パス解決拒否の代替試験は成功）でした。[人工6検査の保存例](artifacts/research/20261015-fixed20-synthetic-contract-v1/preview/manifest.json)と[検証ログ](artifacts/research/20261015-cpu-continuation-validation-v1/unittest.log)を新runへ保存し、[今回の集約記録](experiments/cpu-continuation-audit-20261015.json)とledgerへ追記しました。実精度の新しい測定値はありません。

---

**10月14日14:59 JST：自己最良Public 0.956を維持し、S6＋F006両fold E12・固定20％の正式提出57169291を受理確認しました。目標0.960以上は未達・継続中です。** [最新の記録](experiments/F006-bothfold-formal-Submit57169291-Public956-checkpoint-20261014.json)。可視3検査・20出力を再計算してRoot受理後、Submitを1回実行。V22Rは公式COMPLETE・自己0.956、P028・P024・F006両foldはPublic結果待ちで、旧提出は再送しません。本日4件・残枠1・累計27件は前回公式値と今回受理1件からの導出で、提出後の公式枠再照会値ではありません。

F006の可視3検査成功はhidden完走やPublic精度の確認とは別です。F007も可視20出力はRoot受理済みで正式Submitは0。次は同じF006重み・同じ固定20％で、親モデルだけをS6から自己0.956のV22へ切り替える候補を検証します。V22作者報告0.956と自己値は区別し、Docker差37c64…／2757e…、公開重みのGold選択履歴の不確かさを保持します。MRI-COREは条件付き調査候補で、重み取得・実装・追加学習は開始していません。以下は過去時点の記録です。


**10月14日14:33 JST：自己再現V22Rの公式Public 0.956を確認し、現在の最良値として採用しました。従来0.954から+0.002、目標0.960以上は未達で継続中です。** [採用記録](experiments/V22R-official-Public956-Root-adoption-20261014.json)。Submit57164195を自己アカウントの公式Get・ListともにCOMPLETE・0.956・エラーなしで確認。作者V22の報告0.956と自己値0.956は別記録です。P028・P024はPublic未返却で再提出せず、本日3件・残枠2・累計26件です。

V22は元コード6セル・Input3固定版・推論式を保持しましたが、Dockerは作者要求37c64…から実保存2757e…へ変わっており、R7単独の改善や非公開精度・公開重みのGold cleanを断定しません。次はS6＋F006両foldの固定20％を正式対照として検証し、同じF006重み・同じ係数でV22へ親モデルだけ切り替えて比較します。F006の新native64・内容SHAとF007可視20出力はRoot受理済み。MRI-COREは条件付き調査候補に留め、重み取得・実装・学習は開始していません。以下は過去時点の記録です。


**10月14日14:21 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [最新の記録](experiments/F007-visible-pass-F006-bothfold-native64-pass-20261014.json)。14:13 JSTの公式Get／ListではP028・V22R・P024のraw状態とPublicは未返却、エラーなし。本日3件・残枠2・累計26件です。

F007は可視3件・20出力の検証を通過しRoot受理、正式Submitは0です。F006両fold E12も64 MRI・256 forwardと新重みの内容SHA確認を通過しRoot受理しました。F006の提出推論・Publicはこれから検証します。次はF006を優先し、V22との組み合わせは公式スコアを見て判断します。ローカルMRIデコードと追加学習は0、CPU復元とKaggle MRI推論は実施済み。旧エラーを含む以下の記録は過去時点のものです。


**10月14日13:45 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [今回の正本](experiments/F006-bothfold-CPU-private-ready-F007-finalmask-error-20261014.json)。13:35 JSTの公式Get／ListはP028・V22R・P024のraw状態／Public未返却、エラーなし。本日3件・残枠2・累計26件です。

F006両fold E12の新4モデルは実CPU復元と独立検証を通過しRoot受理。CLS0・REG0・REG1の実指紋は保持し、CLS1だけを置き換えました。fold1モデルの非公開Dataset12573715/V1はREADY・2ファイル名／サイズ一致を読み取りで確認済み。Create応答guardの停止を保存し再送せず解決しました。新しい推論ソースへ伝播中で、Kaggle上の内容SHA・新native64・Serving・Publicはまだ未検証です。

F007のWindowsパス修正job138608301/V1はS6側の可視3件を完了後、最終mask処理でKeyError serving_shasとなりERROR。Submit0のまま、最終CSVまでのメタデータ契約を人工データで検証しています。元の数値コード・旧エラー・native64記録は保持。ローカルMRIデコードと追加学習は0、必要なCPU復元・Kaggle MRI推論は実施済み。以下は過去時点の記録です。


**10月14日13:18 JST：自己最高Public 0.954、目標0.960以上は未達。3件を正式提出し採点待ちです。** [最新正本](experiments/three-submitted-two-native64-pass-F007-visible-error-20261014.json)。13:11 JSTの公式Get／ListではP02857163437・V22R57164195・P02457166365のraw状態とPublicは未返却、エラーなし。本日3件・残枠2・累計26件を確認しました。

P024新重みとF007残差モデルは、それぞれ64 MRI・256 forward・FP32集約・内容SHA・alias0・fallback0・資源／終了処理の検証を通過しRoot受理済み。P024は可視3件／20出力も再計算PASSし、Submit1受理。可視CSVがP028と同SHAでも、3件だけの順位一致でhidden予測同一とは判断しません。F007提出用job138605901V1はコード・Input9・設定一致を確認しましたがERROR終了。Submit0、原因ログを調査中でServing／Publicは未資格です。

弱ラベルの直接比較ではF006 E12がF007 E12を両foldで上回り、事前固定50:50のCLS混合はAUCを下げました。比率探索はせずF006を優先。fold1 E12もCPU単体復元を通過し、両fold置換の新4モデル検証を準備中です。旧f005 E9とf006自身のBCE E9は別checkpointとして保持します。ローカルのMRIデコード・追加学習は0、必要なCPU復元とKaggle MRI推論は実施済み。以下は過去時点の記録です。


**10月14日12:25 JST：自己最高Public 0.954、目標0.960以上は未達。改良モデルの新64検査テストは実行中です。** [最新正本](experiments/native64-Source4-running-Source3-error-preserved-20261014.json)。P02857163437・V22R57164195は正式提出済み。12:23 JSTの公式Get／Listではraw状態・Publicは未返却、エラーなし、本日2件・残枠3・累計25件です。

P024旧64検査job138598572V1は、REGモデル読み込みの作業ルート指定でdriver起動前にERROR。実MRI推論の資格は0です。旧コード・ログを保持し、作業ルート・明示実行フラグ・producer情報転記の3か所を独立5テストで修正確認しました。新Source4 job138601076V1は公式raw／typed RUNNING。モデル・前処理・数値処理は保持し、完走・結果・内容SHAはこれから確認します。

F007の残差モデルも独立レビューと実API容量確認を通過し、Save1でkernel138600811V1が返りました。返却を扱うguardが停止したため、追加Saveせず実保存refとコードを読み取りで照合中です。自己Public・新native64・Servingの資格は発行していません。以下は過去時点の記録です。


**10月14日11:51 JST：自己最高Public 0.954、目標0.960以上は未達。2候補を正式提出し採点待ちです。** [最新正本](experiments/two-submissions-private-ready-Root-CPU-accepted-20261014.json)と[実行方針](docs/goal-096.md)。最新公式Get／ListではP02857163437・V22R57164195のraw状態とPublicは未返却、エラーなし。SDK既定PENDINGを公式状態と推定しません。本日2件・残枠3・累計25件をraw／typedで確認しました。

P028親guard対処版とV22公開recipeの自己候補は、可視推論・12確率CSV・順位計算の再検証を通過し、各1回Submit受理。V22は元6コード・3Input版を保持していますが、要求した作者Docker37c64と実保存2757が異なります。この差を束縛して記録し、作者環境の厳密再現とは主張しません。作者0.956と自己Public未返却を区別します。

P024モデルDataset12571388とF007 head Dataset12571495は、実照会で非公開・V1・READY・正しい2ファイル名／サイズを確認済み。Create後の旧guard停止は保存し、再Createせず読み取りで保存実体を確定しました。remote内容SHAと内部版IDはAPI未公開なので未検証。Kaggle推論の起動前にmounted archive／member SHAを検査する境界で確認します。

P024新4モデルはRoot委任CPU復元と独立査読を通過し、Rootがcomponentを受理。共有データauditと各resource provenanceを区別し、bytesパスの監査漏れも修正しました。モデル・前処理・数値処理は保持。新native64／Serving／Public資格はまだ未成立で、F007の新9Input接続も準備中です。F006固定10 AUC候補を優先し、弱差をPublicへ加算しません。

今回のローカルMRIデコード・追加学習は0、Kaggleの可視MRI推論は実開始・完了済み。非公開送信・実行・手動提出はユーザーの今回＋今後の明示承認範囲で進めています。MRI・Report本文・教師行・認証情報の外部送信、自動アップロード・自動提出は追加していません。以下は過去時点の記録です。


**10月14日11:09 JST：自己最高Publicは0.954。目標0.960以上は未達です。** [今回の正本](experiments/goal-resume-comparisons-P028-submitted-20261014.json)と[実行方針](docs/goal-096.md)。最新公式照会でP026は0.953、自己最高0.954を確認しました。

旧記述の学習中／待機を再監査し、F006・F007両fold各12epoch、F008両fold各24epochの完了を確認。計96epochのcheckpoint／CSVとBCE bestのSHA、CSV契約を照合してPASSしました。F006固定10 AUC副選択E12の固定50:50弱比較差はfold0 +0.004010、fold1 +0.002311で条件付き区間も正。BCE主選択は保持。F00824はfold1の改善が不確かで、追加延長を優先しません。weak差をPublicへ加算せず、全12指標・欠損・選択後の検証再利用を別記しています。

P028の修正版はprivate kernel138592375／V1の可視3件・20出力をRootが再検証してPASS、正式提出57163437を1回受理。hidden完走・新Publicはまだ未確認です。提出直前の公式残枠は5で、提出後の新しい枠を未照会のまま断定しません。

ユーザーは今回と今後の必要な自己rsrakiへの非公開送信・実行・手動提出を明示承認済み。過去の審査拒否を保持し、未回答待ち状態は解消しました。P024は公式Google JSON upload endpointへの許可不足を修正し2ファイルのPUTが成功しましたが、Create1後の返却guardで受付未確定。同じCreateは再送せず、保存状態を読み取りで確認します。MRI・Report本文・教師行・認証情報は送信しません。

Versia7 V22／script358019432は作者linked COMPLETE／0.956、3Input版を固定して自己再現を準備中。作者値と自己値を区別し、公開重みの上流gold利用が不明な部分も別監査に記録しました。現在の公式公開LB先頭100件は最高0.966、0.960以上54件。全件集計や利用可能重みの証拠とは扱いません。以下は過去時点の記録です。


**10月13日03:34 JST更新：自己最高Publicは0.954、残り提出枠は1です。** [F007接続受理・F008待機・F006 E6の正本](experiments/F007Source3-reviewed-F008-waiting-F0061-E6-local-20261013-v2.json)。03:06のRoot委任local_analysisによる公式Get／ListはP026のraw・typed COMPLETE／0.953／エラーなしで一致し、Rootが独立受理しました。本日カウンタ4・全23件・日枠リセット未観測。P026はS6 control 0.954を上回らず不採用とし、次のnative候補は最高S6 controlを基準に進めます。P026基準への切替・Reader係数探索・追加Submitは0。目標未達・作業を継続します。

F006 fold1は6/12epoch（weak BCE 0.29386907・AUC12補助0.86122754・各208更新／skip0・累積12,386.50秒）まで完了し、学習は継続中です。定率6epochまで旧条件との入力・設定一致を確認。差は小さいものの、厳密再現・改善を証明しません。残る7〜12はlate cosine区間で、固定12epoch完了前の採用は行いません。

F007接続Source3はRootの独立レビューを通過しましたが、source準備のみの受理で、新Input・native64・Serving・Public・実metadata容量の資格は未発行です。F008の手動キューsession45576はWAITING_FOR_F0061、process object固定済み・子プロセスなし・F008学習未開始。F006 fold1の両プロセスの終了0と固定12epoch完了後に、同じ旧BCE CLS・zero残差から24epochをfold0→fold1の順に実行する予約です。4090はこの順序で予約し、追加GPU jobは開始しません。Source2のSTOPで次の起動を抑止し、稼働中子プロセスは停止しません。外部送信の権限はありません。

02:51にRootが[公式公開リーダーボード](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)を1回読み、取得上位100件は最高0.966・最小0.956、厳密な0.960超は28件でした。全件集計ではなく、次ページは未取得です。高スコアだけでは公開Notebook・重みの利用可能性や自己達成を証明しません。Notebook一覧401／Source403を全SDK障害とは扱いません。

02:29の匿名公開metadataでは、[Versia7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) V21の作者linked／bestは0.954、[Kiyoshi template](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template)の最新runは取消・linkedなし／bestはV9の0.951、[Dinosaur-v5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) V45は0.950でした。自己再現値とは区別します。EV本文は旧指定404、公式SDK形式への修正指定403で未取得・静的比較HOLD。同score／Inputだけで重複・撤回・非公開・認証失効や公開重み全体の不在を断定しません。

P024モデル、P028コード、F007モデル輸送の非公開送信3件の回答は未受領で、送信・再試行0を維持しています。P024／P028の具体的承認不足によるプロセス開始前の自動審査拒否を保存し、旧REG承認を新3件へ転用しません。ローカル作業と既存学習を継続します。以下は過去時点の記録です。


**10月13日01:06 JST：自己最高Publicは0.954、厳密な0.960超えは未達です。** [最新公式観測・F007四モデルCPU復元・学習状況の正本](experiments/official01p06-F007-new-four-CPU-F0061-running-20261013.json)。Root委任のlocal_analysisがP026のGet／List／limitsを読み、RootがrawとSHAを確認しました。P026のraw状態・Public・エラー項目はまだ未返却で、SDK既定PENDINGを公式状態とは扱いません。本日カウンタ4・残枠1・全履歴23件で、日枠リセット時刻は未観測です。

F007は両foldの残差12epochを完了し、Rootが新しい四モデルをCPUで1回復元してPASSしました。generic factory1回、元checkpoint厳密適用4回、残差head安全読込2回、6組のstorage alias検査で共有0、REGは元objectを保持。forward／GPU／教師／MRI／ネットワークは0で、新Kaggle Input・native64・可視提出推論・Public精度の資格には進んでいません。元checkpointを保持した3メンバー・816,882Bのモデル用輸送はローカル準備だけで、外部未送信です。

F006 fold1はsession41581で2/12epochまで進行中（Root stdoutのweak BCE 0.340669・AUC12補助0.77125048）、まだ完了・採用していません。F008は旧CLS設定12epochを保ち残差期間だけ24epochにする準備とdefault HOLDまでで、実学習0。先読み高速化案も未実装・未計測で、既存一件の読込比率約63%を全epochの律速や速度改善とは扱いません。P024の新モデル送信とP028の新Notebook Saveは、自動審査がプロセス開始前に拒否した具体的承認待ちで、外部送信・再試行0を維持しています。以下は過去時点の記録です。


**10月13日：F007 fold1は12epochを完了し、F006 fold1の学習をRootが開始しました。** [最新公式照会・完了比較・実GPU確認・学習開始の正本](experiments/official00p06-F007-fold1-complete-F006-fold1-training-20261013.json)。F007 session59615は終了コード0、元BCE規則の最早最良はE12で、weak BCE 0.27517597・固定10 AUC補助値0.88163248。Rootが全12 checkpoint／予測SHAと4207 NPZ・7,918,878,170Bの件数／サイズを受理しました。特徴7GB本文の再hashや、今回の凍結head tensor別復元は行っていません。全891件の保存CLS確率誤差0.0とnative oracle最初1件を区別します。

fold1の固定50:50弱教師比較差は+0.00137217（861供給group・1,000回、条件付きbootstrap区間+0.00021828〜+0.00274822）。fold0も弱比較では改善していますが、checkpoint選択後の診断であり、患者独立性、全2fold OOF、Public改善は証明しません。Rootのfold0実CPU復元に続くRTX4090の人工GPU確認は、3有効窓・wrapper2回＋原AST参照1回、計9 encoder窓で参照logit誤差0.0・padding誤差0.0でした。実MRI／教師は0で、新4モデル共有・2T4 native64／Serving／精度資格には移していません。

RootはF007 fold1完了と人工GPU確認後、開始前に定義したpolicyを維持してF006 fold1の固定12epoch学習を起動（session41581、新run）。主規則は最早最小weak BCE、副規則は全固定10項目が有限な最早最大平均AUC。元bestを副規則で上書きせず、同epochなら元checkpointを再利用して重複候補を作りません。Gold／Publicによる選択は0で、完了・途中採用・Public改善はまだ未確認です。

00:06 JSTのRoot委任local_analysisによるP026個別Get／List／limits読み取りとRootの独立確認では、P026はraw状態・Public未返却、エラーなし。SDK既定PENDINGを公式状態とは扱いません。既知の自己最高はP020・P021の0.954、P022は前回確認のhiddenエラー。本日カウンタ4・全履歴23件・残枠1を維持し、JSTの日付変更から日枠リセットを推定しません。P024の新モデル送信とP028の新Notebook Saveは具体的承認待ち・開始前審査拒否／外部送信0を維持し、別送信で迂回しません。厳密な0.960超えは未達・作業を継続します。以下は時点ごとの履歴です。


**10月12日23:30 JST：F007 fold1のcacheをRootが受理し、局所残差だけの12epoch学習を開始しました。** [cache完了・残差開始・次の計画の正本](experiments/f007-fold1-cache-complete-CLS-exact-residual-training-started-20261012.json)。抽出session71660は終了コード0、4207 NPZ・7,918,878,170Bの全path／存在／サイズ／件数とCLS headのraw SHAを確認。全891検査・12所見の保存CLS確率との最大誤差は0.0です。native logitの直接oracleは最初1件でexact、全891件のfull native forward照合ではありません。7GB特徴本文の再open／hashや凍結head tensorの別復元とは区別します。

Rootは旧f005 fold1の元BCE最良E9を起点に、encoderと元CLS headを凍結したまま、局所残差だけをconstant LR3e-4で学習開始（session59615、新run）。固定12epoch・元の最早最小weak BCE選択・Gold非使用を維持し、途中checkpointの採用、fold1の効用、Public改善はまだ主張しません。モデルproducerとデータ事前検査producerの役割、元source／run／configを保持しています。

F006 fold1は開始前に固定12epochのBCE主選択と固定10 AUC副選択をRootが別規則で定めましたが、実学習はまだ0です。F007 fold1完了と人工GPU確認の後に進める計画で、所要約6時間は見積もり、fold1の実測ではありません。同epochなら元checkpointを再利用し、別モデルや重複提出を作りません。元bestを副規則で上書きせず、Gold／Publicで選択しません。

最新公式は23:06のP020・P021自己Public 0.954、P022はhiddenエラー、P026はraw状態／Public未返却。本日受理4件・残枠1、厳密な0.960超えは未達です。P028の具体的Notebook SaveとP024の新モデル送信は、それぞれ自動審査が開始前に拒否して具体的承認待ち・外部送信0を維持。ローカル学習の許可を外部送信の承認とは扱わず、影響しない学習と採点確認を継続します。以下は時点ごとの履歴です。


**10月12日：P028の実ソース946,931Bは上限950,000B内でローカル検証に通りましたが、外部Saveは承認待ち・実送信0です。** [具体的コード保存と審査停止の正本](experiments/p028-concrete-Notebook-Save-auto-review-held-20261012.json)。Rootが新しい非公開Notebook rsraki/p022m-0e93fd1399c445dbb769ccb1e07bf6edのmetadata／ソースを生成し、defaultのオフラインDTO検証は終了コード0。モデル・数値処理を維持した元S6の検査別補完guard修正で、まだ新Kaggle実行・可視資格・Submitはありません。

Rootの明示Save操作はCLI開始前に自動承認審査が拒否しました。理由は「新しい具体的コードと宛先への明示承認不足」。審査文はDataset送信と表現していますが、実際の予定APIはprivate NotebookのSaveKernelであり、新Dataset作成ではありません。この区別と元拒否記録を保全し、認証／Input照会／Quota照会／Save・remote送信は全て0、attempt dirも未作成です。Rootの具体的承認依頼への回答を待ち、同じ操作・別ref・encoded送信で迂回しません。P024の新104,308,640Bモデル／別宛先への具体的承認待ちも独立して維持します。

最新公式は23:06のP020・P021自己Public 0.954、P022はhiddenエラー、P026はraw状態／Public未返却、本日4件・残枠1。F007 fold0はRootが12epoch完了と実CPU復元を受理し、弱比較は固定50%差+0.00158535ですがPublic改善を証明しません。fold1のcache抽出session71660は継続中で、完了・全891件照合・残差学習はまだ未受理／未開始です。影響しない採点確認とcache作業を続け、厳密な0.960超えは未達です。以下は時点ごとの履歴です。


**10月12日23:06 JST：P020とP021は自己Public 0.954、P022はhidden再実行エラー、本日の残り提出枠は1です。** [公式結果・F007完了・P028修正準備の正本](experiments/official23p06-P021public954-P022error-F007complete-P028source-20261012.json)。Rootが委任したlocal_analysisの読み取り6RPCとRootの独立確認で、P021のmirror版57103408はGet／Listともraw・typed COMPLETE／0.954／エラーなし。作者の旧V1最高0.956とは別の再現値です。P02257102110はCOMPLETEという処理終了状態でもerrorDescriptionがあり、Publicはありません。P02657107853はraw状態・Public未返却で、SDK既定PENDINGを公式状態とは扱いません。自己最高0.954・本日受理4件・全履歴23件・残枠1、厳密な0.960超えは未達です。

P022の静的診断では、元S6が所定の検査別補完で有効CSVを出しても、追加親guardがarm失敗件数ゼロを要求して停止する条件を確認しました。hiddenの原因は未確定で、未知のreader例外や全件の時間・資源制約も残ります。P028としてこの親guardだけを修正するSource3をRootが査読し、元controller／親の実ASTと人工CSVで補完時のfullN bytes保持、worker異常・未知source拒否を確認しました。元モデル・前処理・Native固定20%・rank・mask・失敗12文字列復元は保持。P026には同じゼロ件条件がなく、P023／P024の共有親には波及します。新P028は手動tools準備までで、このsnapshotのSave／可視資格／Submitは0です。

F007 fold0はRootが12epoch完了・終了コード0と全12 checkpoint／予測SHAを受理し、元BCE規則の最早最良はE12、weak BCE 0.26977043・固定10 AUC補助値0.88233461です。固定50:50の弱教師比較差は+0.00158535（812供給group・1,000回の条件付きbootstrap区間+0.00054275〜+0.00263536）。患者独立性・Public改善は証明しません。Rootの実CPU1モデル復元で元prefix／suffix／CLS tensorの一致と残差15,180 parameterを確認しましたが、新4モデル共有・Kaggle native64／Servingの資格には移していません。全821件の保存CLS確率誤差0.0とnative oracle最初1件は区別し、7GB特徴本文の再hashとは扱いません。

F007 fold1は旧f005 BCE E9を起点にRootがcache抽出を開始（session71660）。23:05時点で1331 NPZ、cache完了・全891検査12所見照合・残差学習は未完／未開始です。Rootは弱教師比較でより大きな差を示すf006の別AUC12候補を優先しますが、その新非公開ペイロード送信は具体的承認待ちで外部送信0。P028の同モデル修正を可視検証して最後の枠を判断し、Goal作業を継続します。以下は時点ごとの履歴です。


**10月12日：F007 fold0のcacheをRootが受理し、局所残差だけの12epoch学習を実開始しました。** [cache受理と学習開始の正本](experiments/f007-cache-complete-CLS-exact-residual-training-started-20261012.json)。4207 NPZ・7,918,878,170B（約7.38GiB）の抽出が終了コード0で完了し、全821検査・12所見の保存CLS確率は最大誤差0.0。nativeとの直接logit oracleは最初の1件でexact、全821件のfull native forward検査とは分けて記録します。Rootがencoderと元CLS headを凍結し、旧f005 BCE fold0／E8・constant LR3e-4のまま残差だけを学習開始（session77307）。weak／Gold／Publicを区別し、学習完了・効用・精度改善はまだ未確認です。

最新公式は22:06の自己最高Public 0.954・本日受理4件・残枠1。P026提出57107853は受理済みで新Publicを主張しません。P024の具体的新モデル／宛先への送信は自動承認審査で拒否され、承認待ち・外部送信0を保全しています。実cache・残差学習と既提出採点を継続し、厳密な0.960超えは未達です。以下は時点ごとの履歴です。


**10月12日22:06 JST：P026の正式提出57107853が受理され、本日4件・残り提出枠1です。** [公式照会・提出・送信審査の正本](experiments/official22p06-P026-submitted-P024-upload-review-held-20261012.json)。Rootが3fold raw平均と最終固定20% CSVを独立受理し、22:01に保存kernel138327844／V1を手動Submit1回・再送0で提出。Rootが委任したexecution_auditの22:06読み取り5RPCではP020の自己Public 0.954を維持、P022／P021のPublic・raw状態は未返却、エラーなし。全履歴23件の末尾と残枠1をraw・typed双方で確認しました。P026への個別Getはこの元3候補observerには含まれず、新しい4候補入口を準備中です。厳密な0.960超えは未達・作業を継続します。

P024の新AUC12モデル2ファイル・104,308,640Bを新しい非公開Dataset rsraki/n392-aucf0-816d6df4187b4878a3ccd7234ffa14ddへ送る操作は、自動承認審査が「この新ペイロードと宛先への具体的なユーザー明示承認が不足し、API提出までという目標だけでは許容できない」として拒否しました。CLI開始前のためStart／PUT／Create・remote送信は全て0、attempt dirも未作成。Rootの具体的承認依頼を待ち、過去のREG2承認を流用せず、同じ送信や別経路の送信は行いません。元Root intent・査読・審査履歴は保全し、外部承認成立や新Input／推論資格とは扱いません。

既存4提出の採点確認と、RootのF007 fold0／旧BCE E8 cache抽出は継続します。最後のRoot進捗は21:45の511 feature files・cache complete=Falseで、残差学習はまだ0。この送信停止だけでGoal全体をblockedとせず、影響しない作業を続けます。以下は時点ごとの履歴です。


**10月12日21:43 JST頃：F007 fold0の特徴cache抽出をRootが実開始しました。** [実Originと開始記録](experiments/f007-original-BCE-fold0-cache-extraction-started-20261012.json)。f006の12epoch完了・終了コード0を確認後、旧f005の元BCE最良E8を厳密に束縛したOriginを発行し、RTX4090でCLS＋16局所特徴の抽出を起動（session90205）。モデルのproducerと現f006のデータ事前検査QAは別々に記録し、元run・checkpoint・sourceは保持しています。cacheの完了、最初1件のnative oracleと全821検査・12所見の保存予測一致はまだ未受理。残差学習・fold1・競技Submitは開始していません。最新公式結果は下記21:36の自己最高0.954・残枠2、厳密な0.960超えは未達・作業を継続します。


**10月12日21:36 JST：自己最高Public 0.954、残り提出枠2を維持し、f006は固定12epochを完了しました。** [公式結果・完了比較・P026可視結果の正本](experiments/official21p36-f006-complete12-P026-visible-20261012.json)。Rootが明示委任したexecution_auditの実読み取り5RPCで、P020はGet／Listともraw・typed COMPLETE／0.954／エラーなし。P022／P021はPublicとraw状態未返却で、SDK既定PENDINGを公式状態とは扱いません。本日3件・全履歴22件の末尾確認。厳密な0.960超えは未達・作業を継続します。

f006の元BCE選択はE8のままです。旧CLSを置き換える固定50:50の弱教師比較では、BCE候補の固定10 AUC差−0.00034757（供給group bootstrap 95%区間−0.00105968〜+0.00025010）、別規則のAUC候補E12は+0.00401012（+0.00182578〜+0.00631689）でした。812供給group・各1,000回が有効ですが、同じ弱検証集合でcheckpointを選んだ後の条件付き診断で、患者独立性・Public改善を証明しません。RootはE12を実CPUで厳密再構築して別bundleへ封印済み。新Kaggle入力・native64・可視推論・競技Submitの資格には移していません。

P026は元P020に公開2.5D readerを固定20%追加し、Rootの保存kernel138327844／V1がCOMPLETE。全3検査成功・失敗0、3foldで9forward、約15.2秒、モデルSHAを確認しました。11結果の取得はSDK環境のNumPy不足でheldとなりましたが、元記録を保全し、Rootが同じ取得bytesを主.venvで再検算して全N最終CSV・36値のFP32契約を確認。3fold raw平均の独立再計算はこのsnapshotで未受理、Root提出認定・Submitは0です。F007は旧f005 BCE fold0／E8を起点にCLSを保ち局所残差だけを比較する方針とSource4を準備済み。実Origin・cache・学習はまだ0、fold1はfold0の効用判断後です。以下は時点ごとの履歴です。


**10月12日20:36 JST：自己最高Public 0.954を維持し、残り提出枠は2です。** [公式結果・E10・CPU監査の正本](experiments/official20p36-f006E10-P025CPU-audit-20261012.json)。Rootが委任したexecution_auditの実読み取り照会で、P02057098752はGet／Listともraw・typed COMPLETE／0.954／エラーなし。P02257102110とP02157103408はスコア・raw状態未返却で、SDK既定PENDINGを公式状態とは扱いません。本日受理3件・全履歴22件の末尾を確認し、厳密な0.960超えは未達です。

f006はE10／12の学習中で、weak BCE 0.29639006、固定10 AUC 0.88149432、12項目AUC 0.88164164、LR倍率0.325です。旧同epochとの固定10 AUC差は−0.00387266で、改善・途中採用は主張しません。元BCE選択は暫定E8のまま保ち、固定12epoch完了後にBCEと別候補のAUC選択を比較します。稼働中のtrainer・CLI・configは維持します。

P025の公開2.5D readerは、RootのKaggle CPU監査で3重みのSHA、安全なweights_only読込、各219 state／30,741,112 FP32要素の有限値と、元公開6ソースのUTF8本文・SHA・構文を確認しました。CPU監査だけで、Competition入力・MRI・教師・Gold実データ・forward・GPU・Submitは0、ローカル重み取得も0です。Gold利用・checkpoint選択とreader実装のソース監査を続け、推論・精度・提出資格には格上げしません。P023／P024の採用・次のSubmitは未決定、F007も準備段階です。目標未達・作業を継続します。以下は時点ごとの履歴です。


**10月12日19:36 JST：P020のS6単独を自己Public 0.954で再現しました。** [公式証拠と比較順序](experiments/p020-official-public954-three-candidate-comparison-20261012.json)。提出57098752はGet／Listのraw・typedがともにCOMPLETE／0.954／エラーなしで、作者報告と自己再現を区別します。自己最高は0.951→0.954。P02257102110（固定Native20追加）とP02157103408（latest V2 mirror）はスコア未返却で、typed既定PENDINGを公式raw状態とは扱いません。公式残枠2・本日3・全22件末尾確認、追加Submit0。旧HTTP履歴と19:06以前のsnapshotを保持します。

比較順序はP022のNative追加効果とP021の公式結果→完了f006のBCE対固定10 AUC選択→次候補です。f006は保存E8でweak BCE 0.27404308／固定10 AUC 0.88090016、旧同epoch比はBCE +0.0002848／AUC +0.0003312、改善・途中採用0。[説明図](artifacts/research/20261012-f006-epoch8-progress-plot-v1/f005-f006-epoch8-three-panels.png)と[secondary方針](experiments/f006-fixed10-AUC-selection-policy-20261012.json)を残し、元BCE選択・RNG・稼働中コードは維持します。[CLS fold0だけを置き換える隔離loader](artifacts/research/20261012-f006-fold0-isolated-mixed-runtime-source-v1/preparation.json)はsource-only準備済みで、実CPU／native64／最終可視CSVの受理はまだ0です。目標未達・作業を継続します。以下は履歴です。
**10月12日18:43 JST：P021の追加手動Submitで具体的なDataset制約を確認し、新しい入力mirror版を準備しています。** [現状と次の比較](experiments/p021-http400-mirror-preparation-pair-review-20261012.json)。18:16の全21件・terminal raw／typed履歴にP021の同説明提出がなく、本日の受理2件・残枠3を確認した後、Rootの明示判断で18:25に追加1回を試行。HTTP400は「Notebook uses non-versioned datasets」を返しました。元17:24と合わせPOST計2回・自動retry0、旧heldと汎用helperの受付不明記録を保持し、旧原因や特定Dataset名を断定しません。成功済みP020／P022のcompact-v1／13はr6／2と6モデルSHAが完全一致するため、モデル・元latest V2の50:50／rank計算を保つSource V3は準備完了・Root査読待ちです。新V1のSave・実結果・競技Submitはまだ0、旧Notebookへの追加POSTは行っていません。

P020提出57098752とP022提出57102110は18:07／18:19の公式照会でもraw採点状態・Public未返却、自己最高0.951・残枠3。次は入力mirror版の保存・実結果確認と、P020のS6単独対P022の固定Native20追加の公式スコア比較です。[固定50:50のペア効用](artifacts/research/20261012-f005-fixed-mean50-pair-utility-v1/checks-and-compact-summary.json)では固定10所見のfold0が修正8,681／悪化8,553、fold1が13,166／9,230で、24所見の効用差が元AUC差と一致しました。弱教師の選択済みepochの探索的診断で、係数探索・Public主力のOOF比較・患者独立性の断定は行いません。

[18:15の公開追加調査](artifacts/research/20261012-post-P022-public-stronger-primary-web-v1/analysis.md)は、0.960超の採点版と利用可能な同一Input／重みが揃う新候補を観測範囲で確認できませんでした。取得不能は不存在と扱わず、大量の公開コピーや単純な画像拡大を次の改善要因にしません。f006は最新比較E6・1,272更新／skip0・LR倍率1を保持し、固定12epochとepoch7以降の減衰比較を優先します。目標未達・作業を継続します。以下は履歴です。
**10月12日18:06 JST：P022の可視推論・最終CSVを受理し、正式提出57102110へ進みました。** [実結果と採点待ち](experiments/p022-visible-accepted-single-Submit-score-pending-20261012.json)。保存kernel138296990／V1のSource・COMPLETEと20個の指定出力を照合し、全3UID・成功3／失敗0、元4モデル平均・全N固定20%rank混合・MCL／Synovitisの基準復元・後処理を独立確認しました。Root受理 `98f1390f…ec4b`、最終CSV `7115f865…a7a8`。18:01:11に手動Submit1回が公式57102110として受理され、再送0。可視結果の確認であり、hidden完走・Public改善はまだ未証明です。

18:02の読み取り専用照会はraw／typed日枠3。P021に対応するrefは取得20件では見つからず受付不明・再送0を維持し、4→3の減少をP021受理の根拠にはしません。P020は17:50の専用照会で採点状態・Public未返却、自己最高0.951。次はS6単独P020と固定Native20追加P022の公式スコアを比較します。f006の最新保存比較はE6で、固定12epoch継続・初6epochのLR倍率1を保持し、epoch7以降の減衰と最早最小weak BCE選択を完了後に評価します。精度改善・厳密な0.960超えは未達・作業を継続します。以下は履歴です。

**10月12日17:41 JST：P021の可視出力を受理しましたが提出受付は不明、P022は新ソースのSaveが受理され結果待ちです。** [最新の実証拠](experiments/p021-visible-qualified-submit-unknown-p022-Save-accepted-20261012.json)。P021のRoot受理 `4eeafc57…3ec5f` は全3検査・12項目、2T4・各4モデル・fallback0、原latest V2の固定50:50／OA3項目の処理／各arm自身の成功集合でのrank／`%.9g` を確認し、全36値をFP32で独立再計算しました。作者最良0.956とlatest V2の未採点・自己Publicを区別します。17:24の競技Submit1回はHTTPErrorで受理不明・heldを保持。17:36の読み取り専用照会 `7cc6c894…27f92` は取得20件に対応refなし・残枠4ですが、追加ページ未読のため全履歴の不存在や受理拒否を断定せず再送0を維持します。SDK要求6項目は公式実装と一致し、複数pendingの一律禁止や今回の原因は確定していません。

P022 Source V2はbuilderの輸送codecだけを変更し、他8ソース・原5コードセル・元ZIP／全metadataのbytesとSHA、4モデル／family50／native20／mask／UID復旧を保持。joint LZMAのRoot査読 `97d8c344…604d5` と新しいgenuine Root V2 authorityを束縛し、最終APIソース946,315Bが内部950,000B上限とdefault gateを通過しました。17:37に手動Save1回、kernel138296990／V1／ref `rsraki/p022m-3f74be0b51a04af3b5b0c9ee99cf116b`、受付 `60a1e750…5f2ce`。実完走・保存版／出力照合・最終CSV・競技Submit・Publicは未受理です。旧Root V1の1,272,956B・local build HOLD・Save0は上書きせず保持します。

f006 fold0はE5 snapshot `2d509d78…f627`、固定12epochを継続。weak BCE 0.30152797、weak AUC12 0.84894347、累計1060更新／skip0、LR倍率1で、f005同epochとの差は数値観測に限り改善を主張しません。P020提出57098752は17:11の専用公式照会でraw状態・Public未返却、自己最高0.951です。本記録更新は通信・GPU・MRI・教師／モデル本文読込0、稼働中trainer／CLI／config変更0。厳密な0.960超えは未達・作業を継続します。以下は履歴です。

**10月12日17:11 JST：P019の実UID復旧を受理し、P021の保存実行は公式COMPLETE、P022はSave前のサイズ制限で停止しています。** [実証拠と次の順序](experiments/p019-qualified-p021-complete-p022-source-size-hold-20261012.json)。P019（kernel138289877／V1）はSource9の独立査読 `fe349696…7ac82` と同じソースで、故障注入64件が成功63・構造失敗1・252forward・約155秒、通常64件が失敗0・256forward・約134秒。Root実受理 `03ddfc5b…c7a2b` は原NoDICOMの出所、失敗UIDのforward0、全UID補完、失敗12文字列、成功4モデル・入力SHA・FP32平均・2T4・資源後処理を確認します。最終rank／mask／CSVはCPUの別証拠で、この実診断からServingや精度を受理しません。

P021公開latest V2は手動Save1回（kernel138292308／V1、受付 `5a674ae6…e2ac9`）後、17:11に公式CLIでCOMPLETEを確認。実出力・CSV・hidden完走・競技Submit・自己Publicは未受理です。P022はP019実受理metadata注入後のAPIソースが内部950,000B上限を超え、ローカルbuildでHOLD・Save0。元metadataを保持した圧縮Source V2を準備し、Rootの差分確認後に保存実行へ進みます。P020提出57098752は17:11公式Get／一覧でraw採点状態・Public未返却、SDK既定PENDINGを観測状態とせず、残り日枠4・自己最高0.951を維持します。

RTX4090のf006 fold0はsession38827、4／12epoch完了・running。weak BCE 0.30298274、weak AUC12 0.83599448、累計848更新・AMP skip0で、LR倍率はまだ1です。固定12epochと最早最小weak BCE選択を保ち、LR減衰の比較はepoch7以降、fold1判断はfold0完了後に行います。途中checkpoint採用・Gold／Publicによる学習選択0、trainer／CLI／config編集0。厳密な0.960超えは未達・作業を継続します。以下は履歴です。

**10月12日16:20 JST：P018の実256件検査をRootが受理しました。** [実結果と次の比較](experiments/p018-native256-qualified-p022-source-preparation-20261012.json)。kernel138273767／V1は公式COMPLETE。2枚のT4で実MRI256件・1,024forward、約633秒、fallback0、入力SHA・FP32 fold／family平均・後処理終了を確認し、推論動作の受理を発行しました。P020提出57098752は16:19にGet／一覧を再照合した時点でもスコア・明示状態未返却、残枠4・自己最高0.951を維持。P022はS6＋Native20のソース独立査読に合格し、次はP019の実fault64／normal64とP022全体のServing確認へ進みます。精度改善・hidden完走・厳密0.960超えは未証明です。

**10月12日16:15 JST：P022のS6基準＋Native20をソースのみで準備しました。** [準備記録](artifacts/research/20261012-p022-S6-native20-source-v1/report.json)。P019 V2のNative4モデル／family50／native20／MCL・Synovitis mask／UID構造的復旧を保ち、基準S5予測器を採点済み作者S6 V3の原5セルへ変更。Raptor実行と最終70:30セルは完全一致、compact Inputのみ公開V13へ変更し、他7役のInputは維持しました。cleanupの親SHA・5セルSHA・source variantを新コピーで束縛し、元の所有プロセス管理・GPU解放閾値・推論計算を保っています。source588,500Bと全Python／生成セルの構文確認まで完了し、API・GPU・MRI・モデル本文取得0。P020の可視3件の受理はコンポーネント参照に限り、P018実256・P019実64・P022 Servingの必要資格とruntime authorityはnull、独立ソース査読待ちです。自己Public・精度改善・実完走を発行せず、最高0.951とgoal activeを維持します。

**10月12日15:59 JST：P020の公式提出57098752は存在を確認済み、採点結果は未返却、残り日枠4です。** [実証拠と進捗](experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。Input13／1／1と保存kernel138279894／V1を維持し、全3検査・12項目・2T4・各4モデル・失敗0、70:30平均tie百分位rankの最終36値をFP32で独立受理しました。取得側の省略値・終了コード型の修正後は、元reader／held／8bodyの実SHAを保持してRootが再通信なしで検算し、再ダウンロード・ローカルモデル取得・MRI取得0です。日枠5と自己最高0.951を提出直前に公式確認し、手動Submit1回・再送0。15:59のGet／一覧は同57098752、raw状態・Publicなし、SDK既定PENDINGは未観測状態を埋めません。作者0.954は自己再現値ではなく、最高0.951とgoal activeを維持します。P018 native256とf006 fold0の履歴も保持しています。

**10月12日15:53 JST：P020の全可視結果を受理し、公式提出57098752へ進みました。** [実証拠と提出記録](experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。kernel138279894／V1、APIソースSHA `4a70f118…32503` とInput13／1／1を保持。Source・COMPLETEのraw／typed確認後、許可したCSV・receipt8件だけを取得し、全3検査・12項目、2T4・FP16、S6各4モデル、両枝とワーカーの失敗0を確認しました。原70:30の平均tie百分位rankと最終FP32を36値すべて独立再計算し、提出CSV SHA `2a174e07…eec68` が一致。終了コードを配列と仮定した取得側の誤りは原writerの辞書型に合わせて修正し、旧reader SHA／held／8GETを残してRootが通信なしで再受理しました。新規ダウンロード・モデル取得・MRI取得0です。提出直前に保存ソース・COMPLETE・raw Decimal GPU残量・自己最高0.951・日枠5を公式に再確認し、同V1を手動1回提出、再送0。自己Publicは未返却で0.951を保持し、作者0.954・Gold由来公開重みの履歴と区別します。P018の256件検査とf006 fold0学習の履歴は継続、厳密な0.960超えは未達でgoalはactiveです。

**10月12日15:38 JST：配布が復元された公開S6をP020として非公開保存し、実結果を待っています。** [正本と根拠](experiments/p020-public954-restored-S6-private-Save-current-research-20261012.json)。最新公開Codeをscore順100件・実行日順40件、14候補の保存sourceとlinked scoreで照合しました。0.954の4候補は原5セルまで同じS6／Raptor70:30で、独立した4改善とは数えません。P020は公開Input `compact-v1/13`・`SWA/1`・`Ortho/1`、Original31481／T4／OFF／private、306,470B。Rootの独立10項目を通過後、Input metadata3回・raw Decimal GPU残量98,637.918秒の確認1回・Save1回、競技Submit0です。作者0.954と自己測定を区別し、完走・hidden成功・Public改善は結果待ちです。公開重みのGold58選択・prior／anchor履歴を保持して監査し、自分のGold fit／選択／評価は0、Reportを推論特徴に使いません。Notebookの正確なlicense名は取得したendpointでは未確認で、元creditsを保持しモデル再配布0です。

P018 native256（kernel138273767／V1）はRootの直近観測でRUNNING、64件の成功から256・Servingの資格を転用しません。RTX4090のf006 fold0はE1完了：weak BCE 0.37770310、AUC12 0.74299619、212steps・skip0、1783.93秒。旧同fold E1との差はBCE約+2.1e-6／AUC約+2.5e-6で、終盤LR減衰の効果はまだ判断できません。固定12epochと最早最小weak BCE選択を維持し、途中採用・trainer／CLI／configの追加変更0です。自己最高0.951、厳密な0.960超え未達のまま並行継続します。以下は履歴です。

**10月12日14:54 JST：P018のremoteバイト・64件検査が合格し、256件検査とf006 fold0の学習を並行して進めています。** [実結果と実行状況](experiments/p018-native64-f006-fold0-progress-20261012.json)。REG2件はそれぞれ193memberの一致を確認。native64（kernel138271136／V1）はMRI64件・256forward・154.986秒・2T4・fallback0でRootの実資格を発行しました。native256（kernel138273767／V1）は保存受理済みで、実結果は未確認です。Serving未実行・競技Submit0、今回のPublicは未評価です。P016・P017の公式Public 0.951を自己最高として保持し、厳密な0.960超えは未達、goalはactiveです。

f006はRTX4090のrun `artifacts/runs/f006-late-cosine-fold0-20261012-v1`、session38827でrunningです。weak訓練3,386件・検証821件を固定し、Epoch1〜6はLR一定、7〜12はcosineで初期値の10%まで減衰します。全144窓とprocessed115のGPU QA、resource gateは合格。epochごとのmetricsと重みSHAを記録します。P018のPublic測定後に学習する順序を変更し、Kaggleの待ち時間と約6時間の独立した学習を並行させましたが、Public改善は未証明です。Gold／Publicを使う学習・選択・調整は0、固定12epoch完了後に同foldのf005と比較してfold1を判断します。今回の記録更新ではtrainer・CLI・config・P018の推論計算を変更していません。

**10月12日13:44 JST：0.960超えを目標に再開し、既存提出の結果と本日の枠を確認しました。** [現状・根拠・次の順序](experiments/goal-resume-analysis-20261012.json)。P016（57044321）・P017（57042712）は公式COMPLETE／Public 0.951／エラーなし。P014・P015は隠し再実行エラー・スコアなしであり、成功扱いにしません。13:36の残枠は5回、GPU残量101,707秒（約28.3時間）。Public LBの取得50件は最高0.966、0.960超27件で、全参加数・自分の順位・メダル境界は未推定です。

ユーザーが新REG2件・4ファイル計208,822,289Bのprivate送信を明示承認し、PUT4／Create2を実施。作成応答のref欄省略で旧確認コードは停止しましたが、再送せず各3回の独立照会で両方のprivate／V1／READY／正2ファイルを確認しました。新Dataset IDは12526561／12526584。元の未確定Create記録を保持し、remote byte・64／256件・Serving・新提出は未検証です。次は既存P018のCLS／REG固定50:50、基準80:自作20、MCL／Synovitis基準復元を実測します。弱検証fixed10 AUCの増分は両fold +0.004344／+0.006284ですが、Public改善は未証明です。P017は全体復旧経路があるため同点だけではhiddenのnative完走を断定しません。後続はUID単位の構造読込復旧、epoch別重み保持、学習率減衰だけの比較を順次検討します。Gold学習・選択0、自己最高0.951、厳密な0.960超えは未達、goalはactiveです。以下は履歴です。

**10月11日01:08 JST：目標は未達のまま、承認・外部状態待ちのblockedへ更新しました。** [停止条件と再開手順](experiments/p018-specific-upload-three-turn-blocked-audit-20261011.json)。新REG2件の具体的送信承認が3連続ターンで未回答、準備済みソース・独立確認は完了しています。新送信・作成0で、承認なしに実入力→64／256件→提出推論へ進みません。P016v5／P017v9の公式照会と同P016番号の再確認v6はHTTP401でした。認証の原因は未確定で、提出失敗・ジョブ停止とは扱わず、再起動・再提出0です。前回00:57の一覧では2提出の存在とスコア未返却を確認済。具体的承認とKaggle API認証の復旧、または公式結果が届いたら同じ目標で再開します。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日01:00 JST：追加学習は、準備済みの候補をPublicで測定してから判断します。** [学習曲線と公式照合](experiments/p018-next-training-decision-official-list-crosscheck-20261011.json)。4モデルとも採用epoch後に訓練損失は下がり、検証損失は悪化しているため、epoch延長や学習範囲拡大を今始める根拠は不足しています。fold2・last4は現在の学習入口の資格外です。P016・P017は00:57の公式提出一覧でも存在を確認しましたが、スコアと明示状態は未返却。確認APIの指定項目・endpointに不足は見つかりません。新REG2件は具体的な送信承認の回答待ち・作成送信0。ソース準備は完了し、次は入力バイト検証→実MRI64／256件→提出推論確認です。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日00:48 JST：P018の入力確認を実APIの形式に合わせて修正し、次の実行用ソースを準備しました。** [根拠と次の順序](experiments/p018-metadata-api-contract-fixed-source-ready-20261011.json)。既存の正常なprivate V1を公式APIで1回読み、Dataset情報からファイル一覧が省略される形式を確認しました。専用のV1一覧APIへ分けた修正版は独立12項目、提出用の親は20項目、保存入口は12項目、64／256件の入口もソース査読を通過しました。過去の隠し提出エラー全体の原因を確定したものではありません。実CPU4・人工GPU8回の成功を保持し、新REG入力・Kaggle実MRI・提出推論は未検証です。新規privateアップロード2件の具体的承認は回答待ち、送信・作成0。P016・P017の00:45公式照会はスコア・明示状態未返却。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日00:12 JST：P018の新規非公開モデル2件は、今回の宛先・内容への明示承認待ちです。** [現状と追加調査](experiments/p018-private-upload-approval-pending-current-public-research-20261011.json)。自動承認審査が新UUID宛の未公開モデル・コード約209MBの送信を停止したため、質問済みの回答を待っています。作成API・送信0、以前のCLS fold1承認とは別です。実CPU4・人工GPU8回の成功は保持し、アップロードを待つ間に新しい提出用の親と64／256件診断のソースを準備します。新REG Inputのバイト検証入口は独立12項目、phase契約は11項目のソース査読を通りましたが、Kaggle実行資格は未発行です。追加確認した公開4-arm現V57の作者スコアは0.936で、自己0.951を超える根拠はありません。P016・P017は00:10の公式照会でもスコア・明示状態未返却。自己最高0.951・厳密な0.960超え未達です。以下は履歴です。

**10月10日23:51 JST：P018の実4モデルは、RTX4090の人工入力確認も成功しました。** [実GPU確認の記録](experiments/p018-mixed-realGPU4-synthetic-QC-20261010.json)。元の選択済み4モデルで144窓と欠損96窓を各4回、計8回推論し、全画像欠損4拒否・encoder入場0・全6組の共有なし・推論前後の重み一致を確認しました。観測最大allocated 653,356,544 B／reserved 767,557,632 B、検査全体6秒、独立10項目・Root12項目が合格。これは人工画像と1枚の4090の確認で、2枚のT4・実MRI64／256件・提出推論・Public改善は未検証です。次は新規非公開REG2 Inputの登録とバイト検証、その後Kaggleで新しい4モデルの64／256件と親の復旧経路を確認します。P016・P017は23:44の公式照会でもスコア・明示状態未返却、自己最高0.951・厳密な0.960超え未達を維持します。以下は履歴です。

**10月10日23:42 JST：P018の4モデル再構築が実CPUで成功しました。** [実確認とアップロード停止の記録](experiments/p018-mixed-realCPU4-bytecode-fixed-upload403-20261010.json)。Pythonが封印ツリーへキャッシュを追加する起動不具合を修正し、モデル・計算式・厳密なファイル検査を保持しました。新しい展開先の193ファイル×2を元のハッシュと照合し、4重みの厳密読み込み・6組の共有検査・Root9項目を通過。forward・MRI・GPU・教師・Report・Gold利用は0です。4モデルの人工GPU／Kaggle64・256件／提出用統合は未検証です。非公開アップロードは作成前に停止し、追加の公式1回照会はHTTP403でした。存在しないDatasetとは判定せず、新規作成専用APIを確認します。P016・P017は直近23:20の照会でも公式スコア・明示状態未返却、自己最高0.951・厳密な0.960超え未達です。以下は履歴です。

**10月10日23:09 JST：P016修正版の公式提出57044321が受理されました。** [提出の記録](experiments/p016-overlay-Submit57044321-20261010.json)。実完走・CSV一致を確認したkernel138035248/V1を、提出入口の独立14項目と本物の通信なし確認後に提出しました。直前のSource・COMPLETE状態・GPU残量・提出枠を各1回確認し、提出1回・再試行0です。提出前の公式枠は2回、提出後の残り1回は推定で、改めて提出直前に確認します。hidden完走・スコアは未確定です。両Regionalモデルの新規非公開アップロードと4モデル混合のCPU/64/256・統合検証を次に進めます。自己最高0.951・厳密な0.960超え未達を維持します。以下は履歴です。

**10月10日23:01 JST：P016修正版の保存実行が正常完了し、提出前の実確認を通りました。** [完走と両foldの記録](experiments/p016-overlay-visible-complete-regional-both-folds-20261010.json)。kernel138035248/V1の公開3検査についてSource8項目・出力18件53,900 B・数式と後処理17項目・独立10項目を確認し、最終CSV830 Bも原bytes一致です。追加推論24.87秒、代替予測0、後処理の解放1回を確認しました。hidden完走・Publicは未確認で、提出入口を準備中です。Regionalは両foldとも固定12epoch完了、最小weak BCEでfold0 epoch8・fold1 epoch10を選択しました。固定50:50平均の弱検証10項目AUCは旧CLS比でfold0 +0.004344・fold1 +0.006284、観測確率BCEも両foldで改善しました。両モデルの厳密CPU再構築と推論ファイルの作成を完了しましたが、4モデル統合のKaggle検証・新規非公開アップロードは未実施です。P017提出57042712は22:56の公式応答でもスコア・明示状態未返却です。自己最高0.951・厳密な0.960超え未達を維持します。以下は履歴です。

**10月10日22:40 JST：P016修正版を新規非公開kernel138035248・Version 1で保存実行しました。** [保存実行と補完性の記録](experiments/p016-overlay-Save138035248-regional-mean50-20261010.json)。後処理へ渡す環境情報の形式だけを厳密に変換する修正で、旧21ソース・モデル・元の予測セルを保持しています。ソース査読13項目、保存入口の査読10項目、本物の成果物4件とSDK変換を確認し、入力4件・GPU残量101,969.405秒を直前照会してSave1回、競技提出0回です。修正版全体の実完走資格は、まだ取得していません。Regional fold1の弱検証では新旧モデルの固定50:50 FP32平均により固定10項目AUCが旧0.880062450から0.886345997、観測確率BCEが0.298079359から0.283321486へ改善しました。係数探索・Gold利用は0で、fold0完了後に同じ比較を行います。Public改善は未証明です。公開Source追加2件は既重みの再利用で、新checkpointの根拠はありませんでした。P017提出57042712は22:35時点で公式スコア未返却、自己最高0.951・厳密な0.960超え未達です。以下は履歴です。

**10月10日22:16 JST：P016の保存実行が失敗した原因を、当該Version 1の実ログで確認しました。** [失敗原因と学習完了の記録](experiments/p016-serving-platform-error-regional-fold1-complete-20261010.json)。kernel138030227の基盤3検査の推論完了後、追加推論前の後処理で、親の環境情報と再利用したcleanupの項目名・機種表記が違うため厳密比較に拒否されました。時間超過とは判断せず、接続部分だけの修正を準備しています。旧ソース21件とCPU・64件・256件の実診断は保持し、新しい提出用統合には改めて保存実行が必要です。過去のhidden提出エラーまで同じ原因とは断定しません。P017提出57042712は22:15時点でも公式スコア・明示状態が未返却です。Regional fold1は12epoch完了、弱BCEによる選択epoch10、BCE0.277968890、AMP更新スキップ累計2回。完了検査を通りましたが新モデルのKaggle実行資格は未取得です。fold0は11/12epochで学習を継続しています。自己最高0.951、厳密な0.960超えは未達です。以下は履歴です。

**10月10日21:55 JST：正常版P017の公式提出57042712が受理されました。** [提出とP016保存実行の記録](experiments/p017-normal-Submit57042712-p016-Save138030227-20261010.json)。保存済みkernel138018511・Version 1・submission.csvを、正常結果と実故障復旧の確認、提出ヘルパーの独立査読14項目、全体のローカル確認後に提出しました。直前のSource・COMPLETE状態・GPU残量・提出枠をそれぞれ1回確認し、提出APIは1回、再試行0です。提出前のヘルパーのフォルダ指定と承認記録の欠落で止まった旧記録は保持し、保存済みNotebookや予測計算は変更していません。採点待ちで、スコア・hidden成功は未確認です。P016も新規非公開kernel138030227・Version 1の保存実行が受理されました。Input4件、GPU残量102,220.268秒を直前確認し、Save1回・競技提出0回です。完了出力を確認してから次の提出を判断します。Regional学習は固定12epochを継続中。自己最高0.951、厳密な0.960超えは未達です。以下は履歴です。

**10月10日21:42 JST：P017の故障復旧とP016のCPU・64件・256件検証を実結果から確認しました。** [実検証と提出用ソースの記録](experiments/p017-fault-root-accepted-p016-phase3-materialized-20261010.json)。故障診断は追加モデル読み込み後のInvalidDicomError、所有プロセス終了、元CSV633 Bの完全復元を確認し、検査24項目・独立確認12項目に合格しました。JSONの参照先ミス1か所を修正して保存済み15出力を再解析し、追加API・GPU実行は0です。失敗した推論の一時ファイルは残り、モデルの生存状態は未観測です。この限定的な復旧確認から、全hidden障害への対応や精度改善は断定しません。正常版P017の提出準備を進めます。P016の256件は追加推論133.86秒、fallback0、独立確認8項目に合格し、同じcheckpoint・新ソースの3段階検証を受理しました。提出用ソースを生成しましたが、保存実行・競技提出はまだです。Regionalはfold0 10/12・fold1 11/12epochで、同epoch旧CLSより弱BCEは低く固定10項目AUCは低い状態です。fold1 E11でAMP更新スキップ1回を記録し、部分採用せず12epochを続けます。自己最高は0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日20:52 JST：P017の可視3検査は正常に完了し、予測計算の照合に合格しました。** [正常枝の実確認と故障診断の保存記録](experiments/p017-healthy-visible-accepted-fault-Save-native256-running-20261010.json)。kernel138018511・Version 1のSource8項目、23出力、独立レビュー11項目を確認し、基盤70:30、追加モデルの平均、80:20の順位処理、MCL・Synovitisの基盤値維持を再計算して数値差0、最終CSV834 Bの完全一致を確認しました。復旧は使われていません。これは可視3検査の確認で、hidden完走・精度改善は未確認です。壊れたDICOMを追加推論へ渡す専用非公開診断をkernel138022868・Version 1で保存し、モデル読み込み後の失敗、所有プロセス終了、封印済み基盤CSVの完全な復元を実証します。この診断はCSVを診断名へ変更し、競技へ提出しません。正常版P017の提出も、その実証を待って進めます。P016の256検査もkernel138020795・Version 1で保存実行済み、取得器の確認10項目に合格しましたが、実256検査の結果確認は未完了です。学習は固定12epochを継続し、途中モデルは採用していません。自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日20:24 JST：P016の64検査は実行・結果確認を通過しました。** [実64検査と256検査の準備の記録](experiments/p016-actual-native64-qualified-native256-materialized-20261010.json)。保存kernel138015216・Version 1のSource全8項目と公式COMPLETE、必須10出力の取得ハッシュ、独立した実結果確認8項目を照合済みです。直接224画素・full94の追加推論は64件を約43.21秒で完了し、fallback0、12項目の有限予測、2枚のTesla T4、基準CSV維持、基準処理の清掃1回と子プロセス終了を確認しました。追加推論のTorch最大予約メモリは各8,600 MiB、プロセスRSSの標本合計は約12.47 GBで、共有ページの重複を含み厳密なピークではありません。取得側の旧100件ページ制限は187出力に対応できず停止していましたが、同じ実行結果を回収でき、GPU再実行はしていません。新256検査は本物の64証拠と別途固定した元基準64証拠を入力したsource864,077 Bで準備済み、実Save・実256の確認はこれからです。P017はkernel138018511・新規非公開Version 1のSaveを受理済みで、正常時と復旧時の分岐別の結果確認を準備中です。並行学習はfold0 E8・fold1 E9、skip0、同epochの旧CLSより固定10項目AUCは−0.003813・−0.003962、weak BCEは+0.006544・+0.003491で、途中採用せず固定12epochを継続します。動作確認は精度改善の証拠ではなく、自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:55 JST：起動設定を修正したP016の64検査を、新しい非公開Notebookで保存実行しました。** [実Saveと並行作業の記録](experiments/p016-actual-native64-V2-Save-p017-source-20261010.json)。kernel138015216・保存Version 1・修正版source695,688 B。実Input4件とGPU残量106,748.302秒を直前に確認し、Saveは1回、競技Submitは0回です。モデル・原5セル・CPU復元・予測式は維持しています。実行結果を取得する新しい取得器も12項目の確認を通過し、実64/256・Serving・精度の確認は完了待ちです。並行学習はfold0 E7・fold1 E8でskip0。旧モデルの同epochに対し固定10項目AUCは−0.007869・−0.009082、weak BCEは+0.004657・+0.000997で、途中採用はせず固定12epochを継続します。追加モデル失敗時の基準CSV復元を扱うP017は、本物の認可・査読情報を入れたsource949,769 Bで容量確認済みですが、実Serving・Save・Submitはまだです。自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:32 JST：診断ログの取得まで完了し、別の取得側の問題も切り分けました。** [実診断の記録](experiments/p016-native64-actual-diagnostic-Input-order-20261010.json)。Kaggleの保存ソースは全15セル・実行環境・4つのInput参照が一致していましたが、APIがInputの表示順を入れ替えるため、取得側の順序一致チェックだけが失敗していました。元のfalse判定は保存し、新しい取得器では重複・追加・Version 2を拒否する4参照の集合照合に変更します。実ジョブの失敗原因は別に確認したPython起動時のsitecustomize衝突です。モデル・原5セル・CPU復元を維持した最小修正版の独立査読後、新しい64検査を実行します。新たな64/256・Serving・精度の確認はまだ済んでいません。自己最良0.951、0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:23 JST：P016の64検査は追加処理のPython起動段階で失敗し、原因を実ログで特定しました。** [失敗記録と最小修正](experiments/p016-native64-startup-exception-baseline-guard-20261010.json)。元0.951の基準処理は64件・fallback0で完了し、清掃1回まで成功（約765秒）。追加childの検索パスに元基準専用sitecustomizeを混入したため、基準用routing環境がない新childが0.114秒で終了しました。追加モデル・画像読込は未実行、64/256・Serving資格は未取得。v1ソースと証拠を保持し、モデル・原5セル・CPU復元を変えず、参照先を除去するwrapper v2と実Python起動の再発テストを準備します。取得器もexact_source検査で別HOLDになり、原因を分けて調査中です。P014/P015のhidden原因は依然不明で、そのServingに同じ混入パスはありません。P017の明示的な基準予測への切替と固定12epoch学習を並行し、自己最良0.951・0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日19:07 JST：学習はfold0 E6／fold1 E7、P016の64検査は実行中です。** [学習比較とP017の事前記録](experiments/f005-regional-E6E7-p017-runtime-policy-20261010.json)。両foldともskip0で固定12epochを継続。旧CLSの同epochよりweak BCEは+0.003856／+0.012150、固定支持数10項目の平均AUCは−0.004308／−0.010178です。fold0のAUC12は高いものの少数ラベルの影響があるため改善とは認定しません。P017は、元0.951の予測完成・封印・清掃と追加処理の安全な終了を条件に、追加Nativeの実行失敗時だけ元CSVの原bytesへ戻す処理を別の一要因として事前登録しました。モデル・係数・正常時の計算は変更せず、元予測の失敗・入力やモデルの整合性違反は隠しません。P017はソース準備中で実資格・Save・Submit未取得。現在の自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日18:56 JST：P016の新64検査診断をKaggleの非公開Version 1で保存実行しました。** [実Save記録](experiments/p016-actual-native64-private-Save-20261010.json)。kernel138009389・source695,007 B・Original31481・T4x2・Internet OFF・指定4 Input V1。実Input確認4回とGPU残量確認1回の後、Save1回を受理（残量107,531.917秒）。元0.951の5セルと新CPU証拠を固定し、ソース確認12項目と単発Save確認14項目を通過しています。実64の完了・予測・メモリ・保存コードの照合はこれからです。256・Serving・競技提出資格は未取得、競技Submit0。P014/P015はhidden例外・スコアなし、同一再提出は行いません。固定12epochの学習を継続し、自己最良0.951・0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日18:48 JST：P014・P015は両方ともhidden再実行で例外となり、スコアが出ませんでした。** [結果と次の実験](experiments/p015-hidden-exception-p016-native64-authorized-20261010.json)。COMPLETEという状態名でもerrorDescriptionがあり、採点成功とは扱いません。例外の型・場所・実実行時間は不明で、同じNative構成の再提出は進めません。Servingで既知Gold/train UIDとの重複を拒否する処理は見つからず、根本原因は調査中です。P016は同じSWA重みの直接224px切り出しだけを変える候補で、新CPU復元確認と64診断ソース22項目の確認後、元0.951の全5セルを保つ64検査版を準備しました。695,007 B、private・Original31481・T4x2・Internet OFF・4 Input V1です。実64/256・提出資格はまだ未取得。固定12epochの学習は継続し、途中checkpointは採用しません。自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日18:38 JST：P016の新しい実CPU確認が合格しました。** [実結果と学習比較](experiments/p016-actual-CPU-qualified-regional-E5E6-20261010.json)。Kaggleの保存コード・private/CPU/OFF/Original31481・SWA Inputと、293MBの固定重みのSHAを照合。486テンソルを厳密に復元し、独立15項目・root3項目を確認、モデル1・forward0・MRI0です。取得器が空の成功ログを拒否した問題はv2だけで修正し、実GETの0バイト/空SHAも確認。Notebookの再実行・再Saveは行っていません。GPU64/256・提出・hidden精度は未確認で、次は新CPU資格を接続した元5セル保持の64件診断です。局所モデルはfold0 E5・fold1 E6、skip0で固定12epochを継続。同epoch CLSとの差は固定10項目AUCで−0.005331/−0.010638、BCEも悪いため改善と判断しません。P014はhidden例外、P015は18:28時点で未返却。自己最良0.951、厳密な0.960超え未達です。以下は各時点の履歴です。

**10月10日18:26 JST：P016の実CPU診断を非公開Version 1で保存実行しました。** [実Saveと確認範囲](experiments/p016-actual-private-CPU-Save-p014-hidden-audit-20261010.json)。kernel138006054、GPU無効・Internet無効・Original31481、公開SWA V1だけを接続。新ソース14項目・CPU Notebook18項目の独立確認と実SDK設定検査を通過し、Save1回を受理、公式CLIではCOMPLETEを確認しています。実モデル復元の出力・保存コード・Inputの照合はこれからで、CPU/64/256/提出資格はまだ認定していません。次は新CPU結果を確認し、元0.951版の5セルを保った64件診断へ進みます。fold1学習は6/12 epochまで継続。P014はhidden例外・スコアなし、P015は18:17時点で未返却。追加Nativeの入力・期限等の境界差は確認しましたが、hiddenの具体的原因は未確定です。自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日18:07 JST：P014は隠しデータの再実行で例外となり、スコアは出ませんでした。** [公式応答と次の確認](experiments/p014-hidden-rerun-exception-p016-source-hold-20261010.json)。公式statusはCOMPLETEですがerrorDescriptionがあり、採点成功ではありません。表示されたAssertionErrorは取得ツール側のスコア検査で、隠し実行の例外型ではありません。可視3件・CPU・64/256件の成功からhidden完走を保証できず、時間超過・OOM・入力欠損のどれかはまだ断定できません。再提出せず、追加Nativeの入力・デコード・期限・検査単位の失敗条件を監査します。P015は結果未返却です。P016は単独native64/256入口のimport不足を独立確認し、v1をHOLDで保存して最小修正のv2を準備中。数値型の曖昧な説明は[別記録](experiments/p016-numerical-type-wording-clarification-20261010.json)で訂正し、旧helper・90:10・all12・元基準コードは変更していません。固定12 epochの学習は継続、自己最良0.951・厳密な0.960超え未達です。以下は各時点の履歴です。

**10月10日17:54 JST時点：局所モデルはfold0 4／12・fold1 5／12epochまで進み、両方例外・更新skipなしで継続しています。** [最新実観測と公開候補照合](experiments/f005-regional-epochs45-versia7-duplicate-20261010.json)。weak BCE0.312486／0.309305、AUC12 0.824600／0.831306です。同epochの旧CLSより固定支持10項目のAUC平均は−0.011181／−0.011783、weak BCEも高く、改善はまだ認定しません。新P016のソース組立・独立査読と、その後の新CPU→64→256確認を進めています。追加取得したVersia 7は、セル境界の改行以外の5コードセルが現在の基準と一致し、新モデル取得へ進む根拠はありません。P014／P015は17:54の公式照会でもスコア・エラー未返却。自己最良0.951、厳密な0.960超え未達で作業を継続します。以下は各時点の履歴です。

**10月10日17:29 JST：既存SWA V1の別224px重みを使うP016候補のソース準備を開始しました。** [事前固定した条件と既提出の確認](experiments/p016-direct224-onefactor-preregistration-20261010.json)。同じ重みの旧P011は自己Public0.950で改善なしでした。今回は中央307pxから224pxへのresizeを、中央224pxの直接切り出しへ変え、旧singleview・full94等は保持します。現在の自己0.951の元5セル・70:30基準予測を保護し、全12項目を基準90%・新枝10%の順位融合に固定しました。Gold由来の項目別係数は移植せず、未返却のP014／P015スコアを見る前に条件を記録しています。新Inputや学習は不要ですが、新CPU／64／256件検証が必要で、旧cropやRegionalの実行資格は転用しません。新ソース・実推論・Save／Submitは未完了です。P014／P015は17:29でもスコア・エラー未返却、自己最良0.951を維持します。

**10月10日17:18 JST時点：局所モデルはfold0 3／12・fold1 4／12epoch、修正版の統合Notebookはソース査読15項目に合格しました。** [最新進捗・ソース検査・公開候補の記録](experiments/f005-regional-epochs34-integrated-v2-public-source-20261010.json)。weak BCEは0.339705・0.327385、AUC12は0.808560・0.806197、両foldともAMP skip0です。固定12epochを継続し、途中checkpointは採用していません。新Notebook v2は最終mask失敗時の途中CSV撤去を修正し、数式・元5セル・66ソースを維持しました。送信ソースは671,063バイトで、950,000バイトという内部容量ガードの範囲内です。ソース検査の合格であり、実モデル・CPU／64／256件・提出推論の確認はこれからです。

同fold・同epochの旧CLSと比べた固定支持10項目のAUC差は−0.014212／−0.025869で、weak BCEも両foldとも悪く、現段階の改善は未確認です。追加の公開候補DINOsaur V5は主CoAtNet重みが現在の0.951基準と同一で、現版のPublicスコアは未確認でした。別の224px cropモデルをLateral OAだけ13%加える枝はありますが、補完性は未実証で、Goldによる過去の選択・入力版の未固定・例外時の代替予測も残るため、候補全体への置換は保留します。モデル取得・実行は0です。P014・P015は17:18の公式観測でもスコア・エラー未返却。自己最良0.951を保持し、厳密な0.960超えを目指す作業を継続します。以下は各時点の履歴です。

**10月10日16:50 JST時点：局所モデルはfold0 2／12・fold1 3／12epochまで進み、学習を継続しています。** [実観測と公開資料調査](experiments/f005-regional-epochs23-public-search-20261010.json)。weak BCEは0.343500・0.331905、更新skip0。現段階では両foldとも旧モデルの同じepochより悪く、改善は未確認です。今回調べた公開一次資料3候補からも、取得へ進める0.955以上の候補は確認できませんでした。統合Notebook v1は正確な送信ソース669,729バイトですが、最終mask拒否時に直前のCSVが残る経路でソース査読HOLD。最終呼出だけを修正するv2を準備中で、実モデル・提出の資格は未取得です。P014・P015は16:49の公式観測でもスコア・エラー未返却。自己最良0.951を保持し、厳密な0.960超えを目指す作業を継続します。

**台帳の評価種別を訂正しました。** [訂正記録](experiments/ledger-metric-source-correction-20261010.json)。Gold欄に誤記されていたweak AUC3件を、新設した末尾の`weak_macro_auc_12`列へ同じ数値文字列で移しました。台帳は15列です。既知のGold評価2件・他の全セル・元の実験JSONとrunは保持し、訂正前のCSVも保存しています。移した値はいずれもfold1で、epoch9の選択値とepoch1・2の途中値であり、2fold平均ではありません。

**10月10日16:28 JST：局所特徴モデル用の2列support mask接続は、ソース検査12項目を通過しました。** [記録](experiments/f005-regional-support-mask-peer12-20261010.json)。MCL・Synovitisを基準値に戻し、他10列のuniform20文字列を保持する元のCSV処理・支持数ルールは維持しています。新しいRegionalの完走記録と同じ認可を要求する接続を確認しましたが、新Notebookの組立・実CPU／64／256件・提出推論は未検証です。fold0は1／12、fold1は2／12epochで学習継続中、例外なし。P014・P015は16:28時点でもスコア・エラー未返却。自己最良0.951を保持し、12epoch完了後の評価と厳密な0.960超えを目指す作業を継続します。

**10月10日16:17 JST：新しい親推論接続の独立ソース査読22項目が合格しました。** [接続の記録](experiments/f005-regional-parent-peer22-20261010.json)。基準予測の5セル・時計・所有ファイルの保護・clear1回・平均・CSVの数学を維持し、旧CLS資格を転用せず専用Regionalの転送と推論を接続する実装です。実モデル・Kaggle資格は未成立で、2列のsupport maskを新しい完成記録へ接続する作業が残っています。両学習はfold0 1／12、fold1 2／12で継続中、P014／P015は16:16でもスコア・エラー未返却。自己最良0.951と厳密な0.960超えを目指す作業を継続します。

**10月10日16:10 JST：局所モデルはfold0が1epoch、fold1が2epochまで進み、両方例外なしです。** [実epoch・転送ソースの記録](experiments/f005-regional-parallel-epochs12-transport-source12-20261010.json)。fold0はBCE0.377602／AUC12 0.761979、212更新・skip0。fold1はBCE0.342996／AUC12 0.757316、各208更新・skip0。履歴の時間は累積値で、並行時の実epoch間隔は約37分でした。全12epochの短縮幅は未証明。初期のfold0 AUC上昇は陰性1件のSynovitisに依存し、以前から支持数不足として除いたMCL／Synovitis以外の10項目では両foldとも改善未確認です。途中checkpointは採用せず固定12epochを継続します。専用BIN転送ソースは独立12項目合格、親推論接続はソース準備中で実CPU／64／256／提出の資格は未成立。P014／P015は16:06でもスコア未返却、自己最良0.951、厳密な0.960超えは未達です。

**10月10日15:43 JST：局所モデル専用の推論ソース24項目が合格し、Notebook容量候補は698,443バイトになりました。** [ソース・容量の記録](experiments/f005-regional-driver-source24-capacity-20261010.json)。最初の圧縮packetでは改行変換により元ファイルのハッシュが変わったため、失敗記録を保持して修正し、6ファイルの復元バイト一致を独立確認しました。不要な旧CLS用情報を外して共通コードと基準予測の5セルを維持し、API上限950,000バイトまで251,557バイトの余裕を実測しました。これは容量だけの候補で、Regional用の転送・後段接続と実CPU／64／256件推論は未完成、実行・提出できません。2foldは固定12epochで学習継続中。fold1の初期BCE0.3769は従来0.3683より悪く、改善をまだ主張しません。P014／P015は15:33の公式照会でも採点待ち、自己最良0.951、厳密な0.960超えは未達です。

**10月10日15:23 JST：局所特徴fold1の1epoch目が完了し、同じ設定のfold0を別プロセスで並行学習中です。** [実epochと並行開始](experiments/f005-regional-epoch1-parallel-fold0-started-20261010.json)。fold1はweak BCE0.376899／AUC12 0.706897、208更新・AMP skip0、観測最大reserved約1.20GB、epoch約1,881秒。初期結果なので途中checkpointは採用しません。fold0は3,386／821件、fold1は3,316／891件、source13・CLI2・config・教師・cache・Gold除外は同じです。PCの空き資源を確認して同GPUで重ねましたが、並行時の全期間ピークや短縮幅は未証明。両runは例外なしで固定12epochを継続します。P014／P015は15:23のAPIでも採点待ち、自己最良0.951、厳密な0.960超えは未達です。

**10月10日15:16 JST：局所モデル専用の保存・単fold CPU復元ソースが独立38項目を通過しました。** [準備・調査の記録](experiments/f005-regional-sealer-source38-medical-lead-20261010.json)。未完成run・旧CLS・人工資格は重みへのアクセス前に拒否し、完成12epochの最小weak BCE選択後に実CPU検査と元checkpointのバイトを維持した封印へ進む実装です。現時点はソース準備の合格で、実regional CPU・封印・Kaggle推論・精度の資格は未成立。fold1は実学習中で最初のepoch評価待ち、P014／P015は15:12のAPIでも採点待ち、自己最良0.951を保持します。OrthoFoundation-MSK-5Mの固定commit・公開LFS情報も確認しましたが、モデル本体の取得・配布条件・この大会での効果は未確認で採用していません。

**10月10日14:50 JST：専用局所特徴モデルのfold1をRTX4090で実学習中です。** [開始の根拠と実記録](experiments/f005-regional-dense-QA50-fold1-started-20261010.json)。固定12epoch・seed20261002・同じweak教師とgroup foldを使用し、学習3,316件／検証891件。新しいrunへ保存し、完了後の最小weak BCEでcheckpointを選びます。元DICOMの一括取得・新しい画像キャッシュ作成・Gold fit/selectionは0です。

局所表現は各窓のCLS＋16領域のpatch平均で、headと学習パラメータ数は維持します。省メモリVJPと通常のdense双方で微小勾配の再現差があり、AdamWによる増幅を確認しました。VJP固有の原因という解釈は撤回し、失敗3件を保持。新しいdense経路でCUBLASとPyTorchの決定的GPU計算を初回CUDAより前に設定し、人工50項目・独立33項目・両foldの事前照合を通過しました。144窓・全3,633,816学習パラメータ・effective16／10／4の勾配、重み、optimizer状態は差0。許容差・学習率・epsilonは変更していません。旧CLSとは実行設定にも差があるため、今後の差を表現だけの効果とは断定しません。

P014提出57031186とP015提出57031981は14:39の公式照会でスコア・エラー未返却でした。自作局所モデルの実学習完了・CPU復元・64／256件推論・提出・精度改善はまだ未確認で、旧資格は転用しません。自己最良0.951、0.955／厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日14:30 JST：P015は可視実行・独立58項目を通過し、公式提出57031981が受理されました。** [実結果と次の検証](experiments/f005-p015-submitted-regional-QA-held-20261010.json)。保存V1/kernel137973395の実22本文を取得し、基準2列の6値・他10列の30文字列・最終834バイトが完全一致。元モデル・前処理・2fold平均は維持、代替0・所有プロセス終了・clear1回も確認しました。提出直前の残枠は4件、API提出は1回のみ。P014提出57031186とP015はスコア未返却で、自己最良0.951を保持します。

次の一要因候補はCLSに16領域のpatch平均を加える局所特徴です。同じweak教師・fold・seed・last2・12epoch設定を使う専用学習経路を実装しましたが、実学習は未開始。人工RTX4090検証は144窓のlogit・masked BCE・全学習パラメータの勾配一致を通過し、effective16のAdamW更新後の一致で停止しています。失敗記録を保持し、許容差を安易に広げず原因を調査します。Gold・Publicによる自作checkpoint選択は0、公開重みのGold由来は別管理、hidden時間上限と厳密な0.960超えは未確認です。以下は各時点の履歴です。

**10月10日14:06 JST：P015の非公開V1/kernel137973395がRUNNINGです。** [28番Notebook](notebooks/public/28_submit_p015_native_support_mask.ipynb)の変更はMCL・Synovitisの2列を元のP012基準予測へ戻す一要因です。他10列は完走したuniform20出力をそのまま保存し、モデル・前処理・2fold平均は変更しません。両foldの正負支持数が各20件以上という閾値は、今回weak教師から立てた将来仮説で、過去の事前規則ではありません。[保存・ソース照合記録](experiments/f005-p015-native-support-mask-running-20261010.json)。P014提出57031186は公式スコア未返却です。P015の完走・CSV検証・競技提出・Publicも未確認。旧37項目合格は旧uniform20の資格であり、新CSVへ転用しません。自己最良0.951と元Notebookを保持し、厳密な0.960超えは未達です。

**10月10日13:41 JST：P014の提出が公式APIで受理されました。提出番号57031186、採点待ちです。** [提出受理と検証](experiments/f005-native-twofold-p014-submitted-20261010.json)。[27番Notebook](notebooks/public/27_submit_p014_native_twofold_compact.ipynb)は正常完走し、CSV・JSON・ログ20本を取得。独立37項目が合格し、元70:30・自作2fold平均・最終80:20を各36値で再計算して差0、最終CSVも834バイトのCRLF/.17g形式で完全一致しました。自作モデル追加の最初の公式測定です。以前のP013自己0.950は公開モデル同士の合成であり、自作nativeの評価ではありません。自己最良0.951を保持し、今回のhidden完走・Public・0.955／厳密な0.960超えはまだ未確認です。

**10月10日13:21 JST：圧縮したP014提出候補の非公開Version 1が保存され、エラーなしのRUNNINGを確認しました。** [保存受理とソース照合](experiments/f005-twofold-p014-compact-serving-running-20261010.json)。[27番Notebook](notebooks/public/27_submit_p014_native_twofold_compact.ipynb)のAPIソースは927,057バイト。77ファイルの圧縮復元SHA、元5セル、固定20%の計算を維持し、実際に保存された全21セルと6個の入力を8項目で照合しました。次は完走出力20本の取得、CSV・2fold平均・順位計算・後処理の確認、保存済みV1の競技提出です。今回の競技Submitは0、Publicは未評価、自己最良0.951を維持します。

**10月10日13:02 JST：自作2foldのCPU・64件・256件の検証が12項目すべて合格しました。** [実結果と保存エラー](experiments/f005-twofold-qualified-p014-source400-20261010.json)。256件は512回のforward、約575秒、両T4の観測最大635MiB、代替予測0件でした。P012の元5セルを保持し、自作モデルを固定20%で加える提出候補を作成しましたが、2.76MBのソースのSaveはHTTP400でした。参照先の照会は404で、保存版なし・競技Submit0を確認しました。サイズが原因という判断は推測です。圧縮して復元SHAを照合する修正を進めています。26番Notebookは失敗候補として保持します。自己最良0.951、0.955と厳密な0.960超えは未達です。

**10月10日 02:04 JST：自作二foldの実CPU・64件検証が完了し、同じ構成の256件が実RUNNINGです。** [保存版と実出力](experiments/f005-twofold-CPU64-qualified-native256-running-20261010.json)。decoderは実在する非公開V1へ参照を修正し、manifestの内容は変更0。新ソースの独立27項目を通過後、CPUは厳格な2モデル復元、64件は128forward・214秒・両T4の観測最大635MiB・代替0・後処理終了を実出力で確認しました。256件は新V1/kernel137904159として保存受理され、同じpair・入力・コードを公式RUNNINGと照合済みです。競技Submit0、256件全体／Serving／Publicの資格はまだありません。自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:36 JST：256件の実検証とfold1学習・CPU復元・非公開入力登録が完了しました。** [実結果と次の修正](experiments/p012-baseline256-qualified-f005-fold1-complete-20261010.json)。P012は実18本文・15項目合格、2559秒、3072予測値のfloat32差0、代替0・子2群終了・clear1回・GPU解放を確認。fold1は固定12epoch完了、最小BCEのE9で弱BCE0.27969／AUC12 0.86695、全12所見の改善方向が両foldで再現しました。paired13・CPU8項目合格、明示承認後にモデル2ファイルをprivate V1へ登録済。二foldのSave前検査で、decoderのローカルpackage改訂v2をKaggle Version2と混同した参照を検出しました。Version2一覧は404、実Version1のmanifestは元pinと完全一致。内容を変えず参照を修正中です。native Save0・競技Submit0、二fold／提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:13 JST：256件の保存版はエラーなしでRUNNING、fold1は11/12epochです。** 01:10の公式raw/typed状態と保存ソース6項目は一致しました。二foldの準備・単発Save手順は[独立ソース確認](artifacts/research/20261010-f005-twofold-native-root-independent-v1/review.json)22/22を通過していますが、実CPU・64件・256件の二fold資格はまだありません。最終epoch完了後、事前指定の最小weak BCE選択→CPU復元→fold1非公開入力登録→二foldの実検証を順に進めます。途中checkpointは未採用、競技Submitは追加していません。自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:03 JST：作者0.954の新S6候補を確認しましたが、入力の配布終了により保留します。** [一次資料と取得結果](experiments/public954-candidate-retired-input-20261010.json)。作者の保存V1・採点script356507041とソース一致、対応NetとV10 manifestも一致しました。現行公開V11は183バイトのREADMEだけで、モデルと旧版を利用不可にしたと明記。S6モデル取得0・アップロード0・自己0.954未確認です。既に固定した非公開S5入力の自己0.951を基準に、256件検証と自作二fold追加を継続します。fold1は11/12epoch、途中checkpoint未採用。二foldの実資格と0.955／厳密な0.960超えは未達、今回の競技Submit0です。この配布終了を過去全エラーの原因とは扱いません。

**10月10日 00:33 JST：実64件の合格を受け、同じコードの256件検証を開始しました。** [保存版と実行状態](experiments/p012-baseline256-cleanup-running-20261010.json)。新しい非公開V1（kernel137894797）はRUNNING、元5セル・同じ入力3版・Original/T4/Internet OFFを保存ソース6項目で確認。64件の実合格証拠を固定し、256件の合格は完了出力から別に検査します。競技Submit0、二fold・提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:27 JST：P012の64件診断は実完了し、GPU解放を含む15項目に合格しました。** [実検査の記録](experiments/p012-baseline64-cleanup-qualified-20261010.json)。固定70:30の768値は元式とfloat32差0、代替0、子2群終了、clear1回、両T4の確保量・予約量0。実行803秒、18本文の取得SHAを検査しました。以前の2件のERRORは追加診断checkerの不具合として保持し、成功資格へ転用していません。同じコードで256件を次に検証します。fold1は10/12epochで継続中、途中checkpointは未採用。学習の弱点解釈ではSynovitisの陰性が1件しかなく、weak改善をPublic改善とは断定しません。二fold・提出版の実資格は未確認、今回の競技Submit0、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:03 JST：再診断V2は成功ログの誤判定でERRORとなり、後段契約全体を実形式で再照合しています。** [取得した実ログ・13本文・原因](experiments/p012-baseline64-V2-worker-log-contract-error-20261010.json)。元予測は64件・代替0で完了し、固定70:30の768値は実CSVで再度一致。GPU観測はRaptor3763/3643MiB、S510225/9987MiBのsampled peakです。追加checkerが要求したPASS文字は元workerログには存在せず、WORKER_DONE JSONとfallback0が本来の成功記録でした。元モデル・数値は保持し、この形式と残りのseal→cleanupを実出力・元producerに合わせてV3で確認しています。封印・GPU解放は未到達、ERRORのCSVは提出しません。fold1は9/12epochで正常継続、途中checkpoint未採用。今回の競技Submit0、自己最良0.951、二fold/Serving資格と0.955／厳密な0.960超え未達を維持します。

**10月9日 23:47 JST：診断checkerを修正し、新しい非公開64件診断を再実行中です。** [修正・実取得・新Saveの記録](experiments/p012-baseline64-checker-repaired-rerun-20261009.json)。旧ERRORから必要な8本文だけを取得し、4CSVの全64件・13列・有限12値、固定70:30の768セルfloat32一致、代替0、子2群の終了を確認しました。新checkerはGPU記録listのbool化と実S5先頭50件preview／全worker CSVの区別を修正。独立19項目・作者8＋独立2テストを通過し、元モデル・5セル・数値は維持しました。23:40に新private Save1回受理（kernel137888437/V1）、23:42 raw/typed RUNNINGと保存ソース6項目一致を確認。GPU解放を含む実完了はまだ未確認で、旧ERRORから資格を転用しません。fold1は8/12epoch、途中checkpoint未採用。競技Submit0、自己最良0.951、二fold・Serving資格と0.955／厳密な0.960超えは未達です。

**10月9日 23:33 JST：64件診断は予測完了後の新しい診断checkerでERRORとなりました。** [実ログと原因](experiments/p012-baseline64-diagnostic-checker-error-20261009.json)。公式raw/typed ERRORと保存ソース一致を確認。元Raptor・S5は64件すべて、代替0、70:30融合を完了したと実ログに記録されています（786.9秒）。追加診断の`require(selected)`がlistを厳密なTrueと比較して必ず拒否する不具合が原因です。実S5のper_studyはtop-level・先頭50件previewであり、旧checkerの仮定も修正が必要と分かりました。原版・エラー証拠を保持してV2 checkerを準備中です。モデル・数値は変更せず、修正版64→256を実検証します。GPU解放・封印は未到達で、残った診断CSVは提出しません。ローカルfold1は8/12epochで継続。自己最良0.951、今回の競技Submit0、二fold・Serving資格と0.955／厳密な0.960超えは未達です。以下は時点ごとの履歴です。

**10月9日 23:25 JST：P012の64件GPU解放診断を非公開保存版V1で実行中です。** [最新記録](experiments/p012-baseline64-started-nativefold1-progress-20261009.json)。診断ソースは独立30項目、二fold統合入口は33項目を確認し、元5セル・数値を保持しています。標準SDKの認証を更新した後、診断Saveを1回受理。23:22の公式raw/typed状態はRUNNING、保存ソース・宣言設定6項目は一致しました。CSV・子プロセス・GPU解放の実検証は完了待ちで、二foldや提出版の実行資格とは扱いません。RTX4090のfold1は8/12epoch、途中checkpointは未採用。完了後に事前指定の最小weak BCEで選択し、二fold自身のCPU→64→256と親への統合を検証します。今回の競技提出は0回。自己Public最良0.951、P013は0.950・エラーなし、0.955／厳密な0.960超えは未達です。

**10月9日 22:32 JST：P012（自己Public0.951）を基盤に、実診断へ向けた取得・保存検査を修正しました。** [最新記録](experiments/f005-twofold-metadata-repairs-20261009.json)。JSON保存でfoldキーが整数から文字列になる比較不具合を修正し、型や本文の差は拒否するGetter V4を独立19項目、偽Save証跡を使わない保存前検査を24項目、結果集約converter V2を16項目で確認しました。数値・モデルソースは変更していません。fold1は6/12epochで継続中、途中checkpointは未採用です。必要な3つのInputはAPIで現Version1を確認しましたが、これは全モデル実bytesの検査や実行資格ではありません。P012の64/256件GPU解放診断と二fold統合のソースを準備中で、実二fold資格・追加モデルの提出は未実施です。P013の公式結果は0.950・エラーなし、自己最良0.951と0.955／厳密な0.960超え未達を維持します。

**10月9日 22:04 JST：P013は公式COMPLETE・Public0.950・エラーなしで採点を完了しました。** [結果と次の親の決定](experiments/p013-own950-p012-native-parent-decision-20261009.json)。提出ref57006968／保存Version1を照合し、再提出は行っていません。自己最良0.951のP012（Notebook24・元5code）を次の自作二fold追加の親に選択しました。元の0.950対照とP013は記録として保持します。追加率は事前宣言の20%・全12所見で固定し、比率探索は行いません。二fold自身の実CPU→64→256と親への統合は未資格です。学習は固定12epochを継続し、0.955・厳密な0.960超えは未達です。

**10月9日 21:56 JST：自己Public最良0.951を維持し、二fold追加の実検証準備を進めています。** [最新記録](experiments/f005-twofold-source-progress-20261009-2155.json)。学習中のfold1は5/12epochで、途中checkpointは採用していません。全体8.3時間の時計・固定20%融合CSVは独立検査28項目、取得ツールの型修正は39項目・ソース本文記録の追加は13項目を確認しました。清掃監視の誤識別と元タプルキーへの対応はV2で修正し、独立35項目を確認しました。新しい公開3候補もソースと採点情報を比較しましたが、新S6の正確な再構築や0.951超えの再現候補は確認できず、採用していません。P013は最後の21:08照会で状態未記載・スコア未返却です。二fold自身の実CPU・64件・256件と提出版統合は未資格で、ソース検査の成功を精度改善とは扱いません。

**10月9日 21:03 JST：二fold提出ルーターの[ソース準備](artifacts/research/20261009-f005-online-last2-twofold-serving-router-preparation-v1/preparation.json)と[独立査読31項目](artifacts/research/20261009-f005-online-last2-twofold-serving-router-independent-v1/review.json)が完了し、既存37件の数値ソースはバイト同一です。** [単fold0の実CPU・64/256資格47項目](artifacts/research/20261009-f005-online-last2-actual-CPU64-256-root-review-v1/review.json)は二foldへ転用せず、二fold自身の実CPU→64→256資格は未成立です。fold1の弱教師学習は20:56の観測で3/12epochを完了し、固定12epochを継続中です。自己Public最良は0.951、P013は最後の20:33照会でstatus未記載・SDK既定PENDING・スコア未返却でした。親全体の時計・GPU清掃・最終CSVの統合は現在ソース準備中で、二foldの実資格取得とは別工程です。ソース準備の合格を精度改善や提出資格とは扱いません。

**10月9日 20:48 JST：単一foldの自作モデルはCPU・64件・256件の実診断47項目を合格しました。** [実行資格と次の設計の記録](experiments/f005-online-last2-engineering-and-serving-plan-20261009.json)。この資格はfold0だけで、2fold統合や提出版の資格には転用しません。2fold設計V2はfold1の封印先だけを短い `b1` に修正し、既存215ファイル名の最長パス247文字と7つのCLI定義を静的に確認しました。旧設計V1と独立25項目の査読は保持し、統合推論のルーティングは実装中・未完成です。

自己Public最良は公式 **0.951** を維持し、0.955・厳密な0.960超えは未達です。P013は20:33の観測でraw status省略・SDK既定PENDING、スコア未返却です。fold1の時刻付きスナップショットは 2/12epoch（weak BCE 0.340666・AUC12 0.771251）で、固定12epochを継続中です。公開S6は学習コードと4重みの変更を確認しましたが、その新しい推論コード本文・互換性は未確認で、実行可能な新候補とは扱いません。教師なし64件の予測差は確認しましたが、学習から独立であることを確認していない単一foldのcohortであり、精度改善やPublic0.951との補完性の証明ではありません。


**10月9日20:17 JST：自己Public最良は0.951に更新しました。** [公式採点・次の提出・自作モデル進捗の記録](experiments/p012-own951-p013-submitted-native-progress-20261009.json)。P012の保存Version 1は公式COMPLETE、Public **0.951**、エラーなし（ref56998356）。作者報告だけでなく、自分の提出で確認した値です。元p005の0.950とNotebook14は対照として保持します。目標0.955、厳密な0.960超えはまだ未達です。

P013は、基準p005へS5予測を固定20%で加える別候補です。実64件・256件の診断後、提出版の全3件・12所見、代替予測0、GPU解放、基準融合差0、提出CSVのバイト一致を独立20項目で確認し、同じ保存Version 1から1回提出が受理されました（ref57006968）。20:15の照会ではraw status省略、SDK既定PENDING、スコア・エラー未返却です。Input順序のAPI返却だけを取得ツールが厳しすぎる条件で比較していた不具合は、全22セルと環境の一致を確認して修正しました。モデル・入力の集合・予測計算は変えていません。

自作DINOv2最終2層モデルはfold0の固定12epoch学習とCPU再構築を完了し、新しいモデル自身の実64件・256件診断もそれぞれ10項目合格しました（320／1,280配列SHA、全12予測有限、両T4、代替0、約178／701秒）。256件の実COMPLETEを20:16に確認し、両GPUの観測最大は563MiBです。以前の凍結モデルの資格は流用していません。Serving・Public、将来の2fold合成モデルの資格はまだ未確認です。fold1は同条件の12epoch学習を継続し、最新保存スナップショットは1/12epoch（weak BCE 0.368269、weak AUC12 0.711189）。weak値をPublicへ換算せず、12epoch完了後に凍結対照と比較します。自分のGoldラベルの学習・選択利用は0、公開重みのGold/事前分布利用は別監査です。以下は過去の観測履歴です。


**10月9日19:35 JSTの公式観測を含む進捗：last2 fold0の固定12epoch学習とCPU再構築が完了し、fold1の比較を開始しました。** [学習・配布・診断の最新記録](experiments/f005-last2-fold0-complete-CPU-privateInput-fold1-progress-20261009.json)。fold0の選択weak BCE/AUC12は凍結対照0.323206/0.794448からlast2の0.273758/0.877128へ改善し、最小BCEのepoch8を選択しました。12epochで実2,543更新・AMP skip1、選択checkpointのCPU厳格再構築8項目に成功。これは弱教師検証の結果で、Public精度への換算や患者独立性の保証ではありません。fold1の凍結対照も12epochを完了し、選択epoch10のweak BCE/AUC12は0.324100/0.800933。last2 fold1は同じ固定条件で開始済みで、途中checkpointによる採用判断は行いません。

last2 fold0の新しい非公開Input V1は19:38のKaggle CPU診断で215 memberの全SHAと厳格なモデル再構築に成功しました。専用の元online producerを使う実CPU8項目の証跡を拘束し、以前の凍結モデルの資格は転用していません。CPU診断はMRI読取・モデルforward0で、続くnative64/256・提出推論は未資格。p013は実64件を受理済み、256件は19:35の公式観測でRUNNING。p012提出ref56998356は19:27のAPIでも状態・スコア・エラーが未返却でした。自己Public最良0.950を維持し、0.955・厳密な0.960超えは未達です。以下の記述は各観測時点の履歴です。


**10月09日18:56 JST：p013の実64件診断を受け入れ、同じ修正版の256件を実行中です。** [実64件・修正・次の診断の根拠](experiments/p013-cleanup-repair-and-last2-runtime-progress-20261009.json)。旧64件はS5開始前のCUDA清掃guardで停止、数値ログがなく当時の原因は未確定。新V3はcuBLAS作業領域の解放後に従来の清掃を行い、両T4のallocated/reservedが0、free各15.49GBとなりました。閾値と元14の7code／S5 core2／80:20を保持。実64件は全体942.90秒（S5追加区間716.54秒）、独立38項目成功、Reader選択384読取の失敗0、S5代替0、現在実行の元融合と固定20%の独立再計算一致。別実行ではReader CSV byte同一、CoAt6・基準融合10の順位セル差があり、完全な数値/精度同等性は断定しません。新256件はV1/kernel137854919として保存受理、18:52 JSTにRUNNING。256・hidden・精度の資格は未取得、p013競技提出0。

RTX4090のlast2 fold0は11/12epoch、weak BCE0.311269／AUC12 0.877300、実2,332更新・AMP skip0。12epoch完了前にcheckpointを使わず、最小weak BCEの選択ルールを維持。last2専用CPU入口・封印・転送・2fold driver、およびProfileに依存しない提出用清掃のソース査読は完了しましたが、実last2 CPU/64/256へ資格を転用しません。提出済みp012 ref56998356は18:53 JSTでもスコア・エラー未返却。自己最良0.950、0.955／厳密な0.960超えは未達。以下は観測履歴です。


**10月09日17:03 JST：次の候補はソース準備の段階です。** [次候補の準備と現在の観測](experiments/f005-last2-and-fixed-S5-addition-source-preparation-20261009.json)。提出済みref56998356は16:56 JSTの公式読取で採点・エラー未返却、自己最良0.950を維持。RTX4090のlast2は7/12epoch、weak BCE 0.279831、AUC12 0.870949、条件変更・AMP skipなし。完了12epoch前にcheckpointを使いません。last2専用CPU入口V2は元8ソースの直接固定を含む独立8項目を通過し、実モデル再構成・推論は未実施。p013は元14の7codeとS5 core2を保持し、80:20と最終順位化を固定した未資格候補です。6人工CSV/構造テストの成功は実T4・64/256件・hidden・精度の成功を意味しません。Reader依存環境の一致、silent decode omissionの観測範囲、全体時間/資源を別途確認します。S5子は親開始から8.3時間の残時間を使用しますが、元14の非同期強制停止は未実装。追加の限定一次資料調査で新しい0.955以上の完全再現レシピは未確認です。0.955・厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:22 JST：S5 V9の非公開固定再現は可視実行・独立検査を完了し、公式API提出が受理されました（ref56998356）。** [実完了・検査・提出の記録](experiments/p012-private-S5-fixed-V1-visible-complete-submission-accepted-20261009.json)。承認を受けた自分の非公開snapshot V1を接続し、Notebook24の原版5code/70:30/guardを保持。全3件・12所見を両T4で推論し、両モデルfallback0、約174秒、独立9項目・全36float32順位融合一致。保存V1から1回提出、初回APIではstatusが省略されSDK既定PENDING（採点・新自己Public未確認）。旧承認待ち／画面でのV9固定作業は不要になりました。RTX4090の独立last2は6/12epoch、BCE0.285281/AUC12 0.851437で継続。自己基準0.950を保持し、0.955／厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:12 JST：具体的なユーザー承認を受け、S5 V9の非公開固定Inputを作成し、原版5コードセルの再現を保存実行しました。** [承認・実作成・保存実行の記録](experiments/p012-private-S5-snapshot-approved-and-run-started-20261009.json)。`rsraki/rsna-knee-compact-v1`はAPIで非公開Version1・8ファイル名/サイズ・元manifest byte一致を確認。新規Notebook24はInput参照と説明だけを変更し、元5code/70:30/guardを保持。新private SaveはV1でRUNNING、可視出力検査・採点はまだ。過去のアップロード承認待ちは解消しました。独立last2は6/12epoch、weak BCE0.285281・AUC12 0.851437。固定12epochの学習を維持。自己Public最良0.950、0.955／厳密な0.960超え未達。以下は観測履歴です。

**10月09日16:00 JST：S5 Version9の使用中6ファイルと元manifestを取得・ハッシュ一致まで確認しました。** [実取得と審査停止の記録](experiments/p012-S5-private-snapshot-ready-review-rejected-20261009.json)。原版5コードセルのまま使える約885MBの非公開入力を準備。新規private Datasetへのアップロードは自動承認審査が具体的な送信内容の承認・宛先所有権確認不足として拒否したため、実アップロード0・Dataset作成0・再試行0。ユーザーへこの具体的なprivateアップロードの承認を質問済み。Kaggle画面で元Input V9をPinして保存実行する別経路も回答待ち。元last2学習は実プロセスをpollし継続、5/12epoch・weak BCE0.301530/AUC12 0.848946。固定条件を保持し、自己Public最良0.950／厳密な0.960超え未達。以下は観測履歴です。

**10月9日15:26 JST：実64件・256件の入力一致診断は成功。Helios作者0.951原版の再現は、APIによる旧Input版固定が反映されず停止しました。** [実行・調査の最新記録](experiments/public951-native256-and-next-review-20261009.json)。64件は320配列SHA一致・204秒、256件は1,280配列SHA一致・541秒、両T4のsampled peak各563MiB、代替0。同じ凍結対照headの動作検証で、精度/hidden/最後2層モデルの資格とは区別します。Heliosに合うcompactはV9（外a5/使用部分c7）で、原版guardは正しいです。しかし標準Save APIの`datasetDataSources`に`/9`を付けても実際は最新V10（209a…）が接続され、約74秒でguardが停止。Kaggle画面でV9をPinし保存実行した版の確認が次です。23は予測コード未変更、自己スコアなし・競技提出0。Ryan V70は推論モデルでなく順位表（5,374チーム、0.955以上56・0.960超え7・最大0.964）。RTX4090最後2層は4/12epoch・実848更新・AMP skip0、weak BCE0.302986/AUC12 0.835976で継続中。自己基準0.950を保持し、0.955・厳密な0.960超えは未達。過去段落は観測履歴です。

**10月9日14:36 JST：新しい64件診断は入力パス指定の不備で停止。原因を実ログで確認しました。** [実エラー記録](experiments/f005-native64-actual-mount-path-error-20261009.json)。旧flat競技rootを固定したためtrain_series.csvが見つからず、MRI処理・モデル読込前に約21秒で停止。競技Inputの実mount場所はこのログでは未観測です。構造CSV／画像ディレクトリの存在と固定CSV SHAで一意にrootを選ぶ新wrapperを準備し、同じモデル・64件で再診断します。CPUでの183ファイル一致・厳密再構築成功は保持、64／256件の資格は未取得。last2 fold0の固定12epoch学習は変更せず進行中。元p005自己0.950を保持、0.955／厳密な0.960超は未達。以下の結果待ちは過去の観測時点の履歴です。

**10月9日14:25 JST追記：last2 fold0は2/12epoch、実424更新／AMP skip0まで継続しています。** [実観測](experiments/f005-last2-fold0-actual-epoch2-observation-20261009.json)。weak BCE0.339634／AUC12 0.785699、学習記録累積3506.667秒。早期weak値で採用を決めず、固定12epochと別foldの比較を続けます。Kaggle実MRI64件の保存実行は受理済み、結果未確認、自己Public最良0.950のままです。

**10月9日14:20 JST：最終2層モデルの実1epoch更新とKaggle CPU読込を確認し、実MRI64件診断を起動しました。** [現在の根拠](experiments/f005-last2-firstepoch-privateCPU-native64-root-progress-20261009.json)。last2 fold0は実212更新／AMP skip0、1epoch累積1748.733秒、weak BCE0.377701／AUC12 0.742994。同じepochの凍結対照より改善しましたが固定12epoch実行中で、最終採用・Public改善は未判断。新しい自分の非公開Input V1を作成し、実Kaggle CPUで183ファイルSHAと厳密再構築に成功。T4×2の64件診断はV1/kernel137811090として保存受理、実結果待ちです。次は64→256の入力一致・時間・資源検査、並行してfold0比較→固定fold1比較。自己最良p0050.950を保持し、0.955／厳密な0.960超は未達。作者0.951は確認しましたが取得SourceとV7のIDが不整合で非採用。アップロードの自動承認拒否は所有・内容・既存許可の証拠で解消、現在承認待ちなし。以下は各観測時点の履歴です。

**10月9日13:15 JST：凍結対照fold0を完了、元last2 fold0の実学習を開始しました。** [root記録](experiments/f005-frozen0-complete-last2-actual-start-root-review-20261009.json)。全4,207件／21,035特徴配列のCPU受入後、固定12epochを372.260秒で完了。選択epoch11、weak BCE0.323206／AUC12 0.794448、実更新2,544／AMP skip0。全CSVの独立14項目、欠損を除く5,727教師セルの指標再計算、選択checkpointのCPU厳密再構築が成功しています。同じteacher・fold・seed・head・lossで最後の2block学習だけを変更する元last2を開始。次は完了後のfold0対照比較→固定fold1 pairです。自己最良は元p005 Public0.950のまま、ownGold学習・選択0、0.955／厳密な0.960超は未達。以下は各観測時点の履歴です。

**10月9日12:39 JST：p011 ref56986152は公式COMPLETE／自己Public 0.950でした。** [実採点記録](experiments/p011-scored-review-20261009.json)。隠し再実行の採点成功は確認できましたが、返却された3桁精度で基盤と同点なので追加stock224 10%は採用せず、元14/p005を保持します。目標0.955／厳密な0.960超は未達です。凍結CLSのFP32cacheは別producerとして固定し、[実人工144窓のGPU一致5項目と実抽出開始](experiments/f005-frozen-FP32-actual-qualification-and-cache-start-20261009.json)を確認。dropout RNG・logits・勾配・AdamW・GradScaler・検証出力がbyte一致しました。実cache抽出は12:34開始、完了後の全4,207件CPU検査を通してからfrozen fold0を学習し、元last2 fold0と比較します。正式fitはまだ0、ownGold学習・選択0。以下は各観測時点の履歴です。

**10月9日12:25 JST：全件索引完成、通常fold0 preflight成功。** [root11項目](experiments/native392-full-index-and-frozen0-preflight-root-review-20261009.json)で4,207 UID／21,035配列／386,196有効窓／24 shardと、学習3,386件・検証821件を確認しました。index SHA `7d373ae85098758a8bc322b4e65079de07e998f7b7cd42a5a2d26cc4577eae31`。正式fitはまだ0。凍結CLSのFP32特徴cacheは別producerとして準備中で、元計算との実人工GPU同値確認が済むまでは実抽出・cached fitに使いません。p011は[12:22の公式照会](artifacts/research/20261009-p011-scoring-status-tools-v1/observation-v3/review.json)でもraw status／Public／errorが未返却、SDK既定PENDINGを完了状態とは扱いません。元14と自己Public0.950を保持し、学習条件と実行時間を混同せず比較します。以下は過去の観測履歴です。

**10月9日11:43 JST：全4,207件の加工済み画像cacheを取得・検査・配置しました。** [全24 shardの受入8項目](experiments/native392-all24-canonical-root-acceptance-20261009.json)成功、計21,035配列。既存の全件index作成コマンドで画像SHAとUIDの全件整合を確認中です。正式fitはindex完了後。p011 ref56986152は11:37の公式照会で明示status／scoreが未返却（SDK既定PENDING）、[未確定の観測](artifacts/research/20261009-p011-scoring-status-tools-v1/observation-v1/review.json)として記録しました。自己最良0.950と元14を保持しています。

**10月9日11:30 JST：p011 Version 1を競技へ一回提出し、ref56986152として受理されました。** [受理記録](experiments/p011-submission-accepted-20261009.json)。直前の公式提出枠は5、正常終了とCSV契約を確認して送信。採点・隠し再実行は未確認、自己最良0.950を維持します。残23/24 cacheの実取得も開始しました。以下はそれぞれの観測時点の履歴です。

**10月9日11:28 JST更新：Kaggle認証が復旧し、p011提出候補の保存実行と可視出力検査まで完了しました。** 自己最良はp005の**0.950**を維持、0.955／0.960超は未達です。11:05にprivate `rsraki/p011srv-8105f0434cc183692525c2dd8ccdc9eb` Version 1を保存実行し、[保存ソース17項目](experiments/p011-serving-savedV1-binding-review-20261009.json)を確認。[可視出力の独立33項目](experiments/p011-serving-visible-output-independent-review-20261009.json)と[root NumPy12項目](experiments/p011-serving-visible-output-root-review-20261009.json)は成功しました。元14の全7 codeを保ち、別の224モデルを固定90:10で追加した候補です。次はfresh日次提出枠とV1正常終了を確認して一回提出、隠し再実行の採点を待ちます。可視3件成功を精度改善・隠し全件成功とは扱いません。

画像cacheはremote全24 shardがCOMPLETE。PCでは01〜22の**3,857検査／19,285配列**を実検査・canonical配置済み、残23/24の350件を取得する入口を準備中です。全4,207件index完成後にRTX4090で凍結encoderと最終2block学習を同じfold・seed・教師・head・loss・12epochで比較します。正式fitはまだ0です。最新の限定公開調査でも、採点値・版・コード・重みを結び付けられる0.955以上の新レシピは確認できず、不存在とは断定しません。[調査記録](artifacts/research/20261009-post950-public955-web-update-v1/research.json)。以下の認証停止等は過去の観測履歴です。

**10月9日04:07 JST：認証待ち／目標未達**。第三連続goal turnでも[公式API観測](artifacts/research/20261009-kaggle-connectivity-probe-v1/observation-v3/receipt.json)はHTTP401。[停止判定の監査](artifacts/research/20261009-goal-authentication-hold-audit-v3/receipt.json)で同一障害3回・独立確認・準備済み入口・実保存/提出0を照合し、再認証または外部認証状態の復旧が必要と判断しました。自動継続をblockedへ更新する条件を満たしています。自己最良0.950、0.955/strict>.960未達、元14/候補22V3のSHA不変。remote CPU jobsの停止や完了は追加確認できておらず、再起動していません。認証復旧後、fresh quota→保存実行→可視CSV確認→fresh提出枠→提出で再開します。ledger150→151の旧byte prefixを保持しました。以下は各確認時点の履歴です。

更新日 2026年10月9日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、**自己採点済みp005のPublic 0.950を基準に、次の0.955・最終0.960超を目指す**ことです。p002自己0.937と過去runも保持し、公開レシピ改善と独立した自作学習の評価を分けます。

## 最新：自己Public0.950確定、基準を固定して一要因の改善へ

**10月9日04:03 JST再確認**：[公式APIの新観測](artifacts/research/20261009-kaggle-connectivity-probe-v1/observation-v2/receipt.json)もHTTP401、Kaggle API再認証待ちです。保存実行/提出は0、自己最良0.950を保持。cache20/22の既metadataにも再利用できる署名URLはなく、取得を進められていません。[地域モデル接続前の不足](experiments/regional-triplet-training-connection-review-20261009.json)として、CLS同形状の重みを混同しない専用schema/表現識別をsave/load/apply/factoryで検査する必要を確認。現factoryはNative固定で地域モデル未接続、既学習器への変更/実fit0です。ledger148→150は[旧byte prefix保持](artifacts/research/20261009-goal-authentication-hold-audit-v2/receipt.json)。

**10月9日03:50 JST更新**：自己最良p005は**0.950**、0.955・厳密な0.960超は未達。p011のprivate実256件がCOMPLETE、[root NumPy24項目](experiments/p011-profile256-completed-root-review-20261009.json)・[独立47項目](experiments/p011-profile256-actual-results-independent-review-20261009.json)で元融合差0、追加前後の基準CSV byte同一、固定90:10候補の再構成SHA一致を確認しました。main512/Reader192/追加256×94窓に例外/fallback0、工程計977.913秒/最大sampled GPU8,893MiB。queue/import等を除く動作検査で、精度/全件隠し成功は未確認です。

提出用22V3の[ソース24項目](experiments/p011-serving-handoff-independent-source-review-20261009.json)・[確定リクエスト11項目](experiments/p011-serving-qualified-request-independent-review-20261009.json)も成功し、元14/Original T4×2/OFF/Input両Version1を保持。private Serving Save・競技提出はともに0です。直近の[公式API照会](artifacts/research/20261009-kaggle-connectivity-probe-v1/receipt.json)がHTTP401で拒否され、ユーザーへKaggle API再認証を依頼しました。自動承認の拒否やNotebook推論失敗とは区別します。再認証後、fresh GPU枠→一回保存実行→保存ソース/可視CSV→fresh日次枠→提出の順で再開します。

学習cacheはPC01〜19計**3,332件/16,660配列**のまま、remote20/22 COMPLETE・summary各28成功、21の最終観測RUNNING。23/24も保存V1を受理し全24 jobは起動済みですが、その保存ソース/実結果はまだ未確認。20/22の追加取得は認証エラーで0bytes、残875件/full4207 index/正式fit未完です。[f005の4学習コマンド](artifacts/research/20261009-f005-fast16-root-commands-v1/commands.json)を用意し、全24検査後に凍結対照→last2をfold0/1で比較します。[CLS＋16領域の新部品](experiments/regional-triplet-component-preparation-20261009.json)はCPU8・既存汎用重みの人工CUDA2窓に合格、教師/MRI/optimizer0。精度・full144資源は未測定、現学習器への接続はしていません。[進捗と認証停止の記録](experiments/p011-profile256-authentication-hold-progress-20261009.json)、ledger141→148の[旧prefix保持](artifacts/research/20261009-p011256-auth-hold-progress-record-v1/receipt.json)。

再認証後の[単発保存入口](artifacts/research/20261009-p011-serving-root-launch-v1/launch_once.py)も準備済みです。4件の人工SDK/停止条件検査とrootの通信なし確認モードが成功し、確定済みnonce・actual256証拠・元ソースを保持します。実保存と提出は認証復旧後に行います。

以下は各確認時点の履歴です。

**10月9日03:19 JST更新**：自己最良p005は**0.950**、0.955・厳密な0.960超は未達。p011のprivate実64件はCOMPLETE、[root24項目](experiments/p011-profile64-completed-root-review-20261009.json)と[独立45項目](experiments/p011-profile64-actual-results-independent-review-20261009.json)で元融合差0・追加前後基準CSV byte同一・固定90:10候補再構成SHA一致を確認。追加64×94窓に例外/fallback0、工程計230.089秒/最大sampled GPU8,893MiB。起動・queueを除く計測であり、精度/全件提出成功の証明ではありません。旧p010 gateは閉じたまま。03:12:34 JSTに[新256件版](notebooks/public/22b_profile_p011_independent224_256_v1.ipynb)をprivate保存実行として受理：`rsraki/p011diag256-e5e8cdc760af73b43bbd386184635afb` V1/kernel137720781。[起動独立24項目](experiments/p011-profile256-root-launcher-independent-review-20261009.json)、fresh quota1/Save1、競技提出0、結果待ちです。

学習cacheはPC**01〜19の3,332検査/16,660配列**を検査・配置済み。[canonical19のroot9項目](experiments/native392-shards01-19-canonical-stage-root-review-20261009.json)は全19の実stat/summaryと新17〜19の実配列QA13を照合し、[既16の独立264項目](experiments/native392-shards01-16-canonical-stage-independent-review-20261009.json)をSHA固定再利用しました。20〜22は[保存100項目](experiments/native392-shards20-22-saved-binding-review-20261009.json)成功/03:07観測RUNNING、23/24未起動。残875検査/full4207 index/正式fit未完。[現学習入口14項目](experiments/f005-fast16-fullcache-fit-entry-review-r2-20261009.json)は両config/CUDA/資源gateを受理、全24成功後に凍結対照／last2をfold0/1で比較します。p011実256の[保存ソース17項目](experiments/p011-profile256-savedV1-binding-review-20261009.json)も成功しましたが、03:14 raw status省略/SDK既定QUEUEDはworker開始の証拠ではありません。ledger131→141は[旧byte prefix保持](artifacts/research/20261009-p01164-cache19-progress-record-v1/receipt.json)。

[追加web調査](experiments/post950-strategy-primary-update-20261009.json)も4検索では採点済み0.955以上の新公開レシピを確認できず、不存在とは断定しません。[今後の順序・採用条件](docs/roadmap.md)を更新。02:13公式照会は日次5件・残0、更新時刻未確認。以下は各時点の履歴です。

**10月9日01:56 JST追記**：加工済みcache13の[実175件/875配列](experiments/native392-shard13-completed-cache-review-20261009.json)も13項目成功、元ZIP不変のcanonical配置まで完了。PC01〜13は**2,282検査/11,410配列**です。ledger130→131の[旧prefix保持receipt](artifacts/research/20261009-native392-shards11-13-progress-record-v1/receipt.json)。全4,207件indexと正式fitは未完、14〜16のRUNNINGは保存照合時点の観測、残8未起動です。

**10月9日01:54 JST追記**：p010 A/A savedV4の実32件・10phaseを[独立root17項目](experiments/p010-AA-completed-compact-root-review-v4-r2-20261009.json)で検証しました。40 bounded GET・2,470,284bytes、source/statusは既取得readsetを再利用して追加0、metadataは10 RPC。全10入力SHA、実重みSHA、base各64回のviews/stock各32回のforward、16CSV、6比較が一致。原main通常A/Aはraw最大差0.00032717・377セル差・LM順位2セル/1組変化。固定GPUでもbase0.00017780、stock0.00021404の差が残り、両固定反復の順位差は0でした。GPU割当だけを原因と断定しません。sampled GPU最大8,895MiB、process tree RSS最大18,024,181,760bytesは共有ページの重複とサンプリング誤差を含みます。Reader/融合の実推論0、精度未評価、全6の旧raw1e-7 gateは失敗を保持しp010候補は未資格です。2反復から一般的変動上限や移植の数学的同等性を導きません。PyTorchの[公式再現性資料](https://docs.pytorch.org/docs/2.11/notes/randomness.html)も確認しましたが、backend条件の因果比較はしていません。

native392はremote01〜13 COMPLETE・各summary28検証済み、PC全配列検査とcanonical配置は01〜12計**2,107検査/10,535配列**。[配置root14項目](experiments/native392-shards01-12-canonical-stage-root-review-20261009.json)。13は取得済み・全配列検査中。14〜16は各175 whole-group・[保存100項目](experiments/native392-shards14-16-saved-binding-review-20261009.json)成功、観測時全3 RUNNING。残8未起動、全4207 indexと正式fitは未完です。

短期の別候補p011は[新しい独立stock224枝のgate設計](experiments/p011-independent-stock224-gate-proposal-20261009.json)をrootでレビューし、ローカル実装を開始しました。原14全7 codeを保持し、元融合成功・CPU CSV確定後に二度目のdecodeとstock単独推論を行い、全12の固定90:10を試す案です。基準CSVの追加前後byte同一、明示した欠損処理、実64→256の二度decode込み時間/資源の検証が必要。p010の許容差を事後に緩めた候補とは扱いません。実候補推論・提出・採用はまだです。

[新しい限定公開調査14項目](experiments/post950-public955-delta-primary-research-v2-20261009.json)でも実採点0.955以上の公開レシピは未確認。新ApexV5の採点V1は0.950で、前6 codeが14とbyte/AST一致、最後の融合式と5係数だけが変化しています。不在の証明ではありません。ledger124→130は[旧byte prefix保持](artifacts/research/20261009-AA-and-cache13-progress-record-v1/receipt.json)。以下01:24以前の内容はその時点の履歴です。

**10月9日01:24 JST更新：自己最良はp005の0.950です。** p009は公式COMPLETE・自己Public0.948で非採用。[採点10項目](experiments/p009-scored-review-20261008.json)。画角140→160の一要因で表示スコアは低下しました。教師/尺度ずれ等の機序や未丸めAUCを断定せず、画角140の基準14を維持します。短期の微調整p006/p007/p008も0.950で改善確認なし。0.955・厳密な0.960超は未達です。

**現在の実進捗**：p010は新[A/A診断保存V4の全21セル・設定36項目](experiments/p010-AA32-savedV4-binding-review-20261009.json)が一致、01:16の公式raw/typed状態はCOMPLETEです。通常GPU割当とUID固定割当の各反復結果をこれから取得・独立分析します。実行完了だけでraw1e-7/基準byte同一の事前gate、候補採用、精度向上は成立しません。原14/19/21/V1〜V3/V4_r1と過去の失敗証拠を保持します。

native392は01〜10のremote作成・PC取得・全ZIP/NPZ/配列検査を完了、**1,757検査/8,785配列**です。[canonical配置のroot8項目](experiments/native392-shards01-10-canonical-stage-root-review-20261009.json)も成功。加工済みZIPだけを元SHA・更新時刻保持で同一ボリュームのhardlinkへ配置し、SUMMARYはbyte同一コピー。リンクは同じ書込み可能inodeを共有するため、両方を不変の入力として扱います。元DICOMのPC取得・ZIP展開はしていません。11〜13は各175 whole-group・[保存100項目](experiments/native392-shards11-13-saved-binding-review-20261009.json)成功、01:16観測RUNNING。残11未起動、全24の成功後に初めて全4,207件indexを作成し正式fitへ進みます。

f005は旧chunk2のsource/config/handoffを保持して、両armにchunk16を適用する[新入口の準備47項目](experiments/f005-native-fast16-preparation-validation-20261009.json)と[実加工済み115件のroot13項目](experiments/f005-native-fast16-processed115-root-review-20261009.json)に成功。人工full144窓のforward/backward平均はchunk2→16でfrozen6.27倍、last2 7.80倍、last2最大予約GPUメモリ約1.115GiB。[計測84項目](experiments/f005-window-throughput-review-20261009.json)。eval出力最大差1.78814e-5、gradient同値性は未計測でbyte同一とは扱いません。実教師を使うoptimizer更新0、全epoch時間・学習精度は未測定です。実資源validatorは両configを受理しましたが、全件indexの条件は維持します。最初の比較は同じ入力/教師/fold/seed/headでencoder凍結対最終2block学習です。

ledgerは旧byte prefixを保持し118→124行へ追記。[記録receipt](artifacts/research/20261009-native392-fast16-progress-record-v1/receipt.json)。00:10公式照会では日次5件・残0、枠の更新時刻は未確認です。以下の00:35以前の表示は、その時点の履歴です。

p010診断V2の二次停止は、[実ログとsource/人工再現12項目](experiments/p010-profile32-second-failure-review-20261008.json)で古いp009データaliasのreplay確認に絞りました。新V3は全4経路の実32件推論後、original/bounded main CSVの不一致で停止。[実失敗監査14項目](experiments/p010-profile32-third-failure-review-20261009.json)と[限定9CSV数値監査12項目](experiments/p010-profile32-third-failure-CSV-review-sanitized-r2-20261009.json)を完了。bounded基準はLMの2件が順位入替、original/shared mainと融合はbyte同一。224共有/単独は300セル・最大raw差0.0005064458でも全12所見の平均percentile順位と固定90:10の仮最終順位は同一でした。raw1e-7/基準byte同一の事前gate合格とは扱いません。rawbase/device/backend/実resourceのreceipt欠落を残し、[A/A反復の新診断](experiments/p010-AA-precision-diagnostic-plan-20261009.json)を準備中です。通常経路反復とUID固定GPU割当反復を区別し、各phase結果を比較guard前に保存します。旧checkerのfloat64対float32 ordinal厳密比較の欠陥は別に修正予定。原14/19/21/V1〜V3を保持、競技未提出です。

native392は分割01〜07のremote作成がCOMPLETE、各保存ソース・要約28項目も成功。**PCで全配列まで検証済みは01〜04の計704検査/3,520配列**です。02/03の転送接続切断は旧partialの実byte位置とSHAを保ち新ファイルへコピー、16/7個の最大128MiB Rangeで復旧しました。[02の実176検査](experiments/native392-shard02-completed-cache-review-20261008.json)、[03の実176検査](experiments/native392-shard03-completed-cache-review-20261008.json)とも全ZIP/member/880配列SHA・shape・dtype・mask・coverage・padding等13項目成功。例外の転送カウンタは再開位置に使っていません。05〜07は最初から[有限Range取得のroot13項目](experiments/native392-full-range-root-review-20261009.json)を経てPC取得中。08〜10は各175 whole-group・[保存100項目](experiments/native392-shards08-10-saved-binding-review-20261009.json)成功・取得時RUNNING、残14未起動。全4207 index/正式fitは未完。正式fit前にwindow_chunkだけを変える人工full144の4090速度計測を実施中、学習config/源/handoffは保持します。

[独自学習の設計監査](experiments/f005-native-contrast-design-review-v2-20261008.json)では、既存headは既に所見別二段attentionと確認。最初の一要因はencoderの最終2block解凍です。その後は教師/系列coverageの根拠を確認し、必要ならCLSから局所patch、支持数の偏りに対する所見別loss、強いbackboneを別々に比較します。弱ラベルBCEと競技macro12の重みは異なり、Synovitisのfold0陰性1件等を単独の採用理由にしません。fold0/1は選択に既使用、他foldが未使用かは履歴監査前に断定しません。公開all-weak学習済みモデルと混ぜたweak評価を独立OOFと呼びません。

[公式公開候補13件の追加監査](experiments/post950-official-keyword-public-audit-20261008.json)では最大確認値0.950。新0.950の計算は元14と同一、タイトル0.957の別版の公式採点は0.939でした。この限定範囲で再現可能な0.955超を確定しておらず、公開レシピが存在しないと断定しません。

以下の10月8日23:34以前の表示は、その時点の履歴です。

**23:34 JST追記**：p010診断savedV2は23:31のexactV2照会でraw/typed ERROR、failureMessageは空。次の実ログを確認中で、修正成功や追加224完走を主張しません。分割02/03/04 CPUは23:34のexactV1照会でも全3件raw/typed RUNNING・failureなし。以下の23:28にあるRUNNINGは、その時点の観測です。

**23:28 JST更新：自己最良はp005の0.950、目標0.955・厳密な0.960超は未達です。** p006/p007/p008の表示スコアがすべて0.950だったため、元の14を基準に保持します。p009画角160の競技提出ref56956158は23:25照会でもscore/error/raw status未返却。SDK既定PENDINGだけではworker段階を判断できません。確認済み公式日次残は0、同版を再提出しません。

p010別解像度224の診断savedV1は公式ERROR。[実ログ14項目](experiments/p010-profile32-first-failure-review-20261008.json)で、original32のmain完走後、Reader子Pythonが計測用`p007_metrics`をimportできず停止したと特定しました。原融合fallbackを成功として扱わず、診断guardが停止しています。追加224・bounded/shared比較は未到達で、精度・VRAMは未確認。親のsys.pathが子Pythonへ伝わらない経路だけを、新しい診断V2で補正。[独立30項目・CPU34検査](experiments/p010-profile32-independent-prelaunch-review-v2-20261008.json)成功後、23:25:06 JSTに同private refの**新savedV2/kernel137683716**として起動を受理。[保存版34項目](experiments/p010-profile32-savedV2-binding-review-20261008.json)も成功、23:26 raw/typed RUNNINGです。失敗V1、基準14、提出19、Serving21は保持。実T4結果が通るまで競技候補にはしません。

native392は先頭176/880配列の実検査成功に続き、**分割02・03・04をprivate CPUで起動済み**です。[保存版独立100項目](experiments/native392-shards02-04-saved-binding-review-20261008.json)は全source/whole-group分割/設定一致、取得時全3件RUNNING・failureなし。実出力QC・取得はまだです。管理対象CPU最大3件を守り、残20件は空きを確認して同じ正本V2で進めます。全4207検査のcomplete indexと配列検証が揃うまでf005正式学習を開始しません。

次の優先順は、①p009採点とp010実32件で短期候補の効果・追加計算を判断、②全キャッシュ完成後に固定教師/fold/seedでFrozen対照とDINO最終2block学習だけを比較、③両foldの所見別weak support/BCE/AUCと不足ラベルを確認し、有望な独自モデルだけを別提出または事前固定の少量融合で評価、です。0.955には12所見AUCの合計で約0.060、0.960には約0.120の改善が必要。weak AUCをPublicへ換算せず、公開重みが学習済みの検査を混ぜた評価を独立OOFと呼びません。教師17セルの追加監査から訂正採用は0件なので、推測で教師を変更しません。

以下の22:46以前の表示は、その時点の履歴です。

**22:46 JST更新：p009画角160のsavedV1を競技提出しref56956158を取得しました。** [保存31項目](experiments/p009-serving-saved-binding-review-20261008.json)・[可視CSV15項目](experiments/p009-serving-visible-output-review-20261008.json)・[提出binding22項目](experiments/p009-submission-binding-review-20261008.json)が合格、可視3件の固定融合差0/例外fallback0。22:36:51 JST受理、直前残1、22:42公式日次5件/残0。採点score/error/rawstatusは初回未返却でworker段階やhidden runtimeを推定しません。基準p005・自己0.950を維持します。

native392先頭176は[SUMMARY25独立項目](experiments/native392-shard01-summary-independent-review-range-v1-20261008.json)に加え、**取得したprocessed archiveの[13独立項目](experiments/native392-shard01-completed-cache-review-20261008.json)**も合格。4,741,650,198bytes/4.416GiB、外側ZIP/176NPZ/880配列のSHA・安全な圧縮size・shape/dtype/fingerprint/mask/位置/flags/coverage/ゼロpaddingを確認。valid53〜124/計15994、local直読解凍SHA/QC59.999秒、Kaggle27.13分/peakRSS1,508,470,784bytes/byteQC514/予算skip14/一時0。PCへ取得したのはprocessed176だけ、元DICOM・教師・正式fitなし。残23本を同じ正本V2で最大3件ずつ進めるone-off準備中、全4207 completeindexができるまで正式fitを開始しません。[ledger98→101](artifacts/research/20261008-p009-submit-native176-record-v1/receipt.json)は旧byte prefixを保持。

[22:42公式Public上位100件](experiments/current-public-leaderboard-head-20261008.json)は首位0.964、0.955以上73件、厳密な0.960超13件。目標を達成している他チームがある事実と、再現できる公開レシピの有無を分けます。[追加web調査](experiments/post950-public-followup-20261008T2234.json)では作者0.941の独立追加例を確認したが、今回の範囲で再現可能な0.955超を確定していません。旧記事のpublic ceilingや順位を現時点へ流用しません。別解像度224追加p010は人工CPU/CUDA検査後の独立review中で、producer例外のqueue accountingを起動前に補正中。candidate_gate=false、実T4/実MRI/競技提出は未開始です。

以下の22:27以前の表示は、その時点の履歴です。

**22:27 JST更新：p008半月板10%も公式COMPLETE・自己Public0.950です。** [採点10項目](experiments/p008-scored-review-20261008.json)。p006/p007/p008は表示改善がなく、追加計算を要するため非採用。3桁表示の同値を未丸めAUC/予測同値と断定せず、基準14/p005を保持します。0.955・厳密な0.960超は未達成。

p009画角140→160の修正診断savedV2は公式COMPLETE、[実32件72項目](experiments/p009-profile32-completed-review-v3-20261008.json)成功。両main各64forward/320px/全gradFalse、mask差0、reader24forward/3model/B4、2固定rank融合の独立差0。main160観測52.96秒/GPU各3,319MiB、順序・cache条件があるため速度改善や精度改善の根拠にはしません。可視logの重複表示は実呼出し数ではないため、元checkerを保持して実hook/source/CSVに基づくv3へ補正。[root17項目](experiments/p009-serving-prelaunch-root-review-20261008.json)から正本19をprivate `rsraki/p009-coat-fov160` **savedV1/kernel137675752**として22:08:33 JSTにAPI起動。保存source/環境と可視CSVを確認してから競技提出し、診断CSVは提出しません。p008採点とp009診断/serving起動を[ledger95→98](artifacts/research/20261008-p008-p009-progress-record-v1/receipt.json)へ追加、旧byte prefix保持。

native392先頭176 CPUは公式COMPLETE。取得clientのGetKernel DTO/source場所、basename対fullpath、信用できないListKernelFiles sizeに対応し、旧失敗観測を保持。HEADがbyte証拠を返さなかったため、別one-offの4MB制限SUMMARY GETと**archive body消費0のRange headers照会**で実総長4,741,650,198bytesを確認、JSON archive.bytesと一致。[compact取得receipt](artifacts/research/20261008-native392-shard01-range-metadata-v1/archive-http-receipt.json)。JSON報告は176/176・1627.635秒・byteQC514/予算skip14・一時0で、独立source/plan照合とarchive/176NPZ検査の前です。残23本と正式fitは待機、元DICOMをPCへ取得しません。

以下の21:42以前の表示は、その時点の履歴です。

**21:42 JST更新：p007鏡映も公式COMPLETE・自己Public0.950、表示改善なしで非採用です。** [採点証拠10項目](experiments/p007-scored-review-20261008.json)。p006 k24とp007 mirrorはどちらも自己0.950で、計算量の少ない元のp005/k16/plainを維持します。未丸めAUCの同値は断定しません。p008 ref56950968は21:42照会にscore/error/rawstatusがなく、SDK既定PENDINGだけではworkerの状態や所要時間不明です。0.955および厳密な0.960超は未達成です。

p009の画角140→160比較は、保存V1が最初の140側で停止。[実ログによる原因確認](experiments/p009-profile32-first-failure-review-20261008.json)で、診断helperの別exec名前空間に作られた関数をPoolが解決できないPicklingErrorと特定しました。採点Notebook19や従来の隠し失敗の原因と混同しません。登録ModuleType内で実mainを実行する修正版19a/v2を作り、[8成功・Windowsfork1skipの回帰検査](experiments/p009-profile32-artificial-validation-v2-20261008.json)、[root18項目](experiments/p009-profile32-prelaunch-root-review-v2-20261008.json)、[独立35項目](experiments/p009-profile32-independent-prelaunch-review-v2-20261008.json)に合格。予測literal・基準14・候補19は不変、失敗V1のソース・ログも保持しています。同じprivate refの保存**Version2/kernel137665364**をAPI起動し、[保存版54項目](experiments/p009-profile32-savedV2-independent-binding-review-20261008.json)で全source/settings一致。fresh GPU残17.568h・Original/T4/OFF/Input1+1を記録。21:36 raw status未返却、SDK既定QUEUEDだけではworker状態不明です。38分のread-only selector監査でもv1/v2別sourceを取得でき、指定あり/なしのstatusはともに未返却。Kaggle実fork人工3taskと実32件が通るまで提出候補19を競技へ出しません。[追加ledger receipt](artifacts/research/20261008-p009-corrected-profile-record-v1/receipt.json)は91→92行です。

自作f005の[学習CLIと実processed115窓GPU QA](experiments/f005-native-training-preparation-validation-20261008.json)を完了し、[独立47項目・人工CPU10 tests](experiments/f005-native-training-independent-review-20261008.json)も合格。microbatch1/累積16、欠損を除く観測数で分母を統一し、末尾3件・AMP skip・Gold除外前の教師読取拒否・strict checkpointを確認しました。既存processed16の最大115valid検査はfreeze peak reserved196MiB、last2 checkpoint704MiB、保存再読込logits差0。実教師読取0・optimizer更新0・実学習未開始で、同時16件のGPU容量や精度向上の証拠ではありません。全4207のcomplete indexを必須にして、16件だけで誤って実学習できないようにしています。

24 shardのwhole-group画像exporterとfull indexは[正本V2/人工9tests/16準備項目](experiments/native392-full-shards-preparation-validation-v2-20261008.json)、[独立31項目](experiments/native392-full-shards-independent-prelaunch-review-v2-20261008.json)、[root16項目](experiments/native392-shard01-root-prelaunch-review-20261008.json)が合格。4207検査/4102groupを175〜176検査に配分し、最大37件のgroupも分割しません。教師/Report/Gold対象を画像exportに使わず、12GiB archive cap・4GiB native temp・1GiB helperを20GB内に収めます。独立監査で見つけたfinal elapsed・exclusive partial・index予算照合を修正し、未実行V1を保持しました。先頭**176検査/140groupだけ**をprivate CPU `rsraki/native392-cache-shard-01-of-24` saved**V1/kernel137670873**としてAPI起動、[保存ソース52項目](experiments/native392-shard01-saved-binding-review-20261008.json)で正本localV2/設定と一致。実QC/容量/時間/cleanup確認前に残23本は起動しません。PCへの元DICOM/全件processed cache取得・正式学習は未開始です。[cache準備/起動ledger receipt](artifacts/research/20261008-native392-shard01-progress-record-v1/receipt.json)は92→94、[p007採点receipt](artifacts/research/20261008-p007-scored-record-v1/receipt.json)で95行、以前のbyte prefix保持です。

[固定SWA Inputの追加重み監査22項目](experiments/publication-swa-v1-additional-checkpoint-audit-20261008.json)では、384版と同seed42の224版のみ、別seed/fold/個別epochはなし。224作者保存V2のlinked Public0.945は確認済みですが自己再現値ではありません。作者がGold58をcheckpoint選択に再利用した点を記録し、こちらのGold fit/選択へ流用しません。同じ384pixel前処理を共有できる静的根拠があり、別候補p010の同volume→base320 pairと独立stock224 single-view→事前固定90:10rank全12列を設計中。base出力不変・stock224厳密load・paired32資源と予測の一致・固定Input/環境の実検証は未gateです。現行p009/初回cacheのQAより先にremote起動しません。

以下の20:44以前の表示は、その時点の履歴です。

**20:44 JST更新：p006は公式COMPLETE・自己Public0.950、表示改善なしで非採用です。** [10項目の採点証拠](experiments/p006-scored-review-20261008.json)。未丸めAUCの同値を断定せず、計算量の少ないp005/k16を基準として維持します。p007は可視COMPLETE・[11項目のCSV/融合監査](experiments/p007-serving-visible-output-review-20261008.json)を通過し、20:38:57 JST保存V1を競技提出、ref**56951892**を取得。[独立19項目の提出binding](experiments/p007-submission-binding-review-20261008.json)も合格、直前残枠2。以後の残1は算術で、次回提出前に公式残数を再取得します。20:42のp007と20:35のp008はscore/error/rawstatus未返却です。

native392 CPU16は公式COMPLETE、162.109秒、前処理reference byteQC46系列・大きな2系列は比較予算によるskip、一時ファイル0、教師/Report読取0。[取得したprocessed cacheの独立検査](experiments/native392-pixel16-completed-review-20261008.json)で16NPZ/80配列のSHA・shape/dtype/全metadata/zero paddingを確認。圧縮397,707,087bytes=379.283MiB、valid窓53〜115・計1,406、ローカル直読＋解凍＋hash/QC合計4.720秒。全4207への平均単純換算は97.390GiB・CPU約11.84hですが母集団の上限ではなく、全件を一Notebookで処理せずshard化が必要です。PCへ取得したのはprocessed16のみで、元DICOM/全件cache/実label fitは未開始。次は最大valid検査のGPU負荷とmicrobatch1/累積16の損失等価・strict checkpointを検証します。

以下の20:30以前の表示は、その時点の履歴です。

**20:30 JST現在：自己最良は0.950、p006/p008は採点結果未返却です。** p006 ref56949480・p008 ref56950968の20:22照会にはscore/error/raw statusがなく、SDK既定PENDINGからworker段階や実行時間は判断しません。同じ候補は再提出しません。p008提出自体の[独立28項目](experiments/p008-submission-binding-review-20261008.json)も成功しました。

p007鏡映の実32件は[50/50結果検査](experiments/p007-profile32-completed-review-20261008.json)成功。全6slot/B4、plain24forward・mirror48forward/変換8回、全grad無効、固定融合最大差0。mirror72.05秒/plain59.84秒、GPU観測最大8,071MiBで、精度は測っていません。このgateから修正16 localv2をprivate `rsraki/p007-convnext-mirror` remote保存V1/kernel137659997として実行開始し、[21root検査](experiments/p007-serving-prelaunch-root-review-20261008.json)と[保存ソース48項目](experiments/p007-serving-saved-binding-review-20261008.json)を確認。可視完走・CSVの独立照合後に同V1を競技提出する順序です。

中期のnative392 pixel16も[人工8検査](experiments/native392-pixel16-preparation-validation-20261008.json)、[独立38項目](experiments/native392-pixel16-independent-prelaunch-review-v3-20261008.json)、[root15項目](experiments/native392-pixel16-root-launch-review-20261008.json)を通過し、private CPU `rsraki/native392-pixel16-diagnostic` 保存V1/kernel137660730をAPI起動しました。初期helperのshutil import欠落を起動前に修正し、正本18a/v3 SHAを固定。Competition＋decoder V2のみ、教師/encoder/学習/提出CSVなし。実16件の前処理byte QC・archive hash・容量・mask・一時ファイル解放が通るまで全件取得は開始しません。

[オンラインDINO最終2block実装](experiments/dinov2-last2-implementation-validation-20261008.json)は既存generic重みの出所hashを照合し、人工RTX4090で19契約と14 unittestが成功。全144窓のcheckpoint OFF/ONでlogits/勾配差0、peak reserved約4,270→806MiB、optimizer更新0・実label fitting0。全144validの純計算0.978秒/検査を3,386検査へ単純換算すると約55.2分/epochで、I/O・転送・optimizer・検証を含む学習時間ではありません。まず実キャッシュから負荷とFrozen対照との等価性を確認します。

次の主系FOV140→160候補は[静的監査](experiments/post950-coat-fov160-feasibility-20261008.json)を経て準備中。外側constantだけでは埋込fastreadに効かず、実workerが読む設定を照合します。320モデル/重み形状を保持してnominal FOV116.67→133.33mmとなる仮説で、尺度ずれの精度効果は未確認。採点済み改善を優先し、今日の残枠と候補結果を見て提出可否を決めます。[外部作者の独立reader例](experiments/post950-external-independent-reader-review-20261008.json)でも異なる表現を追加して得点が下がる例があり、多様性だけで弱いモデルを融合しません。

以下は各時点の履歴で、現在の状態は上記を優先します。

### 18:43 JST以降：API実行を許可されたp006の診断を開始

**20:15 JST更新：半月板専門枝p008も競技へ提出しました。** `rsraki/p008-apex-meniscus10` 保存Version1のref**56950968**、直前の公式日次残数3。[受理記録](experiments/p008-submission-accepted-20261008.json)。[保存版56項目](experiments/p008-serving-saved-binding-review-v2-20261008.json)と[実出力44項目](experiments/p008-serving-visible-output-review-20261008.json)が合格し、公式COMPLETE、specialist_used=true/stock1回、両T4 strict233 FP32・20forwardを実行。固定90:10と元Apexの融合CSVを独立計算してmaxdiff0、他10列の文字列とUID順保持。visible3件では両Meniscusの順位変更0ですが、10%融合で順位が維持される数学的ケースであり枝未使用ではありません。追加phase30.43秒、stock22.79秒。V1はfull12配列/個別maskを外部保存しないため、これらの一致はhash-bound sourceの実行時検査で、外部数値再計算とは区別します。p008もscore/error未返却です。

p007の32件診断も20:11:37 JST公式COMPLETE/no failureを確認。保存版の[独立50項目](experiments/p007-profile32-saved-binding-review-20261008.json)は成功し、全13cells/10codecellsと元source、修正no_grad、同32 UID、private/Original/T4/offline/受理Input pinsを照合。実mask・24/48forward・8mirror・GPU/RSS・融合CSVの独立監査を進めています。診断版は競技提出しません。

中期は[392px DINO最終2blockの微調整監査](experiments/dinov2-last2-finetune-feasibility-20261008.json)を20項目確認し、まずfold/groupを守った16件pixelcache診断と人工CUDAのfreeze/last2 profilerを実装中。既存192画像/保存CLSから高解像空間情報は復元できません。同じ392px/3系列×48窓ならraw上限は1検査63.31MiB・16検査約0.989GiB、全4207は約260.09GiB。全件取得/実label fittingは未開始で、実pilotの時間・byte QC・VRAMを先に確認します。元DICOMはPCへ全取得せず、学習用キャッシュだけを扱います。自己採点済み最良0.950、0.960超目標は未達成です。

**19:55〜20:00 JST更新：次の2候補をKaggleで実行検証中。** [17のp008](notebooks/public/17_submit_p008_apex_meniscus10.ipynb)をprivate `rsraki/p008-apex-meniscus10` 保存Version1/kernel137655738としてAPI起動し、19:59:39の公式statusはQUEUED、failureなし。元7codecellsを完全保持し、専門枝の固定10% rankをMeniscus2列にだけ追加。[オフライン26整合項目・38 unittest](experiments/p008-apex-menis10-validation-20261008.json)と[rootの独立25項目](experiments/p008-prelaunch-root-review-20261008.json)が成功。GPU quota rawの起動直前残は63,741.477秒（約17.71時間）、SDK duration JSONの24h切り落としを避けて計算しました。専門枝が実使用された可視結果・併用メモリ・CSV照合後に競技提出を判断します。

p007は未実行v1を保存し、runの `@torch.no_grad()` を保持した修正v2へ更新。[新16 unittest＋人工CUDA本番run](experiments/p007-convnext-mirror-validation-v2-20261008.json)はcaller grad=True・外側inference_mode無しでも12forward全grad=False、numpy/CSV出力成功。rootは[20独立項目](experiments/p007-profile32-prelaunch-root-review-20261008.json)で新sourceと同一32 UID・既存profilecodeを直接比較し、[16a診断](notebooks/public/16a_profile_p007_plain_mirror_32.ipynb)をprivate `rsraki/p007-profile-32-plain-mirror` 保存Version1/kernel137656226としてAPI起動。これは修正後localpackage v2を使った最初のremote Version1で、診断CSVは提出しません。p006は19:56:39時点でもscore/error未返却、自己最良0.950・厳密0.960超目標は維持します。

**19:36 JST更新：p006の保存済みVersion1を競技へ提出し、ref56949480を取得しました。** [受理記録](experiments/p006-submission-accepted-20261008.json)。直前の公式日次残数は4、19:36:43の照会ではPublic/error/raw statusは未返却です。SDKの既定PENDINGを実際のworker稼働と解釈しません。自己最良0.950・14基準は保持し、0.955/0.960超は未達成です。

[256検査の結果](experiments/p006-profile256-completed-review-20261008.json)は公式COMPLETE、76項目合格、両融合CSVの独立rank再計算最大差0。全256件でreader6slotが有効、最大batch4を実処理し、k24のsampled GPU peakは11,561MiB。main313.74秒、reader16/24は234.96/273.52秒、k24 process-tree RSS合計は約10.81GiB。固定順・共有ページ重複を含む観測で、隠し時間・全native decode・AUC改善の保証ではありません。提出用15はprivate `rsraki/p006-convnext-k24` V1として完走し、[保存ソース/環境44項目](experiments/p006-serving-saved-binding-review-20261008.json)と[可視3件のCSV/融合22項目](experiments/p006-serving-visible-output-review-20261008.json)が成功しました。

別候補のMeniscus10専門枝はprivate `rsraki/public0033-meniscus-profile64` V1で実64件・T4×2・stock FP32/batch32/10windowを完走。[独立結果監査32項目](experiments/public0033-profile64-result-review-20261008.json)。checkpointの実SHA/bytes、233 strict state、両GPUの実forward、raw2/full12の一致を照合。decode/cache108.32秒、stock96.41秒、契約完了まで219.25秒、GPU最大観測7,511MiB/個。298/384slotが有効、選択3557frameのdecode失敗0ですが、実証は非圧縮ExplicitVRLittleEndianのみです。全weakfit重みのため独立OOFや精度根拠ではありません。14を基に固定90:10のMeniscus2列のみ追加する別提出版を準備し、他10列・UIDを保持します。

**19:43 JST：p007鏡映候補は実行前監査で保留。** helperを `def run` の直前へ挿入した際、元の `@torch.no_grad()` がhelperに移り、runの勾配無効化が抜ける生成不具合を発見しました。人工検査の外側inference_modeがこれを覆い隠していました。Kaggle実行/提出はまだ行っておらず、生成位置を修正し、runのAST装飾子と実際のrun呼び出しでgrad無効を検証する別v2へ準備し直します。[保持v1の独立AST監査](experiments/p007-prelaunch-decorator-defect-20261008.json)。旧package/証跡は履歴として保持。14と提出済みp006にはこのhelper挿入がありません。

19:44:35 JSTのp006照会は引き続きscore/error未返却です。[提出自体の独立46項目](experiments/p006-submission-binding-review-20261008.json)は合格。19:44の[最新公開調査](experiments/post950-public-refresh-20261008.json)では公式score順100件と上位20版の採点情報を再照合し、表示0.950超は未確認。これは調査範囲内の小数3桁表示で、非公開/未索引候補の不存在証明ではありません。タイトルの精度表示だけで変更先を決めません。

ユーザーの明示依頼を受け、APIでprivate Notebookの実行・検証後の競技提出まで進めます。目標は**自分の公式Publicが0.960を厳密に超えること**で、現在の採点済み最良は0.950のままです。14・既存採点版・重みを保持し、[15のp006](notebooks/public/15_submit_p006_convnext_k24.ipynb)はConvNeXt `run` の既定窓数16→24だけを変更しました。[実装](experiments/p006-convnext-density-implementation-20261008.json)、[人工検証](experiments/p006-convnext-density-validation-20261008.json)：8 unittest、RTX4090の人工encoderを用いた元Dataset/head forward、欠損/短系列、提出契約が成功。実重み・実MRIの精度はこの検査で確認していません。

[15aの32検査診断](notebooks/public/15a_profile_p006_k16_k24_32.ipynb)を18:43:50 JSTにAPIで保存・実行しました。private `rsraki/p006-profile-32-k16-k24` Version1、kernelId137644181、18:44の公式statusはQUEUED。goldを除いた固定weak UIDのみでCoAtNet1回とConvNeXt k16/k24を比較し、教師/Reportを読みません。観測はmask、実際の融合成功、同一UID/列/順位融合式、時間/RSS/GPUメモリで、精度比較や独立OOFではありません。診断のsubmission.csvは完了時に削除し、診断版を競技へ提出しません。32検査の保存環境と結果が通れば同じcohort規則の256検査へ進み、その後serving版を照合して1候補を提出する順序です。

**19:00 JST更新：32検査は公式COMPLETE、summary passed。** 両reader全32件で6 slotが有効、batch4の最大slot数を実際に検査。rootの標準ライブラリによる6CSVのUID/列/finite/range/hashと、元所見係数のrank融合再計算は37項目成功・最大差0。[実MRI診断監査](experiments/p006-profile32-completed-review-20261008.json)。CoAtセル56.52秒、reader16は61.29秒、24は62.25秒。固定16→24順でinstall/OScache効果も含むため純forward速度比ではありません。sampled device peakは16が8,069MiB、24が11,561MiB。元native DICOMの全件無skipや隠し完走・AUC利得は証明しません。

32検査のgate後、[15bの256検査](notebooks/public/15b_profile_p006_k16_k24_256.ipynb)を19:00 JSTにprivate `rsraki/p006-profile-256-k16-k24` Version1としてAPI起動し、19:01 QUEUEDを確認。診断CSVを競技へ提出していません。[16のp007鏡映候補](notebooks/public/16_submit_p007_convnext_mirror.ipynb)も別の一要因としてオフライン準備：base k16を維持し、valid sagittal slot0/3の隣接チャンネル逆順＋位置反射、coronal/axial幅反転を使う等率2view。7 unittestと人工CUDA契約は成功、実精度/実資源は未確認で、k24と組み合わせていません。別表現のMeniscus10は[単一枝adapter](scripts/public0033_streaming_adapter.py)を人工pixel/契約で検証し、T4×2/FP32/batch32の64検査診断を準備中です。[専門枝の現在の制約](experiments/meniscus10-integration-feasibility-v4-20261008.json)

18:35時点の元APIでGPU割当残18.10時間、18:33の日次提出残4。SDKのduration JSON化が丸1日を落とすため、quotaはraw値/`total_seconds()`で判定します。この時点では競技提出前でした。API保存応答が`/code/owner/slug`を返すため初回のローカル照合で例外になりましたが、保存Version1は受理済みでQUEUE→RUNNING→COMPLETEを直接確認し、再起動せず照合関数を修正しました。これより下の「未開始」「方針相談」の記録は前のターン時点の履歴です。

10月8日18:03:45 JST、公式GetSubmissionでref56943135/own Version1/script356323020を**COMPLETE・Public文字列0.950・エラーなし**と確認。作者0.950の再現に加え、自分の採点済み値として確定しました。[採点監査](experiments/p005-scored-review-20261008.json)。271位はユーザー報告でAPI直接確認とは区別、未丸めAUC/個別Public AUC/Private/隠し時間・reader fallback率は未提供です。旧pending監査を履歴として保持します。

[改善ロードマップ](experiments/p005-improvement-roadmap-20261008.json)を準備しました。Apexは既にCoAtNetのSWA・96slice/K94・解剖学的mirror2viewと3fold EMA ConvNeXtの所見別rank融合です。一般的なTTA追加や同じ重みの再配合から始めず、まず**ConvNeXt窓数16→24だけ**のp006を1候補として設計します。重み/主系/入力版/decoder/係数を固定し、人工variableK契約→Kaggle32/256件のcoverage/time/memory→保存版の限定比較へ進む計画です。推論・精度未測定、ConvNeXtのforward1.5倍でも全体時間/VRAM倍率は不明。[構成と変更候補](experiments/p005-growth-architecture-review-20261008.json)

最新の公式スコア順100件・上位17版と公開検索では、実採点済み公開0.955以上を確認できませんでした。[候補再調査](experiments/post950-public-research-20261008.json)。中期は表現の異なる単一の専門枝を調査します。公開Meniscus10 V2のDINOv2-Baseは仮説候補ですが、全weakfitでOOF不可、既存cacheと非互換、standalonePublic/追加利得未確認。適合しなければ汎用初期化からfoldを守った空間2.5D/patchモデルを1系統作り、末尾fine-tuneを別要因にします。既存CLScacheからpatch情報は復元できず、192px画像のupscaleもnative情報を戻しません。

0.950→0.955は12所見AUC差の合計+0.060、0.960には+0.120が必要です。3所見で各+0.020・他が維持なら平均+0.005という算術例で、達成予測ではありません。自作f002のMCL/PF OAの弱点をApexの隠し弱点と同一視せず、単調な最終score補正や閾値変更だけでAUC向上を作れない点も踏まえます。

公式最新条件は[制約監査](experiments/competition-constraints-20261008.json)：日次5提出・最終選択最大2、最終提出**10月23日08:59 JST**、9時間/Internet OFF。10/8～10は基準固定とp006、11～15は1専門枝の監査/小規模検証、16～19は有望要因だけ確認、20～22は採点済み候補を固定する目安です。自分のgoldは学習/係数/crop選択へ使わず、公開全weakfitを独立CVと扱いません。今回の依頼は方針相談で、Notebook/重み変更・新MRI処理/取得・新学習・外部提出を開始していません。

## 待機中の3作業の履歴：f004は差し替え見送り

以下の採点待ち表示は18:03の採点確認前の履歴です。

ユーザーの実行依頼を受け、10月8日にローカルRTX4090でf004のfold0・1を24epochずつ完了しました。f003との変更はdropout0.2→0.4だけ、特徴/教師/seed/fold/LR/BCE選択を維持。新規runへ保存し、過去run・提出重みは保持しています。BCE最小の選択epochは15/14、head fittingは383.444/383.596秒で、CLI全体の399.96/396.30秒とは区別します。

| fold | f003 BCE → f004 BCE | f003 weak AUC12 → f004 | 補助11平均の差 |
| --- | --- | --- | --- |
| 0 | 0.316232 → 0.315904 | 0.823060 → 0.815188 | −0.002223 |
| 1 | 0.317469 → 0.320704 | 0.815430 → 0.814692 | −0.000543 |

両foldでAUC12/補助11が下がり、fold1はBCEも悪化。事前の採用条件を満たさず、f004へ差し替えません。rootは両モデル・両foldの選択CSVから標準ライブラリだけでBCE/AUCを独立再計算し一致を確認しました。[保存結果の独立確認](artifacts/research/20261008-waiting-work-execution-v1/selected-prediction-verification.json)。[完成比較](experiments/f003-f004-fold01-review-20261008.json)は236項目成功、各fold3000回のgroup bootstrapで補助11差の95%区間は両方0をまたぎます。PF OAの点改善は両foldですが全体の置換根拠にはしません。弱いReport教師での比較で、Publicスコアの比較とは別です。

教師17セルの二次レビューも完了。閾値未満2・明示的正常1・対象への直接言及なし3・未確定11。整理できた6は既存0/欠損と整合し、**教師訂正の採用は0件**です。11の根拠不足を推測で補わず元教師を維持します。[二次レビュー](docs/label-review.md#17セルの二次レビュー結果10月8日)。gold/保護fold除外・原値/入力hash保持など20項目を確認し、原文はprivate artifactに保存しました。

17:45:41 JSTにp005提出ref56943135を最後に公式SDKで確認。スコア/エラー未返却、raw status省略にSDKが既定値PENDINGを当てており、queue/worker実行中や隠し成否は分かりません。[最終状態](experiments/p005-status-final-20261008.json)。この結果が出るまで別公開対照・混合係数・長い学習へ広げず、自己p002の0.937と提出p005の保存版を維持します。今回実施した実学習は取得済み特徴だけで、MRIdecode/新モデル・DICOM取得/外部upload・submitはありません。台帳は旧53行を保持して実学習・比較・教師再確認・状態照会の5イベント追記、58行です。[全体実行記録](experiments/waiting-work-execution-review-20261008.json)

## p005提出と待機中の準備記録

以下は今回のf004実学習・二次レビュー前の準備時点の記録です。

ユーザーが公開0.950原版のコピーを提出しました。10月8日17:02 JSTの公式SDK照会で、提出ref **56943135**、`rsraki/rsna-knee-apex-grandmaster-stack` **Version1/script356323020**と保存Outputを照合。作者採点済みV1の全7 code cell・全source cell、Original container digest、T4×2・Internet OFF、3 Input参照名が一致しました。可視ログの3checkpoint検出・reader CSV・Apex Fusion SUCCESS、CSV13列/UID順/36有限確率、所見別rank融合式の独立再計算差0など29項目が成功しています。両公開DatasetはV1のみですが、保存Inputの内部版ID/競技bundleはAPIに出ず直接未確認です。**自己Public・採点終了状態はまだ確認できません**。[今回の保存提出監査](experiments/p005-submitted-review-20261008.json)

待ち時間は、提出済み版を保持して[教師の未確定17セルの再確認資料](docs/label-review.md#p005の採点待ちに使える17セルの再確認資料)を用意しました。原文・記入欄と所見ごとの確認事項、gold/保護fold除外を再検査し、教師は未変更です。独自headのGPU比較を行う場合は、準備済み[f004](configs/experiments/f004-dinov2-attention24-dropout40.json)のdropout0.2→0.4だけを変えた両fold比較が任意の次工程です。前回f003のhead fittingは約6分/foldで、事前検査は別です。現在f004両runは存在せず、assistantは学習を開始していません。**この独自head学習はApex提出の一部ではなく、提出例外の修正やPublic向上を証明しません。** [操作と判定](docs/frozen-features.md#次の学習候補f004と教師監査)、[優先順位と結果後の分岐](experiments/p005-waiting-work-plan-20261008.json)

13のKaggle1,300件MRI診断は独自f002の提出経路を継続する場合の任意作業です。p005の採点完了の前提ではありません。別公開版の同時提出・係数探索・epoch追加は今回結果と補完性を確認してから判断します。採点成功ならその保存版を新基準候補として記録、自己0.937と比較します。一般例外なら同じ保存版を繰り返し提出せず、原版との再現条件と実行段階を再確認します。

## 10の失敗後に用意した公開対照とf002診断

10月8日15:46 JSTの公式SDK照会で、提出ref **56904146** は `rsraki/notebook428d1e17f8` **Version1/script355977821**を使用し、配布10と全source cell一致・GPU有効・Internet OFF・正しい3 Inputsを確認しました。添付summaryは保存Outputとbyte同一、可視3検査を20.300秒で完了、CSVのUID順・13列・36確率・hash等22項目を独立に検証しました。それでも提出は一般例外・Publicなしです。head埋込みによる接続回避は可視実行で機能しましたが、隠し推論は直っていません。同じ10の再提出は保留します。[提出版・CSV監査](experiments/f002-embedded-head-hidden-failure-review-20261008.json)

件数を3に固定する実装、Report依存、可視CSVの破損は見つかりません。人工1,300検査・5,842系列metadataでは動的UID順・全15,600確率・提出契約が成功しました。ただしMRI/encoderはmockで、隠し完走の証拠ではありません。元画像系列全体の配列重複と、選択系列のspacing/物理位置/強度が不適合なら検査全体を停止する経路を再現しました。元readerは既に学習側4,207件を完走しており、**これらを今回の真因と断定しません**。隠しtracebackとworker実行時間は取得できず、9時間超過とも断定できません。[提出契約の独立監査](experiments/serving-contract-independent-review-20261008.json)、[停止経路の人工再現](experiments/serving-strict-stop-review-20261008.json)

p005は **作者Public0.950の採点済みVersion1/script356192954** の固定対照です。[14の固定ソース](notebooks/public/14_submit_p005_public950_control.ipynb)、[準備手順](docs/public-baselines.md#10月8日の優先工程p005の公開0950固定対照)。ユーザーは原版をCopy & Editして上記保存Versionを提出しました。14はコードを固定/照合するコピーで、原版の正しいコピーなら追加Import・別実行は必須ではありません。作者採点と自分の再現値を分け、自己0.950や隠し完走を保証しません。

最初の候補監査では0.950/0.949のOAI履歴を理由に保留しました。その後、最新公式Rulesの外部公開models許可・事前学習入力のライセンス例外・重み配布cardの大会利用許可を再確認し、**元OAIを取得せず公開checkpointを固定利用する研究として条件付き採用**へ更新しました。Hostの個別承認や作者元訓練の適格保証とは扱いません。[再監査と判断の更新](experiments/public-pretrained-eligibility-20261008.json)。0.946/0.945は必須private Inputsで完全再現不能です。[元の候補・版・利用履歴監査](experiments/public-candidates-20261008.json)

**p004：作者Public0.944の採点済みVersion2/script355300588**も独立対照として用意しました。[12の固定ソース](notebooks/public/12_submit_p004_public944_control.ipynb)、[手順](docs/public-baselines.md#10月8日の次工程p004の固定対照)。元29 code cell・19 Input版・Original31430/Python3.12を維持します。CP311/CP312 wheelsのため最新Python3.13へImportするだけでは再現条件が揃いません。Copy & Editが表示中V2を確実に複製するか未確認なので保存後に照合します。現在はInput数の少ないp005を先に実測し、12を同時に提出しません。

f002はメモリ配送だけを変えたreaderを別ファイルに準備しました。全系列の正規化配列をRAMへ重複生成せず、temporary memmapと必要スライスの正規化を使用します。元のpercentile/物理順/letterbox/系列選択は維持。人工画像では旧readerと入力・特徴配列がbyte同一、固定学習済みheadのRTX4090予測差0、NumPy2.5.3/2.1.3で検証しました。memmapのpartitionでRSSが増える可能性と追加disk/IOは残り、OOM解消を保証しません。[13の1,300件診断](notebooks/public/13_profile_f002_memory_1300.ipynb)は提出用ではなく、実MRI/Kaggle未実行です。[診断手順](docs/frozen-features.md#10月8日以降のf002診断)

両Datasetの選択版がVersion1であることは、今回ユーザーからも確認できました。内部版IDのAPI直接観測とは区別します。長い追加学習や教師変更は提出経路と根拠の確認後へ回し、待ち時間の小さい比較は上記f004を任意で実行できます。f003のfold0のみ延長利得・fold1不変、教師150件/30セル一次レビュー・17セル未確定という状態を記録しています。今回も実モデル/MRI追加取得・実MRI処理・実学習・外部アップロード/提出をassistantは行っていません。p002自己0.937と過去runを保持します。

## 履歴：09は同じheadの接続で再発停止、10はhead Inputを不要にする

以下の「次は10」は10月7日時点の判断です。10は現在、可視成功後の隠し例外を確認済みで、公開p005を提出して採点結果を待っています。

10月7日14:58 JSTの公式SDK照会で、ユーザー指定`rsraki/notebook6372fe6490` **Version1/script355963229**は配布09と全cell一致、必要4 Inputs・GPU有効・Internet OFFを確認しました。それでもPython開始前のhead Dataset12402506/内部版20385564接続で`mkdir ... Read-only file system`となり、ERROR・Output0・log`[]`です。08 Version1の同じ停止→CPU確認/08 Version2成功→09停止という再発を確認し、単に接続し直す対応では安定化していなかったと判断します。21項目の監査が成功しています。[再発監査](experiments/f002-recurrent-mount-review-20261007.json)

次の提出候補は[10_submit_f002_embedded_head.ipynb](notebooks/public/10_submit_f002_embedded_head.ipynb)。監査済みf002 fold0 epoch11の小さいheadをNotebookへ同一byteで埋め込み、サイズ/SHA256検査後にworkingへ復元します。変更は重みの配送方法だけで、09の前処理・encoder・head・decoder・推論は維持。**新しいprivate NotebookにCompetition＋既存encoder/code＋既存decoderの3 Inputsだけを接続し、旧head Datasetを追加しません。** パスだけ変えて旧headをInputsに残すと、同じ起動前mountが発生し得ます。[現在の操作](docs/frozen-features.md#head接続の再発後は10を使う)

10は497,221byte、重み3ファイルの同一byte復元・新規6 unittest/Ruff・人工GPU17件の最大確率差0・root独立16項目が成功。[準備監査](experiments/f002-embedded-head-handoff-20261007.json)。元08/09・16core sourceと過去runは不変。Kaggleの実行はユーザーが行い、採点成功はまだ検証していません。

p002自己Public0.937は今回も公式APIで確認し、成功保存版も不変。p003作者0.943は自己再現値ではなく、p003と独自f002の自己Publicはありません。今回の起動失敗、以前の隠し`Notebook Threw Exception`、学習精度を分け、全体共通原因や9時間超過を断定しません。10のKaggle起動/隠し完走/精度も未確認で、他Inputsのmountまでは回避しません。[公式資料の調査](experiments/kaggle-runtime-primary-research-20261007.json)

08の256件完走は有効な既存証拠なので再実行を前提にしません。p003は64件で未到達だった**DINO cache133件のmemmap分岐**を発見し、[11の256件診断](notebooks/public/11_profile_p003_stress_256.ipynb)を準備しました。22推論セル・14 Inputs・モデル/混合は不変、reference再計算OFF、資源/child/型付きeventの観測を追加。11人工テスト・Ruff成功、実MRIは未実行です。Radの669/892件境界は256件では未到達。元作者の一部CoAt失敗後の継続と、こちらの追加guardの停止も異なるため、元予測式のコピーだけから自己0.943を保証しません。[静的監査・限界](experiments/p003-scale-audit-20261007.json)、[11の独立診断手順](docs/p003-speed.md#新たに確認した規模分岐と元作者版との差)

直近は10を優先し、11は独立したp003原因調査としてその後またはGPU枠に余裕があるときに実行します。追加学習より提出経路を確認します。f003のfold0のみ延長利得、fold1不変という結論と教師未変更を維持。f004は準備のみです。headを含む生成10は個別Git除外し、重みを履歴へ入れません。

## 履歴：08の256件診断が完了、当時は09へ進む判断

以下の再試行・「次は09」の記述は各作業時点の履歴です。現在のユーザー操作は上の10・3 Inputsです。

`rsraki/notebook245423d388` **Version2/script355957807**を10月7日14:35 JSTに公式SDKでCOMPLETE・GPU有効・Internet OFFと確認。配布08と全source cellが一致し、添付`F002_STREAMING_SUMMARY.json`は保存Outputとbyte同一でした。256検査をskip/fallbackなしで完了、固定f002 fold0 epoch11/checkpoint/feature契約、decoder導入・圧縮4形式の人工画素一致、保存CSVの256 UID/order・12所見・3,072確率・hashを独立に検査し32項目が成功しました。[完走監査](experiments/f002-streaming-profile-review-20261007.json)

処理計測748.383秒（12分28.4秒）、約2.923秒/検査。Tesla T4、CPU peak RSS4.426GiB、選択GPUのPyTorch peak reserved364MiB、最後の空きdisk19.501GiB。計測はrun_streaming内で、前段setup/decoder installやqueueを除き、旧経路との速度比較ではありません。先頭headerの集計768系列は非圧縮ExplicitVRLittleEndianで、実圧縮MRIの網羅的検証や隠し完走は未確認です。

次は[09](notebooks/public/09_submit_f002_streaming.ipynb)を**別の新しいprivate Notebook**へImportし、同じ4 Inputs・T4×2・Internet OFFでSave & Run Allします。公開test全件のsummary passed/complete、processed=studies、receipt.valid=trueを確認してから、その09保存版のsubmission.csvをユーザーが手動提出します。08/profile_predictions.csvは提出しません。[具体的な操作](docs/frozen-features.md#08の256件診断が成功した後の09手順)。元head再アップロード・再学習・06再実行は不要。07の隠し例外原因と今回対策の隠し成否、自己Publicは未確認のままです。

## 履歴：CPUでのhead内容確認の成功

ユーザー提供`F002_HEAD_MOUNT_CHECK.json`はpassed/complete、元の`rsna-f002-head-fold0-v1`を接続し、best.pt・config.json・checkpoint-provenance.jsonのサイズ/SHA256が監査済みZIPと全件一致しました。Python3.13.15、model_loaded=false・training_performed=false。この確認では元headの接続問題は再現せず、当時は元headへ戻す判断でした。[当時の添付監査](experiments/f002-head-mount-success-review-20261007.json)。その後の公式SDKで`rsraki/notebookb1f7bfaac7` Version1/script355955526のsourceが配布08aと一致・COMPLETEを確認しました。ただし実設定はCPU・4 Inputs・Internet ONで、head単独の分離試験を実施した証拠にはなりません。[保存設定の補足監査](experiments/f002-recurrent-mount-review-20261007.json)

次は[08](notebooks/public/08_profile_f002_serving_256.ipynb)を元の4 Inputs・新しいGPUセッション・Internet OFFでSave & Run Allし、`F002_STREAMING_SUMMARY.json`の256件完走を確認します。head単独接続の成功は4 InputsのGPU起動・decoder/MRI・09隠し完走を保証しません。08確認後に09へ進みます。[次の具体的操作](docs/frozen-features.md#head単独確認の成功後に行う操作)

## 履歴：08のhead Input接続失敗

`rsraki/notebook245423d388` Version1/script355951711は、10月7日13:53 JSTの公式SDKで**mount data ERROR・Output0**でした。配布08の全cellと一致、GPU有効・Internet OFF・必要4 Inputsは揃っています。head Dataset12402506/内部版20385564はAPIready・Version1、3ファイルのmetadata取得も成功。停止箇所はKaggle管理下の`/tmp/kglt/...`へのmkdirでRead-only file systemとなっており、**Python・decoder・MRI処理・モデルロードには未到達**です。内部host/cache/storageの原因とKaggle全体障害は未確認。[監査記録](experiments/f002-input-mount-failure-review-20261007.json)

当時はセッション停止→head参照の再追加→08再試行、続く場合に[CPU・head単独08a](notebooks/public/08a_check_f002_head_mount.ipynb)で切り分ける方針でした。その後、上記のユーザー添付でhead単独の接続・3ファイルhash一致を確認しました。元GPU版08の再試行・256件完走は未確認です。[復旧手順の履歴](docs/frozen-features.md#08のinput接続エラーを復旧する)。07の隠し例外は別件のままです。

## 07の隠し例外・08/09・学習比較の状況

公式SDKの**10月7日13:04 JSTの確認**で、提出ref **56897486**、Version2/scriptVersionId355923367はCOMPLETE・errorDescriptionあり・自己Publicなしでした。提出11:16 JSTから約1時間48分以内に失敗を確認しましたが、queueと実行時間の内訳は不明です。保存可視版はGPU有効・Internet OFF、配布sourceと一致し3検査を完走。取得ログには隠しtracebackが含まれず、原因を9時間超過やGPU設定に断定しません。[失敗の事実・原因候補・対処記録](experiments/f002-hidden-failure-review-20261007.json)

次は**[08の256検査診断](notebooks/public/08_profile_f002_serving_256.ipynb)→確認後に[09の提出候補](notebooks/public/09_submit_f002_streaming.ipynb)**です。前処理moduleとf002 fold0 epoch11を固定し、圧縮decoderのoffline導入・自作数値画像チェック、検査ごとの特徴→予測、失敗phase/系列header/資源の記録を追加しました。新規private Inputは[約5.33MBのdecoder ZIP](artifacts/kaggle/dicom-decoders-py313-v2-20261007/rsna-dicom-decoders-py313-v1.zip)のみで、encoder/head再アップロードは不要です。Kaggle Linux Python3.13と実MRI/隠し完走は未検証。人工17検査・実checkpointのGPU推論差は最大1.1921e-7、圧縮4形式のWindows人工画素一致、関連21 unittestは成功しました。[ユーザーの操作](docs/frozen-features.md#今行う08の診断と09の提出準備)

ユーザーが**f003・24 epochのfold0/1を完了**しました。最初の12 epochはf002と予測byte同一、136項目の監査が成功。fold0はBCE選択11→15でBCE0.319130→0.316232、weak AUC12 0.813290→0.823060、補助11平均0.811771→0.820611。fold1はepoch12・予測・model tensorが元f002と同一で延長利得はありません。後半はtrain BCE低下とweak BCE上昇が両立し、延長が全foldで有効とは扱いません。[完成監査](experiments/f002-f003-fold01-review-20261007.json)、[fold0 bootstrap](experiments/f002-f003-fold0-bootstrap-20261007.json)。09の重みは変更せず、次の小さな比較として**dropoutだけ0.2→0.4**にした[f004設定](configs/experiments/f004-dinov2-attention24-dropout40.json)と[計画](experiments/f004-dropout40-plan-20261007.json)を準備しました。実学習は開始していません。

教師側はgold・保護fold0/1の連結groupを先に除いた**実150検査キュー・重点30セルの一次根拠レビュー**を作成しました。MCL9、PF OA7、Effusion7、Synovitis7で、17セルは程度/範囲/言語等が未確定。誤ラベル数・母集団の誤り率とは解釈せず、教師は未変更です。大会の重症度・量・急性/既往の条件を固定して再確認し、その後に教師だけの別比較へ進みます。[監査状況](docs/label-review.md)、[原文を含まない集計](experiments/training-report-evidence-review-20261007.json)

p002の採点済みPublic0.937を保持します。作者0.943は自己再現値ではありません。p003の一般例外とf002の共通原因は未確認。新encoder/patch特徴・全fold・混合提出にはまだ進まず、まず診断と提出経路を確認します。今回assistantは実MRI処理・学習・実データ/モデル追加取得・外部アップロード・提出を開始していません。

## 履歴：f002の保存推論が完走、次はVersion2の手動提出

以下は提出前の監査と操作の履歴です。手動提出は現在完了しています。

ユーザーが実行設定を修正し、`rsraki/notebookdd86797040` **Version2 / scriptVersionId355923367**でf002単独推論を完了しました。公式SDKでもCOMPLETE・GPU有効・Internet OFF、配布07と全5 source cell一致、3 Inputs接続、Output8件を確認。添付`F002_INFERENCE_SUMMARY.json`は保存版Outputとbyte同一でした。[可視推論監査](experiments/f002-visible-inference-review-20261007.json)

Tesla T4で公開test3検査・9系列の特徴抽出と予測が成功し、summaryはpassed/complete、cache_complete=true、studies=cached_studies=3、receipt.valid=true。計測は特徴抽出31.2499秒、head/CSV検査0.3927秒、推論セル全体31.7866秒です。前段import/setupやKaggle待ち時間を含むNotebook全体の時間ではありません。独立にCSV本体の12所見順・3 UID/order・36個のfiniteな[0,1]値とhashを検査し、f002 fold0 epoch11のcheckpoint/特徴/前処理契約一致を含む19項目が成功しました。

直前のVersion1はGPU無効・Internet ONで、最初のCUDA guardが停止しOutput0件でした。今回のVersion2でその設定による可視停止は解消しました。[失敗版の記録](experiments/f002-saved-run-failure-review-20261007.json)。p003の隠し例外とは別件です。

当時は同じ保存Version2の`submission.csv`を手動提出する方針でした。その後ユーザーが提出し、隠し例外になったため、現在の操作は上記08→09です。公開3件の完走から隠しtest時間/完走/0.943を保証しません。[当時の提出前確認](docs/frozen-features.md#07の完走確認と次の提出)

## 履歴：fold1でもAttentionの改善を確認、f002単独推論へ

ユーザーがfold1のf001/f002を完了しました。両runと全24 epoch予測・best/last checkpointを監査し、121項目が成功。学習3,316/検証891、12 epoch・各208更新、seed20261002、RTX4090、特徴/教師/source/環境一致、gold非使用、両方BCE最良epoch12でした。[fold1監査](experiments/f001-f002-fold1-review-20261006.json)

mean→attentionでweak BCE **0.334235→0.317469**、12所見AUC **0.792571→0.815430**、Synovitisを除く補助11平均 **0.783086→0.808136**。9/12所見で改善し、Synovitisは0.896907→0.895670とほぼ同じなので、今回はその不安定な所見に依存した改善ではありません。861 supplied groups・3,000回のpaired bootstrapで、11平均差+0.025050の95%区間は[+0.012249,+0.037048]、12平均差の区間は[+0.010083,+0.035736]。全抽出で両平均を定義できました。ただしcheckpoint選択の不確実性・患者独立性・Publicへの一般化は含みません。[不確実性集計](experiments/f001-f002-fold1-bootstrap-20261006.json)

**f002をKaggleで単独評価する候補へ進めます。** 両foldでACL・内側半月板・Baker's等8所見の改善が続き、MCLとPF OAの低下も続きました。fold1 MCLは陽性16/陰性700、AUC0.691964→0.664554、PF OAは0.796471→0.772121です。所見別混合や教師変更は加えず、この弱点を記録してまず推論時間と自己Publicを測ります。選択epoch12は固定日程の末尾なので、収束完了を示す結果とは扱いません。head fittingはf001190.744秒・f002198.853秒で、提出推論時間ではありません。

初回の単独推論は、これまでのfold0基準に揃えて**f002 fold0のBCE選択epoch11**を固定しました。fold1は改善方向の確認に使い、異なる検証集合のAUCを比べて高いfoldのcheckpointを選びません。[設定済み07](artifacts/kaggle/f002-fold0-single-v1-20261006/07_predict_f002_fold0_single.ipynb)と[private head Input用ZIP](artifacts/kaggle/f002-fold0-single-v1-20261006/rsna-f002-head-fold0-v1.zip)を用意しました。既存encoder/code Inputを再利用し、元06・ローカル学習の再実行は不要です。実MRI/隠しtestの完走と独自Publicは未確認で、p002 Public0.937を保持、p003の例外は未解決です。[次の具体的操作](docs/frozen-features.md#次に行うf002単独推論の操作)

## 履歴：f001/f002のfold0学習完了、次はfold1で確認

ユーザーが4090でf001 meanとf002 attentionの各12 epochを完了しました。両run.jsonと全24 epochの予測CSV・best/last checkpointを監査し、117項目が成功しました。特徴/教師/固定fold/seed/環境/実装が一致し、config差はpoolingと仮説文のみ。各epoch 212更新、学習3,386・検証821、gold非使用、BCE最良epoch11が正しく採用されています。[比較監査](experiments/f001-f002-fold0-review-20261006.json)

mean→attentionでweak BCE 0.332475→0.319130、12所見AUC 0.770865→0.813290、補助11平均（Synovitis除外）0.800035→0.811771。9/12所見で改善しましたが、Synovitisは陰性1/陽性100で、12平均差の74.6%を占めます。MCLは陽性11/陰性672、AUC 0.755005→0.657197。所見別に一様な改善とは扱いません。

固定selected予測の812 supplied groupsを3,000回paired bootstrapした補助11平均差は+0.011736、95%区間[-0.002762,+0.024256]で0を跨ぎます。12平均の区間は1,873回の定義可能な抽出に条件付けられるため、頑健な全体改善の証明に使いません。同じfoldでのcheckpoint選択も区間の対象外です。[不確実性集計](experiments/f001-f002-fold0-bootstrap-20261006.json)

**f002を暫定候補として、同じ12 epoch・seed・教師・特徴・BCE選択のf001/f002をfold1で比較します。** fold1は学習3,316/検証891、Synovitis陽性97/陰性25です。まだ別fold・別seed・Publicでの改善は未測定で、完全未観測のholdoutとは呼びません。06の再実行、epoch数/教師の変更、07の提出は現在の次工程ではありません。[次のコマンド](docs/frozen-features.md#fold0の比較結果と次のfold1確認)

head学習時間はf001 164.681秒、f002 179.893秒。Kaggleでの画像→特徴作成やローカルcache事前検査を含まないhead fittingの時間で、提出推論の時間へ換算しません。p002 Public0.937を保持し、p003の隠し例外は別の未解決問題として扱います。

## 履歴：05の2回目の例外と、06全件特徴の取得・検証完了

05の提出ref `56870280`、`rsraki/notebook59f34ce4e6` Version 1、scriptVersionId `355633504`は、公式SDKでerrorDescriptionあり・Publicなしと確認しました。配布05と保存版の全セルが一致し、可視3検査はREADY passed・256.538秒、4 CoAt系統のfallbackは0でした。取得できた実行ログはこの可視実行のもので、隠しtracebackと正味時間は未取得です。ユーザー報告の約4時間を9時間制限超過と解釈せず、**同じ05の再提出は保留**します。64件の数値一致は例外解消や本番規模の完走を証明しません。[2回目の失敗監査](experiments/p003-second-failure-review-20261006.json)

ご指定の`rsraki/notebookbd6efc1cf7`からコードとOutputを取得しました。Version 2、出力scriptVersionId `355640726`の06はPILOT=Falseのみを変更した全件実行で、4,207検査・12,621系列・386,196中心、特徴作成12,548.213秒（3時間29分8秒）、NPZ 281,589,228 bytesでした。全NPZ・studies.csv・export.jsonを`data/features/f001-weak-sv355640726/`へ保存し、metadata 26項目と全payloadのサイズ・SHA-256・shape・dtype・finite値・mask・fingerprintを確認しました。固定weak/foldとの一致・gold/group除外も成功しました。[全件監査](experiments/f001-full-feature-review-20261006.json)

これは**学習前の特徴作成の完了**です。headの実学習・checkpoint・weak評価・新しいPublicはまだありません。次はこのcacheから4090でf001 mean→f002 attentionを同一fold0・seed・教師・12 epoch・BCE選択で学習します。06を再実行する必要はありません。p002の採点済み0.937を保持し、p003の診断と独自モデルの比較を進めます。[今実行するコマンド](docs/frozen-features.md#全件の取得検証完了と今実行する操作)

## 履歴：06の32検査pilot確認完了、次は全weak特徴

ユーザー提供の`export.json`を21項目で確認し、32検査・96系列・2,953中心の特徴作成完了、全系列の物理順・正のPixelSpacing、固定fold0の学習側32件との一致、実装/入力hashとローカルInput梱包記録との一致を確認しました。Tesla T4で特徴作成83.018秒、NPZ計2,152,567 bytes。ユーザーは最後の表示画像に「明らかな崩れは見当たらない」と回答しました。画像自体やNPZは未提供で、目視はユーザー報告として区別します。[pilot確認記録](experiments/f001-pilot-review-20261006.json)

次は同じ06・Inputsで`PILOT=False`、`SHARD_COUNT=1`、`SHARD_INDEX=0`とし、**新しい保存実行**でweak 4,207検査の特徴を作ります。単純換算約3時間2分・NPZ約283MBは計画用の目安で、全件実測や上限ではありません。Outputの`frozen-weak-v1-shard0/`全体をローカルで検証してから4090学習へ進みます。pilotは学習に使いません。05の提出結果は別途確認し、まだ採点済みとは扱いません。[次の具体的な設定](docs/frozen-features.md#32検査pilotの確認結果と次の操作)

## 履歴：p003の例外確認と速度・独自学習の改善

ユーザー提供の`P003_SPEED_SUMMARY.json`を確認しました。新05診断はT4×2・64検査/366系列で完了し、変更したRaptor 2系統の入力・生予測・順位が完全一致、追加CoAt 4系統のfallbackは0でした。最終診断CSVの報告hashも旧04と一致します。総時間30分32.4秒には照合用の7分3.7秒が含まれ、単純差引き約23分28.6秒は旧04の23分47.8秒とほぼ同程度です。**大幅高速化や隠し例外解消は未確認**です。次は提出用05の保存実行へ進み、完了記録を確認して手動提出・採点を待ちます。追加学習や06/07は不要です。[今回の診断監査](experiments/p003-speed-profile-review-20261006.json)、[次の操作](docs/p003-speed.md#05の診断を実行した後にすること)

提出`56855132`（scriptVersionId `355483987`）は自己Public未測定で、公式SDKとユーザー画面の双方で隠し再実行の未処理例外を確認しました。**9時間超過とは未確定**です。SDKのCOMPLETEは処理終了を表し、errorDescriptionがある今回は採点成功を意味しません。保存版1の全セルは配布02と一致し、可視3検査のP003_READYはpassed・332.006秒。隠しtracebackと実時間は取得できていません。[失敗監査](experiments/p003-failure-review-20261006.json)

新しい[05・64検査診断](notebooks/public/05_profile_p003_speed_64.ipynb)は、Raptor GPU1の二つの入力方式を検査ごとに続けて処理してdecode cacheを再利用し、元経路の入力tensor・生予測との一致、GPUメモリ、処理時間を記録します。[提出候補05](notebooks/public/05_submit_p003_speed.ipynb)は別Notebookです。全モデル・係数・精度・失敗拒否・8時間内部予算を保持します。今回のnative処理のPyTorch最大予約メモリは1.55GiB、CoAt各worker最大は2.72GiBでした。これは各処理の計測で、GPU全体使用量や隠しtestのメモリ上限を保証しません。

精度改善は[凍結DINOv2特徴の学習経路](docs/frozen-features.md)を追加しました。Kaggleの元DICOMから392px・最大3系列×48枚の特徴を作り、4090で平均集約f001とAttention集約f002を一要因比較します。汎用encoderの出所・hash、特徴と前処理の共通契約、固定fold、gold非使用、欠損mask、CUDA必須、Report不要推論を維持します。新しい[ラベル監査キュー](docs/label-review.md)ではfold0/1とgold連結groupを保護して学習側の根拠を確認します。

今回は実装・一次資料調査・人工データの検査を実施し、実データ/モデルの追加取得、実MRI処理、実学習、外部アップロード、代理提出は実施していません。0.943は作者報告、0.937は自己実測です。[調査・比較順序・採否条件](docs/research/p003-improvement-20261006.md)。以下の「次は旧02提出」という記述は10月5日時点の履歴です。

## 最新：0.937の確定と次候補p003

23:37 JST追記：ユーザーが64検査の時間診断をKaggleで実行し、提供された`P003_PROFILE.json`は`profile_complete_not_for_submission`でした。T4×2、64検査・366系列、事前検査込み1,427.763秒（23分47.8秒）。DINO20の共有経路比較は全て差0、追加CoAt4系統の完了記録があり、元ソース／派生セルhashと実行順も一致しました。次は**同じ14 Inputs・T4×2・Internet OFFで提出用p003を別の新しいNotebookとして保存実行**し、P003_READY確認後に手動提出へ進めます。A5精度診断は先行必須ではありません。[今回の判断記録](experiments/p003-profile-review-20261005.json)。隠しtestの時間・Publicは未測定で、個別receiptや予測実体はこの添付から再検査していません。

Input追加で名前検索に出ない場合のため、[最新手順](docs/p003-next-step.md#add-inputの検索で見つからない場合)に基本URL貼り付けとDataset／Notebook Output／Model／Competitionの区別を追記しました。配布NotebookのImportだけでInputsが自動追加されるわけではありません。

公式SDKでp002の提出ref 56840796を **COMPLETE / Public 0.937** と確認しました。保存版1（scriptVersionId 355324155）の16セルは前回handoffと一致し、可視実行の全receiptもpassedです。約8時間はユーザー報告の経過時間であり、可視3検査の164.699秒や隠しtest正味時間とは区別します。[採点監査](experiments/p002-scored-20261005.json)、[固定成功版](notebooks/public/01_submit_p002_scored.ipynb)。

次候補は **haideptry V2のp003**。作者Public 0.943を確認した指定版ソースを取得し、実効14 Inputs、追加CoAtNet 4系統、確率平均後の全体順位化、失敗時の提出拒否を固定しました。[提出Notebook](notebooks/public/02_submit_p003.ipynb)、[実行手順・分析・0.955へ向けた順序](docs/p003-next-step.md)、[Input契約](docs/research/p003-input-contract-20261005.json)。自己Publicは未測定で、0.945到達を保証するものではありません。

先に[64検査の全体時間診断](notebooks/public/04_profile_p003_64.ipynb)で48件超の逐次CoAt分岐を確認し、その後p003を別のprivate Notebookで保存実行・手動提出します。[A5 BF16/FP16/FP32診断](notebooks/public/03_a5_precision_probe.ipynb)も用意しました。いずれもT4 x2・Internet OFFでユーザーが実行します。新候補の実MRI処理・重み取得・学習・アップロード・代理提出は今回行っていません。自作ResNet18の長期化は優先しません。

追加公開CoAtには作者のgold58によるcheckpoint選択履歴があります。凍結済み競技用モデルとして比較し、自分のgold利用や独立OOFとは分けます。0.955に向けた残差集約の新規学習は、特徴cacheと元学習除外UIDの対応を整えてから別実験にします。以降の0.924基準・p002未採点記述は、今回の更新前の履歴です。

19:29 JST追記：ユーザーがp002を提出し、約7時間後もNotebook runningと報告。公式SDKで提出ref 56840796、12:07:34 JST、`submission.csv`、status=PENDING、Public未確定を確認しました。経過約7時間22分は提出からの壁時計時間で、実際の隠しtest実行時間や進捗はAPIから取得できません。今回の多モデル構成で数時間かかることはあり得ますが、正常進行や9時間以内の完走を保証しません。現提出を維持し、結果確定後に採点と時間を記録します。[状態確認](artifacts/research/p002-status-20261005T102826Z/submission-detail.json)。下記の準備完了時点の未提出記述は履歴です。

## 今回の方針変更

**開発の主軸を公開学習済みモデルへ移します。** r001のPublic 0.749を公式APIで確認し、0.924の成功版ソースも保存しました。自作ResNet18は補完効果を調べる副系統として保持し、BNの別fold追試・50/100 epoch化・全fold展開・192px物理crop全件化を先に進めません。[提出監査](experiments/submission-audit-20261005.json)、[段階別計画と採否条件](docs/research/public-model-strategy-20261005.md)。

最初の新候補p002はDINOsaur V4のV32です。**第1段階の準備を完了し、ユーザーが第2段階のKaggle保存実行・手動提出へ進める状態にしました。** [提出用Notebook](artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)を新しいprivate NotebookへImportし、[操作手順](docs/public-baselines.md#ユーザーが行う操作)に従って指定5 Inputs・T4 x2・Internet OFFでSave & Run Allします。実行成功と`P002_READY.json`を確認して最終`submission.csv`を手動Submitします。実MRIでの互換性・完走・本番所要時間・Publicは未確認です。

元19参照を有効5 Inputsへ整理し、公開資産47ファイルの期待サイズ・SHA-256を固定しました。元10セルの推論ソース・前処理・係数を保持し、ファイル検査・各モデルの完走・fallback拒否・最終CSV検査を追加しています。重みpayloadのhashはKaggleで実行時に照合します。[Input対応表](docs/research/p002-effective-inputs-20261005.json)、[引渡し・検査記録](experiments/p002-handoff-20261005.json)。歴史的Input版と作者の採点CSVは未確認で、今回固定した組合せを作者0.937の再現済みモデルとは扱いません。公開0.940/0.941系は保存コード等の不一致があり、参考候補に留めます。[候補選定](docs/research/public-candidate-review-20261005.json)。

公開重みのgold履歴は競技用候補の監査として記録し、自分の学習・checkpoint選択・係数調整にgoldを使わない方針は維持します。公開モデルの学習に露出した検査を後から分割しても独立OOFにはなりません。今回は調査・未実行Notebookの準備・人工データでの検査・手順更新を実施しました。実データ/モデル取得・実MRI処理・追加学習・アップロード・代理提出は行っていません。以下の10月4日までの未測定・次工程の記述は当時の履歴です。

## 現在の状態

- GitHubの既存リポジトリに、実データなしで開発できる構成を追加。
- 調査の正本は [docs/competition.md](docs/competition.md)。原文は [ユーザー提供分析](docs/research/user-analysis-20261002.txt)。
- 今後の判断と三週間の計画は [docs/roadmap.md](docs/roadmap.md)。ユーザーはKaggleでキャッシュを作成し、ローカルGPUで学習する経路を選択済み。
- 初回の既存提出を10月3日に公式APIで確認。status=COMPLETE、実測Public 0.924、提出日時は10月2日05:57:22 UTC。提出URLのscriptVersionIdは354569007、該当Notebookの現在のVersionは1。保存Input版と実行時間、global IDと表示版の直接照合は未確認。自作ResNetのスコアとは別に記録した。
- 全件画像キャッシュの転送・検査後、研究用weakラベルの取得・監査・固定分割を実施。1 epochの動作確認と5 epochの基準実験が別runで完了し、自作重みとコードをローカルで梱包した。
- 10月3日、ユーザーの依頼でprivate Notebook `rsraki/rsna-knee` のVersion 1（ユーザー提示のscriptVersionIdは354838181）へAPIでアクセス。全件exportのcomplete=true、4,407検査、4,423ファイル、7,785,521,448 bytes、現在のsrcとのhash一致を確認した。
- 大きいZIPを避ける [download_cache_output.py](scripts/download_cache_output.py) を追加。取得先は `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/`。版確認、4並列、ファイル単位のhash検証・再開、取得後のexport検査に対応する。
- 全4,423ファイルを取得し、4,407検査のexport検査がvalid=trueで成功。サイズ・hash・UID・元CSV・src・前処理が一致。取得記録は親ディレクトリのtransfer.jsonでtransfer_complete=true。元DICOM・公開モデルは取得していない。
- 自作コードの検証範囲は [validation.md](docs/validation.md) に記録。
- ベースライン設定は [configs/baseline.json](configs/baseline.json)。学習コマンドはCUDAがなければ停止する。
- 現在の端末はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブの空き約485GBに対して公式元画像は569.76GBのため、全取得しない。
- Python 3.12.13の専用 `.venv` にtorch 2.10.0+cu128、torchvision 0.25.0+cu128が導入され、CUDA利用可・RTX 4090認識を10月3日に確認。学習環境とは別の `artifacts/tools/kaggle-venv/` にKaggle CLI 2.2.4を導入。CSV監査ツールの新規12件を含むunittest 31件が成功。
- [00_prepare_cache.ipynb](notebooks/00_prepare_cache.ipynb) と転送確認用 [verify_cache_export.py](scripts/verify_cache_export.py) を追加。Notebookは未実行テンプレート。
- 両Notebookはコードzipと展開済みInputの両形式に対応。Kaggleでの全件キャッシュ作成と転送、CUDA版PyTorch導入、ローカルの初回実学習は完了。
- coverageでは4,407検査すべて24 window、選択された13,221シリーズすべてphysical_position順、読み取りエラー・fallbackは0。ユーザーがKaggleで元画像とcacheを目視比較し、確認した範囲では問題なしと回答。比較した件数・UIDは未記録。
- 人工ノイズ画像でAMP・backward・optimizer・checkpoint保存・Reportなしの提出CSVまでGPUパイプラインを確認。これは精度や実データの所要時間の検証ではない。
- 公開ラベル7候補を監査し、vmohitrao Dataset Version 3を研究用に採用。10月4日に公式RulesとHost回答を取得し、研究・学習目的とCC BY-NC 4.0の帰属等を守って大会用途へ採用を進める判断に更新した。外部LLM抽出は条件付き許可、NC制限だけで大会側が禁止するものではない。根拠は [利用条件監査](docs/research/kaggle-source-eligibility-20261004.json)。gold非使用は作者READMEの宣言に基づき、非公開の開発ログを独立監査した事実ではない。
- `data/labels/vmohitrao-v3/` と `data/manifests/v1-vmohitrao-research/` にラベル・出所・groups・固定分割を保存。元train hashと4つのCSV hash、全state/value/maskが一致。weak 4,207件、gold 58件、全欠損138件とgold連結4件を除外。5-foldは821・891・888・778・829件。患者独立性は未確認。
- e001-smoke-fold0が完了。学習3,386／検証821件、weak検証BCE 0.430481、選択後のgold 12クラスmacro AUC 0.487178、epoch時間205.33秒。checkpointはweak BCEで選択。これはPublic LBではない。
- e002-baseline-fold0の5 epoch基準実験が完了。変更はepoch数1→5だけであり、再開せず同じseed・foldで初期化から実行。最良はepoch 1で、weak検証BCE 0.430481、選択後のgold macro AUC 0.487178。事前検査と最後のgold評価を含むtrain関数の所要時間は1,082.83秒（約18分、Python importを除く）。epoch数を増やしてもweak検証は改善せず、現行ランダム初期化モデルは実装上の比較基準として保存する。
- `artifacts/kaggle/e002-baseline-fold0/` にbest.pt、コードzip、hash manifest、RESEARCH-ONLY.jsonを保存。展開したzipのsrcが学習時のhashと一致し、実checkpointを使った人工test 3検査×12所見の提出契約がvalid=true。Reportを使わず予測できる。実testのDICOM decode、Kaggle全体時間、新規アップロード・提出は未実施。
- 提出の優先順位を判断する追加CSV診断を実施。学習側の所見別陽性率だけを全検査へ出す定数予測は、同じ検証821件・観測5,727セルのBCEが0.415327で、自作モデルの0.430481より低い。一方、自作モデルのweak macro AUCは0.557023、定数予測は0.5であり、画像の順位付け信号が全くないとは断定しない。e001/e002のweak・gold予測CSVはそれぞれhashが完全一致。集計は [提出判断の追加診断](experiments/e002-submission-review-20261003.json)。
- 10月4日、追加ログの学習不変性、cache互換性、共通正規化、ImageNet初期化、BN統計固定を実装。全71 unittestと人工CUDA検証が成功。少数train診断q001は初期eval BCE 0.694520→最良0.124106（epoch 32）、最後のtrain-mode BCE 0.000188。適合は進むがevalとの差が残り、画像ラベルの正確さや汎化の証明ではない。
- A/B/Cの5 epoch対照実験が完了。Aは旧e002の全epoch BCE・採用予測を完全再現。正規化だけを変えたBはweak BCE 0.411992／AUC 0.644652、encoder初期化だけを変えたCはepoch 2で0.371395／0.766895。CはBより12/12所見のAUCが改善し、次のローカル比較基準に暫定採用した。各約17〜18分、gold監査は無効・Public未測定。[比較集計](experiments/e003-e005-controlled-summary-20261004.json)、[採用理由](experiments/e005-decision-20261004.json)。
- Cの `artifacts/kaggle/e005-pretrained-imagenet-fold0/` にcode zip・best.pt・hash/帰属記録・未実行提出Notebookを準備済み。凍結コード＋実A/B/C重みで人工3検査×12所見・Reportなしの契約が成功し、CのImageNet初期化ファイル不要も確認。Kaggleへの新規アップロード・実test decode・採点は未実施。
- 続行の依頼を受け、BCE/AUC/last/milestone保存、実optimizer更新・AMP skip、進捗記録を実装。全97 unittest・Ruff・凍結コードの人工CPU/GPU提出契約が成功。e006〜e009の4本を新規runで完了し、CSV・入力/保存hash・選択規則・実更新の監査がvalid=true。[完成集計](experiments/e006-e009-extended-summary-20261004.json)。実checkpoint内部も別途CPUで照合した。合計約2時間47分（cache事前検査込み、import除く）。モデル保存は推論専用で、正確な学習再開は未対応。
- fold 1でもB→CのAUCは0.638538→0.732050、10/12所見が改善。通常BNの20 epochはAUC最良epoch 8で0.784410、BN固定はepoch 9で0.803614。ただしBNの補助11平均差の95%区間は0を跨ぎ、ACL・PF OA・Lateral Meniscusの大きな悪化とMedial Meniscus等の改善が入れ替わる。[対照実験と判断範囲](docs/controlled-experiments.md)。通常BN e008の最初5 epochの旧C予測は全hash完全一致。両20 epochの終点は最良値を下回り、単純な長期化の継続を優先しない。
- 上記の所見別の入れ替わりを見た後、2モデルの固定50:50 rank平均を一度だけCSV診断。weak AUC 0.830249、補助11平均0.815727で両単体を上回った。[固定比率診断](experiments/e008-e009-rank50-review-20261004.json)。BN単体との差の補助11区間は[+0.015129,+0.043443]だが、同じfoldの選択済みモデルによる後付け診断で、別fold・画像ラベル・Publicの改善は未確認。係数/所見weightの探索とgold利用はしていない。
- [r001提出資産](artifacts/kaggle/r001-bn-rank50-fold0/) に2重み・凍結zip・共通rank helper・hash/帰属記録・未実行Notebookを保存。確定weak診断と同じ関数で保存済み人工CPU/GPU出力を合わせ、提出契約とrank専用6チェックが成功した。[提出手順](docs/kaggle-submit.md)。固定BN単体の [e009 v2](artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) も用意した。ユーザーがInputをアップロードし保存実行したが、rank helperの参照先が違って最初のcellで停止。自作Publicはまだ未測定。
- 再依頼で保存済み `rsraki/rsna-knee-r001-rank50` Version 2と現在のコードInputをread-only確認。コードInputの15ファイルは全hashが正しいが、NotebookがInput直下のrank50.pyを展開source内へ探していた。[原因・修正記録](experiments/r001-kaggle-pathfix-20261004.json)。[修正版Notebook](artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) を準備し、実Input配置を写した人工環境で元の失敗を再現、修正後の初め2 cell・CPU checkpoint内部検査が成功。最小修正はRANK_MODULEの1行で、ZIP/重みの再アップロードや再学習は不要。
- ユーザーが保存実行したVersion 3は公式APIでCOMPLETE、GPU有効・Internet OFF。提示ログは表示用3検査の前処理警告0、両GPU推論と最終3×12提出CSVがvalid=trueで、source/重み/helper/fingerprintも候補資産と一致。Outputにsubmission.csv（852 bytes）を確認した。cell内計測47.95秒、提示ログ最終時刻68.2秒。本番の隠しtest採点は未実施で、自作Publicは未測定。[Version 3確認記録](experiments/r001-kaggle-v3-review-20261004.json)。次は変更せずVersion 3のOutputから最終CSVを手動Submitする。
- 今回の採用範囲・不確実性・次の順序と資産hashは [判断記録](experiments/e006-e009-decision-20261004.json)、検査結果は [完了時の検証記録](experiments/e006-e009-verification-20261004.json)。台帳のr001は追加学習ではなくCSV診断として記録した。
- 公開方式の固定契約を追加監査。DINOsaur V35の最終出力はV32の0.937候補と異なり、一部公開資産にはgold学習/選択と混在ライセンスがある。現在のgold除外方針へそのまま移植しない。[再現契約](docs/research/public-reproduction-contract-20261004.json)。重み・MRI・新ラベルの取得や外部書き込みは行っていない。

## 次にすること

06全件特徴の取得・検証、f001/f002のfold0/1比較、f002のVersion2可視推論とCSV監査は完了しました。現在は保存済みの単独候補を手動提出して自己Publicを測る段階です。[次の操作](docs/frozen-features.md#07の完走確認と次の提出)

1. p002 Public 0.937を保持する。同じ05を再提出せず、失敗版・可視receipt・ログと隠し例外を区別して保存する。
2. 完走済み`rsraki/notebookdd86797040` Version2の`submission.csv`をユーザーが手動提出する。GPU有効・Internet OFF、source/Inputs/checkpointは監査済みの保存版を維持する。
3. f002単独の自己Publicまたは正確な例外文面と提出ref/Notebook版を記録する。現時点では未採点。可視完走・weak AUCから隠し完走やPublic0.943を保証しない。
4. p003は64件を上回るラベル不要の規模診断と追加CoAt系統の切り分けを行い、失敗UID・子処理returncode・CPU/GPUメモリ・phase・tracebackを根拠に修正する。現時点では修正版を確定していない。モデル省略やfallback許容で完了扱いにしない。
5. 教師監査は事前にfold0/1を保護した150検査キューから始める。規則を固定した別版教師との比較を、集約やencoder変更と混ぜない。
6. f002の実推論と単独Publicを確認後、採点済みp002との補完性と追加時間を検討する。混合が妥当なら事前固定した5%/10%候補を比較し、p003は完走・採点後に検討する。fold混合・所見別係数・epoch延長・教師/encoder変更を今回の単独測定に重ねない。Gold58や公開重みへの後付けfoldを独立評価とは呼ばない。

10月4日の添付分析の再監査は、続行依頼前の調査履歴である。HEAD 05b0710と保存済みweak評価を照合し、C2−B1の補助11所見差+0.120629の95%区間[+0.078707,+0.162513]、C5−C2差+0.010232の区間[−0.005521,+0.025498]を計算した。後者を追加実験の根拠とし、その後に上記4本の実学習へ進んだ。公式Hostは画像由来の正解とReportの不一致を認めており、weak AUCとKaggle採点を区別する。[当時の調査記録](docs/research/strategy-audit-20261004.json)。

ラベル利用条件の本文確認は完了。研究・学習目的と帰属等を維持し、出所・版・変更点を提出Inputにも記録する。現在の目的で作者への追加許可を新たな必須条件にしない。実際に入賞した場合はWinner公開条文の不整合を確認し、商用化へ目的を変える場合は利用条件を再検討する。

初回提出の採点成功と実測PublicはAPIで確認済み。My Submissionsと保存NotebookでInput版・実行時間を補完する。以後のprepare・trainには `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` を使用する。

キャッシュ作成後の既存コマンドは [after-cache.md](docs/after-cache.md)、現在の判断・比較設計は [10月5日の計画](docs/research/public-model-strategy-20261005.md) を使う。[10月4日の実験計画](docs/research/next-experiments-20261004.md) は履歴として保持する。現行全件Notebookの18GB容量ガードでは256px以上が事前停止する。224pxのraw画像量は約15.92GB、288pxは約26.32GBで、実測192px圧縮export約7.79GBとは区別する。

goldの結果は既に見ているため完全未観測の最終holdoutとは呼ばず、学習・checkpoint選択・prompt調整・ensemble係数合わせには使わない。10月20〜22日は最終候補の再実行・選択に充てる。

## 次へ進む条件

- 初回提出：Notebook版、入力資産の版、採点成功、スコア、実行時間を台帳に記録。
- データ準備：件数、ラベル欠損、シリーズ失敗、前処理画像の目視、使用ラベルの出所が確認済み。
- 自作モデル：学習・weak検証・gold検証の役割を分け、goldをcheckpoint選択に使わず、漏洩監査済み。
- 改善：同じfoldとseedで一要因ずつ変更。12クラス全体と弱いクラス、速度を合わせて判断。
- 最終候補：Kaggleの隠しテストで採点成功し、十分な時間余裕を持つ。未採点の最終変更に依存しない。

## 作業場所

Kaggleは元MRIの前処理と最終提出、このPCはキャッシュからのGPU学習に使う。ユーザーの「進められるところまで進める」依頼と目視確認を受け、準備から実データでのローカル研究へ移行した。Gitへ戻すのはコード、設定、集計と実験記録のみ。レポート、Study UID一覧、分割CSV、画像キャッシュ、重みは `data/` と `artifacts/` 以下へ保存する。

`pj-kaggriculture/PROJECT.md` と `ptcc_pokemon_ai_buttle/experiments` の、実装・実験・生成物を分ける運用を参考にした。ゲーム用の提出形式や対戦評価はMRIコンペへ持ち込んでいない。
