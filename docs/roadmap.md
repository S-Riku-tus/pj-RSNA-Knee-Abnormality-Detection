# RSNA Knee の参加計画と次の作業

**10月09日16:22 JST：S5 V9の非公開固定再現は可視実行・独立検査を完了し、公式API提出が受理されました（ref56998356）。** [実完了・検査・提出の記録](../experiments/p012-private-S5-fixed-V1-visible-complete-submission-accepted-20261009.json)。承認を受けた自分の非公開snapshot V1を接続し、Notebook24の原版5code/70:30/guardを保持。全3件・12所見を両T4で推論し、両モデルfallback0、約174秒、独立9項目・全36float32順位融合一致。保存V1から1回提出、初回APIではstatusが省略されSDK既定PENDING（採点・新自己Public未確認）。旧承認待ち／画面でのV9固定作業は不要になりました。RTX4090の独立last2は6/12epoch、BCE0.285281/AUC12 0.851437で継続。自己基準0.950を保持し、0.955／厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:12 JST：具体的なユーザー承認を受け、S5 V9の非公開固定Inputを作成し、原版5コードセルの再現を保存実行しました。** [承認・実作成・保存実行の記録](../experiments/p012-private-S5-snapshot-approved-and-run-started-20261009.json)。`rsraki/rsna-knee-compact-v1`はAPIで非公開Version1・8ファイル名/サイズ・元manifest byte一致を確認。新規Notebook24はInput参照と説明だけを変更し、元5code/70:30/guardを保持。新private SaveはV1でRUNNING、可視出力検査・採点はまだ。過去のアップロード承認待ちは解消しました。独立last2は6/12epoch、weak BCE0.285281・AUC12 0.851437。固定12epochの学習を維持。自己Public最良0.950、0.955／厳密な0.960超え未達。以下は観測履歴です。

**10月09日16:00 JST：S5 Version9の使用中6ファイルと元manifestを取得・ハッシュ一致まで確認しました。** [実取得と審査停止の記録](../experiments/p012-S5-private-snapshot-ready-review-rejected-20261009.json)。原版5コードセルのまま使える約885MBの非公開入力を準備。新規private Datasetへのアップロードは自動承認審査が具体的な送信内容の承認・宛先所有権確認不足として拒否したため、実アップロード0・Dataset作成0・再試行0。ユーザーへこの具体的なprivateアップロードの承認を質問済み。Kaggle画面で元Input V9をPinして保存実行する別経路も回答待ち。元last2学習は実プロセスをpollし継続、5/12epoch・weak BCE0.301530/AUC12 0.848946。固定条件を保持し、自己Public最良0.950／厳密な0.960超え未達。以下は観測履歴です。

**10月9日15:26 JST：実64件・256件の入力一致診断は成功。Helios作者0.951原版の再現は、APIによる旧Input版固定が反映されず停止しました。** [実行・調査の最新記録](../experiments/public951-native256-and-next-review-20261009.json)。64件は320配列SHA一致・204秒、256件は1,280配列SHA一致・541秒、両T4のsampled peak各563MiB、代替0。同じ凍結対照headの動作検証で、精度/hidden/最後2層モデルの資格とは区別します。Heliosに合うcompactはV9（外a5/使用部分c7）で、原版guardは正しいです。しかし標準Save APIの`datasetDataSources`に`/9`を付けても実際は最新V10（209a…）が接続され、約74秒でguardが停止。Kaggle画面でV9をPinし保存実行した版の確認が次です。23は予測コード未変更、自己スコアなし・競技提出0。Ryan V70は推論モデルでなく順位表（5,374チーム、0.955以上56・0.960超え7・最大0.964）。RTX4090最後2層は4/12epoch・実848更新・AMP skip0、weak BCE0.302986/AUC12 0.835976で継続中。自己基準0.950を保持し、0.955・厳密な0.960超えは未達。過去段落は観測履歴です。

**10月9日14:36 JST：新しい64件診断は入力パス指定の不備で停止。原因を実ログで確認しました。** [実エラー記録](../experiments/f005-native64-actual-mount-path-error-20261009.json)。旧flat競技rootを固定したためtrain_series.csvが見つからず、MRI処理・モデル読込前に約21秒で停止。競技Inputの実mount場所はこのログでは未観測です。構造CSV／画像ディレクトリの存在と固定CSV SHAで一意にrootを選ぶ新wrapperを準備し、同じモデル・64件で再診断します。CPUでの183ファイル一致・厳密再構築成功は保持、64／256件の資格は未取得。last2 fold0の固定12epoch学習は変更せず進行中。元p005自己0.950を保持、0.955／厳密な0.960超は未達。以下の結果待ちは過去の観測時点の履歴です。

**10月9日14:20 JSTの現行計画**：[元p0050.950を固定し、比較学習と実推論診断を進行](../experiments/f005-last2-firstepoch-privateCPU-native64-root-progress-20261009.json)。

1. 同teacher・group/fold・seed・head/lossで凍結対照と最終2block学習を比べる。凍結fold0は12epoch完了、last2は1epoch完了し残り実行中。固定12epoch後にBCE選択checkpointの対照をレビューし、固定fold1で同じ比較をする。
2. 12所見別のAUC/BCE、観測・陽性数、groupごとの不確かさを確認し、次の変更を一つ選ぶ。局所patch特徴や重要スライスの選択、弱教師ラベルの根拠修正を候補とする。Goldを学習・checkpoint・配合選択に使わず、公開学習済み枝を自作OOFと呼ばない。
3. 学習と並行して、実CPU配布/再構築→実MRI64→256→最終推論の経路を検証する。現在CPU成功、64件のprivate保存実行受理。学習時全5配列SHA・例外0・両T4使用・時間/メモリの確認後に次へ進む。
4. 採用候補を元p005へ事前固定の配合で追加し、公式採点で改善を確認する。係数だけの多数再提出を避け、スコアが改善しなければ元0.950を保持する。0.955は中間目標、厳密な0.960超は最終目標で、現在どちらも未達。

作者公式0.951の候補は取得Source/V7 manifestのID不一致が判明し、直コピー・最新Input差替を採用しない。今回の独自学習のweak値からPublic到達値は推定しない。以下は各観測時点の履歴です。

**10月9日13:15 JSTの進捗**：凍結対照fold0の学習を完了、[元last2 fold0を実行中](../experiments/f005-frozen0-complete-last2-actual-start-root-review-20261009.json)。次はlast2の固定12epoch完了→fold0比較→固定fold1 pair→必要なら局所patch特徴を一要因で比較します。選択モデルの実native392推論をKaggleで64/256件検査してから元p005への事前固定配合を提出。公開枝を自作OOFとは扱わず、weak指標から0.955到達を断定しません。p011は公式0.950で非採用、自己最良は元p0050.950、目標0.955／厳密な0.960超は未達。以下は過去の観測履歴です。

**10月9日12:39 JSTの進捗**：p011は[自己Public0.950](../experiments/p011-scored-review-20261009.json)で完走しましたが、追加モデルの精度gainが確認できないため元p005を維持します。次は計算一致を実GPUで確認した凍結CLSの全4,207件特徴抽出を完了し、全件CPU受入→frozen fold0実学習→last2 fold0実学習→対照比較→固定fold1 pairへ進みます。正式fitはまだ0、係数だけの再提出はしません。0.955／厳密な0.960超を目標に、局所特徴の改善はこの対照結果の後に一要因で検証します。以下は過去の観測履歴です。

**10月9日12:25 JSTの進捗**：全24 shard・4,207件の取得／配列検査／index作成と[通常CUDA preflight](../experiments/native392-full-index-and-frozen0-preflight-root-review-20261009.json)を完了。次は凍結CLS特徴のFP32 cacheについて元modelとの実人工GPU計算一致を確認し、条件を保って凍結headを学習。その対照と最終2 block学習をfold0で比較し、固定済みfold1 pairへ進みます。計算一致を満たさなければ既存online凍結CLIを使います。正式fitはまだ0、p011 ref56986152のPublicは未確認、自己最良0.950は保持。以下の古い待ち状態は各観測時点の履歴です。

**10月9日11:30 JSTの進捗**：認証は復旧。p011の可視出力[root12](../experiments/p011-serving-visible-output-root-review-20261009.json)・[独立33](../experiments/p011-serving-visible-output-independent-review-20261009.json)を確認して一回提出、[ref56986152](../experiments/p011-submission-accepted-20261009.json)で採点待ちです。独自モデル用cacheはremote全24 COMPLETE、PC22 shard／3,857件の検査・配置を完了。残350件の取得・配列検査・全件indexを済ませてから、予定した凍結対照／最終2blockの比較学習を始めます。以下の「再認証後」等は以前の待ち状態です。

**10月9日の現行計画は、自己Public0.950のp005を固定し、補完モデルの追加と自作画像モデルの学習を並行比較すること**です。p005 ref56943135/Version1/script356323020のCOMPLETE・Public0.950を公式APIで確認しました。[採点記録](../experiments/p005-scored-review-20261008.json)。次0.955、最終0.960超は目標で達成保証ではありません。保存Notebook14、Input両Version1と過去runを保持します。

| 期間 JST | 優先作業 | 次へ進む条件 |
| --- | --- | --- |
| 10/9～10/10 | p011の独立224枝を元p005へ固定90:10で追加。実64→256件の動作・時間・資源検査後に採点 | 64件は[root24項目](../experiments/p011-profile64-completed-root-review-20261009.json)と[独立45項目](../experiments/p011-profile64-actual-results-independent-review-20261009.json)成功。256件も[root24](../experiments/p011-profile256-completed-root-review-20261009.json)・[独立47](../experiments/p011-profile256-actual-results-independent-review-20261009.json)成功。次はAPI再認証後のServing可視/隠し実行・採点。基準のCPU CSVを保持し、実例外を隠さない |
| 10/9～10/15 | native392全4,207件キャッシュを完成し、RTX4090で同条件の凍結対照／最終2block学習を比較 | 全24 shardの配列検査・group/Gold除外・全件indexが必要。まずfold0の両arm、次にfold1。教師/seed/head/loss/epochを固定、欠損ラベルはmask |
| 10/16～10/19 | 有望な自作モデルだけ追加foldと提出推論を準備。必要なら系列・教師・poolingのうち一要因を次に改善 | 両foldのweak BCE/AUC・各所見の陽性陰性数を確認。weak改善をPublicへ換算せず、汎用初期化から得た未学習fold予測だけを独立OOFとする |
| 10/20～10/22 | 採点済み基準と有望改善版を固定 | 未採点の遅い変更で基準を置き換えず、最大2候補を選択 |

日程は実学習の全epoch時間を未測定の提案です。10月8日に取得した[公式条件](../experiments/competition-constraints-20261008.json)は日次5提出・最終選択最大2、最終**10月23日08:59 JST**。10月9日のweb再取得ではtimeline本文を読めず、日程を新たに確認したとは扱いません。Apexは既に主CoAtNetのmirror2view/密な94窓とConvNeXt3foldを融合しています。p006/p007/p008は自己0.950、p009画角変更は0.948で非採用。今後は異なる解像度・表現の補完性を測ります。追加4検索でも採点済み0.955以上の再現レシピは未確認で、不存在の証明ではありません。[調査結果](../experiments/post950-strategy-primary-update-20261009.json)

独立した検証が必要な学習は汎用初期化で行います。公開allweakfitの重みを自分のfoldへ当てても独立CVではありません。p011の64/256件はラベルを使わない動作検査で、Public利得を測っていません。自分のgold画像ラベルは学習・checkpoint・融合係数選択に使わず、公開重みの作者Gold58選択履歴は別に記録します。教師17セルの見直しから採用した訂正は0件で、根拠なしに欠損を陰性へ変えません。

0.950→0.955は12所見AUC合計で約0.060の改善に相当します。3所見で各0.020改善しても平均の増分は0.005です。これは優先順位を考える算術で、実現予測ではありません。weakで低いMCL・Lateral Meniscus等を調べる際も、基準p005のPublic所見別誤りが判明したとは扱わず、ラベル数・ラベル欠損・画像系列・fold間の再現性を先に確認します。提出は事前に変更と採用条件を記録した候補に絞り、表示0.950の同値だけで基準を置き換えません。

次の表現候補として[CLS＋16領域の部品](../experiments/regional-triplet-component-preparation-20261009.json)と[独立18項目](../experiments/regional-triplet-component-independent-review-20261009.json)を準備しました。DINOの正規化patch特徴を4×4の領域に平均し、既存所見別headへ渡します。headの学習パラメータは83,352のまま、画像cacheを再利用できます。人工CPU8＋独立2検査・既存汎用重みの人工CUDA2窓は成功していますが、精度やfull144窓の資源は未測定です。まずf005の凍結／last2対照を完了し、採用した学習条件を固定してCLS／領域追加を一要因比較します。tensor形状が同じ旧CLS checkpointも意味は異なるため、専用configとcheckpoint識別を用意してから学習器へ接続します。

以下は10月5日以前の計画履歴です。

**10月5日の現行計画は [公開モデルを主軸にする段階別計画](research/public-model-strategy-20261005.md)**。p001 Public 0.924とr001 Public 0.749をAPI確認し、次候補をDINOsaur V4 V32へ絞った。公開候補の版固定と手動比較→採用方式の入力・予測一致→必要な互換cache→一要因改善の順に進める。自作BNの追加fold、長期化、192px crop全件化は先に進めない。10月5〜7日に候補準備、8〜12日に実測と同値性、13〜19日に必要な改善と再現、20〜22日に最終固定を目安とする。利用条件は [監査記録](research/kaggle-source-eligibility-20261004.json)、進捗は [PROJECT.md](../PROJECT.md) を参照する。

以下は10月2日開始時の準備計画を履歴として保持したもの。「台帳はヘッダーのみ」「CUDA未導入」などは当時の状態であり、現在の未完了事項や新しい実行指示ではない。

2026年10月2日 JST時点の判断。リポジトリの資料、ソース、設定、テスト、Notebook、実験台帳を確認し、公式競技説明と公開資産を再調査した。現在の端末はRTX 4090を搭載する。ユーザーの選択に従い、**Kaggleで画像キャッシュを作り、このデスクトップで学習し、Kaggleで提出推論する**。まず公開モデルで採点を通し、比較基準を確保する。

## この競技で行うこと

複数のMRIシリーズを持つ一つの検査から、靱帯・半月板損傷、変形性関節症など12所見の予測値を出す。評価は12項目の平均ROC-AUCで、陽性を陰性より高く順位付けする能力を測る。0.924を診断正解率92.4%と解釈しない。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

学習時の読影レポートから教師信号を作れるが、テスト時にはReportがない。提出Notebookは未知のMRIから予測を作る必要があり、手元で作ったCSVをそのまま提出する経路ではない。例示テストは3検査、本採点は約1,300検査である。[公式データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)

## 現在の準備と不足

| 対象 | 確認した状態 | 次に必要なこと |
|---|---|---|
| CSV契約とラベル準備 | 列順、欠損、gold除外、同一レポートと供給groupの連結を実装済み | 実CSVの件数・欠損・各foldの陽性数を監査する |
| MRI前処理 | 幾何情報で並べ、最大3シリーズから2.5D窓を作る | 実DICOMのdecoder、方向、左右、失敗例、所見の保持を確認する |
| 自作画像モデル | ランダム初期化ResNet18と所見別Attention | 工学的な比較基準として動作と速度を測る |
| 学習と推論 | CUDAチェック、weak BCEでのcheckpoint選択、Report不要の推論を実装済み | CUDA、AMP、backward、実学習と実採点を確認する |
| 公開モデル | CoAtNet単体モデルなどを調査済み | 特定Notebook版とInput版を固定して自分で採点する |
| 実験記録 | 台帳はヘッダーのみ | 初回採点から実測結果を記入する |

このリポジトリは実験の土台であり、精度が確認された解法はまだない。既存の12テスト成功の記録は旧環境での人工データ検証で、実MRI・CUDA学習・採点の成功を表すものではない。今回の確認は [validation.md](validation.md) に分けて記録する。

## RTX 4090 をどこで使うか

端末の読み取りによる確認値はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブは約1.02TB、空き484,710,727,680 bytes、約484.7GBである。公式Data Explorerは569.76GB、819,640ファイルと表示する。現在のCドライブでは元画像の全展開が収まらず、圧縮アーカイブとキャッシュの保存領域も別に必要になる。[公式データページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)

| 場所 | 担当する作業 | 理由 |
|---|---|---|
| Kaggleの公開モデルNotebook | 初回提出と公開モデルの再現 | 元画像や重みをローカルへ移さず、採点経路を確認できる |
| Kaggleのキャッシュ作成Notebook | 競技Inputを直接読み、選択・縮小した画像を保存 | 約570GBの元画像をこのPCへ全取得する必要をなくす |
| このPCのRTX 4090 | キャッシュから学習、検証、実験比較 | 自分のGPUで反復できる。適切なバッチと所要時間は実測する |
| Kaggleの提出Notebook | 隠しtestの前処理と推論 | 競技が指定する環境で未知データを処理する |

Kaggle Notebookはクラウドの計算機であり、ブラウザをこのPCで開いてもRTX 4090は使われない。このGPUを使うにはローカルPythonから学習を実行する。Windows PowerShellで開始できるため、初回実験の前提としてWSLやLinuxへの移行は求めない。

提出の9時間制限は提出Notebookの実行条件であり、学習と提出を一つのNotebookへまとめる設計は必要ない。10月4日にRules本文を公式SDKで取得した。競技由来キャッシュはprivateで保持し、公開Inputには元MRI・レポートを入れない。研究目的と利用資産の帰属等を守る。[Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)、[Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

## 最初の二日間にすること

1. **参加登録と提出環境を確認する。** ルールに同意し、アカウント画面でGPU利用可否、残り枠、日次提出回数、最終選択数を確認する。固定の無料GPU時間を前提にしない。
2. **公開単体CoAtNetモデルを複製して採点する。** [作者Notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) のスコアを出した版と現在版を区別し、Inputs、GPU、オフライン依存を合わせる。Save and Run All後に手動でSubmitし、採点成功まで確認する。[再現手順](public-baselines.md)
3. **採点記録を残す。** 自分のPublic LB、Notebook版、Input版、checkpoint、全体実行時間、変更点を記入する。作者報告0.924は自分の結果欄へ入れない。
4. **専用Python環境のCUDA対応を整える。** 今回はPythonと軽量チェック用依存だけを用意した。CUDA版torchとtorchvisionは未導入である。公式の対応する組を入れ、`doctor` と人工画像のforwardで確かめる。[環境手順](gpu-start.md)
5. **Kaggleで10検査のキャッシュを確認する。** [00_prepare_cache.ipynb](../notebooks/00_prepare_cache.ipynb) へ競技InputとコードzipをAttachし、入力版を記録する。coverageと原画像・縮小画像を比較してから全件へ進む。

初回提出の成功と、ローカル学習準備は独立して進められる。初回採点が動かなければInputsと版を直し、モデル変更を重ねない。

## キャッシュを移して学習する手順

全件キャッシュはKaggleのprivate Outputから手動でダウンロードし、`data/exports/` 以下の新しいディレクトリへ展開する。Notebookはcache、学習CSV、config、前処理ソース、SHA-256一覧を保存する。転送後にhash、完全性、前処理fingerprintを検査してから学習する。具体的なコマンドは [gpu-start.md のキャッシュ経路](gpu-start.md#kaggleのキャッシュでローカル学習する) を使う。

現行configの画像配列は、4,407検査と仮定すると `4407 × 3シリーズ × 8窓 × 3枚 × 192 × 192 × uint8`、約10.9GiBである。これはコードの形状からの計算で、実測ファイルサイズではない。npz圧縮後のサイズ、デコード時間、実際の検査件数は最初の実行で確認する。参加者にも約11GiBのキャッシュ化報告があるが、その6枠・224pxのキャッシュを本コードへそのまま読み込むことはできない。[作者のキャッシュ実験](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)

Kaggleの保存可能なOutputは公式ドキュメントに20GBと記載されるため、現行Notebookは18GBを目安に検査し、出力内に同容量のzipを追加作成しない。画面の現在の制限も確認する。実行時間がセッション制限を超える場合は、保存できたprivate OutputをInputにし、同じコード・configで`RESUME_CACHE`から再開する。必要なら対象検査を分割する機能を別途追加する。[Notebookの仕様](https://www.kaggle.com/docs/notebooks)、[OutputをDatasetにする手順](https://www.kaggle.com/docs/datasets)

弱教師ラベルは出所、版、利用条件、goldへの調整の有無を確認して用意する。現実装には公開ラベルの自動取得やLLMによる抽出器はない。空欄テンプレートを作るだけでは学習できない。`prepare`でgoldを予約し、同一レポートと供給groupをまとめてからfold 0をまず1epochで動かす。この1epochも全fold学習・検証検査を処理する。転送、ラベル監査、学習、提出と改善の具体的な操作は [手順4〜9](after-cache.md) を使う。

最初のキャッシュは自作ResNetの動作と検証用に使う。メダルを目指す主力の改善は、再現できた公開モデルの学習コード・前処理と、汎用事前学習encoderを使った独立のfold学習を候補にする。公開競技重みが全weak検査を学習済みなら、その重みを自分のfoldで評価・fine-tuneしても独立検証にはならない。公開重みを使った比較は提出再現として区別する。

## 三週間の優先順位

以下は10月2日開始の提案であり、公式日程や学習時間の保証ではない。遅れた場合は実験数を減らし、最終採点の余裕を残す。

| 期間 JST | 到達したい状態 | 次へ進む条件 |
|---|---|---|
| 10月2〜4日 | 公開モデルで初回採点、CUDA準備、10検査cache確認 | 成功した提出が一つある |
| 10月5〜8日 | 全件cache、ラベル監査、fold 0の比較基準 | 入力hash、分割、weak検証、速度が記録されている |
| 10月9〜15日 | 同じfoldとseedで少数の改善実験 | 改善した要因を説明でき、他クラスを悪化させていない |
| 10月16〜19日 | 有望な条件を追加foldで確認、必要なら2モデルを比較 | 検証の独立性と前処理込みの提出時間に問題がない |
| 10月20〜22日 | 採点成功済み候補を固定し、最終候補を選ぶ | オフライン再実行と隠しtestの採点が成功している |

参加・チーム統合期限は **10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。本プロジェクトでは10月22日中に選択を終える。未採点の変更を最後の候補にしない。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

改善順は、実画像の前処理監査、事前学習encoder、シリーズ・窓の見せ方、ラベルの誤り、少数ensembleを出発点にする。ボトルネックが計測できたら順序を変える。モデル・ラベル・cropを同時に変えず、変更理由を事前に一行書く。1epochの時間を測って候補runの予算を決め、最初から5foldすべてを回さない。

## 改善を判断する基準

- **weak検証BCE**でepochを選ぶ。レポート由来ラベルへの適合を測るため、公式AUCと同じ意味ではない。
- **goldのクラス別AUCと陽性数**で画像由来ラベルとの整合を確認する。学習やcheckpoint選択には使わず、何度も合わせ込んで小さな差を採用理由にしない。
- **Public LB**は外部評価として確認する。提出ごとに仮説を持ち、連続した微調整で公開部分に合わせ込まない。
- **前処理込みの時間と失敗率**も採用条件にする。約1,300件を9時間で割ると平均約25秒/検査だが、起動や長い検査も含まれるため上限ぎりぎりを狙わない。

12クラスは同じ重みで評価されるため、平均だけでなく弱いクラスも見る。soft labelを0.5で丸めたAUCを公式値と同一視しない。患者対応が未確認なら患者独立と呼ばない。金標準への露出が不明な公開ラベル・重みでは、独立性も未確認として扱う。

## 残る確認事項

現在の順位・メダル境界、Rulesの細目、公開Notebookの厳密な版と再現性は未確認。0.924や添付の0.94台からメダル獲得を予測しない。メモリ容量は端末の読み取りが拒否されたため未確認で、全cacheをRAMに載せる方式は前提にしない。現行Datasetは検査単位で読み出す。

最初に必要なのは、参加登録、公開モデルの採点成功、CUDA利用可否、実画像10検査の監査である。この四つが揃ってから長時間学習へ進む。調査の取得状況は [sources.json](research/sources.json)、実際に行ったチェックは [validation.md](validation.md) に保存する。
