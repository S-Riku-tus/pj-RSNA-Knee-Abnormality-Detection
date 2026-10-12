# RSNA Knee プロジェクトの現在地

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
