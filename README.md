# RSNA Knee Abnormality Detection

**10月11日01:08 JST：目標は未達のまま、承認・外部状態待ちのblockedへ更新しました。** [停止条件と再開手順](experiments/p018-specific-upload-three-turn-blocked-audit-20261011.json)。新REG2件の具体的送信承認が3連続ターンで未回答、準備済みソース・独立確認は完了しています。新送信・作成0で、承認なしに実入力→64／256件→提出推論へ進みません。P016v5／P017v9の公式照会と同P016番号の再確認v6はHTTP401でした。認証の原因は未確定で、提出失敗・ジョブ停止とは扱わず、再起動・再提出0です。前回00:57の一覧では2提出の存在とスコア未返却を確認済。具体的承認とKaggle API認証の復旧、または公式結果が届いたら同じ目標で再開します。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日01:00 JST：追加学習は、準備済みの候補をPublicで測定してから判断します。** [学習曲線と公式照合](experiments/p018-next-training-decision-official-list-crosscheck-20261011.json)。4モデルとも採用epoch後に訓練損失は下がり、検証損失は悪化しているため、epoch延長や学習範囲拡大を今始める根拠は不足しています。fold2・last4は現在の学習入口の資格外です。P016・P017は00:57の公式提出一覧でも存在を確認しましたが、スコアと明示状態は未返却。確認APIの指定項目・endpointに不足は見つかりません。新REG2件は具体的な送信承認の回答待ち・作成送信0。ソース準備は完了し、次は入力バイト検証→実MRI64／256件→提出推論確認です。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日00:48 JST：P018の入力確認を実APIの形式に合わせて修正し、次の実行用ソースを準備しました。** [根拠と次の順序](experiments/p018-metadata-api-contract-fixed-source-ready-20261011.json)。既存の正常なprivate V1を公式APIで1回読み、Dataset情報からファイル一覧が省略される形式を確認しました。専用のV1一覧APIへ分けた修正版は独立12項目、提出用の親は20項目、保存入口は12項目、64／256件の入口もソース査読を通過しました。過去の隠し提出エラー全体の原因を確定したものではありません。実CPU4・人工GPU8回の成功を保持し、新REG入力・Kaggle実MRI・提出推論は未検証です。新規privateアップロード2件の具体的承認は回答待ち、送信・作成0。P016・P017の00:45公式照会はスコア・明示状態未返却。自己最高0.951、厳密な0.960超え未達です。以下は履歴です。

**10月11日00:12 JST：P018の新規非公開モデル2件は、今回の宛先・内容への明示承認待ちです。** [現状と追加調査](experiments/p018-private-upload-approval-pending-current-public-research-20261011.json)。自動承認審査が新UUID宛の未公開モデル・コード約209MBの送信を停止したため、質問済みの回答を待っています。作成API・送信0、以前のCLS fold1承認とは別です。実CPU4・人工GPU8回の成功は保持し、アップロードを待つ間に新しい提出用の親と64／256件診断のソースを準備します。新REG Inputのバイト検証入口は独立12項目、phase契約は11項目のソース査読を通りましたが、Kaggle実行資格は未発行です。追加確認した公開4-arm現V57の作者スコアは0.936で、自己0.951を超える根拠はありません。P016・P017は00:10の公式照会でもスコア・明示状態未返却。自己最高0.951・厳密な0.960超え未達です。以下は履歴です。

**10月10日23:51 JST：P018の実4モデルは、RTX4090の人工入力確認も成功しました。** [実GPU確認の記録](experiments/p018-mixed-realGPU4-synthetic-QC-20261010.json)。元の選択済み4モデルで144窓と欠損96窓を各4回、計8回推論し、全画像欠損4拒否・encoder入場0・全6組の共有なし・推論前後の重み一致を確認しました。観測最大allocated 653,356,544 B／reserved 767,557,632 B、検査全体6秒、独立10項目・Root12項目が合格。これは人工画像と1枚の4090の確認で、2枚のT4・実MRI64／256件・提出推論・Public改善は未検証です。次は新規非公開REG2 Inputの登録とバイト検証、その後Kaggleで新しい4モデルの64／256件と親の復旧経路を確認します。P016・P017は23:44の公式照会でもスコア・明示状態未返却、自己最高0.951・厳密な0.960超え未達を維持します。以下は履歴です。

**10月10日23:42 JST：P018の4モデル再構築が実CPUで成功しました。** [実確認とアップロード停止の記録](experiments/p018-mixed-realCPU4-bytecode-fixed-upload403-20261010.json)。Pythonが封印ツリーへキャッシュを追加する起動不具合を修正し、モデル・計算式・厳密なファイル検査を保持しました。新しい展開先の193ファイル×2を元のハッシュと照合し、4重みの厳密読み込み・6組の共有検査・Root9項目を通過。forward・MRI・GPU・教師・Report・Gold利用は0です。4モデルの人工GPU／Kaggle64・256件／提出用統合は未検証です。非公開アップロードは作成前に停止し、追加の公式1回照会はHTTP403でした。存在しないDatasetとは判定せず、新規作成専用APIを確認します。P016・P017は直近23:20の照会でも公式スコア・明示状態未返却、自己最高0.951・厳密な0.960超え未達です。以下は履歴です。

**10月10日17:18 JST：自己Public最良0.951、提出2件はスコア未返却です。** [最新記録](experiments/f005-regional-epochs34-integrated-v2-public-source-20261010.json)。新しい局所モデルはfold0 3／12・fold1 4／12epochで学習継続中。両foldとも同epochの旧CLSよりweak BCE・固定支持10項目AUCが悪く、改善はまだ確認できません。修正版の統合Notebookはソース査読15項目に合格しましたが、実モデル・CPU／64／256件・提出推論は未検証です。追加の公開候補にも現版スコア・補完性の未確認点があり、現在の基準を維持します。厳密な0.960超えを目指す作業を継続します。以下は各時点の履歴です。

**10月10日14:50 JST：局所特徴モデルの人工GPU検証50項目・独立33項目と両foldの事前照合が合格し、fold1の実学習を開始しました。** [実QA・修正・学習開始の記録](experiments/f005-regional-dense-QA50-fold1-started-20261010.json)。CLSに16領域のpatch平均を加え、画像キャッシュ・weak教師・seed・last2・12epoch設定を維持します。通常の逆伝播でも微小勾配差がAdamWで増幅されたため、専用経路でGPU計算を固定しました。144窓・16／10／4件の勾配、更新後重み、AdamW状態は再実行と差0、観測最大予約メモリ1.27GB。実学習は3,316件／検証891件で実行中、最小weak BCEを12epoch完了後に採用します。旧CLSとはGPU設定にも差があり、効果を局所表現だけには断定しません。P014提出57031186・P015提出57031981は採点待ち、自己最良0.951と元Notebookを保持します。今回の精度改善・厳密な0.960超えは未確認です。以下は各時点の履歴です。

**10月10日14:30 JST：P015も公式提出57031981が受理され、P014と2本が採点待ちです。** [28番Notebook](notebooks/public/28_submit_p015_native_support_mask.ipynb)は正常完走し、実22本文を取得、独立58項目に合格しました。MCL・Synovitisの6値は元の基準値と一致、他10列の30文字列はuniform20と一致、834バイトのCSV全体も一致。新しい局所特徴モデルは人工GPU検証で144窓の予測・損失・勾配まで合格しましたが、AdamW更新後の一致検査で停止しました。実学習は開始せず原因を調査します。[提出と検証の記録](experiments/f005-p015-submitted-regional-QA-held-20261010.json)。自己最良0.951を保持し、今回のhidden完走・Public・厳密な0.960超えは未確認です。以下は各時点の履歴です。

**10月10日14:06 JST：2列だけを変えるP015比較版を非公開V1で保存・実行中です。** [28番Notebook](notebooks/public/28_submit_p015_native_support_mask.ipynb)はMCL・Synovitisを元の0.951基準予測に戻し、他10項目の自作2fold・20%合成を維持します。弱教師の正負支持数に基づく新しい仮説で、Gold・Publicによる係数調整は0。18コードセルの構文確認と保存ソース・6入力の8項目照合を通過しました。[実行記録](experiments/f005-p015-native-support-mask-running-20261010.json)。完走・2列の数値一致・他10列の一致・提出・採点はまだ未確認です。全12項目20%合成のP014提出57031186も採点待ちで、自己最良0.951を保持します。

**10月10日13:41 JST：P014の提出が公式APIで受理されました。提出番号57031186、採点待ちです。** [提出受理と検証](experiments/f005-native-twofold-p014-submitted-20261010.json)。[27番Notebook](notebooks/public/27_submit_p014_native_twofold_compact.ipynb)は正常完走、実出力の独立37項目が合格。自己0.951の基準予測に自作2foldを固定20%で加えた最初の公式測定です。自己最良0.951を保持し、今回のPublic・厳密な0.960超えは未確認です。

**10月10日13:21 JST：圧縮したP014提出候補の非公開Version 1が保存され、エラーなしのRUNNINGを確認しました。** [保存受理とソース照合](experiments/f005-twofold-p014-compact-serving-running-20261010.json)。[27番Notebook](notebooks/public/27_submit_p014_native_twofold_compact.ipynb)のAPIソースは927,057バイト。77ファイルの圧縮復元SHA、元5セル、固定20%の計算を維持し、実際に保存された全21セルと6個の入力を8項目で照合しました。完走出力を検証して競技提出へ進めます。今回の競技Submitは0、Publicは未評価、自己最良0.951を維持します。

**10月10日13:02 JST：自作2foldのCPU・64件・256件の検証が12項目すべて合格しました。** [実結果と保存エラー](experiments/f005-twofold-qualified-p014-source400-20261010.json)。256件は512回のforward、約575秒、両T4の観測最大635MiB、代替予測0件でした。P012の元5セルを保持し、自作モデルを固定20%で加える提出候補を作成しましたが、2.76MBのソースのSaveはHTTP400でした。参照先の照会は404で、保存版なし・競技Submit0を確認しました。サイズが原因という判断は推測です。圧縮して復元SHAを照合する修正を進めています。26番Notebookは失敗候補として保持します。自己最良0.951、0.955と厳密な0.960超えは未達です。

**10月10日 02:04 JST：自作二foldの実CPU・64件検証が完了し、同じ構成の256件が実RUNNINGです。** [保存版と実出力](experiments/f005-twofold-CPU64-qualified-native256-running-20261010.json)。decoderは実在する非公開V1へ参照を修正し、manifestの内容は変更0。新ソースの独立27項目を通過後、CPUは厳格な2モデル復元、64件は128forward・214秒・両T4の観測最大635MiB・代替0・後処理終了を実出力で確認しました。256件は新V1/kernel137904159として保存受理され、同じpair・入力・コードを公式RUNNINGと照合済みです。競技Submit0、256件全体／Serving／Publicの資格はまだありません。自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:36 JST：256件の実検証とfold1学習・CPU復元・非公開入力登録が完了しました。** [実結果と次の修正](experiments/p012-baseline256-qualified-f005-fold1-complete-20261010.json)。P012は実18本文・15項目合格、2559秒、3072予測値のfloat32差0、代替0・子2群終了・clear1回・GPU解放を確認。fold1は固定12epoch完了、最小BCEのE9で弱BCE0.27969／AUC12 0.86695、全12所見の改善方向が両foldで再現しました。paired13・CPU8項目合格、明示承認後にモデル2ファイルをprivate V1へ登録済。二foldのSave前検査で、decoderのローカルpackage改訂v2をKaggle Version2と混同した参照を検出しました。Version2一覧は404、実Version1のmanifestは元pinと完全一致。内容を変えず参照を修正中です。native Save0・競技Submit0、二fold／提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:03 JST：作者0.954の新S6候補を確認しましたが、入力の配布終了により保留します。** [一次資料と取得結果](experiments/public954-candidate-retired-input-20261010.json)。作者の保存V1・採点script356507041とソース一致、対応NetとV10 manifestも一致しました。現行公開V11は183バイトのREADMEだけで、モデルと旧版を利用不可にしたと明記。S6モデル取得0・アップロード0・自己0.954未確認です。既に固定した非公開S5入力の自己0.951を基準に、256件検証と自作二fold追加を継続します。fold1は11/12epoch、途中checkpoint未採用。二foldの実資格と0.955／厳密な0.960超えは未達、今回の競技Submit0です。この配布終了を過去全エラーの原因とは扱いません。

**10月10日 00:33 JST：実64件の合格を受け、同じコードの256件検証を開始しました。** [保存版と実行状態](experiments/p012-baseline256-cleanup-running-20261010.json)。新しい非公開V1（kernel137894797）はRUNNING、元5セル・同じ入力3版・Original/T4/Internet OFFを保存ソース6項目で確認。64件の実合格証拠を固定し、256件の合格は完了出力から別に検査します。競技Submit0、二fold・提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:27 JST：P012の64件診断は実完了し、GPU解放を含む15項目に合格しました。** [実検査の記録](experiments/p012-baseline64-cleanup-qualified-20261010.json)。固定70:30の768値は元式とfloat32差0、代替0、子2群終了、clear1回、両T4の確保量・予約量0。実行803秒、18本文の取得SHAを検査しました。以前の2件のERRORは追加診断checkerの不具合として保持し、成功資格へ転用していません。同じコードで256件を次に検証します。fold1は10/12epochで継続中、途中checkpointは未採用。学習の弱点解釈ではSynovitisの陰性が1件しかなく、weak改善をPublic改善とは断定しません。二fold・提出版の実資格は未確認、今回の競技Submit0、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:03 JST：再診断V2は成功ログの誤判定でERRORとなり、後段契約全体を実形式で再照合しています。** [取得した実ログ・13本文・原因](experiments/p012-baseline64-V2-worker-log-contract-error-20261010.json)。元予測は64件・代替0で完了し、固定70:30の768値は実CSVで再度一致。GPU観測はRaptor3763/3643MiB、S510225/9987MiBのsampled peakです。追加checkerが要求したPASS文字は元workerログには存在せず、WORKER_DONE JSONとfallback0が本来の成功記録でした。元モデル・数値は保持し、この形式と残りのseal→cleanupを実出力・元producerに合わせてV3で確認しています。封印・GPU解放は未到達、ERRORのCSVは提出しません。fold1は9/12epochで正常継続、途中checkpoint未採用。今回の競技Submit0、自己最良0.951、二fold/Serving資格と0.955／厳密な0.960超え未達を維持します。

**10月9日 23:47 JST：診断checkerを修正し、新しい非公開64件診断を再実行中です。** [修正・実取得・新Saveの記録](experiments/p012-baseline64-checker-repaired-rerun-20261009.json)。旧ERRORから必要な8本文だけを取得し、4CSVの全64件・13列・有限12値、固定70:30の768セルfloat32一致、代替0、子2群の終了を確認しました。新checkerはGPU記録listのbool化と実S5先頭50件preview／全worker CSVの区別を修正。独立19項目・作者8＋独立2テストを通過し、元モデル・5セル・数値は維持しました。23:40に新private Save1回受理（kernel137888437/V1）、23:42 raw/typed RUNNINGと保存ソース6項目一致を確認。GPU解放を含む実完了はまだ未確認で、旧ERRORから資格を転用しません。fold1は8/12epoch、途中checkpoint未採用。競技Submit0、自己最良0.951、二fold・Serving資格と0.955／厳密な0.960超えは未達です。

**10月9日 23:33 JST：64件診断は予測完了後の新しい診断checkerでERRORとなりました。** [実ログと原因](experiments/p012-baseline64-diagnostic-checker-error-20261009.json)。公式raw/typed ERRORと保存ソース一致を確認。元Raptor・S5は64件すべて、代替0、70:30融合を完了したと実ログに記録されています（786.9秒）。追加診断の`require(selected)`がlistを厳密なTrueと比較して必ず拒否する不具合が原因です。実S5のper_studyはtop-level・先頭50件previewであり、旧checkerの仮定も修正が必要と分かりました。原版・エラー証拠を保持してV2 checkerを準備中です。モデル・数値は変更せず、修正版64→256を実検証します。GPU解放・封印は未到達で、残った診断CSVは提出しません。ローカルfold1は8/12epochで継続。自己最良0.951、今回の競技Submit0、二fold・Serving資格と0.955／厳密な0.960超えは未達です。以下は時点ごとの履歴です。

**10月9日 23:25 JST：P012の64件GPU解放診断を非公開保存版V1で実行中です。** [最新記録](experiments/p012-baseline64-started-nativefold1-progress-20261009.json)。診断ソースは独立30項目、二fold統合入口は33項目を確認し、元5セル・数値を保持しています。標準SDKの認証を更新した後、診断Saveを1回受理。23:22の公式raw/typed状態はRUNNING、保存ソース・宣言設定6項目は一致しました。CSV・子プロセス・GPU解放の実検証は完了待ちで、二foldや提出版の実行資格とは扱いません。RTX4090のfold1は8/12epoch、途中checkpointは未採用。完了後に事前指定の最小weak BCEで選択し、二fold自身のCPU→64→256と親への統合を検証します。今回の競技提出は0回。自己Public最良0.951、P013は0.950・エラーなし、0.955／厳密な0.960超えは未達です。

**10月9日 20:48 JST：単一foldの自作モデルはCPU・64件・256件の実診断47項目を合格しました。** [実行資格と次の設計の記録](experiments/f005-online-last2-engineering-and-serving-plan-20261009.json)。この資格はfold0だけで、2fold統合や提出版の資格には転用しません。2fold設計V2はfold1の封印先だけを短い `b1` に修正し、既存215ファイル名の最長パス247文字と7つのCLI定義を静的に確認しました。旧設計V1と独立25項目の査読は保持し、統合推論のルーティングは実装中・未完成です。

自己Public最良は公式 **0.951** を維持し、0.955・厳密な0.960超えは未達です。P013は20:33の観測でraw status省略・SDK既定PENDING、スコア未返却です。fold1の時刻付きスナップショットは 2/12epoch（weak BCE 0.340666・AUC12 0.771251）で、固定12epochを継続中です。公開S6は学習コードと4重みの変更を確認しましたが、その新しい推論コード本文・互換性は未確認で、実行可能な新候補とは扱いません。教師なし64件の予測差は確認しましたが、学習から独立であることを確認していない単一foldのcohortであり、精度改善やPublic0.951との補完性の証明ではありません。


**10月9日20:17 JST：自己Public最良は0.951に更新しました。** [公式採点・次の提出・自作モデル進捗の記録](experiments/p012-own951-p013-submitted-native-progress-20261009.json)。P012の保存Version 1は公式COMPLETE、Public **0.951**、エラーなし（ref56998356）。作者報告だけでなく、自分の提出で確認した値です。元p005の0.950とNotebook14は対照として保持します。目標0.955、厳密な0.960超えはまだ未達です。

P013は、基準p005へS5予測を固定20%で加える別候補です。実64件・256件の診断後、提出版の全3件・12所見、代替予測0、GPU解放、基準融合差0、提出CSVのバイト一致を独立20項目で確認し、同じ保存Version 1から1回提出が受理されました（ref57006968）。20:15の照会ではraw status省略、SDK既定PENDING、スコア・エラー未返却です。Input順序のAPI返却だけを取得ツールが厳しすぎる条件で比較していた不具合は、全22セルと環境の一致を確認して修正しました。モデル・入力の集合・予測計算は変えていません。

自作DINOv2最終2層モデルはfold0の固定12epoch学習とCPU再構築を完了し、新しいモデル自身の実64件・256件診断もそれぞれ10項目合格しました（320／1,280配列SHA、全12予測有限、両T4、代替0、約178／701秒）。256件の実COMPLETEを20:16に確認し、両GPUの観測最大は563MiBです。以前の凍結モデルの資格は流用していません。Serving・Public、将来の2fold合成モデルの資格はまだ未確認です。fold1は同条件の12epoch学習を継続し、最新保存スナップショットは1/12epoch（weak BCE 0.368269、weak AUC12 0.711189）。weak値をPublicへ換算せず、12epoch完了後に凍結対照と比較します。自分のGoldラベルの学習・選択利用は0、公開重みのGold/事前分布利用は別監査です。以下は過去の観測履歴です。


**10月09日16:22 JST：S5 V9の非公開固定再現は可視実行・独立検査を完了し、公式API提出が受理されました（ref56998356）。** [実完了・検査・提出の記録](experiments/p012-private-S5-fixed-V1-visible-complete-submission-accepted-20261009.json)。承認を受けた自分の非公開snapshot V1を接続し、Notebook24の原版5code/70:30/guardを保持。全3件・12所見を両T4で推論し、両モデルfallback0、約174秒、独立9項目・全36float32順位融合一致。保存V1から1回提出、初回APIではstatusが省略されSDK既定PENDING（採点・新自己Public未確認）。旧承認待ち／画面でのV9固定作業は不要になりました。RTX4090の独立last2は6/12epoch、BCE0.285281/AUC12 0.851437で継続。自己基準0.950を保持し、0.955／厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:12 JST：具体的なユーザー承認を受け、S5 V9の非公開固定Inputを作成し、原版5コードセルの再現を保存実行しました。** [承認・実作成・保存実行の記録](experiments/p012-private-S5-snapshot-approved-and-run-started-20261009.json)。`rsraki/rsna-knee-compact-v1`はAPIで非公開Version1・8ファイル名/サイズ・元manifest byte一致を確認。新規Notebook24はInput参照と説明だけを変更し、元5code/70:30/guardを保持。新private SaveはV1でRUNNING、可視出力検査・採点はまだ。過去のアップロード承認待ちは解消しました。独立last2は6/12epoch、weak BCE0.285281・AUC12 0.851437。固定12epochの学習を維持。自己Public最良0.950、0.955／厳密な0.960超え未達。以下は観測履歴です。

**10月09日16:00 JST：S5 Version9の使用中6ファイルと元manifestを取得・ハッシュ一致まで確認しました。** [実取得と審査停止の記録](experiments/p012-S5-private-snapshot-ready-review-rejected-20261009.json)。原版5コードセルのまま使える約885MBの非公開入力を準備。新規private Datasetへのアップロードは自動承認審査が具体的な送信内容の承認・宛先所有権確認不足として拒否したため、実アップロード0・Dataset作成0・再試行0。ユーザーへこの具体的なprivateアップロードの承認を質問済み。Kaggle画面で元Input V9をPinして保存実行する別経路も回答待ち。元last2学習は実プロセスをpollし継続、5/12epoch・weak BCE0.301530/AUC12 0.848946。固定条件を保持し、自己Public最良0.950／厳密な0.960超え未達。以下は観測履歴です。

**10月9日15:26 JST：実64件・256件の入力一致診断は成功。Helios作者0.951原版の再現は、APIによる旧Input版固定が反映されず停止しました。** [実行・調査の最新記録](experiments/public951-native256-and-next-review-20261009.json)。64件は320配列SHA一致・204秒、256件は1,280配列SHA一致・541秒、両T4のsampled peak各563MiB、代替0。同じ凍結対照headの動作検証で、精度/hidden/最後2層モデルの資格とは区別します。Heliosに合うcompactはV9（外a5/使用部分c7）で、原版guardは正しいです。しかし標準Save APIの`datasetDataSources`に`/9`を付けても実際は最新V10（209a…）が接続され、約74秒でguardが停止。Kaggle画面でV9をPinし保存実行した版の確認が次です。23は予測コード未変更、自己スコアなし・競技提出0。Ryan V70は推論モデルでなく順位表（5,374チーム、0.955以上56・0.960超え7・最大0.964）。RTX4090最後2層は4/12epoch・実848更新・AMP skip0、weak BCE0.302986/AUC12 0.835976で継続中。自己基準0.950を保持し、0.955・厳密な0.960超えは未達。過去段落は観測履歴です。

**最新（10月9日14:25 JST）：最終2層モデルは2/12epoch・実424更新／AMP skip0まで進み、実Kaggle CPU読込も成功しました。** [学習の観測](experiments/f005-last2-fold0-actual-epoch2-observation-20261009.json)・[配布/CPUと64件診断開始](experiments/f005-last2-firstepoch-privateCPU-native64-root-progress-20261009.json)。64件の実GPU診断は結果待ち、自己最良0.950を保持し目標0.955／0.960超は未達。[次の順序](docs/roadmap.md)。以下は過去の観測履歴です。

**最新（10月9日13:15 JST）：全4,207件の特徴cache検査と凍結対照fold0の12epoch学習を完了し、最終2層を学習する比較を開始しました。** [実学習記録](experiments/f005-frozen0-complete-last2-actual-start-root-review-20261009.json)。凍結対照はweak BCE0.323206／AUC12 0.794448、Publicとは区別します。自己最良p005の0.950を保持、0.955／0.960超は未達。以下は過去の観測履歴です。

**最新（10月9日12:39 JST）：p011も自己Public 0.950で採点成功、改善がないため元p005を維持します。** [採点結果](experiments/p011-scored-review-20261009.json)。全4,207件のindexとCUDA学習入口を確認し、凍結特徴の計算一致を実人工GPUで検証しました。[実特徴抽出を開始](experiments/f005-frozen-FP32-actual-qualification-and-cache-start-20261009.json)、完了・受入後に比較学習を始めます。正式fitはまだ0、0.955／0.960超は未達。以下は過去の観測履歴です。

**最新（10月9日12:25 JST）：全4,207件の学習索引を作成し、CUDA学習入口の確認が成功しました。** [索引とfold0 preflightの記録](experiments/native392-full-index-and-frozen0-preflight-root-review-20261009.json)。凍結特徴を一度だけ計算する実装を元の計算と比較してから、実学習へ進みます。正式fitはまだ0。p011 ref56986152は公式スコア未確認、自己最良0.950を保持しています。以下は各観測時点の履歴です。

**最新（10月9日11:43 JST）：全4,207件の加工済み学習cacheを検査・配置しました。** [受入記録](experiments/native392-all24-canonical-root-acceptance-20261009.json)。全件index作成後にRTX4090で比較学習へ進みます。p011はref56986152で採点待ち、自己最良0.950を保持。[現在地](PROJECT.md)。以下は各確認時点の履歴です。

**最新（10月9日11:30 JST）：追加モデル候補p011を提出し、採点待ちです。** [ref56986152の受理記録](experiments/p011-submission-accepted-20261009.json)。自己最良0.950を保持し、残350件の加工済み学習cacheを取得中です。[現在地](PROJECT.md)・[比較の順序](docs/roadmap.md)。以下は各確認時点の履歴です。

**最新（10月9日11:28 JST）：自己Public0.950を保持し、次候補p011の保存実行・可視検査を完了。** [root12](experiments/p011-serving-visible-output-root-review-20261009.json)・[独立33](experiments/p011-serving-visible-output-independent-review-20261009.json)成功。Kaggle認証は復旧、次はfresh提出枠確認と採点です。学習cacheはPC3,857件、remote全24 COMPLETE、残350件／全件index／正式fit未完。[現在地](PROJECT.md)・[今後の順序](docs/roadmap.md)。以下は過去の観測履歴です。

**最新（10月9日03:50 JST）：自己Public0.950を保持し、p011実256件の動作検査を完了しました。** [root24](experiments/p011-profile256-completed-root-review-20261009.json)・[独立47](experiments/p011-profile256-actual-results-independent-review-20261009.json)成功、工程計977.913秒/sample GPU8,893MiB、精度は未評価。提出リクエストは準備済み、直近API401でKaggle再認証待ち、ServingSave/競技提出0。学習cacheはPC3,332件、remote20/22 COMPLETE、全24 job保存受理済み/full4207 index/fit未完。[現在地](PROJECT.md)・[実証と未完](docs/validation.md)。以下の旧「最新」は各記録時点の履歴です。

**最新（10月9日03:19 JST）：自己Public0.950を維持し、p011の実64件検証を完了、別256件版をprivate保存実行として起動しました。** [root24項目](experiments/p011-profile64-completed-root-review-20261009.json)・[独立45項目](experiments/p011-profile64-actual-results-independent-review-20261009.json)で元融合差0・基準CSV前後byte同一・固定90:10候補再構成SHA一致。追加64×94窓に例外/fallback0、工程計230.089秒/最大sampled GPU8,893MiB。精度/全件提出は未確認。学習cacheはPC01〜19計3,332件/16,660配列を検査・配置済み、20〜22 remote観測RUNNING、残2未起動/full4207 index/fit未完。256版は保存ソース17項目成功、実結果待ちです。[参加計画](docs/roadmap.md)・[現在地](PROJECT.md)。以下の旧「最新」は各記録時点の履歴です。

2026年のRSNA膝MRIコンペを進めるためのリポジトリです。**Kaggleで画像キャッシュを作成、このPCのRTX 4090で学習、Kaggleで提出推論**を行います。Cドライブの空き約485GBに対して公式元画像は約570GBのため、元画像は全取得せずキャッシュを移します。データ・重み・認証情報はGitに入れません。

全件画像キャッシュ（4,407検査、約7.79GB）の取得・検証、weakラベルの監査と固定分割、RTX 4090での基準学習を完了しています。**weak 4,207検査・gold 58検査、fold 0の学習3,386／検証821検査**です。10月3日の5 epoch基準実験は約18分、最良checkpointはepoch 1でした。公開モデルを使った既存提出の実測Public 0.924は公式APIで確認済みで、自作モデルのスコアとは区別します。

**最新（10月9日01:56 JST）：自己Public0.950を維持します。** p006/p007/p008は公式0.950、p009画角160は[公式0.948](experiments/p009-scored-review-20261008.json)で、いずれも非採用。基準14/p005を保持します。p010の[実32件/10phase診断17項目](experiments/p010-AA-completed-compact-root-review-v4-r2-20261009.json)では、同じ入力・同じGPUでもraw値の差が残りました。原mainの通常反復はLM順位2セル変化、固定反復は順位差0、全6の旧1e-7 gate失敗を保持し候補未資格。精度や原因、一般的変動上限の証明ではありません。画像キャッシュ01〜13はremote作成、取得、全配列検査とcanonical配置を完了し、**計2,282検査/11,410配列**です。ZIPのbyteコピー・展開はせず、元SHA・更新時刻を保つhardlinkを使用。[14〜16の保存100項目](experiments/native392-shards14-16-saved-binding-review-20261009.json)も成功、観測RUNNING、残8未起動。全4,207件index/正式fitは未完です。[fast16学習入口](experiments/f005-native-fast16-processed115-root-review-20261009.json)は人工144窓・実加工済み115件のラベルなしCUDA検査と両configの資源検査を通過。人工速度改善を全epoch時間や精度の改善とは扱いません。p011の元14全7 codeを保持する独立stock224追加枝はローカル実装中、実候補推論・提出はまだです。00:10公式日次5件・残0、枠の更新時刻は未確認。限定公開調査では0.955以上の再現レシピはまだ確認できず、自己0.955/厳密な0.960超は未達成です。[現在の判断と履歴](PROJECT.md)。

08 Version2の256件成功は保持します。人工1,300件の提出契約は成功しましたが、MRI/encoderはmockで隠し完走の証拠ではありません。[提出経路の独立監査](experiments/serving-contract-independent-review-20261008.json)。f002には元画像全体の配列重複と不適合系列で停止する経路を確認し、メモリ配送だけを変えた[13の診断](docs/frozen-features.md#10月8日以降のf002診断)を別に準備しました。p003/f002の自己Publicはなく、作者0.943/0.944を保証しません。p002自己Public0.937を保持。0.945〜0.950も版別採点・Input・外部重み条件を[監査](experiments/public-candidates-20261008.json)しました。生成10/13は重みを含むため個別Git除外です。

f003の24 epoch学習はユーザーが両foldを完了し、136項目の監査が成功しました。fold0 weak AUC12は0.813290→0.823060、fold1のBCE選択予測は元f002と同一で、延長の利得は揃っていません。[完成比較](experiments/f002-f003-fold01-review-20261007.json)。[f004のdropoutだけ0.2→0.4](configs/experiments/f004-dinov2-attention24-dropout40.json)を今回両foldで実行しましたが、両fold平均AUC低下で採用を見送りました。教師17セルの二次レビューは6セルの原文状態を整理、11セル未確定、訂正採用0です。[根拠監査](docs/label-review.md)。実データ/モデル追加取得・MRI処理・外部アップロード/提出は行っていません。

**提出の最新（10月6日）：05も隠し再実行で例外になりました。同じ05の再提出は保留します。** 提出ref `56870280`、scriptVersionId `355633504`、自己Publicなし。保存版の全セルは配布05と一致し、可視3件は256.538秒で完走しましたが、隠しtracebackは取得できず原因は未確定です。ユーザー報告の約4時間は隠し推論の実測時間ではなく、9時間超過とは断定しません。[2回目の失敗監査](experiments/p003-second-failure-review-20261006.json)、[診断の方針](docs/p003-speed.md)。自己採点済みの基準は[p002 Public 0.937](notebooks/public/01_submit_p002_scored.ipynb)。作者0.943は設計上の参考値で、自己再現値として扱いません。以下は以前の履歴を含みます。

10月4日に公式RulesとHost回答を取得し、vmohitrao Version 3ラベルを研究・学習目的とCC BY-NC 4.0の帰属等を守って採用する判断に更新しました。[利用条件の根拠](docs/research/kaggle-source-eligibility-20261004.json) に本文の確認日と出典を保存しています。

## 最初に読む資料

1. [PROJECT.md](PROJECT.md) — 現状、次にすること、完了条件。
2. [今後の参加計画](docs/roadmap.md) — 環境の判断、最初の二日間、締め切りまでの優先順位。
3. [GPU端末での開始手順](docs/gpu-start.md) — Windows環境とKaggleキャッシュを使った学習手順。
4. [キャッシュ作成後の手順4〜9](docs/after-cache.md) — 転送検査、ラベル監査、fold 0学習、採点、改善、最終選択の操作と完了条件。
5. [大会と調査結果](docs/competition.md) — 公式に確認した条件、添付分析との照合、未確認事項。
6. [公開モデルの再現手順](docs/public-baselines.md) — 最初の採点に使う候補と記録項目。
7. [モデルと検証の設計](docs/experiment-design.md) — ラベル、前処理、分割、改善の順序。
8. [Kaggleへの提出手順](docs/kaggle-submit.md) — オフライン推論の準備。

10月5日、固定50:50 rank候補r001の**Public 0.749を公式APIで確認**しました。weak AUC 0.830249とは対象・正解が異なります。今後は**Public 0.924の公開学習済みモデルを基準に、公開候補一つの推論比較を最優先**にします。次候補DINOsaur V32は指定5 Inputsを固定し、元の推論ソースに開始・完了検査を加えた[提出用Notebook](artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)を準備済みです。次はユーザーが[操作手順](docs/public-baselines.md#ユーザーが行う操作)に沿ってT4 x2・Internet OFFで保存実行し、成功後に手動提出します。実MRIでの完走と自己採点は未確認です。[段階別計画](docs/research/public-model-strategy-20261005.md)、[提出監査](experiments/submission-audit-20261005.json)。自作ResNet18の追加学習は補完根拠が得られるまで保留します。

10月4日までのA/B/C・fold 1対照・通常BN／固定BNの20 epochは完了済みです。当時の全97 unittest・Ruff・人工CPU/GPU提出契約の成功は [検証履歴](docs/validation.md)、weak評価の詳細は [対照実験](docs/controlled-experiments.md) に残します。公開重みのgold履歴を監査し、自分の学習・選択にgoldを使わない方針を維持します。

## 大会の要点

MRI検査ごとに12所見の連続値を予測し、12項目の平均ROC-AUCで評価されます。提出はKaggle Notebookで、インターネット無効、実行9時間以内、出力名は `submission.csv` です。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

参加・チーム統合は **2026年10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。期限は変更される可能性があるため、提出前に公式ページを確認してください。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

レポートは学習ラベルを作るための情報として扱い、推論コードはMRIと検査IDだけを使用します。提供ラベルの少なさ、レポート由来ラベルとの違いが今回の設計上の焦点です。

## このPCでできるチェック

Python 3.11〜3.13を使用してください。基本パッケージは標準ライブラリだけで動きます。この端末にはPython 3.12.13の `.venv` と画像依存があり、torch 2.10.0+cu128・torchvision 0.25.0+cu128でRTX 4090のCUDA利用可を確認しています。以下は準備済み環境で実行できます。

```powershell
.\.venv\Scripts\python.exe -m rsna_knee doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/cache-v1
```

新しい端末でのvenv作成と、Windowsの `python` がアプリ実行エイリアスになる場合の対処は [gpu-start.md](docs/gpu-start.md) を参照してください。

## 構成

```text
configs/             再現可能なJSON設定
src/rsna_knee/       軽量なCSVツールとGPU用パイプライン
scripts/             提出コードの梱包
notebooks/           Kaggleキャッシュ作成・提出Notebookのひな形
docs/                調査、実行手順、設計判断
docs/research/       ユーザー提供分析の原文と出典一覧
experiments/         実験台帳と記入用テンプレート
tests/               人工データによる契約・分割・前処理チェック
data/                実データ、ラベル、キャッシュ、分割（Git対象外）
artifacts/           重み、予測、ログ、提出用zip（Git対象外）
```

自作モデルは **ResNet18と所見別Attention**。ランダム初期化を比較基準に、ImageNet事前学習を固定条件で検証しました。公開CoAtNet/DINOの提出再現と、自作モデルのweak検証は分けて記録します。
