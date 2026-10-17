# RSNA Knee Abnormality Detection

**2026-10-16 18:11 JST：修正版P032-v6を正式提出し、提出57262934の受理を確認しました。** 192件診断・可視3件とも失敗0、独立FP32再計算とnative実寄与の検証を通過。採点結果は未返却で、自己最高0.957を保持しています。[提出記録](experiments/p032-v6-formal-Submit57262934-score-unreturned-20261016.json)、[現在地](PROJECT.md)、[検証範囲](docs/validation.md)を参照してください。以下は各時点の履歴です。

追加で確認した作者0.958の固定係数だけのP033も可視検証を完了しましたが、こちらは未提出です。[資格と比較範囲](experiments/p033-visible-pass-parent-weight-vector-unsubmitted-20261016.json)。次の判断は既存57262934の採点結果に基づいて行います。

**2026-10-16継続：自己Public 0.957を保持し、F006追加候補のspawn不具合を修正しました。** 修正版v6は54テスト・ruff成功。KaggleのCPU起動確認も9.866秒で完了しましたが、GPUの旧2実行が枠を占有しているため、修正版の実推論と正式提出はまだです。旧2本のStop／Cancelを依頼済みで、成功済み提出57225898は変更していません。[現在の状態と次の実行](PROJECT.md)、[検証範囲](docs/validation.md)、[実行記録](experiments/p032-v6-ready-CPU-startup-pass-GPU-capacity-wait-20261016.json)を参照してください。以下は各時点の履歴です。

**2026-10-16追記：次が最後に近い提出という意向を反映し、[最終候補を絞る総合計画](docs/research/20261016-final-submission-plan-v1/report.md)を保存しました。** 自己0.957を保護し、追加正式提出は原則1候補という計画上の仮定で準備します。F006固定20%の運用・rank契約を優先確認し、条件が整わなければ0.957を保持。前回の複数候補の連続採点は標準経路から外します。これは実際の残枠の観測でも新たな提出実績でもありません。

**2026-10-16：提出57225898（P029M）の自己Public 0.957をユーザーが確認済み。次の目標は0.960以上です。** 当方の今回の公式API再照会ではなく、[ユーザー確認記録](experiments/user-confirmed-public957-research-20261016.json)として更新しました。[調査レポート](docs/research/20261016-goal096-review-v1/report.md)はR6 TTA比較→R7 bundle比較と、並行するF006 hidden失敗・rank統合の診断を推奨。今回は調査・文書更新のみで、モデル／MRI取得・学習・Kaggle保存・提出は実行していません。以下は各時点の履歴です。

**10月15日19:21 JSTの採点確認後：公開0.957候補の自己推論P029Mは可視検証を通過し、正式提出57225898を1回受理されました。自己最良の確認値は引き続き0.956、目標0.960超は未達です。** 新提出の公式Get/Listには状態・エラー・Publicがまだ返っていません。同じ固定R6でTTAだけOFFのP030Mも可視検証を完了し、両候補とも3検査×12ラベル・36値のfloat32再計算一致・検査単位failure 0を確認。P030Mは未提出でP029Mの結果を待ちます。可視全体453.7→405.2秒は1回ずつの観測で、精度改善やhidden完走の証明ではありません。公式枠は本日1件／残4／累計28。[最新正本](experiments/p030m-visible-pass-P029M-submit-unreturned-20261015.json)、[次の判断順序](docs/goal-096.md)、[検証範囲](docs/validation.md)。以下は各時点の履歴です。

**10月15日17:47 JST観測：自己最良は0.956。P029 V2は最初のR6 V1 manifest検査でERROR停止し、科学推論・MRI処理・モデル読込・正式Submitは0です。** [停止の正本](experiments/p029-first-guard-failure-20261015.json)。期待manifestが見つからないか複数だったため、実mountがR7だったとは断定しません。次は公開R6 V1をKaggleのCPU側だけで取得・全ハッシュ照合する別stageを準備しています。自己認証済みで、[17:21の公式照会](experiments/authenticated-progress-refresh-20261015.json)は自己0.956、旧3提出はhidden失敗、提出可能数5・累計27。目標0.960超は未達です。詳細は [PROJECT.md](PROJECT.md)、検証境界は [docs/validation.md](docs/validation.md) を参照。以下は各観測時点の履歴です。

**10月15日17:32：公式認証が完了し、自己最良0.956を再確認。公開0.957候補P029を非公開Notebookで実行中です。** 公開LBは1029位／6025チーム。旧P028/P024/F006両foldはhidden再実行エラーでスコアなしと確定しました。日次残枠は公式5回です。P029は元科学コードを保持し、誤った最新版R7を推論しないR6 V1 manifest guardを追加。正式提出・自己0.957再現はまだです。現在の表示0.960は47〜66位ですが最終メダル保証ではありません。目標0.960超への実行順序と出典は [PROJECT.md](PROJECT.md)、資格とテストは [docs/validation.md](docs/validation.md) を参照。以降の更新は各時点の履歴です。

**2026-10-15の再監査：最終確認記録は自己Public AUC 0.956、目標0.960以上は未達です。** 今回の環境ではCUDA/PyTorch、過去の実モデル・rawログ、Kaggleの接続済みブラウザを利用できず、提出3件の最新採点と現在枠は未確認です。学習記録は両fold計96epoch完了。GPUなしでの進捗証拠監査と固定20%候補のCSV契約確認を進めます。詳しい判断と出典は [PROJECT.md](PROJECT.md)、検証範囲は [docs/validation.md](docs/validation.md) を参照してください。以降の更新ブロックは記載時点の履歴です。

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


**10月12日19:36 JST：S6単独のP020が公式Public 0.954となり、自己最高を更新しました。** [公式採点と次の比較](experiments/p020-official-public954-three-candidate-comparison-20261012.json)。提出57098752はGet／Listともraw・typed COMPLETE、エラーなし。従来の自己最高0.951から+0.003ですが、厳密な0.960超えは未達です。P022の固定Native20追加とP021のlatest V2 mirror版はPublic・raw採点状態未返却。残枠2・本日受理3件・全履歴22件の末尾を公式確認しました。

次はP022／P021の公式比較を優先し、固定12epoch完了後にf006の最早最小BCEと[事前定義した固定10 AUC選択](experiments/f006-fixed10-AUC-selection-policy-20261012.json)を弱教師評価で比較します。最新保存snapshotはE8で、AUCは旧同epochよりわずかに高い一方BCEは高く、改善・採用を主張しません。[E8までの説明図](artifacts/research/20261012-f006-epoch8-progress-plot-v1/f005-f006-epoch8-three-panels.png)は未来のBCE・AUCを予測していません。目標未達・作業を継続します。以下は履歴です。

**10月12日18:43 JST：P021の追加手動提出はHTTP400で、非版管理Dataset使用を理由に拒否されました。** [原因・修復準備・比較結果](experiments/p021-http400-mirror-preparation-pair-review-20261012.json)。今回の追加試行は18:25の1回、元と合わせPOST計2回・自動retry0。旧17:24試行の原因は未確定です。特定Dataset名への帰属を断定せず、6モデルのSHAが一致する版管理mirrorへ入力を置き換え、元latest V2の予測計算を保つ新Source V3は準備完了、Root査読待ち・未Saveです。

P020／P022は18:07／18:19の公式照会でもPublic・raw採点状態未返却、残枠3・自己最高0.951。[弱教師ペア効用](artifacts/research/20261012-f005-fixed-mean50-pair-utility-v1/checks-and-compact-summary.json)は固定50:50混合による修正・悪化を集計しましたが、Public改善の証拠にはしません。[追加公開調査](artifacts/research/20261012-post-P022-public-stronger-primary-web-v1/analysis.md)でも直ちに再現できる0.960超の同一入力候補は未確認です。f006はE6比較を保持して固定12epochを継続。目標未達・作業を継続します。以下は履歴です。

**10月12日18:06 JST：P022の正式提出57102110が受理され、採点を待っています。** [可視結果と提出記録](experiments/p022-visible-accepted-single-Submit-score-pending-20261012.json)。保存版kernel138296990／V1の全3検査は成功3・失敗0。基準S6に固定20%の自作4モデルを加えた最終CSVを独立再計算し、18:01に手動Submit1回・再送0で提出しました。18:02の公式残枠は3、自己最高0.951・新Public未返却です。

次はP020のS6単独とP022の追加モデル候補を公式スコアで比較します。P021はSubmit受付不明・読み取り専用照会で対応refなし・再送0を維持。f006の比較snapshotはE6で、固定12epochを継続し、LR減衰が始まるepoch7以降と完了時のweak BCEを比較します。精度改善・厳密な0.960超えは未達・作業を継続します。以下は履歴です。

**10月12日17:41 JST：P022の圧縮ソースによる保存実行が受理され、実結果を待っています。** [受付・検証・提出状況](experiments/p021-visible-qualified-submit-unknown-p022-Save-accepted-20261012.json)。元ソース・metadataのbytesを保つjoint LZMA方式で、実Root metadata込みのAPIソースを946,315Bへ縮小。新Rootの査読・権限を束縛し、kernel138296990／V1を手動Save1回・再送0で受け付けました。旧版の1,272,956BによるSave前停止は履歴として保持します。P022の実完走・最終CSV・競技提出・自己Publicは未確認です。

P021は全3検査・12項目、原50:50と各arm自身の成功集合によるrankの36値をFP32で受理しました。ただし17:24のSubmit1回がHTTPErrorとなり受理不明です。17:36の読み取り専用照会では取得20件に対応refなし・残枠4、再送0を維持します。P020は17:11の専用照会でスコア未返却、自己最高0.951。f006 fold0は5／12epoch・LR倍率1で学習を継続しています。精度改善・厳密な0.960超えは未達・作業を継続します。以下は時点ごとの履歴です。

**10月12日17:11 JST：P019の実64件によるUID復旧検査をRootが受理し、P021は公式COMPLETEを確認しました。** [現状と確認範囲](experiments/p019-qualified-p021-complete-p022-source-size-hold-20261012.json)。P019は同じ保存版で故障注入63成功・1失敗／252forwardと通常64成功／256forwardを実測し、失敗UIDのforward0・基準12文字列一致を確認。これは復旧と推論の受理で、最終提出CSV・精度改善の受理ではありません。P021（kernel138292308／V1）は手動Save1回後にCOMPLETE、実出力検査・競技提出・自己Publicは未確認です。P022は実受理metadataを組み込むとソースが内部950,000B上限を超え、Save前に停止。元metadataを保持した圧縮版を準備中です。

P020提出57098752は17:11の公式照会でもraw採点状態・Public未返却、残り日枠4・自己最高0.951です。f006 fold0は4／12epoch完了で学習を継続し、LR変更が始まるepoch7より前の差を改善とは扱いません。厳密な0.960超えは未達・作業を継続します。以下は時点ごとの履歴です。

**10月12日16:20 JST：P018の実256件検査が完了し、自作4モデルの推論確認を通過しました。** [実結果と次の候補](experiments/p018-native256-qualified-p022-source-preparation-20261012.json)。2枚のT4、256件・1,024forward、約633秒、fallback0、元入力ハッシュとFP32平均を確認。これは推論の動作確認で、精度改善の証拠ではありません。P020提出57098752は16:19時点でも公式スコア未返却、残り日枠4・自己最高0.951を維持。次はUID単位の構造的読込失敗への復旧を実64件で検証し、復元公開S6に固定20%の自作モデルを加えるP022を提出候補にします。厳密な0.960超えは未達、作業を継続します。

**10月12日15:59 JST：P020の公式提出57098752を確認し、残り日枠は4です。** [実結果と提出の記録](experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。全3検査・12項目、2T4・各4モデル・失敗0、元70:30の36値をFP32で独立確認し、同kernel138279894／V1を手動1回提出・再送0。公式Getと一覧で提出の存在は一致しましたが、raw採点状態とPublicは未返却です。SDK既定PENDINGを公式状態とせず、作者0.954と自己最高0.951を区別します。厳密な0.960超えは未達、goalはactiveです。以下は時点ごとの履歴です。

**10月12日15:53 JST：公開S6のP020が公式提出57098752として受理されました。** [実結果・提出の記録](experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。保存kernel138279894／V1のソース・公開Input13／1／1・COMPLETEを照合し、結果8ファイルを取得。全3検査・12項目、2枚のT4と各4モデル、失敗0を確認し、元の70:30百分位rank混合36値をFP32で独立再計算しました。取得側の省略値・終了コード型の仮定を修正し、元の取得記録を保持したままローカル受理しています。提出直前の公式残枠は5、手動Submitは1回・再送0。作者0.954は自己スコアではなく、公式採点待ちのため自己最高0.951を維持します。厳密な0.960超えは未達、goalはactiveです。以下は各時点の履歴です。

**10月12日15:38 JST：配布が復元された公開S6をP020として非公開保存し、実結果を待っています。** [公開候補の照合と実Save](experiments/p020-public954-restored-S6-private-Save-current-research-20261012.json)。作者0.954の保存V3・script357152922とS6の現V13が対応し、manifestは旧V10と全bytes一致。[29番Notebook](notebooks/public/29_submit_p020_restored_s6_public954_control.ipynb)は元5セル・固定70:30を保持し、入力ハッシュと全test UID／提出CSVの検査2セルだけを追加。kernel138279894／V1のSave1回が受理されました。実完走・競技Submit・自己スコアは未確定です。作者0.955は私有追加Input、0.956採点版はモデルV1の一覧404で再現候補へ転用していません。自己最高0.951、厳密な0.960超え未達、goalはactiveです。以下は各時点の履歴です。

**10月12日14:54 JST：P018の64件検査が合格し、256件検査とf006 fold0の学習を並行して進めています。** [実行状況](experiments/p018-native64-f006-fold0-progress-20261012.json)。REG2件の各193memberについてremoteバイト一致を確認しました。64件検査は2枚のT4で256forward、約155秒、fallback0で合格。256件検査（kernel138273767／V1）は保存受理済みで結果待ちです。Serving・新しい競技提出・今回の精度改善は未確認です。自己最高Public 0.951、厳密な0.960超えは未達で、goalはactiveです。

RTX4090では終盤の学習率減衰だけを変えるf006 fold0を学習中です。訓練3,386件・検証821件、入力・教師・seedを固定し、固定12epochを完了後に最早最小weak BCEで選択します。途中epochは採用しません。P018のKaggle実行・採点待ちと約6時間の学習を並行させますが、Public改善は未証明です。Publicによる追加調整は行わず、fold0の完了・比較後にfold1を判断します。以下は履歴です。

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
