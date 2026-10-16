# 準備内容の検証範囲

## 2026-10-16：自己0.957のユーザー確認と調査の検証境界

提出57225898の自己Public 0.957は、今回のユーザーの明示回答による確認です。当方の認証済みGet/List再照会、未丸め値、現在順位・残枠・GPU quotaの再確認は行っていません。[確認記録](../experiments/user-confirmed-public957-research-20261016.json)。10/15のraw証拠と未返却記録は変更していません。

[調査レポート](research/20261016-goal096-review-v1/report.md)は保存記録・コード・添付分析とWeb一次資料を照合しました。人工順位例2件で、(1)再順位化後の失敗行だけraw親へ戻すとAUC1→2/3となること、(2)同じ百分位尺度でも成功集合内rankと全件rankの混在で逆転することを確認。実際のP029Mの損失やhidden例外原因の証明ではありません。親の再順位化は全成功時にも式を変え、2fold平均も自動的にOOFにならないため、将来比較の契約として明記しました。

最初の反例は実`preview()`と人工CSVで出力を再確認し、MCL／Synovitisの保持も確認しました。[今回の検証記録](research/20261016-goal096-review-v1/verification.json)は人工値・文書SHA・台帳全15列と旧prefix保持を記録します。既存suiteを再実行して成功を主張するものではありません。

今回は実モデル取得・読込、実MRI処理、学習、Kaggle保存・提出、学習／推論コードの変更をしていません。GPU forwardや既存テスト群の再実行による新資格はありません。旧限定53件と広域299件の未解決状態は各観測時点のまま保持します。以下は過去の検証履歴です。

## 10月15日19:21の採点確認後：P029M提出とP030M可視対照の独立照合

19:33:56 JSTの追加公式Getも同じ377バイト・SHA9f435966…dc6cで、status／errorDescription／publicScoreはいずれも欠落したままです。[最終チェックポイント](../experiments/public957-final-checkpoint-20261015-1934.json)。状態欠落とCOMPLETE／失敗／精度0を混同せず、追加Submitもしていません。

P029M V1の通常sessionはCOMPLETE。取得対象はCSV6件・receipt JSON2件・workerログ2件と通常ログだけで、ローカルにモデル本文やMRIは取得していません。[取得receipt](../artifacts/research/20261015-p029m-visible-output-audit-v1/download-receipt.json)。Rootは保存済み可視test.csv／sampleに対して6CSVの3UID・順序・正規12列・有限値・[0,1]を検証しました。各raw384/raw224/oursの平均同順位rank/Nから元の括弧順序で計算しfloat32へ量子化、最終CSV全36値のビットが一致。[Root監査](../artifacts/research/20261015-p029m-visible-output-audit-v1/root-independent-audit.json)、[独立監査](../artifacts/research/20261015-p029m-visible-output-audit-v1/independent-numeric-audit.json)。従来のfixed20プレビューとは順位式が異なり、代用していません。

実stdout eventで9member guard成功は1件。traceback中のソース文字列を数えていません。blend receiptは各arm failure／both_failed全0、R6はscored3・fallback0・worker終了0、bundle fe055406。両workerは4fold・fp16・FLIP_TTA_READY・timeout falseを報告。fluid特徴の欠損1件は規定の欠損処理であり、検査未採点とは区別します。compile/eagerの最終study差と中間group差は別値で、FP32ビット同値や無言fallback皆無を証明したとは扱いません。

提出直前にprivate／GPU／Internet OFF／candidate V1／stage V1／全候補セル一致／現在枠5を再確認し、`57225898`を19:01:51 JSTに1回Submit受理。19:21の公式Get/Listでもstatus／errorDescription／publicScoreがなく、hidden完走・採点成功・自己0.957は未資格です。公式枠は本日1・残4・累計28。[提出レビュー](../artifacts/research/20261015-p029m-submit-preflight-v1/manual-submit-review.json)。提出前GetはDockerのprivate-byod registry＋要求と同じ2757 digestを返しました。旧Getのキー欠落と区別し、古い監査JSONは取得当時の証拠として保持します。

P030M V1/kernel138832933はTTA booleanだけをOFFにした可視対照として1回保存・実行しCOMPLETE。全セル・Input・設定のGet一致、元test/sampleに対する6CSVの3UID順と12列、全36値の再計算一致をRootと独立agentが確認しました。[Root](../artifacts/research/20261015-p030m-visible-output-audit-v1/root-independent-audit.json)、[独立監査](../artifacts/research/20261015-p030m-visible-output-audit-v1/independent-p030m-audit.json)。TTA OFFは実ASSETS_READY false・元OR条件とmanifestにoverrideなし・FLIP_TTA_READY 0・workerのflip_ttaキー不在で確認し、receiptに明示falseがあるとは主張しません。全arm failure 0／R6 scored3／worker終了0・timeout false。R6確率31/36値が変わる一方で3件内の順位は同一、最終CSVバイト同一も確認。hidden／Publicは未確認・Submit0です。

P030Mの全体405.2秒とP029M453.7秒は単回比較で、未変更armの時間も変わっています。48.5秒差をすべてTTAの因果効果とせず、大規模実行や精度へ外挿しません。quota rawは108000秒許可／12237.127秒使用／予約0の観測値。最初のquotaキー名検査はSave前に停止し、実Saveは1回だけでした。typed durationは計算に使っていません。限定unittest53件の52成功・1skip、広域299件の未解決範囲は下記のままです。以下は各取得時点の履歴です。

## 10月15日18:11：固定入力stage完了とP029M起動

stage b1のtemp親未存在による取得前停止を保存し、新しいb2で親作成を修正しました。b2のraw Getではprivate/CPU/Internet ON/Inputなし/currentVersionNumber=1、送信した全セル本文と一致。通常sessionはCOMPLETEで、実stdoutの9ファイル検証成功JSONを確認しました。取得した2213バイトのprovenanceの全9項目を、元manifestから束縛したサイズ/SHAへ突合しています。計886,813,460バイトはKaggle上での既存公開ファイルのコピーで、ローカルモデル取得・デシリアライズ・MRI読込・学習ではありません。[記録](../experiments/p029m-frozen-input-private-run-20261015.json)。

P029M V1/kernel138824627のSAVE_AND_RUN_ALLは18:11:16 JSTに1回受理。送信payloadはcode/configのみ、元数値コードは保持し、固定mount探索1行と全9member guardを追加。古い作者dataSources/runtime metadataは新launchcopyで除去しました。実mount/GPU forward/可視CSV/hidden/Publicの資格はまだfalse/null、Submit0です。P030MのTTA一箇所だけの対照は別途AST/byte/compile/独立レビュー済みですが未実行です。

最終限定53件は52成功・1symlink skip・失敗/エラー0。[53件の全文ログ](../artifacts/research/20261015-p029-stage-fallback-validation-v1/selected-53-unittest.log)、[結果とコードSHA](../artifacts/research/20261015-p029-stage-fallback-validation-v1/validation.json)。旧45件に新tracked stage準備8件を加算したもので、旧artifact内8件は重複計数していません。親ディレクトリ未存在からのmain経路は人工rootとfake transferで実際に検証。元科学コード・ネット・実モデル/MRI・実学習をこの限定suiteでは実行していません。

別途、agentが誤って広く実行した299件はfail9/error50/skip66で全体不成立です。[観測記録](../artifacts/research/20261015-p029-stage-fallback-validation-v1/broad-discovery-299-unresolved.json)。完全stdoutとbaseline比較がないため、全てを既存不具合と断定せず未解決とします。旧artifact不在や依存importに関する例があり、人工DICOM/CUDA学習のtest定義も含むため、この広域実行全体について外部操作/人工処理ゼロを保証しません。対象変更の検証根拠は保存した限定53件です。広域suiteは再実行・追加修正していません。以下の未認証・旧待機・途中RUNNING等は各取得時点の履歴です。

## 10月15日17:47 JST観測：P029 first guardの失敗と認証後raw監査

[P029 V2停止の正本](../experiments/p029-first-guard-failure-20261015.json)はkernel138818102／V2／ERROR・In [1]のメタデータ検査失敗です。期待R6 V1 manifestが一意に見つからず、科学セル・MRIデコード・モデル読込は未開始、正式Submit0・自己Public未測定。実mountのbundle／版を表示していないためR7接続とは断定しません。旧pollのsubstring成功boolはtraceback内のprintソースへの一致で、実stdoutの成功eventは0。guard成功や可視／hidden資格を発行しません。

[認証後の提出・quota独立監査](../experiments/authenticated-progress-refresh-20261015.json)では、raw/typed計15証拠のサイズとSHA、Get/List4件の共有欄、27件の自己identity、最高Public0.956を再計算しました。旧3件はrawにCOMPLETE＋明示errorDescription＋publicScore欠落＋totalBytes0があり、SDK既定PENDING・説明文の旧pending・ゼロ精度には置き換えません。V22Rはraw Get/ListともCOMPLETE・0.956・報告エラーなしです。認証は成功済みで、旧未認証節は履歴です。

17:21 JSTのlimits raw/typedはnumAllowedNow5・numTotal27で一致し、numTodayはrawにないため0と推定しません。GPU quotaのrawはtotal108000秒・used11361.972秒・reserved0秒、残96638.028秒。typedの21600秒や`11361.972.0s`は採用せず、rawからDecimalで計算しました。quota更新予定は10月17日09:00 JST、残量と提出枠は観測時点の値です。

次のKaggle CPU staging案はソース準備・査読中です。公開固定V1のmanifest/member hashをKaggle上だけで検査する取得専用処理と、Internet OFFのMRI推論を分けます。現時点でstage完了・新mount・forward・hidden9時間・Publicの資格はなく、以前のP029 source／失敗runを上書きしません。以下の17:32 RUNNING等は旧観測の履歴です。

## 10月15日17:32：認証後の提出結果とP029起動

公式CLI OAuthの終了0・username `rsraki` を取得し、SDKでも同じアカウントを確認。認証コードやtokenを含み得るCLI標準出力/エラーはメモリ内で破棄し、認証値を手動表示・コピーしていません。認証済みGet4件/List/limits/GPU quotaはraw JSON本文と取得時刻・SHAを新規保存しました。[receipts](../artifacts/research/20261015-authenticated-submission-status-v1/receipts.json)。P028/P024/F006両foldはstatus=COMPLETEだけで成功と判定せず、errorDescriptionあり・Public欠損・totalBytes=0のhidden失敗と確認。V22RのみCOMPLETE・0.956・エラーなし。日次残5/累計27は今回の公式値です。

P029の送信はcode/configのみ。QUICK_SAVE V1とSAVE_AND_RUN_ALL V2を各1回行い、Submitは0。元科学コード7セル不変・作者出力除去・R6 V1 manifest guard先頭追加を送信前後に検証しました。V1 Getのprivate/GPU/Internet OFF/全10セルは一致。ただしInput版は返却参照から版suffixが落ち、内部版IDは未返却です。元Notebook metadataにある版情報を実接続と見なしません。内部設定APIはOAuthで403となり、CommitAndRunの書込は試していません。

V1で誤って指定したDocker registryは未実行のまま保持し、V2では作者実runの完全registry名とdigestに修正しました。[操作レビュー](../artifacts/kaggle/p029-r6-957-20261015-42fbc0e9ae14/batch-review-v2.json)。17:32:44 JSTのP029 V2は公式RUNNING、ログ0バイトで、manifest guard成功・forward・CSV・自己Publicは未資格です。

最終オフラインunittestは45件・44成功・1スキップ・失敗/エラー0。[結果JSON](../artifacts/research/20261015-public957-followup-validation-v1/validation.json)、[ログ](../artifacts/research/20261015-public957-followup-validation-v1/unittest.log)。内訳は監査8成功/1skip、CSV14、既存契約7、P029準備7、P030準備8。これはKaggle実推論の合格ではありません。以下の未認証や旧提出待機は以前の取得時点の記述です。

## 10月15日夕方：公開0.957候補・現在順位・P029準備

匿名公式公開LBの全6025行についてrank集合1..6025、team ID対応、重複なし、自己rsraki/既知Submit57164195を照合しました。自己Public表示0.956・1029位、現Public金22/銀301/銅602が境界。保存時刻16:40:59 JST、正確なHTTP取得時刻は未記録です。Privateキーはなく、個別の旧3提出状態と現在枠は未確認のままです。[順位監査](../experiments/medal-frontier-audit-20261015.json)。

公開Code一覧はID154281/SCORE_DESCENDING/EVERYONE/page1/100を要求し、100行降順・競技参照・linked有無・ID順を検証しました。0.957が3件で、うち1件は非公開Inputあり。`totalCount=0`の不整合があるため全件取得・全公開recipe不存在は証明しません。[一覧監査](../experiments/public-code-score-frontier-20261015.json)。

P029は作者V1/script358332966/Submit57183099/公式表示0.957と一致する元Notebook638918B・SHA449ce778…bd02を完全コピー。最初のHTTP保存はPowerShell byte配列が数値列文字列になったため、その全バイト値から新しいexactファイルへ復元し、受信サイズ一致・JSON構文を確認しました。元の保存物も保持しています。全9セル・出力・metadataを保存し、7コードとembedded7pyはcompileだけを実施しました。元出力は作者の歴史的出力です。

[P029 builder](../scripts/prepare_p029_public957.py)はソース/metadata SHA、linked提出のsource版、固定3Inputと内部版、実Docker、Internet、公開性、R6 V1の9ファイル名/サイズを照合してから新runを作ります。組立時にはR6 V1 manifest本文も未取得でしたが、その後公式CLIで4120バイトのメタデータだけを取得し、期待SHA6508…/bundle fe055406/歴史ソースSHAを突合、別preflightも成功しました。[追加監査](../experiments/p029-r6-v1-manifest-metadata-verification-20261015.json)。これはモデル本体・mounted版・forwardの検証ではありません。最新V2/R7誤接続を拒否する契約を維持し、元のP029 runは上書きしていません。元NotebookのGPU=falseと実run GPU=trueの差も保持し、将来の実保存T4×2を別確認します。

人工unittestはP029追加7件成功。ソース非実行、SHA不一致、Input版/linked版/Docker/Internet/非公開/ファイル一覧不一致拒否、未知magic/非literal source拒否、既存run保全、R7 manifest拒否を検証しました。[独立静的監査](../experiments/p029-public957-static-audit-20261015.json)も原byte・manifest・生成verifier・builder SHAを再計算して準備範囲PASSです。GPU/可視Serving/hidden9時間/自己Publicの資格は発行していません。

元recipeには成功UID集合ごとのrank、失敗時補完、224のFP32 fallback、既存/tmp/_fastread.py再利用等があります。8.3hは全処理hard timeoutではありません。新しいbatchで実経路と終了処理を確認する必要があります。SWAのGold選択露出・OAI条件・上流ライセンスを保持し、Gold clean/患者独立を主張しません。

[P030 builder](../scripts/prepare_p030_tta_control.py)はR6 V1とP029ソース/manifestを固定し、TTA boolean一箇所だけのAST変更、他6コードセル・埋込7モジュールの同一、credit保持、作者旧出力/番号除去を検証しました。追加人工8件は成功。第二の科学的変更・重複/nested代入・非literal・R7 InputやTTAを有効化するmanifest・親SHA不一致・既存runを拒否します。元ソースにraiseを含む人工例でも実行せず準備できることを確認しています。実行/精度資格はfalseのままです。

公式CLI2.2.4/SDK0.1.37をプロジェクトの新 `.venv` へ導入しました。標準認証は未認証・SystemExit、filesystem PermissionError=false。認証値は手動で読まず表示せず、出力を捨てた状態だけ記録しました。[認証状態](../artifacts/research/20261015-standard-kaggle-auth-check-v2/status.json)。ユーザーのログイン意向返信は完了確認と区別します。保存/実行/提出0、自動アップロード/提出コード追加0、MRI/モデル重み本体取得/学習0です。

---

## 2026-10-15：記録監査とCPU準備の検証境界

自己Public最良0.956は、10月14日14:33の公式Get/Listが一致したという追跡済み記録です。今回新しく公式APIを照会して確認した値ではありません。P028 `57163437`、P024 `57166365`、F006両fold `57169291` の最終記録は未返却で、現在の状態は未確認です。10月14日の残枠1は派生値であり現在枠ではありません。

作業開始時の `artifacts/` と `data/` はREADMEのみでした。以前のSHA参照を読みましたが、raw API本文・実checkpoint・予測CSV・V22/F006設計本体の照合はできていません。F006/F007/F008の計96epoch完了、native64/可視成功は過去の監査記録として区別します。モデル取得・MRIデコード・学習・Kaggle保存/実行/提出は今回実施していません。今回のCPU監査出力は新規runへ保存しました。

`nvidia-smi` は利用できず、明示Python 3.11.9にはtorchがありません。CUDA forwardは未実施です。物理GPUがないとの断定はせず、今回利用可能なCUDA学習環境は未確認とします。既存学習CLIのCUDA必須チェックは変更していません。GPUコードは変更せず、今回の追加は標準ライブラリのオフライン準備のみです。

Webでは公式概要・論文・公式コード・作者投稿を調査しました。Kaggle直接openの本文0行、プロフィール取得エラー、ブラウザinventory0件/in-app browser利用不可のため、自己提出スコア・日次枠・順位は更新できません。Rules全文の今回の再取得もできず、[10月4日の監査](research/kaggle-source-eligibility-20261004.json)を現在の新規確認に置き換えません。

F006の弱比較は選択後holdout再利用、fold0方針の観測後定義、固定10と全12の差、全12 bootstrap未定義反復を保持します。患者独立性・未知データ改善は未証明です。今回の人工CSVテストを実Kaggle推論・本NotebookとのFP32同値・Public改善の証拠に流用しません。

今回の [audit v2](../artifacts/research/20261015-recorded-progress-evidence-audit-v2/audit.json) は参照記述子30件（重複参照を別計数）について一致2・欠落28。SHA一致もファイル同一性の確認までで、公式スコアの真正性を再認証するものではありません。記録スコア0.956、現在スコアnull、現在残枠null、達成確率nullを出力します。認証名・認証ディレクトリ・root外への解決・非JSON/モデル参照・上限超過を内容読込前に拒否し、過去の出力ディレクトリも上書きしません。

固定20%プレビューは全N平均同順位rank、二段階rank、UID集合/列順/重複/finite範囲検査、親順序維持、MCL/Synovitisと明示失敗UIDの元セル文字列保持を標準ライブラリで検証します。失敗UIDのnative値も含め入力は全Nの有限確率を要求し、欠損推論値を自動補完しません。入力のfold/モデル同一性・native平均のFP32集約・本Notebookとの等価・25出力・hidden/9時間・Publicは未検証です。

最終確認はPython 3.11.9の標準 `unittest` で30件実行、**29件成功・1件スキップ・失敗/エラー0**。内訳は新監査8成功/1スキップ、新CSVプレビュー14成功、既存 `test_contracts.py` 7成功です。スキップは実symlink作成権限不足で、外部へresolveした参照を内容open前に拒否する模擬試験は成功しました。設計参照欠落時の出力未作成も確認しました。[結果JSON](../artifacts/research/20261015-cpu-continuation-validation-v1/validation.json)と[全文ログ](../artifacts/research/20261015-cpu-continuation-validation-v1/unittest.log)を保存。`git diff --check` は成功しています。

新しい[人工6 UIDのrun](../artifacts/research/20261015-fixed20-synthetic-contract-v1/preview/manifest.json)ではnativeの行順を逆転し、1件の明示失敗を与えてCSVを生成しました。数値契約・元セル保持・ソース/入力/output SHAを確認しただけで、実MRI/教師/モデル予測は含みません。[集約記録](../experiments/cpu-continuation-audit-20261015.json)に入力・コードSHA、環境、採用理由、seedなし/新学習なし、参照fold0+1を記録し、ledgerは旧バイト列を維持して1行追記しました。

---

**10月14日14:59 JST：自己最良Public 0.956を維持し、S6＋F006両fold E12・固定20％の正式提出57169291を受理確認しました。目標0.960以上は未達・継続中です。** [最新の記録](../experiments/F006-bothfold-formal-Submit57169291-Public956-checkpoint-20261014.json)。可視3検査・20出力を再計算してRoot受理後、Submitを1回実行。V22Rは公式COMPLETE・自己0.956、P028・P024・F006両foldはPublic結果待ちで、旧提出は再送しません。本日4件・残枠1・累計27件は前回公式値と今回受理1件からの導出で、提出後の公式枠再照会値ではありません。

F006の可視3検査成功はhidden完走やPublic精度の確認とは別です。F007も可視20出力はRoot受理済みで正式Submitは0。次は同じF006重み・同じ固定20％で、親モデルだけをS6から自己0.956のV22へ切り替える候補を検証します。V22作者報告0.956と自己値は区別し、Docker差37c64…／2757e…、公開重みのGold選択履歴の不確かさを保持します。MRI-COREは条件付き調査候補で、重み取得・実装・追加学習は開始していません。以下は過去時点の記録です。


**10月14日14:33 JST：自己再現V22Rの公式Public 0.956を確認し、現在の最良値として採用しました。従来0.954から+0.002、目標0.960以上は未達で継続中です。** [採用記録](../experiments/V22R-official-Public956-Root-adoption-20261014.json)。Submit57164195を自己アカウントの公式Get・ListともにCOMPLETE・0.956・エラーなしで確認。作者V22の報告0.956と自己値0.956は別記録です。P028・P024はPublic未返却で再提出せず、本日3件・残枠2・累計26件です。

V22は元コード6セル・Input3固定版・推論式を保持しましたが、Dockerは作者要求37c64…から実保存2757e…へ変わっており、R7単独の改善や非公開精度・公開重みのGold cleanを断定しません。次はS6＋F006両foldの固定20％を正式対照として検証し、同じF006重み・同じ係数でV22へ親モデルだけ切り替えて比較します。F006の新native64・内容SHAとF007可視20出力はRoot受理済み。MRI-COREは条件付き調査候補に留め、重み取得・実装・学習は開始していません。以下は過去時点の記録です。


**10月14日14:21 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [最新の記録](../experiments/F007-visible-pass-F006-bothfold-native64-pass-20261014.json)。14:13 JSTの公式Get／ListではP028・V22R・P024のraw状態とPublicは未返却、エラーなし。本日3件・残枠2・累計26件です。

F007は可視3件・20出力の検証を通過しRoot受理、正式Submitは0です。F006両fold E12も64 MRI・256 forwardと新重みの内容SHA確認を通過しRoot受理しました。F006の提出推論・Publicはこれから検証します。次はF006を優先し、V22との組み合わせは公式スコアを見て判断します。ローカルMRIデコードと追加学習は0、CPU復元とKaggle MRI推論は実施済み。旧エラーを含む以下の記録は過去時点のものです。


**10月14日13:45 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [今回の正本](../experiments/F006-bothfold-CPU-private-ready-F007-finalmask-error-20261014.json)。13:35 JSTの公式Get／ListはP028・V22R・P024のraw状態／Public未返却、エラーなし。本日3件・残枠2・累計26件です。

F006両fold E12の新4モデルは実CPU復元と独立検証を通過しRoot受理。CLS0・REG0・REG1の実指紋は保持し、CLS1だけを置き換えました。fold1モデルの非公開Dataset12573715/V1はREADY・2ファイル名／サイズ一致を読み取りで確認済み。Create応答guardの停止を保存し再送せず解決しました。新しい推論ソースへ伝播中で、Kaggle上の内容SHA・新native64・Serving・Publicはまだ未検証です。

F007のWindowsパス修正job138608301/V1はS6側の可視3件を完了後、最終mask処理でKeyError serving_shasとなりERROR。Submit0のまま、最終CSVまでのメタデータ契約を人工データで検証しています。元の数値コード・旧エラー・native64記録は保持。ローカルMRIデコードと追加学習は0、必要なCPU復元・Kaggle MRI推論は実施済み。以下は過去時点の記録です。


**10月14日13:18 JST：自己最高Public 0.954、目標0.960以上は未達。3件を正式提出し採点待ちです。** [最新正本](../experiments/three-submitted-two-native64-pass-F007-visible-error-20261014.json)。13:11 JSTの公式Get／ListではP02857163437・V22R57164195・P02457166365のraw状態とPublicは未返却、エラーなし。本日3件・残枠2・累計26件を確認しました。

P024新重みとF007残差モデルは、それぞれ64 MRI・256 forward・FP32集約・内容SHA・alias0・fallback0・資源／終了処理の検証を通過しRoot受理済み。P024は可視3件／20出力も再計算PASSし、Submit1受理。可視CSVがP028と同SHAでも、3件だけの順位一致でhidden予測同一とは判断しません。F007提出用job138605901V1はコード・Input9・設定一致を確認しましたがERROR終了。Submit0、原因ログを調査中でServing／Publicは未資格です。

弱ラベルの直接比較ではF006 E12がF007 E12を両foldで上回り、事前固定50:50のCLS混合はAUCを下げました。比率探索はせずF006を優先。fold1 E12もCPU単体復元を通過し、両fold置換の新4モデル検証を準備中です。旧f005 E9とf006自身のBCE E9は別checkpointとして保持します。ローカルのMRIデコード・追加学習は0、必要なCPU復元とKaggle MRI推論は実施済み。以下は過去時点の記録です。


**10月14日12:25 JST：自己最高Public 0.954、目標0.960以上は未達。改良モデルの新64検査テストは実行中です。** [最新正本](../experiments/native64-Source4-running-Source3-error-preserved-20261014.json)。P02857163437・V22R57164195は正式提出済み。12:23 JSTの公式Get／Listではraw状態・Publicは未返却、エラーなし、本日2件・残枠3・累計25件です。

P024旧64検査job138598572V1は、REGモデル読み込みの作業ルート指定でdriver起動前にERROR。実MRI推論の資格は0です。旧コード・ログを保持し、作業ルート・明示実行フラグ・producer情報転記の3か所を独立5テストで修正確認しました。新Source4 job138601076V1は公式raw／typed RUNNING。モデル・前処理・数値処理は保持し、完走・結果・内容SHAはこれから確認します。

F007の残差モデルも独立レビューと実API容量確認を通過し、Save1でkernel138600811V1が返りました。返却を扱うguardが停止したため、追加Saveせず実保存refとコードを読み取りで照合中です。自己Public・新native64・Servingの資格は発行していません。以下は過去時点の記録です。


**10月14日11:51 JST：自己最高Public 0.954、目標0.960以上は未達。2候補を正式提出し採点待ちです。** [最新正本](../experiments/two-submissions-private-ready-Root-CPU-accepted-20261014.json)と[実行方針](goal-096.md)。最新公式Get／ListではP02857163437・V22R57164195のraw状態とPublicは未返却、エラーなし。SDK既定PENDINGを公式状態と推定しません。本日2件・残枠3・累計25件をraw／typedで確認しました。

P028親guard対処版とV22公開recipeの自己候補は、可視推論・12確率CSV・順位計算の再検証を通過し、各1回Submit受理。V22は元6コード・3Input版を保持していますが、要求した作者Docker37c64と実保存2757が異なります。この差を束縛して記録し、作者環境の厳密再現とは主張しません。作者0.956と自己Public未返却を区別します。

P024モデルDataset12571388とF007 head Dataset12571495は、実照会で非公開・V1・READY・正しい2ファイル名／サイズを確認済み。Create後の旧guard停止は保存し、再Createせず読み取りで保存実体を確定しました。remote内容SHAと内部版IDはAPI未公開なので未検証。Kaggle推論の起動前にmounted archive／member SHAを検査する境界で確認します。

P024新4モデルはRoot委任CPU復元と独立査読を通過し、Rootがcomponentを受理。共有データauditと各resource provenanceを区別し、bytesパスの監査漏れも修正しました。モデル・前処理・数値処理は保持。新native64／Serving／Public資格はまだ未成立で、F007の新9Input接続も準備中です。F006固定10 AUC候補を優先し、弱差をPublicへ加算しません。

今回のローカルMRIデコード・追加学習は0、Kaggleの可視MRI推論は実開始・完了済み。非公開送信・実行・手動提出はユーザーの今回＋今後の明示承認範囲で進めています。MRI・Report本文・教師行・認証情報の外部送信、自動アップロード・自動提出は追加していません。以下は過去時点の記録です。


**10月14日11:09 JST：自己最高Publicは0.954。目標0.960以上は未達です。** [今回の正本](../experiments/goal-resume-comparisons-P028-submitted-20261014.json)と[実行方針](goal-096.md)。最新公式照会でP026は0.953、自己最高0.954を確認しました。

旧記述の学習中／待機を再監査し、F006・F007両fold各12epoch、F008両fold各24epochの完了を確認。計96epochのcheckpoint／CSVとBCE bestのSHA、CSV契約を照合してPASSしました。F006固定10 AUC副選択E12の固定50:50弱比較差はfold0 +0.004010、fold1 +0.002311で条件付き区間も正。BCE主選択は保持。F00824はfold1の改善が不確かで、追加延長を優先しません。weak差をPublicへ加算せず、全12指標・欠損・選択後の検証再利用を別記しています。

P028の修正版はprivate kernel138592375／V1の可視3件・20出力をRootが再検証してPASS、正式提出57163437を1回受理。hidden完走・新Publicはまだ未確認です。提出直前の公式残枠は5で、提出後の新しい枠を未照会のまま断定しません。

ユーザーは今回と今後の必要な自己rsrakiへの非公開送信・実行・手動提出を明示承認済み。過去の審査拒否を保持し、未回答待ち状態は解消しました。P024は公式Google JSON upload endpointへの許可不足を修正し2ファイルのPUTが成功しましたが、Create1後の返却guardで受付未確定。同じCreateは再送せず、保存状態を読み取りで確認します。MRI・Report本文・教師行・認証情報は送信しません。

Versia7 V22／script358019432は作者linked COMPLETE／0.956、3Input版を固定して自己再現を準備中。作者値と自己値を区別し、公開重みの上流gold利用が不明な部分も別監査に記録しました。現在の公式公開LB先頭100件は最高0.966、0.960以上54件。全件集計や利用可能重みの証拠とは扱いません。以下は過去時点の記録です。


**10月13日03:34 JST更新：自己最高Publicは0.954、残り提出枠は1です。** [F007接続受理・F008待機・F006 E6の正本](../experiments/F007Source3-reviewed-F008-waiting-F0061-E6-local-20261013-v2.json)。03:06のRoot委任local_analysisによる公式Get／ListはP026のraw・typed COMPLETE／0.953／エラーなしで一致し、Rootが独立受理しました。本日カウンタ4・全23件・日枠リセット未観測。P026はS6 control 0.954を上回らず不採用とし、次のnative候補は最高S6 controlを基準に進めます。P026基準への切替・Reader係数探索・追加Submitは0。目標未達・作業を継続します。

F006 fold1は6/12epoch（weak BCE 0.29386907・AUC12補助0.86122754・各208更新／skip0・累積12,386.50秒）まで完了し、学習は継続中です。定率6epochまで旧条件との入力・設定一致を確認。差は小さいものの、厳密再現・改善を証明しません。残る7〜12はlate cosine区間で、固定12epoch完了前の採用は行いません。

F007接続Source3はRootの独立レビューを通過しましたが、source準備のみの受理で、新Input・native64・Serving・Public・実metadata容量の資格は未発行です。F008の手動キューsession45576はWAITING_FOR_F0061、process object固定済み・子プロセスなし・F008学習未開始。F006 fold1の両プロセスの終了0と固定12epoch完了後に、同じ旧BCE CLS・zero残差から24epochをfold0→fold1の順に実行する予約です。4090はこの順序で予約し、追加GPU jobは開始しません。Source2のSTOPで次の起動を抑止し、稼働中子プロセスは停止しません。外部送信の権限はありません。

02:51にRootが[公式公開リーダーボード](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)を1回読み、取得上位100件は最高0.966・最小0.956、厳密な0.960超は28件でした。全件集計ではなく、次ページは未取得です。高スコアだけでは公開Notebook・重みの利用可能性や自己達成を証明しません。Notebook一覧401／Source403を全SDK障害とは扱いません。

02:29の匿名公開metadataでは、[Versia7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) V21の作者linked／bestは0.954、[Kiyoshi template](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template)の最新runは取消・linkedなし／bestはV9の0.951、[Dinosaur-v5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) V45は0.950でした。自己再現値とは区別します。EV本文は旧指定404、公式SDK形式への修正指定403で未取得・静的比較HOLD。同score／Inputだけで重複・撤回・非公開・認証失効や公開重み全体の不在を断定しません。

P024モデル、P028コード、F007モデル輸送の非公開送信3件の回答は未受領で、送信・再試行0を維持しています。P024／P028の具体的承認不足によるプロセス開始前の自動審査拒否を保存し、旧REG承認を新3件へ転用しません。ローカル作業と既存学習を継続します。以下は過去時点の記録です。


**10月13日01:06 JST：自己最高Publicは0.954、厳密な0.960超えは未達です。** [最新公式観測・F007四モデルCPU復元・学習状況の正本](../experiments/official01p06-F007-new-four-CPU-F0061-running-20261013.json)。Root委任のlocal_analysisがP026のGet／List／limitsを読み、RootがrawとSHAを確認しました。P026のraw状態・Public・エラー項目はまだ未返却で、SDK既定PENDINGを公式状態とは扱いません。本日カウンタ4・残枠1・全履歴23件で、日枠リセット時刻は未観測です。

F007は両foldの残差12epochを完了し、Rootが新しい四モデルをCPUで1回復元してPASSしました。generic factory1回、元checkpoint厳密適用4回、残差head安全読込2回、6組のstorage alias検査で共有0、REGは元objectを保持。forward／GPU／教師／MRI／ネットワークは0で、新Kaggle Input・native64・可視提出推論・Public精度の資格には進んでいません。元checkpointを保持した3メンバー・816,882Bのモデル用輸送はローカル準備だけで、外部未送信です。

F006 fold1はsession41581で2/12epochまで進行中（Root stdoutのweak BCE 0.340669・AUC12補助0.77125048）、まだ完了・採用していません。F008は旧CLS設定12epochを保ち残差期間だけ24epochにする準備とdefault HOLDまでで、実学習0。先読み高速化案も未実装・未計測で、既存一件の読込比率約63%を全epochの律速や速度改善とは扱いません。P024の新モデル送信とP028の新Notebook Saveは、自動審査がプロセス開始前に拒否した具体的承認待ちで、外部送信・再試行0を維持しています。以下は過去時点の記録です。


**10月13日：F007 fold1は12epochを完了し、F006 fold1の学習をRootが開始しました。** [最新公式照会・完了比較・実GPU確認・学習開始の正本](../experiments/official00p06-F007-fold1-complete-F006-fold1-training-20261013.json)。F007 session59615は終了コード0、元BCE規則の最早最良はE12で、weak BCE 0.27517597・固定10 AUC補助値0.88163248。Rootが全12 checkpoint／予測SHAと4207 NPZ・7,918,878,170Bの件数／サイズを受理しました。特徴7GB本文の再hashや、今回の凍結head tensor別復元は行っていません。全891件の保存CLS確率誤差0.0とnative oracle最初1件を区別します。

fold1の固定50:50弱教師比較差は+0.00137217（861供給group・1,000回、条件付きbootstrap区間+0.00021828〜+0.00274822）。fold0も弱比較では改善していますが、checkpoint選択後の診断であり、患者独立性、全2fold OOF、Public改善は証明しません。Rootのfold0実CPU復元に続くRTX4090の人工GPU確認は、3有効窓・wrapper2回＋原AST参照1回、計9 encoder窓で参照logit誤差0.0・padding誤差0.0でした。実MRI／教師は0で、新4モデル共有・2T4 native64／Serving／精度資格には移していません。

RootはF007 fold1完了と人工GPU確認後、開始前に定義したpolicyを維持してF006 fold1の固定12epoch学習を起動（session41581、新run）。主規則は最早最小weak BCE、副規則は全固定10項目が有限な最早最大平均AUC。元bestを副規則で上書きせず、同epochなら元checkpointを再利用して重複候補を作りません。Gold／Publicによる選択は0で、完了・途中採用・Public改善はまだ未確認です。

00:06 JSTのRoot委任local_analysisによるP026個別Get／List／limits読み取りとRootの独立確認では、P026はraw状態・Public未返却、エラーなし。SDK既定PENDINGを公式状態とは扱いません。既知の自己最高はP020・P021の0.954、P022は前回確認のhiddenエラー。本日カウンタ4・全履歴23件・残枠1を維持し、JSTの日付変更から日枠リセットを推定しません。P024の新モデル送信とP028の新Notebook Saveは具体的承認待ち・開始前審査拒否／外部送信0を維持し、別送信で迂回しません。厳密な0.960超えは未達・作業を継続します。以下は時点ごとの履歴です。


**10月12日23:30 JST：F007 fold1のcacheをRootが受理し、局所残差だけの12epoch学習を開始しました。** [cache完了・残差開始・次の計画の正本](../experiments/f007-fold1-cache-complete-CLS-exact-residual-training-started-20261012.json)。抽出session71660は終了コード0、4207 NPZ・7,918,878,170Bの全path／存在／サイズ／件数とCLS headのraw SHAを確認。全891検査・12所見の保存CLS確率との最大誤差は0.0です。native logitの直接oracleは最初1件でexact、全891件のfull native forward照合ではありません。7GB特徴本文の再open／hashや凍結head tensorの別復元とは区別します。

Rootは旧f005 fold1の元BCE最良E9を起点に、encoderと元CLS headを凍結したまま、局所残差だけをconstant LR3e-4で学習開始（session59615、新run）。固定12epoch・元の最早最小weak BCE選択・Gold非使用を維持し、途中checkpointの採用、fold1の効用、Public改善はまだ主張しません。モデルproducerとデータ事前検査producerの役割、元source／run／configを保持しています。

F006 fold1は開始前に固定12epochのBCE主選択と固定10 AUC副選択をRootが別規則で定めましたが、実学習はまだ0です。F007 fold1完了と人工GPU確認の後に進める計画で、所要約6時間は見積もり、fold1の実測ではありません。同epochなら元checkpointを再利用し、別モデルや重複提出を作りません。元bestを副規則で上書きせず、Gold／Publicで選択しません。

最新公式は23:06のP020・P021自己Public 0.954、P022はhiddenエラー、P026はraw状態／Public未返却。本日受理4件・残枠1、厳密な0.960超えは未達です。P028の具体的Notebook SaveとP024の新モデル送信は、それぞれ自動審査が開始前に拒否して具体的承認待ち・外部送信0を維持。ローカル学習の許可を外部送信の承認とは扱わず、影響しない学習と採点確認を継続します。以下は時点ごとの履歴です。


**10月12日：P028の実ソース946,931Bは上限950,000B内でローカル検証に通りましたが、外部Saveは承認待ち・実送信0です。** [具体的コード保存と審査停止の正本](../experiments/p028-concrete-Notebook-Save-auto-review-held-20261012.json)。Rootが新しい非公開Notebook rsraki/p022m-0e93fd1399c445dbb769ccb1e07bf6edのmetadata／ソースを生成し、defaultのオフラインDTO検証は終了コード0。モデル・数値処理を維持した元S6の検査別補完guard修正で、まだ新Kaggle実行・可視資格・Submitはありません。

Rootの明示Save操作はCLI開始前に自動承認審査が拒否しました。理由は「新しい具体的コードと宛先への明示承認不足」。審査文はDataset送信と表現していますが、実際の予定APIはprivate NotebookのSaveKernelであり、新Dataset作成ではありません。この区別と元拒否記録を保全し、認証／Input照会／Quota照会／Save・remote送信は全て0、attempt dirも未作成です。Rootの具体的承認依頼への回答を待ち、同じ操作・別ref・encoded送信で迂回しません。P024の新104,308,640Bモデル／別宛先への具体的承認待ちも独立して維持します。

最新公式は23:06のP020・P021自己Public 0.954、P022はhiddenエラー、P026はraw状態／Public未返却、本日4件・残枠1。F007 fold0はRootが12epoch完了と実CPU復元を受理し、弱比較は固定50%差+0.00158535ですがPublic改善を証明しません。fold1のcache抽出session71660は継続中で、完了・全891件照合・残差学習はまだ未受理／未開始です。影響しない採点確認とcache作業を続け、厳密な0.960超えは未達です。以下は時点ごとの履歴です。


**10月12日23:06 JST：P020とP021は自己Public 0.954、P022はhidden再実行エラー、本日の残り提出枠は1です。** [公式結果・F007完了・P028修正準備の正本](../experiments/official23p06-P021public954-P022error-F007complete-P028source-20261012.json)。Rootが委任したlocal_analysisの読み取り6RPCとRootの独立確認で、P021のmirror版57103408はGet／Listともraw・typed COMPLETE／0.954／エラーなし。作者の旧V1最高0.956とは別の再現値です。P02257102110はCOMPLETEという処理終了状態でもerrorDescriptionがあり、Publicはありません。P02657107853はraw状態・Public未返却で、SDK既定PENDINGを公式状態とは扱いません。自己最高0.954・本日受理4件・全履歴23件・残枠1、厳密な0.960超えは未達です。

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


**10月12日20:36 JST：自己最高Public 0.954を維持し、残り提出枠は2です。** [公式結果・E10・CPU監査の正本](../experiments/official20p36-f006E10-P025CPU-audit-20261012.json)。Rootが委任したexecution_auditの実読み取り照会で、P02057098752はGet／Listともraw・typed COMPLETE／0.954／エラーなし。P02257102110とP02157103408はスコア・raw状態未返却で、SDK既定PENDINGを公式状態とは扱いません。本日受理3件・全履歴22件の末尾を確認し、厳密な0.960超えは未達です。

f006はE10／12の学習中で、weak BCE 0.29639006、固定10 AUC 0.88149432、12項目AUC 0.88164164、LR倍率0.325です。旧同epochとの固定10 AUC差は−0.00387266で、改善・途中採用は主張しません。元BCE選択は暫定E8のまま保ち、固定12epoch完了後にBCEと別候補のAUC選択を比較します。稼働中のtrainer・CLI・configは維持します。

P025の公開2.5D readerは、RootのKaggle CPU監査で3重みのSHA、安全なweights_only読込、各219 state／30,741,112 FP32要素の有限値と、元公開6ソースのUTF8本文・SHA・構文を確認しました。CPU監査だけで、Competition入力・MRI・教師・Gold実データ・forward・GPU・Submitは0、ローカル重み取得も0です。Gold利用・checkpoint選択とreader実装のソース監査を続け、推論・精度・提出資格には格上げしません。P023／P024の採用・次のSubmitは未決定、F007も準備段階です。目標未達・作業を継続します。以下は時点ごとの履歴です。


**10月12日19:36 JST：公式採点でP020の自己Public 0.954を確認しました。** [正本と確認範囲](../experiments/p020-official-public954-three-candidate-comparison-20261012.json)。observer `ebf897ef…9932e` はGET3＋LIST1＋枠GET1の読み取り専用実結果。P02057098752はGet／Listのraw・typed COMPLETE・0.954・エラーなしが一致し、自己最高を0.954へ更新します。P022／P021はPublic・raw状態未返却、typed既定PENDINGを公式raw状態と混同しません。raw／typed残枠2・本日3・全22件末尾確認。P020の保存版束縛は既存のgenuine Save／Submit／結果chainで、Getがkernel全識別を再露出したとは主張しません。

f006は保存E8 snapshot `23bbb6b1…12486` のweak指標のみを記録し、旧同epochよりAUCが0.0003312高い一方BCEは0.0002848高いため改善・採用0。[secondary選択](../experiments/f006-fixed10-AUC-selection-policy-20261012.json)はE7既観測後の定義で、元minBCE選択・best.ptを維持、同epochなら同best SHAを使って重複Submit0。[新隔離loader](../artifacts/research/20261012-f006-fold0-isolated-mixed-runtime-source-v1/preparation.json)は人工metadata6件・source compileのみ合格、実CPU／GPU forward／Serving／精度資格0です。次はP022／P021公式比較→固定12epoch完了のweak比較→次候補。本更新の実run／model／checkpoint／教師本文／MRI／GPU／Kaggle API読込・稼働中3source／RNG変更0、過去raw bytes・ledger prefixを保持します。目標未達・作業を継続します。

**10月12日18:43 JST：P021のHTTP400と入力mirror修復準備を記録しました。** [原因と証拠の区別](../experiments/p021-http400-mirror-preparation-pair-review-20261012.json)。18:16の全21件terminal履歴SHA `58894424…24177` は同説明のP021提出なし・本日受理2件／残枠3。Rootの明示判断による18:25:33の追加1回はHTTP400、held SHA `704d11fa…3b2dc` に非版管理Dataset使用の具体的拒否を保存しました。旧17:24の原因は未確定で、汎用helperの `remote_acceptance_unknown:true` も上書きしません。redactされたDataset名から特定Inputを確定せず、認証・日枠・複数pendingを原因と呼びません。計POST2・自動retry0、同一6モデルSHAの版管理mirrorを用いるSource V3は準備完了・Root査読待ち・未Saveです。

P020／P022の18:07／18:19照会はraw status・Public未返却／errorなし、SDK既定PENDINGと区別し、自己最高0.951・日枠3を保持します。[ペア効用集計](../artifacts/research/20261012-f005-fixed-mean50-pair-utility-v1/checks-and-compact-summary.json)は原選択予測／weak manifest／UID順序／欠損・soft・binary maskと支持数を束縛し、手計算3fixture後に実2foldを集計。固定10の修正／悪化は8,681／8,553と13,166／9,230、全24所見で元AUC差と一致・最大残差8.33e−17です。missing／softを0へ置換せず、係数探索・checkpoint／Gold／Report／モデル／MRI読込・GPU／API0。Public改善・公開主力OOF・患者独立性は未証明です。

[追加公開調査](../artifacts/research/20261012-post-P022-public-stronger-primary-web-v1/analysis.md)の確認候補数0は限定的な観測結果で、取得不能を不存在と判定しません。f006 E6 snapshot `a77abda2…decc` は初6epochのLR倍率1、弱BCE 0.28527664・fixed10 AUC 0.87154078、1,272更新／skip0。微小差をLR改善とせず、固定12完了後に同じ選択規則で比較します。本記録更新はagentのKaggle通信・実モデル／MRI／教師本文読込・GPU0、稼働中trainer／CLI／config変更0。目標未達・作業を継続します。
**10月12日18:06 JST：P022の可視結果・最終CSVと正式Submit1回を記録しました。** [正本と確認範囲](../experiments/p022-visible-accepted-single-Submit-score-pending-20261012.json)。Source V2／Root V2／kernel138296990／V1の保存Source・COMPLETEを束縛し、getter V3 `45cad44b…3d4ea` は指定20出力だけを取得。実receipt `be4aa58f…fcf32` に対しRootが再通信なしで再計算し、Root受理 `98f1390f…ec4b` を発行しました。全3UID・S3／F0、原4モデル平均・全N固定20%rank混合・mask・復元順序と最終CSV `7115f865…a7a8` を確認。今回F0のため失敗UIDの最終CSV復元枝は実可視推論では未行使で、既存CPU契約・P019実故障診断と区別します。hidden完走・精度の資格は未発行です。

同保存V1から18:01:11にCreateCodeSubmission1回が57102110として受理、受付SHA `9fdccbd4…0f4bc`・再送0。直前raw GPU残量99,813.289秒・日枠4・自己最高0.951を確認し、18:02 observer `bf3db3b8…d11c9` でraw／typed残枠3を確認しました。P021の取得20件に対応refなし・追加ページ未読で受付不明を維持し、枠減少をP021受理の証拠としません。P020の17:50専用照会はraw状態／Public未返却。f006 E6比較を保持し、初6epochの差や可視推論成功を精度改善とは扱いません。本更新のagent通信・GPU・MRI・教師／モデル本文読込・稼働中3ソース編集0。厳密な0.960超えは未達・作業を継続します。

**10月12日17:41 JST：P021の可視CSV受理と受付不明のSubmit、P022の新Save受付を記録しました。** [根拠の正本](../experiments/p021-visible-qualified-submit-unknown-p022-Save-accepted-20261012.json)。P021 Root受理SHA `4eeafc57…3ec5f` は保存Source／COMPLETE／モデル・補助入力・geometry SHA、全3UID×12列、2T4・各4モデル・fallback0、原latest V2の50:50・OA3項目・各arm自身の成功集合での平均tie百分位rankを確認。CSV `%.9g` の36値はFP32へ戻すと完全一致しました。17:24のCreateCodeSubmission1回はHTTPErrorで受理ref／HTTP status／error bodyが保存されず、held SHA `9b14ac52…041f` のまま再送0。17:36 observer SHA `7cc6c894…27f92` は1回の一覧20件に対応候補なし・追加ページ未読、raw／typed残枠4を確認し、全履歴の不存在や受理拒否を主張しません。公式SDK契約6項目の一致とpending制約調査を別記録に残し、例外型のみから原因を決めません。

P022 Source V2はbuilderだけをjoint XZ／LZMA2へ変更、他8ソース・原5 literal・元ZIP／全raw metadataのSHAと長さを保持し、切断・余剰・展開長・SHA不一致をCPUで拒否しました。Root source peer SHA `97d8c344…604d5` はソース資格に限り、新Root V2 authority SHA `226d3804…2500c` のgenuine metadataで最終946,315B・固定上限950,000B・default gateを確認。Save受付SHA `60a1e750…5f2ce` は17:37、kernel138296990／V1、Input metadata8 RPC・quota1 RPC・Save1・再送0。旧Root V1の1,272,956B HOLD／Save0は保持します。P019 actual UID復旧とP018 component証拠を束縛しましたが、P022 whole Serving／final CSV／Publicの資格はまだfalseです。

f006 E5比較snapshot SHA `2d509d78…f627` はmanifest／seed／fold／group／cache／configの比較に合格し、累計1060更新／skip0・LR倍率1・教師／予測／モデル本文読込0。初6epochの差をLR改善やPublic改善と扱いません。P020専用observation-v5（17:11）のraw採点状態／Public未返却・自己最高0.951を保持します。本更新はRootの既発行receiptを読むだけで、agentの認証API／GPU／MRI／教師／モデル本文読込・稼働中3ソース編集0。厳密な0.960超えは未達・作業を継続します。

**10月12日17:11 JST：P019実fault64／normal64のRoot受理を記録しました。** [証拠と未確認範囲](../experiments/p019-qualified-p021-complete-p022-source-size-hold-20261012.json)。Source9独立査読SHA `fe349696…7ac82`、Root実受理SHA `03ddfc5b…c7a2b`、同kernel138289877／V1・非公開Original／T4／OFF／Input5を照合。指定14 JSONだけを取得し、故障64件はS63＋F1・252forward・155.15秒、通常64件はS64＋F0・256forward・133.79秒でした。原REG0コードとcommon readerのSHA／関数／literal raise行をKaggle出所へ対応させ、NoDICOMの構造失敗だけを認めました。失敗UIDのGPU forward0、全N補完・原12文字列一致、成功UIDの元4forward・入力SHA・FP32 fold／family平均、2T4・queue／owned child／一時資源の終了をRootが独立検算し、認定時API／GPU／MRI／モデル本文読込0です。最終rank／mask／CSVはCPUコンポーネント証拠で、この実診断のServing最終CSV・精度資格はfalseです。

P021はSource査読と手動Save受付SHA `5a674ae6…e2ac9` を保持し、kernel138292308／V1の公式CLI COMPLETEを17:11に観測しました。結果本文・提出CSV・自己Publicの受理はこれからです。P022は実Root metadataを束縛するとソースが内部950,000B上限を超え、local build HOLD・Save0。metadata原bytesを保持する圧縮Source V2を準備し、新ソースの照合後に実Servingを検証します。P020 observation-v5（17:11）はraw status／Public未返却、typed PENDINGはSDK既定、raw／typed日枠4・自己最高0.951。f006はrun metadataのみを読みE4完了・running・848更新／skip0・LR倍率1を確認し、epoch7前の数値差をLR改善と呼びません。本記録更新の通信・モデル本文／教師／MRI読込・GPU実行0、稼働中trainer／CLI／configの変更0。厳密な0.960超えは未達・作業を継続します。

**10月12日16:20 JST：P018実native256の保存版・実結果を取得し、Rootが独立受理しました。** [証拠と未確認範囲](../experiments/p018-native256-qualified-p022-source-preparation-20261012.json)。kernel138273767／V1のraw／typed COMPLETE・保存Sourceを確認後、8個の小容量診断本文だけを取得。実MRI256件・4モデル1,024forward、入力5SHA×256・原phase契約、FP32 fold／family平均の再計算、2T4・fallback0・queue／worker／一時資源の後処理を確認しました。Root受理SHA `8ea46921…f8f00`、raw receipt SHA `6983914f…92b9f`。実推論633.24秒、GPU標本最大891／889MiB、RSS標本合計4.43GBは共有ページ重複を含み、待ち時間や厳密なメモリピークと同一視しません。Serving・Public精度の資格は未発行。P020の16:19照会はGet／一覧で存在を再確認、raw状態／Public未返却・日枠4。P022は独立ソース査読に合格しましたが、P019 fault64／normal64とP022実Servingはこれからです。自己最高0.951、厳密0.960超え未達を維持します。

**10月12日16:15 JST：P022の基準S6への差替えをソースのみで確認しました。** [差分と未発行資格](../artifacts/research/20261012-p022-S6-native20-source-v1/report.json)。P019 V2正本SHA `04b049bd…ccf89` と作者S6 V3 raw SHA `1dc9a5df…d9d33`、P020実可視受理 `45730b06…dcaec` を固定。原5セルのP012との一致は `[true,true,false,false,true]` で、Raptorと最終70:30が同じです。Native数学・reader・rank20・transport・UID recovery・router・diagnosticは元bytesを保持し、maskは親SHA定数だけ、mixed_parentはcleanup source variantの1フィールドだけを変更しました。cleanupの新コピーはS6の親／原5セルとvariantを厳密照合し、strict memory・process ownership・閾値・clear1の操作を保持。compact公開V13と他7役同一を確認し、source588,500B／69member・全Pythonと生成セルのcompileが成功しました。実推論・モデル本文取得・API0、追加GPU数学テスト0です。P020可視3件をP022やhiddenの実証拠に転用せず、runtime authority・P018256／P01964／P022 Servingの必要受理はnull、独立査読待ち、自己Publicと精度改善は未評価です。

**10月12日15:59 JST：P020の提出受理と読み取り専用の公式照合を記録しました。** [検証正本](../experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。保存Source／Input13／1／1／COMPLETE、8body実SHA、全3UID・12列、worker2T4・4モデルずつ・FP16・失敗0と70:30最終FP32の36値一致をRootが受理し、同kernel138279894／V1から提出57098752を1回受理。raw Decimal GPU・日枠5・公式自己最高0.951を直前に確認しました。修正したwire terminal／worker辞書型の元V1・V2・held・取得producer SHAを残し、V3純検算で再API／再ダウンロード0、実通信actorはRootです。15:59のGetSubmissionと一覧はref一致・errorなし、raw状態／Publicは未返却。typed既定PENDINGは公式状態と呼ばず、fresh日枠raw／typed4のみを確定します。hidden完走・自己0.954・精度改善・厳密0.960超えは未証明で、最高0.951とgoal activeを維持し、既存UTF8履歴を保持しています。

**10月12日15:53 JST：P020の可視推論・独立FP32検算を受理し、公式Submit1回を確認しました。** [結果と境界](../experiments/p020-public-S6-visible-acceptance-single-Submit-20261012.json)。raw／typed保存Source・kernel138279894／V1・Input13／1／1・COMPLETE一致、許可した8ファイルだけの実GET、全3UIDと12列・有限確率、入力モデル／補助／geometry SHA、2T4・各4モデル・FP16・controller／worker失敗0を確認。終了コードの原writerは辞書型で、取得側の配列仮定をV3で修正しました。元V2reader `ae308…` とheld `b7b8…`・8bodyの実SHAを変更せず、検算source `69b547…` で全36値の70:30平均tie百分位rank・FP32を再計算し、Root受理SHA `45730b06…dcaec` を発行。CSV小数表現との差最大約2.38e-8はFP32へ戻すと完全一致しました。再API・再ダウンロード0です。提出直前に新しいSource／COMPLETE・raw Decimal GPU・公式自己最高0.951・日枠5を確認し、同kernelV1／submission.csvの公式提出57098752を1回だけ受理、再送0。hidden完走・自己0.954・精度改善は未確認で、最高0.951とgoal activeを維持します。Source／検算担当と実通信／Root受理担当を記録し、既存のUTF8履歴を保持しています。

**10月12日15:38 JST：配布が復元された公開S6をP020として非公開保存し、実結果を待っています。** [実証拠と検証境界](../experiments/p020-public954-restored-S6-private-Save-current-research-20261012.json)。Kaggle公式の保存source／metadataとpublic viewを照合し、restored V3／script357152922の作者Public 0.954を確認。現V13・旧V10・R6V2のmanifestはSHA `1c1211e6…9517e6` まで一致。P020原5コードセルのbytes／順序とcredits、embedded7 source、manifestの学習source4、guardモデル6・SWA・anchor・geometry8をRootが独立10項目で確認しました。追加2セルと原5セルの構文、人工CSV全2件受入れ／部分1件拒否は成功。API source306,470B／SHA `4a70f118…32503`、単発Save受理receipt SHA `f42e088c…85552`。Input3／quota1／Save1、競技Submit0、actual whole-run・hidden・自己Publicの資格は未発行です。

0.955 Notebookは私有student Inputを含み、0.956 V1はbundle `fe055406fad6a12f` のInput一覧404。元sourceのモデルcopy先は一時領域でworkingへのweight exportは検出せず、Outputのモデル本文取得0。最新版V2はS6に差替え、Net・reader・失敗行rank基準・混合比も変わるため作者0.956を引き継ぎません。S6は作者manifestのGold訓練0・official trainer labels false、固定16epochのdeploy後にGold予測を出すsourceを確認しましたが、Raptor58選択と公開prior／anchorのGold履歴を除いたclean OOFとは呼びません。自分のGold fit／選択／評価0、Report特徴0、モデル再配布0。P018256はRootのRUNNING報告のみで成功未確認。f006 E1はLRfactor1・head3e-4／last2 1e-5・steps212／skip0、weak BCEとAUC12は旧同epochとの微差であり採用理由にはしません。既14:54記録と学習source `2ffa41e2…44ef` は保持しました。

**10月12日14:54 JST：REG2のremoteバイトとP018のnative64を検証し、native256は実結果待ちです。** [実確認の記録](../experiments/p018-native64-f006-fold0-progress-20261012.json)。REG2件それぞれ193memberが一致。Root-native64-acceptanceのSHA `95f38c32b670cea3df009128b88894a1427c26dd367a58a2c7dac8bb0efae5cd` を固定し、kernel138271136／V1のMRI64件・256forward・154.986秒・fallback0・2T4を確認しました。native256 kernel138273767／V1は保存受理済みですが実結果未確認です。256・Serving・Submit・Publicの資格を64件の成功から転用していません。

f006の最終configで全144窓・pixel16とprocessed115のGPU QAは6項目合格・失敗0、optimizer0です。元のpixel16・DINO GPU19項目の証拠とSHAを保持し、gate `057f7095e4c9af26db97abbc5f2edd5ef1284a9495ba8dd6da95ef1685cbd369` を固定しました。fold0の実学習はsource `2ffa41e2fc650b61c5ccf2d5cf7c49974aff001025e604764238f6711efd44ef`、weak訓練3,386件・検証821件、固定12epochでrunningです。変更要因は終盤のLR減衰のみで、BCE・欠損mask・入力・教師・seedを保持します。Goldの学習・選択・評価0、途中checkpoint採用0です。P018のPublic測定後に学習する順序を変更して並行実行していますが、Public改善は未証明です。fold0の完了後に比較してfold1を判断し、weakの結果をPublic改善とは扱いません。自己最高0.951、goal active、厳密な0.960超え未達を維持します。

**10月12日13:44 JST：再開後の公式結果とREG2入力metadataを確認しました。** [正本と検証境界](../experiments/goal-resume-analysis-20261012.json)。Kaggle SDK専用環境 `artifacts/tools/kaggle-venv` と通常認証を用い、P016／P017のraw・typed明示COMPLETE、有限Public 0.951、errorなしを同一提出番号で確認。公式枠5件、GPU残量101,707.236秒。最新一覧ではP014／P015のCOMPLETE欄にerrorDescriptionがありスコアなしのため、隠し採点成功と区別しました。Public LBは明示public指定・1ページ50件のみ、最高0.966／0.960超27件で追加ページ0です。

具体的な2宛先・4ファイルへのユーザー承認後、REGのPUT4／Create2を実行しました。旧getterが要求するCreate応答refが未返却で元heldを保持。成功Create receiptを作らず、元intent・成功PUT・heldを検証する限定復旧getterでGetDataset／GetDatasetStatus／ListDatasetFiles(V1)各1回、計6 RPCから両private／V1／READY／正2名前・サイズを確認しました。再送・更新・公開化0、認証情報の表示・コピー0。remote BIN実bytes／193memberの一致、Kaggle実MRI64／256、Servingと新Publicはまだ資格外です。ソース点検ではvisible件数・UID・test SHA固定や256予算のServing転用は検出されませんでした。一方、構造読込1件の失敗で全件を基準へ戻す経路があるため、P017同点から追加モデルの寄与は判定できません。保存済み弱予測の補完性とepoch比較を、Public・Gold・独立P012 OOFと混同しません。自己最高0.951／厳密な0.960超え未達を保持して継続します。以下は履歴です。

**10月11日01:08 JST：3ターン継続した承認阻害を監査し、実returned status=blockedを記録しました。** [監査記録](../experiments/p018-specific-upload-three-turn-blocked-audit-20261011.json)。2つの直前記録と現在のattemptなしを照合し、具体的新REG承認待ちが3連続ターンで継続、Source準備完了後に独立監査もimpasseを確認。旧CLS fold1承認を新REGへ転用せず、Start／PUT／Create0を維持します。P016v5／P017v9は各Get1で401、同P016 handleを新SDK clientでv6 Get1再確認しても401。観測HTTP401は採点ERROR／terminalの証拠ではなく、認証原因・失効は未確定です。この新認証阻害を3ターンと数えていません。既存ジョブの再起動／再提出・認証値の直接読取／表示／コピー0。自己最高0.951でcompleteは未成立、目的は公式Public厳密0.960超えのままです。以下は履歴です。

**10月11日01:00 JST：追加学習の判断と、別経路での公式採点照合を記録しました。** [根拠](../experiments/p018-next-training-decision-official-list-crosscheck-20261011.json)。固定12epochの4run原SHAを再照合し、E12は採用epochよりweak検証BCEが大きくtrain BCEが小さいことを確認。これはepoch延長／容量拡大の改善を保証せず、現fold2／last4もCLI・schemaの資格外です。GetSubmissionのPrimary SDKでrequest ref1項目・正規endpoint・PENDING enum0既定値を確認、指定漏れはありません。ListSubmissions firstpage20を1RPC／HTTP1・64KiB上限・追加ページ／再送なしで読み、57044321／57042712だけ保存。2件は存在しますがraw状態／PublicなしでUNKNOWN、実採点中や隠し完走は未証明です。新REGの承認回答は未取得、attemptなし・Start／PUT／Create0を維持。新モデル・教師・Report・Gold・GPU操作0、自己最高0.951を保持します。以下は履歴です。

**10月11日00:48 JST：実APIのファイル一覧省略を確認し、Source準備と実行結果を区別して記録しました。** [検証記録](../experiments/p018-metadata-api-contract-fixed-source-ready-20261011.json)。既存owned CLS0 GetDataset1／HTTP1・200はraw filesなし、SDK既定files=[]であり、リモートにファイルがない証拠ではありません。新getter V2はGetDataset／status／V1 ListFilesの上限3通信・1MiB制限・追加ページ／再送なし、独立12項目合格。親20・保存入口12・native Wrapperの独立ソース確認も完了しましたが、新private REG ID／実バイト／64／256件／Serving資格は未発行です。541,915Bの親ソースと442,280Bの圧縮native packetは未注入の見積もりで、実メタデータ注入後に950,000B制限を再測定します。新REG送信は自動承認拒否でプロセス前に停止、別質問回答待ち・Start／PUT／Create0。P016v4／P017v8は各公式Get1、raw状態／PublicなしのUNKNOWN。教師・Report・Gold・モデル計算変更0、自己最高0.951を保持します。以下は履歴です。

**10月11日00:12 JST：承認待ちの外部送信と、完了した動作確認を分けて記録しました。** [根拠記録](../experiments/p018-private-upload-approval-pending-current-public-research-20261011.json)。今回の自動承認拒否はツール実行前で、新REG fold0／fold1のattemptフォルダなし・Start／PUT／Create0。特定2宛先・2組のBIN／manifestの質問は回答未取得です。新upload V2の限定独立13項目と通信なしSDK default確認を終えていますが、private／V1／ID／remote bytesは未観測です。バイト検証Wrapper12項目・新phase契約11項目のソース確認をKaggle64／256／Serving資格に転用しません。公開Sourceのbounded公式2読取は現V57と348575417を結び、作者0.936を確認、重み取得・実行0。拡張OrthoFoundation-MSK-5Mの事前学習規模は作者主張で、別repo MITを重みに転用せず、利用条件・RSNA精度は未確認。P016v3／P017v7は各公式Get1、raw状態／PublicなしでUNKNOWN、SDK既定PENDINGは公式状態としません。自己最高0.951です。以下は履歴です。

**10月10日23:51 JST：4モデルの実人工GPU確認を、再実行せず独立確認しました。** [根拠記録](../experiments/p018-mixed-realGPU4-synthetic-QC-20261010.json)。本物の新CPU4 receiptとRoot許可、QC Source2件・mixed Source6件・差分peerを束縛してdefault確認後、RTX4090で実行48529dがexit0。full144各9chunk／masked96各6chunk・各モデル2正常forward、4全欠損拒否、prefix／suffix／head不変、全6組alias0を実観測しました。保存された各familyのFP32 fold平均48値は独立再計算で差0。最終family合成ベクトルは別保存されていないため、凍結QCソースの実比較assertionが通った範囲として記録し、独立CSV比較と混同しません。Torch最大allocated653,356,544 B・reserved767,557,632 B、RSSは未観測です。人工画像の入力ハッシュ保持はnative cacheとの一致の証明ではなく、2T4の8モデル常駐・64／256件・Serving・精度の資格へ転用しません。MRI・教師・Report・Gold・API実行0。両提出の公式スコアは23:44時点でも未返却、自己最高0.951です。以下は履歴です。

**10月10日23:42 JST：新P018 Sourceによる実4モデルCPU復元を確認しました。** [根拠記録](../experiments/p018-mixed-realCPU4-bytecode-fixed-upload403-20261010.json)。起動時のbytecode禁止と子の`-B`だけを変更し、独立した差分17項目と旧数式・モデルクラス・4ソースの一致を確認しました。新Root許可を用いた実行0542b8はgeneric factory1・checkpoint load4・strict apply4・全6組のalias0、元prefix／suffix／head／選択epochの束縛を通過しました。Rootは実receiptの同じphase契約を再計算し、実行後も新REG2ツリーが各193ファイルで余分なpyc0であることを確認しました。旧failed Source・peer・キャッシュを保持し、資格を転用していません。CPU復元はforwardやKaggle実行・精度の証明ではありません。アップロードの旧GetDataset例外は詳細を失っていましたが、新しいbounded probeでHTTP403・JSON codeなし・Get1/HTTP1/Start・PUT・Create0を実観測しました。403をabsenceへ読み替えず、JSON200/code404仮説も未成立です。人工GPU／remote transport／64・256／親統合は未資格、自己最高0.951です。以下は履歴です。

**10月10日23:09 JST：P016修正版の単発提出を公式APIで受理確認しました。** [根拠記録](../experiments/p016-overlay-Submit57044321-20261010.json)。凍結Source→Rootの全項目承認→最終準備の順でSHAを束縛し、本物のdefault確認は認証・API0で成功、独立14項目も合格しました。過去のP017故障資格は要求していません。直前のSource8項目・明示COMPLETE・GPU残量・raw/typedで整数一致した枠2回を照合し、保存版V1のコード提出1回・自動再試行0で57044321を受理しました。新しいCSVアップロードや既存Dataset更新は0です。visible実確認・API受理をhidden完走や精度資格へ転用せず、公式Publicは未取得のままです。以下は各時点の履歴です。

**10月10日23:01 JST：P016の新しい実完走とRegional両foldの完了を確認しました。** [根拠記録](../experiments/p016-overlay-visible-complete-regional-both-folds-20261010.json)。修正版kernel138035248/V1は明示COMPLETE、18件の取得bytes・保存SHA・Source8項目が一致しました。基盤70:30 FP32・追加90:10の1回最終rankは全36値差0、17g/LFの最終CSV830 Bも原bytes一致です。Rootは同じ出力契約17項目を再計算し、独立10項目を束縛してvisible3検査の工学的確認を発行しました。モデルforwardはFP16・raw出力FP32、子24.87秒・watchdog31.52秒、両GPUの解放後allocated/reserved 0を観測しました。親全体のelapsed数値・モデル生存数・一般的hidden完走は未証明です。両Regionalモデルは実CPUで選択重みを厳密再構築し、元checkpoint bytesを保存した193memberの推論用BINを作成しました。長いWindows保存先でコピーが失敗した旧partialを保持し、短い新保存先で同一ソースのまま成功しました。固定50:50平均の弱検証改善は両foldで確認できましたが、同じweakでの探索的比較でありPublic改善ではありません。別foldモデルが当該holdoutを学習しているため、2fold推論平均の独立精度検証には転用しません。4モデル自体のCPU/64/256・統合推論資格は未発行です。P017公式照会は1回でスコアとraw状態未返却、SDK既定PENDINGを公式状態と扱いません。以下は各時点の履歴です。

**10月10日22:40 JST：環境情報の接続修正を新しいP016保存実行へ渡しました。** [根拠記録](../experiments/p016-overlay-Save138035248-regional-mean50-20261010.json)。Serving専用親の差分はクラス名・platform変換呼出し・厳格な変換関数だけで、逆変換すると旧親のraw SHAが一致します。旧21ソース・3段階診断・cleanup検査・数式を保持し、新Root許可と親SHAを配置前に要求します。元18セルと5つの旧metadata本文は一致し、ソース522,033 B、新保存入口の26関数は旧ASTと一致しました。入力4件とGPU残量を直前確認して新kernel138035248/V1のSave1回を受理、実完走・最終CSV・hidden精度は未確認です。fold1補完性は本物の選択済み予測2件と固定weak教師SHA、891 UIDの原順、欠損マスク・支持数を照合し、単独2種と固定50:50 FP32平均のAUC・観測確率BCEを再計算しました。確率BCEとproducer logit BCEを区別し、Gold・Report本文・モデル本体・GPU読取0、係数探索0です。平均の固定10 AUC改善は旧比+0.006283547ですが、同じ弱検証データ上の探索的比較でありPublic改善は未証明です。公開Source2件の取得はコードと版・入力metadataのみで、既モデルと同じ根拠が確認されました。Sourceの作者目標値やソート順位を現版の公式Publicとして扱っていません。P017の22:35応答もraw状態・スコア未返却です。以下は各時点の履歴です。

**10月10日22:16 JST：P016の実ERRORを当該保存版のログから切り分けました。** [根拠記録](../experiments/p016-serving-platform-error-regional-fold1-complete-20261010.json)。同一kernel138030227/V1のSource8・ERROR拘束を再利用し、出力メタデータ1回、許可済みworkerログ2件とAPI内実行ログを回収しました。In[17]の`seal_and_cleanup`で、authorityの`container_digest/docker_image_version_id/enableInternet/T4x2`表記を旧cleanupの`docker_image/docker_version_id/internet/GpuT4x2`表記へ変換せず渡したことが、厳密なplatform比較の拒否原因です。基盤推論の完了はログ上確認でき、追加モデルは開始前でした。元の検査を緩めず親側の変換を準備し、旧凍結21ソース・3段階診断を保持します。修正版Servingの実完走資格はまだありません。以前のhiddenエラー・時間超過・OOMの原因には転用しません。P017の22:15公式照会はスコアとraw状態を未返却、SDK既定PENDINGから採点状態を推定しません。Regional fold1は固定12epochと最小weak BCEの選択epoch10、checkpoint・予測SHA、更新機会とAMPスキップ2回の会計を完了検査で確認しました。モデルロード・新Input・Kaggle資格・Public改善の確認は未実施、fold0は学習中です。以下は各時点の履歴です。

**10月10日21:55 JST：P017正常版の単発競技提出とP016非公開保存実行を受理確認しました。** [正本記録](../experiments/p017-normal-Submit57042712-p016-Save138030227-20261010.json)。P017は正常visible3件と専用実故障復旧を根拠に、frozen Source→Rootの完全な承認JSON→最終準備の順でSHAを拘束しました。フォルダ指定不適合のV2と、Rootが差分説明を全項目と誤って扱ったV3のHOLDは、認証・意図ファイル・API呼び出し0のまま保持しています。V4の全ローカルgateは実際に1回通過し、独立14/14の後にfresh Source8・status COMPLETE・quota・allowance3を各1 RPCで確認、CreateCodeSubmission1回で57042712を受理しました。故障Notebook・復元診断CSVは未提出、hidden/精度の確認は未了です。P016はgenuine3段階確認から生成した別ソースを、metadata/DTO確認と独立6/6後に、fresh Input4・quota1・Save1で非公開kernel138030227V1へ保存実行しました。まだServing出力を取得しておらず、提出や精度の資格へ転用しません。新規モデル・MRI・Report・認証情報アップロード0、自己最高0.951、0.960超え未達を維持します。以下は履歴です。

**10月10日21:42 JST：実故障復旧・CPU/64/256確認と、提出用ソース生成まで進みました。** [正本記録](../experiments/p017-fault-root-accepted-p016-phase3-materialized-20261010.json)。P017故障診断Version 1はGPU4モデルのstrict読み込み後に壊れたDICOMで失敗し、同じ子プロセスのjoinと所有PGの不在、封印済み元CSV633 Bのbyte一致を確認しました。元取得はSource1・status1・metadata4・15 GET/41,114 Bです。CPU componentにないbundle SHAを参照していた検査を外側の正しい項目へ1か所修正し、旧held23/24を保持したままoffline再解析24/24と独立12/12を記録しました。再取得・モデル再実行0、失敗Nativeの一時ファイル削除False、モデル生存状態Noneを保持します。故障診断と復元CSVの競技提出は禁止です。P016はCPU→64→256の同じcheckpoint/source/receiptチェーン、Source8、10出力5,461,003 B、strict486・2 T4・full94・直接224・fallback0・入力前後SHA・finite256×12・clear1・所有PG終了を確認しました。追加133.86秒、Torch最大予約8600/7448 MiB、RSS標本6.42 GBは共有ページ重複を含みます。Rootの3段階8項目確認と提出用ソース生成は済み、提出推論そのものは未確認です。Regional F0E10/F1E11の固定10項目AUC差は旧CLS比−0.003186/−0.005063、F1E11初AMP skip1です。弱BCEによる採用は固定12epoch終了後に行います。精度評価・hidden成功・患者独立性を未確認のまま認定しません。自己最高0.951、0.960超え未達を維持します。以下は履歴です。

**10月10日20:52 JST：P017の正常枝は実可視3検査限定で確認済み、故障復旧は専用実行の結果待ちです。** [記録](../experiments/p017-healthy-visible-accepted-fault-Save-native256-running-20261010.json)。公式COMPLETE・保存Source8項目・23出力63,909 B・独立した実結果レビュー11項目を照合。FP32の基盤70:30、追加2fold平均、固定80:20と平均同値順位、MCL・Synovitisの基盤値6フィールド、他30フィールド、CSV834 Bを再計算し差0・バイト一致を確認しました。今回の分岐はhealthyで、復旧処理の実証やhidden成功ではありません。kernel138022868・Version 1の非公開故障診断は、新規Save1回・既存Input6件・直前GPU残量104,700.365秒を確認して受理されました。壊れた入力は基盤処理後に作り、追加GPU4モデルの読み込み後のtask失敗を検査します。復元後のCSVを診断名へ変え、診断版Submitを禁止します。実プロセス終了・復元バイト一致の確認後に正常版のみ提出を検討します。Native256の新規Saveも受理済みで、取得器のSource確認10項目に合格しましたが、256件の実結果は未取得です。モデル・MRI・Report・認証情報の追加アップロード0、競技Submit0、自己最良0.951、厳密な0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日20:24 JST：P016の64検査は実行・結果確認を通過しました。** [実64検査と256検査の準備の記録](../experiments/p016-actual-native64-qualified-native256-materialized-20261010.json)。保存kernel138015216・Version 1のSource全8項目と公式COMPLETE、必須10出力の取得ハッシュ、独立した実結果確認8項目を照合済みです。直接224画素・full94の追加推論は64件を約43.21秒で完了し、fallback0、12項目の有限予測、2枚のTesla T4、基準CSV維持、基準処理の清掃1回と子プロセス終了を確認しました。追加推論のTorch最大予約メモリは各8,600 MiB、プロセスRSSの標本合計は約12.47 GBで、共有ページの重複を含み厳密なピークではありません。取得側の旧100件ページ制限は187出力に対応できず停止していましたが、同じ実行結果を回収でき、GPU再実行はしていません。新256検査は本物の64証拠と別途固定した元基準64証拠を入力したsource864,077 Bで準備済み、実Save・実256の確認はこれからです。P017はkernel138018511・新規非公開Version 1のSaveを受理済みで、正常時と復旧時の分岐別の結果確認を準備中です。並行学習はfold0 E8・fold1 E9、skip0、同epochの旧CLSより固定10項目AUCは−0.003813・−0.003962、weak BCEは+0.006544・+0.003491で、途中採用せず固定12epochを継続します。動作確認は精度改善の証拠ではなく、自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:55 JST：起動設定を修正したP016の64検査を、新しい非公開Notebookで保存実行しました。** [実Saveと並行作業の記録](../experiments/p016-actual-native64-V2-Save-p017-source-20261010.json)。kernel138015216・保存Version 1・修正版source695,688 B。実Input4件とGPU残量106,748.302秒を直前に確認し、Saveは1回、競技Submitは0回です。モデル・原5セル・CPU復元・予測式は維持しています。実行結果を取得する新しい取得器も12項目の確認を通過し、実64/256・Serving・精度の確認は完了待ちです。並行学習はfold0 E7・fold1 E8でskip0。旧モデルの同epochに対し固定10項目AUCは−0.007869・−0.009082、weak BCEは+0.004657・+0.000997で、途中採用はせず固定12epochを継続します。追加モデル失敗時の基準CSV復元を扱うP017は、本物の認可・査読情報を入れたsource949,769 Bで容量確認済みですが、実Serving・Save・Submitはまだです。自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:32 JST：診断ログの取得まで完了し、別の取得側の問題も切り分けました。** [実診断の記録](../experiments/p016-native64-actual-diagnostic-Input-order-20261010.json)。Kaggleの保存ソースは全15セル・実行環境・4つのInput参照が一致していましたが、APIがInputの表示順を入れ替えるため、取得側の順序一致チェックだけが失敗していました。元のfalse判定は保存し、新しい取得器では重複・追加・Version 2を拒否する4参照の集合照合に変更します。実ジョブの失敗原因は別に確認したPython起動時のsitecustomize衝突です。モデル・原5セル・CPU復元を維持した最小修正版の独立査読後、新しい64検査を実行します。新たな64/256・Serving・精度の確認はまだ済んでいません。自己最良0.951、0.960超えは未達です。以下は各時点の履歴です。

**10月10日19:23 JST：P016の64検査は追加処理のPython起動段階で失敗し、原因を実ログで特定しました。** [失敗記録と最小修正](../experiments/p016-native64-startup-exception-baseline-guard-20261010.json)。元0.951の基準処理は64件・fallback0で完了し、清掃1回まで成功（約765秒）。追加childの検索パスに元基準専用sitecustomizeを混入したため、基準用routing環境がない新childが0.114秒で終了しました。追加モデル・画像読込は未実行、64/256・Serving資格は未取得。v1ソースと証拠を保持し、モデル・原5セル・CPU復元を変えず、参照先を除去するwrapper v2と実Python起動の再発テストを準備します。取得器もexact_source検査で別HOLDになり、原因を分けて調査中です。P014/P015のhidden原因は依然不明で、そのServingに同じ混入パスはありません。P017の明示的な基準予測への切替と固定12epoch学習を並行し、自己最良0.951・0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日19:07 JST：学習はfold0 E6／fold1 E7、P016の64検査は実行中です。** [学習比較とP017の事前記録](../experiments/f005-regional-E6E7-p017-runtime-policy-20261010.json)。両foldともskip0で固定12epochを継続。旧CLSの同epochよりweak BCEは+0.003856／+0.012150、固定支持数10項目の平均AUCは−0.004308／−0.010178です。fold0のAUC12は高いものの少数ラベルの影響があるため改善とは認定しません。P017は、元0.951の予測完成・封印・清掃と追加処理の安全な終了を条件に、追加Nativeの実行失敗時だけ元CSVの原bytesへ戻す処理を別の一要因として事前登録しました。モデル・係数・正常時の計算は変更せず、元予測の失敗・入力やモデルの整合性違反は隠しません。P017はソース準備中で実資格・Save・Submit未取得。現在の自己最良0.951、厳密な0.960超えは未達です。以下は各時点の履歴です。

**10月10日18:56 JST：P016の新64検査診断をKaggleの非公開Version 1で保存実行しました。** [実Save記録](../experiments/p016-actual-native64-private-Save-20261010.json)。kernel138009389・source695,007 B・Original31481・T4x2・Internet OFF・指定4 Input V1。Input確認4回とGPU残量確認1回の後、Save1回を受理（残量107,531.917秒）。元0.951の5セルと新CPU証拠は固定し、ソース確認12項目と単発Save確認14項目を通過。実64の完了・予測・メモリ・保存コードの照合はこれからです。256・Serving・競技提出資格は未取得、競技Submit0。P014/P015はhidden例外・スコアなし。固定12epochの学習は継続し、自己最良0.951・0.960超え未達です。以下は各時点の履歴です。

**10月10日18:48 JST：P015 ref57031981もhidden再実行例外・Publicなしを確認しました。** [結果と64検査版の記録](../experiments/p015-hidden-exception-p016-native64-authorized-20261010.json)。raw/typedともCOMPLETEですがerrorDescriptionあり、hidden traceback・実時間は取得できていません。可視入力・CPU/64/256確認だけではhidden成功を保証できません。共通ServingのGold/train UID排他は未検出。P016新64入口はsource-only独立22項目を通過し、新CPU raw receiptとroot受入れの原bytesを固定した本診断権限をroot発行、5成果物に具体化しました。source695,007 B／private／Original31481／OFF／4 V1。実Save・GPU64・256・Serving確認はこの時点では未実施で、以前の資格は転用していません。モデル・係数・旧5セル・学習ソースは変更せず、自己最良0.951・目標未達を維持します。

**10月10日18:38 JST：新P016の実CPU復元資格を発行しました。** [検証記録](../experiments/p016-actual-CPU-qualified-regional-E5E6-20261010.json)。同一kernel138006054/V1のraw/typed COMPLETEと保存source8項目を確認し、2396Bのraw receipt SHAa87cd104…b786を保持。固定checkpoint3394…/292842755B、inventoryaa6d…、CPU strict486・finiteFP32・12ラベル・factory1・forward0・MRI0、Gold/Report利用falseを独立15項目とroot3項目で確認しました。成功childがstdoutを出さない正常経路のためlogは空。取得器v1の非空要求によるHOLDを保持し、v2の実HTTP GETでContent-Length0・実0B・e3b0…空SHAを確認。モデル・Saveの再実行0です。新64/256・Serving/hidden資格はまだfalse、旧資格を転用しません。局所runは原文E5/E6を保存し、独立12項目でsource13/CLI2/config/seed/固定support10・skip0・累積時間差を確認、rootの両session生存も確認しています。途中採用0、自己最良0.951・厳密な0.960超え未達を継続します。

**10月10日18:26 JST：新P016 CPU診断の実Saveを確認しました。** [記録](../experiments/p016-actual-private-CPU-Save-p014-hidden-audit-20261010.json)。V2 source peer14/14・CPU専用Notebook peer18/18・rootの実SDK DTO変換が成功、245,864 Bの送信ソースを固定。private/CPU/GPU無効/OFF/Original31481/SWA V1のみで単一Saveを受理し、kernel138006054/V1の公式CLI status COMPLETEを観測しました。CLI完了は実モデル復元の証明とはせず、raw出力・保存コード/Input照合・root acceptanceは未取得です。MRI/forward/競技Submitは0。P014の独立11項目監査と人工4ケースでは、追加Nativeの厳格な入力不足・reader例外・期限等の境界差を確認。通常の有効系列経路ではzero-validより先に検査が拒否するため、hidden例外原因をzero-valid・OOM・時間超過等に断定しません。P015は18:17でstatus省略・Public/エラー未返却。固定12 epochの実学習を継続、自己最良0.951・厳密な0.960超え未達です。

**10月10日18:07 JST：P014 ref57031186の公式応答でhidden再実行例外を確認しました。** [失敗記録](../experiments/p014-hidden-rerun-exception-p016-source-hold-20261010.json)。raw/typed双方COMPLETE・errorDescriptionあり・Publicなし。取得ツールのAssertionErrorはローカル検査でありhidden tracebackではありません。約4時間26分の経過は実行時間・タイムアウトの証拠にはせず、例外の発生箇所は未確定です。P015 ref57031981はstatus省略・Public/エラー未返却。P016のsource v1は独立28項目中27肯定、standalone native64/native256でreused import経路がなくhidden_io未解決のためHOLD。最小import修正を新v2に保存して限定再確認後、実CPU→64→256へ進めます。[数値型の訂正](../experiments/p016-numerical-type-wording-clarification-20261010.json)は原preregを保持し、raw FP32・後段の既存Pythonfloat/.17g helperを明記するだけで、数式・係数・コード変更0です。新P016実モデル/GPU/MRI/Save/Submitはまだ0、旧資格は転用しません。自己最良0.951・厳密な0.960超え未達、学習は固定12 epochを継続しています。

**10月10日17:54 JST時点：実学習fold0 E4／fold1 E5の元runバイト・同epoch比較を保存しました。** [検証記録](../experiments/f005-regional-epochs45-versia7-duplicate-20261010.json)。212／208更新、AMP skip0。weak BCE0.312486072／0.309305496、AUC12 0.824600009／0.831306383、固定支持10項目のAUC差−0.011180644／−0.011783377。累積履歴の直近差は2,204.985／2,203.990秒、両runのsource13・CLI2・config一致を確認。固定12epochを維持し、途中checkpoint採用・Gold／Public fit／evaluationは0です。GPU設定・並行実行差があるため表現だけの因果差とは断定しません。

Versia 7の追加source pull1回は、markdown／code／metadataのみを分析し、非空5セルが基準27の元5と境界改行以外で一致。新因子0、現版作者Public・採点版は未確認、compact Input空欄と版未固定もあるため追加取得へ進みません。saved output・モデル／MRI／教師本文の閲覧・実行0。P016の新ソースは最終freeze／独立査読待ちで、実CPU／64／256／Serving資格は未取得です。17:54のP014／P015公式応答は状態欄省略・スコア／エラー未返却で、hidden完走は未確認。自己最良0.951・厳密な0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日17:29 JST：直接224px切り出し候補の事前条件を固定しました。** [P016記録](../experiments/p016-direct224-onefactor-preregistration-20261010.json)。既SWA Input V1（Dataset12431568・internal20440357）のcrop重みSHAと過去P010／P011の実到達をmetadataで照合。同重みの旧P011は公式COMPLETE・自己Public0.950、改善なしです。新しい枝は旧centre307→bilinear224をdirect centre224／offset80／noresizeへ変えるもので、過去のCPU／32／64／256件・Serving資格は転用できません。自己0.951基準を保護し、全12項目の90:10・平均同値順位・float32・UID順を固定。Gold項目別係数・待機中Public結果による調整は0、source実装中・新実モデル復元／forward／GPU／Save／Submitは0です。新CPU→64→256と基準CSVバイト保全・時間・資源の確認後に提出へ進めます。17:29の2件の公式照会は状態欄省略・スコア／エラー未返却でした。

**10月10日17:18 JST時点：fold0 epoch3・fold1 epoch4の実run原文を保存し、統合Notebook v2のソース査読15／15を照合しました。** [最新記録](../experiments/f005-regional-epochs34-integrated-v2-public-source-20261010.json)。weak BCE0.339705079／0.327385289、AUC12 0.808560059／0.806196504、212／208更新・AMP skip0。直近epoch間隔は2,195.144／2,198.282秒で、累積`elapsed_seconds`の隣接差です。source・CLI・configの両fold一致を原文から確認し、固定12epochを継続、途中checkpoint採用0です。旧CLSとはGPU設定・並行実行条件にも差があり、表現だけの改善やPublicへの換算は認定していません。

統合v1の最終mask拒否時に途中CSVが残る問題は、v2の最終呼出だけに元の`stage.fail`を接続して解消しました。独立人工negativeで所有CSV撤去・封印済基準予測／backup／native／completion入力の保持・元の例外の保持を確認。他21セル・元5セル・数式・66ソースは一致、正確なAPIシリアライズ671,063バイト、内部ガード950,000バイトへの余裕278,937バイトです。この値をKaggle公式上限とは扱いません。ソース査読の合格に限定し、実12epoch完成pair・新Input・CPU／64／256件・実Notebook／hidden推論の確認は未完了です。

独立metadata比較でも支持数が新旧一致し、固定10項目のAUC平均差はfold0 E3 −0.014212・fold1 E4 −0.025869、weak BCEは両foldとも旧CLSより高い状態です。固定支持集合は両foldで正負各20件以上を満たす共通10項目で、epochの結果を見て変更していません。Gold・Public評価は0です。追加調査は公開Notebook一覧1回とDINOsaur V5のソース・metadata取得1回。保存outputの閲覧・モデル／MRI／教師取得・実行・Save／Submitは0。取得原文SHAを固定して、現版のPublic／版ID未確認、主CoAtNetと自己0.951基準のSHA一致、別crop224のLateral OA限定13%枝、Gold選択履歴・未固定Input・例外継続を記録しました。順位順一覧から0.955以上とは推定せず、候補全体への置換は保留です。17:18のP014／P015は公式応答の状態欄省略・スコア／エラー未返却で、hidden完走は未確認。自己最良0.951、厳密な0.960超え未達を維持します。以下は各時点の履歴です。

**10月10日16:50 JST時点：fold0 epoch2・fold1 epoch3の実run原文とweak検証値を保存しました。** [実観測・調査・統合HOLDの記録](../experiments/f005-regional-epochs23-public-search-20261010.json)。weak BCE0.343499913／0.331905033、AUC12 0.756164100／0.780217849、212／208更新・AMP skip0。両直近epochの時間は約2,217.53秒で、累積履歴の隣接差です。旧モデルの同fold・同epochとの比較では固定支持10項目のAUC平均差が−0.005349／−0.014080。表現以外にGPU設定・実行重複の差もあり、途中checkpoint採用・改善の認定は0です。公開一次資料の検索2バッチ8クエリ・open4回・3候補は作者報告のみで、自己再現やモデル取得0、利用可能な0.955以上の候補は今回の範囲で未確認です。

統合Notebook v1は18コードセル・ZIP66ファイルで、正確なAPIシリアライズ669,729バイトをrootが再計測しました。初期未資格拒否・元5セル等は査読18項目に合格しましたが、最終maskの所有登録前拒否で既発行uniformCSVが残る1項目を人工negativeで確認しHOLD。v2の限定修正・再査読を進めています。実pair・Input・authority、実CPU／64／256・提出推論の資格は未発行です。16:49のP014・P015公式観測は状態欄省略・スコア／エラー未返却、hidden完走は未確認。自己最良0.951・厳密な0.960超え未達を維持します。

**台帳のGold／weak誤列表示3件を訂正しました。** [訂正根拠](../experiments/ledger-metric-source-correction-20261010.json)。209行を照合し、数値のあるGold欄5件中、明示weakキーと一致した3件だけ末尾の新`weak_macro_auc_12`列へ移しました。数値文字列・行順・他セルを保持し、既知Gold評価2件は変更していません。訂正前CSVを保存し、元の実験JSON・モデル・runの書換え0。列数は14から15へ変更しました。Gold学習不使用というflagだけでGold評価未実施まで推測せず、評価種別の根拠を各JSON/key・fold1・epoch9／1／2として記録しています。

**10月10日16:28 JST：Regional support maskのソース検査12項目が合格しました。** [記録](../experiments/f005-regional-support-mask-peer12-20261010.json)。旧CSV helper7個・定数6個はAST一致、writerを埋めたセルのバイトとJSONも一致。認可不足2probeは作業CSVの読取・書込前に拒否されます。人工3UIDのMCL・Synovitis計6値、他10列計30文字列、tie・UID順・backup・CRLFを確認した7項目の記録を照合しました。実モデル・MRI・教師・GPU・API操作0のソース検査であり、実Notebook統合・新RegionalのCPU／64／256・提出・精度を証明しません。fold0 1／12・fold1 2／12epochは例外なく継続中、途中checkpoint未採用。P014・P015の16:28公式観測は状態欄省略・スコア／エラー未返却で、hidden完走は未確認。Goldによる学習・選択0、自己最良0.951とgoal継続を維持します。

**10月10日16:17 JST：専用Regional parent bridgeのソース査読22項目が合格しました。** [査読の記録](../experiments/f005-regional-parent-peer22-20261010.json)。新しい3phaseの完成review・同じpair・6役割Input・transport peerを必要とし、旧CLS／未発行authorityはconstructorがassets-import／出力前に拒否します。元5 hooks→所有PG／基準予測の封印→元clear1→専用transport→絶対パスのRegional子→元FP32平均／fixed20 writerを確認。既存source54（共通39を含む）・元5のSHAは保持。独立probeもソースの拒否契約だけで、実Torch／GPU／モデル／API0。support maskの新receipt接続・新Notebook組立・実CPU／64／256／Serving資格は未完成です。16:16の公式照会はP014／P015ともスコア・エラー未返却で、自己最良0.951・Gold fit／selection0・goal継続を維持します。

**10月10日16:10 JST：最初の並行epochと弱ラベルの初期比較を記録しました。** [実観測・ソース記録](../experiments/f005-regional-parallel-epochs12-transport-source12-20261010.json)。fold0 E1はBCE0.377602087／AUC12 0.761978944・212 updates・AMP skip0・interval2,247.069秒、fold1 E2はBCE0.342996104／AUC12 0.757316060・208 updates・skip0・累積4,110.913秒、前epochからの差は2,230.186秒です。`elapsed_seconds`を各epoch時間として足しません。runの元バイト2件を新しい観測フォルダへ保存し、source13／CLI2／config／表現／targets／provenanceの同一性を照合しました。両runは未完成・例外なし、固定12epochと最小weak BCE／最早tieを維持します。

同じ旧fold・epochの比較でfold0 macro AUC差は+0.018985ですが、Synovitis差+0.260／12の寄与がそれを上回り、同ラベルの陰性は1件だけです。以前から支持不足のMCL／Synovitisを除いた10項目の初期差はfold0 -0.003380、fold1 -0.000699でした。これは初期の事後診断で、checkpoint選択・係数調整・精度改善の証拠にはしません。実行設定と並行条件も旧CLSとは異なるため表現だけの効果と断定しません。専用transportは独立12項目・stdlib2件・compile2・Ruffに合格し、正2filesとZIP全memberのSHAを展開前に確認する実装です。実remoteバイト保存／CPU／64／256／Serving資格は未成立、専用親bridgeのソース査読とCSV mask receipt接続を準備中。自己Gold fit／selection0、患者独立性未確認、P014／P015は16:06の公式GETでもスコア・エラー未返却、自己最良0.951です。

**10月10日15:43 JST：Regional専用driverの独立ソース査読24項目が合格しました。** [検証記録](../experiments/f005-regional-driver-source24-capacity-20261010.json)。専用schema・元producer13／CLI2／closure4・generic factory1→deepcopy2→strict apply2・CPUのprefix／suffix／head／alias検査・初回CUDA前の決定的設定・旧数値helperの保持を確認。初回packetのCRLF→LF変換による4ファイルのraw SHA不一致はnegative査読を残して修正し、fresh packet-v2の全6ファイルの復元バイト一致を限定再確認しました。6mock・compile・Ruffはソースの検証であり、実Regional forward／実モデル復元／Kaggle資格ではありません。

容量候補では旧authority JSON6と専用launcher1を外し、元70 memberとRegional6を収録しました。共通39・元5 code cells・他21cellsはバイト保持。実Saveと同じJSON serializationのUTF8は698,443バイト、上限950,000まで251,557バイトでrootも独立実測しました。未発行authorityを展開・モデルimport前に拒否。旧ParentStages／late cell20は未接続・到達不能のままで、実行可能な推論Notebookという資格はありません。元Notebook／BIN／進行中学習ソースを変更せず、専用transport・後段接続・完成12epoch後の実CPU→64→256→Servingを準備します。完了レビューCLIも未完成runを重みアクセス前に拒否し、承認ファイルは作られませんでした。これは期待した拒否で、学習エラーではありません。両学習は継続、初期1epochで改善は未確認、自己Gold fit／selection0、患者独立性未確認、15:33のP014／P015は採点待ち、自己最良0.951を維持します。

**10月10日15:23 JST：regional fold1の実1epochを確認後、同じ固定条件のfold0も独立プロセスで開始しました。** [実観測](../experiments/f005-regional-epoch1-parallel-fold0-started-20261010.json)。fold1は208 opportunities／208 updates／AMP skip0、最終short group4、観測weak cell22,851、weak BCE0.376898958／AUC12 0.706897070、epoch1,880.727秒、Torch lifetime peak reserved1,197,473,792バイト。まだ1/12なのでモデル選択・精度改善の資格はありません。

fold0はfresh run・session94738、fold1はsession91809を維持。source13・CLI2・config・representation・execution・targets・generic provenance・教師／cache／QAハッシュを開始前と両runで照合しました。CPU28論理・RAM約32GB／空き約19GB、並行後のGPU全体sample4,097MiB／24,564MiBを確認。optimizer・RNGは各プロセス内、cache読取だけを共有します。これは実行を重ねた記録であり、旧single-process QAから二CUDA contextの全期間資源資格を転用していません。次のepoch時間・例外・資源を確認し、短縮を未測定で断定しません。固定12epochと最小weak BCE／最早tie、自己Gold fit／selection0、患者独立性未確認を維持します。15:23時点でP014／P015の公式スコアは未返却、自己最良0.951です。

**10月10日15:16 JST：regional専用sealer／単fold CPU入口の独立ソース検査38項目が合格しました。** [記録](../experiments/f005-regional-sealer-source38-medical-lead-20261010.json)。作者のstdlib4件を独立再実行し、未成立rootでprovenance・モデルへアクセスしないこと、checkpointの非tensor metadataにUIDが入る場合はfactory前に拒否することを追加2件で確認。13 producer・2 CLI・4 import closureの相対配置、26 memberの全SHA、元Regional factory／load／strict applyの再利用、prefix／suffix／headの照合、元checkpointのバイト維持、公開物の集約metadataと元証拠SHAの区別を査読しました。Ruff・26 source compile成功、producerと進行中の学習は変更0。現在の実CPU復元・実封印・新regional二fold／64／256／Serving資格はすべて未成立です。

追加の一次資料調査では、[OrthoFoundation-MSK-5Mの公開元](https://github.com/ytrsk/OrthoFoundation-MSK-5M)の固定commitとLFS宣言を確認。宣言SHAは取得済み重みのSHAではありません。新releaseの配布条件、実payload、公式backboneとのstrict復元、競技Publicは未確認で保留します。BrainDINOは作者が重み未公開と明記、KneePreMのリンク先で利用可能な専用重みは今回も確定していません。別の論文課題の成績やモデル名を、この大会の0.960超えの証拠には扱いません。P014／P015は15:12の照会でも採点待ち、自己最良0.951と自己Gold fit／selection0を維持します。

**10月10日14:50 JST：決定的dense局所モデルの実人工QA50項目・独立33項目を受け入れ、両foldのpreflight後にfold1学習を開始しました。** [実QA・事前照合・開始記録](../experiments/f005-regional-dense-QA50-fold1-started-20261010.json)。新schemaは`regional392_deterministic_dense_QA_v3`。同じ初期重み・RNGで144窓、16／10／4件蓄積を再実行し、全3,633,816学習パラメータの勾配・更新後重み・AdamW状態、logit・masked BCE・scaler・dropout RNGは差0でした。分母160／100／40、人工更新6回、観測最大reserved1,272,971,264バイト。欠損教師・NaN位置・paddingの除外、凍結prefix、専用checkpointのroundtrip、人工／旧CLS／改変表現の拒否も通過しました。これは人工GPUの学習契約であり、実MRI推論や精度の資格ではありません。

失敗したVJP2件と非決定的dense1件は保持します。両経路で微小勾配差がAdamW更新へ増幅されたため、VJP固有の原因とは断定しません。CUBLAS workspaceをtorch import前に設定し、共通helperで初回CUDAより前に厳格な決定的計算を有効化します。factory／QA／preflight／train／predictは同じ設定を要求し、config／checkpoint／実QA／環境へ拘束。旧失敗QAとsource違いは拒否し、許容差・optimizer・学習率・epsilon・局所表現・旧15ファイルは維持しました。[PyTorch 2.10の再現性資料](https://docs.pytorch.org/docs/2.10/notes/randomness.html)も参照しましたが、異なるGPU・PyTorch版での完全一致は保証しません。旧CLSの実行設定との差は記録し、表現の効果だけとは主張しません。

実preflightはfold0が3,386／821件、fold1が3,316／891件、キャッシュ4,207件で合格、fitting0。fold1は新runで固定12epoch学習中です。途中checkpointの採用・局所モデルの外部登録・提出は0。完了後の新しいCPU→64→256→提出契約が必要です。P014／P015の採点は未返却で、自己最良0.951・自己Gold fit/selection0・患者独立性未確認を維持します。以下は各時点の履歴です。

**10月10日14:30 JST：P015は実22本文・独立58項目に合格し、公式提出57031981が受理されました。** [実検証と提出記録](../experiments/f005-p015-submitted-regional-QA-held-20261010.json)。raw/SDKとも保存V1のCOMPLETEを確認。元70:30のFP32、native2foldのFP32平均、uniform20バックアップの順位式は各36値で差0。MCL・Synovitisの6値は封印済み基準値と差0、他10列の30文字列はバックアップと差0、CRLF/.17gの最終834バイトも完全一致しました。6入力と全セルの照合、実取得SHA、両T4・代替0・clear1回・子2群終了・queue終了を確認し、直前の残枠4件からAPIを1回呼びました。この資格は可視3件の提出契約であり、P014・P015のhidden完走・Public・全件時間上限は未確認です。

次の局所特徴経路の人工QAは、[失敗記録](../artifacts/qa/20261010-regional4x4-streamed-full144-v1/checks.json)を保持しています。RTX4090で3系列×48窓のdense/streamed両経路を比較し、全3,633,816学習パラメータの勾配・logit・masked BCE・dropout RNGを通過しました。effective16の勾配は一致許容範囲ですが、AdamW更新後のパラメータ一致に失敗し、QA全体は不合格です。教師表・実MRIを読まず、実学習0。新しいregional schemaとfactoryを使用し、旧CLSモデル・旧資格は変更していません。更新差の原因を修正するまで学習資格を付与しません。自己最良0.951、自己Gold fit/selection0を保持します。以下は各時点の履歴です。

**10月10日14:06 JST：P015比較版は保存・ソース照合まで合格し、実行中です。** [記録](../experiments/f005-p015-native-support-mask-running-20261010.json)。22セル/18コードの構文確認（writefileは本体をcompile）、人工の同順位・UID乱順チェック5項目、APIの全セル・6入力照合8項目が成功。MCL・Synovitisは封印済み基準値を`.17g`で復元し、他10列は既存uniform20の文字列を維持します。完走後は旧CSVバックアップ・4入力SHA連鎖・新CSV SHA・UID順・12ラベル・2列数値一致・他10列一致を新receiptで検証します。旧uniform20の37項目資格は新CSVの資格ではありません。P014提出57031186とP015のhidden/Public、全件時間上限は未確認。自己Gold fit/selection0、公開重みのGold由来は別管理、自己最良0.951を維持します。

**10月10日13:41 JST：P014の可視提出契約が37項目合格し、公式提出57031186が受理されました。** [提出受理と検証](../experiments/f005-native-twofold-p014-submitted-20261010.json)。V1/kernel137969034はraw/SDKとも明示COMPLETE・エラーなし。実20本文のSHAを照合し、元70:30 FP32・自作2fold FP32平均・最終80:20順位を各36値で再計算して差0、最終CSVは元writerどおりCRLF/.17gの834バイトで完全一致。基準モデルの所有プロセス終了・clear1回・両T4の解放、nativeのqueue残0・一時ファイル削除・代替0も確認しました。直前の保存ソース・完了状態・提出枠5件を確認してAPIを1回呼びました。hidden全件の完走・時間上限・Publicは未評価です。自己最良0.951を保持し、厳密な0.960超えは未達です。

**10月10日13:21 JST：圧縮したP014提出候補の非公開Version 1が保存され、エラーなしのRUNNINGを確認しました。** [保存受理とソース照合](../experiments/f005-twofold-p014-compact-serving-running-20261010.json)。APIソース927,057バイト、77メンバーの復元size/SHA・元5セル・固定20%計算の維持を確認。旧5テストは圧縮V4で成功、V5の変更は950,000バイトの任意上限と同じguard文だけなので再実行せず、Ruff・compile・独立10項目を確認しました。実GetKernelでは全21セル、Original/T4/Internet OFF、6入力のraw/SDK照合8項目に合格しています。実完走の出力・CSV計算・後処理・Publicはまだ未評価です。競技Submit0、自己最良0.951、厳密な0.960超えは未達です。

**10月10日13:02 JST：自作2foldのCPU・64件・256件の検証が12項目すべて合格しました。** [実結果と保存エラー](../experiments/f005-twofold-qualified-p014-source400-20261010.json)。256件は512回のforward、約575秒、両T4の観測最大635MiB、代替予測0件でした。P012の元5セルを保持し、自作モデルを固定20%で加える提出候補を作成しましたが、2.76MBのソースのSaveはHTTP400でした。参照先の照会は404で、保存版なし・競技Submit0を確認しました。サイズが原因という判断は推測です。圧縮して復元SHAを照合する修正を進めています。26番Notebookは失敗候補として保持します。自己最良0.951、0.955と厳密な0.960超えは未達です。

**10月10日 02:04 JST：自作二foldの実CPU・64件検証が完了し、同じ構成の256件が実RUNNINGです。** [保存版と実出力](../experiments/f005-twofold-CPU64-qualified-native256-running-20261010.json)。decoderは実在する非公開V1へ参照を修正し、manifestの内容は変更0。新ソースの独立27項目を通過後、CPUは厳格な2モデル復元、64件は128forward・214秒・両T4の観測最大635MiB・代替0・後処理終了を実出力で確認しました。256件は新V1/kernel137904159として保存受理され、同じpair・入力・コードを公式RUNNINGと照合済みです。競技Submit0、256件全体／Serving／Publicの資格はまだありません。自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:36 JST：256件の実検証とfold1学習・CPU復元・非公開入力登録が完了しました。** [実結果と次の修正](../experiments/p012-baseline256-qualified-f005-fold1-complete-20261010.json)。P012は実18本文・15項目合格、2559秒、3072予測値のfloat32差0、代替0・子2群終了・clear1回・GPU解放を確認。fold1は固定12epoch完了、最小BCEのE9で弱BCE0.27969／AUC12 0.86695、全12所見の改善方向が両foldで再現しました。paired13・CPU8項目合格、明示承認後にモデル2ファイルをprivate V1へ登録済。二foldのSave前検査で、decoderのローカルpackage改訂v2をKaggle Version2と混同した参照を検出しました。Version2一覧は404、実Version1のmanifestは元pinと完全一致。内容を変えず参照を修正中です。native Save0・競技Submit0、二fold／提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:13 JST：256件の保存版はエラーなしでRUNNING、fold1は11/12epochです。** 01:10の公式raw/typed状態と保存ソース6項目は一致しました。二foldの準備・単発Save手順は[独立ソース確認](../artifacts/research/20261010-f005-twofold-native-root-independent-v1/review.json)22/22を通過していますが、実CPU・64件・256件の二fold資格はまだありません。最終epoch完了後、事前指定の最小weak BCE選択→CPU復元→fold1非公開入力登録→二foldの実検証を順に進めます。途中checkpointは未採用、競技Submitは追加していません。自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 01:03 JST：作者0.954の新S6候補を確認しましたが、入力の配布終了により保留します。** [一次資料と取得結果](../experiments/public954-candidate-retired-input-20261010.json)。作者の保存V1・採点script356507041とソース一致、対応NetとV10 manifestも一致しました。現行公開V11は183バイトのREADMEだけで、モデルと旧版を利用不可にしたと明記。S6モデル取得0・アップロード0・自己0.954未確認です。既に固定した非公開S5入力の自己0.951を基準に、256件検証と自作二fold追加を継続します。fold1は11/12epoch、途中checkpoint未採用。二foldの実資格と0.955／厳密な0.960超えは未達、今回の競技Submit0です。この配布終了を過去全エラーの原因とは扱いません。

**10月10日 00:33 JST：実64件の合格を受け、同じコードの256件検証を開始しました。** [保存版と実行状態](../experiments/p012-baseline256-cleanup-running-20261010.json)。新しい非公開V1（kernel137894797）はRUNNING、元5セル・同じ入力3版・Original/T4/Internet OFFを保存ソース6項目で確認。64件の実合格証拠を固定し、256件の合格は完了出力から別に検査します。競技Submit0、二fold・提出版の資格は未確認、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:27 JST：P012の64件診断は実完了し、GPU解放を含む15項目に合格しました。** [実検査の記録](../experiments/p012-baseline64-cleanup-qualified-20261010.json)。固定70:30の768値は元式とfloat32差0、代替0、子2群終了、clear1回、両T4の確保量・予約量0。実行803秒、18本文の取得SHAを検査しました。以前の2件のERRORは追加診断checkerの不具合として保持し、成功資格へ転用していません。同じコードで256件を次に検証します。fold1は10/12epochで継続中、途中checkpointは未採用。学習の弱点解釈ではSynovitisの陰性が1件しかなく、weak改善をPublic改善とは断定しません。二fold・提出版の実資格は未確認、今回の競技Submit0、自己最良0.951、0.955／厳密な0.960超えは未達です。

**10月10日 00:03 JST：再診断V2は成功ログの誤判定でERRORとなり、後段契約全体を実形式で再照合しています。** [取得した実ログ・13本文・原因](../experiments/p012-baseline64-V2-worker-log-contract-error-20261010.json)。元予測は64件・代替0で完了し、固定70:30の768値は実CSVで再度一致。GPU観測はRaptor3763/3643MiB、S510225/9987MiBのsampled peakです。追加checkerが要求したPASS文字は元workerログには存在せず、WORKER_DONE JSONとfallback0が本来の成功記録でした。元モデル・数値は保持し、この形式と残りのseal→cleanupを実出力・元producerに合わせてV3で確認しています。封印・GPU解放は未到達、ERRORのCSVは提出しません。fold1は9/12epochで正常継続、途中checkpoint未採用。今回の競技Submit0、自己最良0.951、二fold/Serving資格と0.955／厳密な0.960超え未達を維持します。

**10月9日 23:47 JST：診断checkerを修正し、新しい非公開64件診断を再実行中です。** [修正・実取得・新Saveの記録](../experiments/p012-baseline64-checker-repaired-rerun-20261009.json)。旧ERRORから必要な8本文だけを取得し、4CSVの全64件・13列・有限12値、固定70:30の768セルfloat32一致、代替0、子2群の終了を確認しました。新checkerはGPU記録listのbool化と実S5先頭50件preview／全worker CSVの区別を修正。独立19項目・作者8＋独立2テストを通過し、元モデル・5セル・数値は維持しました。23:40に新private Save1回受理（kernel137888437/V1）、23:42 raw/typed RUNNINGと保存ソース6項目一致を確認。GPU解放を含む実完了はまだ未確認で、旧ERRORから資格を転用しません。fold1は8/12epoch、途中checkpoint未採用。競技Submit0、自己最良0.951、二fold・Serving資格と0.955／厳密な0.960超えは未達です。

**10月9日 23:33 JST：64件診断は予測完了後の新しい診断checkerでERRORとなりました。** [実ログと原因](../experiments/p012-baseline64-diagnostic-checker-error-20261009.json)。公式raw/typed ERRORと保存ソース一致を確認。元Raptor・S5は64件すべて、代替0、70:30融合を完了したと実ログに記録されています（786.9秒）。追加診断の`require(selected)`がlistを厳密なTrueと比較して必ず拒否する不具合が原因です。実S5のper_studyはtop-level・先頭50件previewであり、旧checkerの仮定も修正が必要と分かりました。原版・エラー証拠を保持してV2 checkerを準備中です。モデル・数値は変更せず、修正版64→256を実検証します。GPU解放・封印は未到達で、残った診断CSVは提出しません。ローカルfold1は8/12epochで継続。自己最良0.951、今回の競技Submit0、二fold・Serving資格と0.955／厳密な0.960超えは未達です。以下は時点ごとの履歴です。

**10月9日 23:25 JST：P012の64件GPU解放診断を非公開保存版V1で実行中です。** [最新記録](../experiments/p012-baseline64-started-nativefold1-progress-20261009.json)。診断ソースは独立30項目、二fold統合入口は33項目を確認し、元5セル・数値を保持しています。標準SDKの認証を更新した後、診断Saveを1回受理。23:22の公式raw/typed状態はRUNNING、保存ソース・宣言設定6項目は一致しました。CSV・子プロセス・GPU解放の実検証は完了待ちで、二foldや提出版の実行資格とは扱いません。RTX4090のfold1は8/12epoch、途中checkpointは未採用。完了後に事前指定の最小weak BCEで選択し、二fold自身のCPU→64→256と親への統合を検証します。今回の競技提出は0回。自己Public最良0.951、P013は0.950・エラーなし、0.955／厳密な0.960超えは未達です。

**10月9日 22:32 JST：P012（自己Public0.951）を基盤に、実診断へ向けた取得・保存検査を修正しました。** [最新記録](../experiments/f005-twofold-metadata-repairs-20261009.json)。JSON保存でfoldキーが整数から文字列になる比較不具合を修正し、型や本文の差は拒否するGetter V4を独立19項目、偽Save証跡を使わない保存前検査を24項目、結果集約converter V2を16項目で確認しました。数値・モデルソースは変更していません。fold1は6/12epochで継続中、途中checkpointは未採用です。必要な3つのInputはAPIで現Version1を確認しましたが、これは全モデル実bytesの検査や実行資格ではありません。P012の64/256件GPU解放診断と二fold統合のソースを準備中で、実二fold資格・追加モデルの提出は未実施です。P013の公式結果は0.950・エラーなし、自己最良0.951と0.955／厳密な0.960超え未達を維持します。

**10月9日 22:04 JST：P013は公式COMPLETE・Public0.950・エラーなしで採点を完了しました。** [結果と次の親の決定](../experiments/p013-own950-p012-native-parent-decision-20261009.json)。提出ref57006968／保存Version1を照合し、再提出は行っていません。自己最良0.951のP012（Notebook24・元5code）を次の自作二fold追加の親に選択しました。元の0.950対照とP013は記録として保持します。追加率は事前宣言の20%・全12所見で固定し、比率探索は行いません。二fold自身の実CPU→64→256と親への統合は未資格です。学習は固定12epochを継続し、0.955・厳密な0.960超えは未達です。

**10月9日 21:56 JST：自己Public最良0.951を維持し、二fold追加の実検証準備を進めています。** [最新記録](../experiments/f005-twofold-source-progress-20261009-2155.json)。学習中のfold1は5/12epochで、途中checkpointは採用していません。全体8.3時間の時計・固定20%融合CSVは独立検査28項目、取得ツールの型修正は39項目・ソース本文記録の追加は13項目を確認しました。清掃監視の誤識別と元タプルキーへの対応はV2で修正し、独立35項目を確認しました。新しい公開3候補もソースと採点情報を比較しましたが、新S6の正確な再構築や0.951超えの再現候補は確認できず、採用していません。P013は最後の21:08照会で状態未記載・スコア未返却です。二fold自身の実CPU・64件・256件と提出版統合は未資格で、ソース検査の成功を精度改善とは扱いません。

**10月9日 21:03 JST：二fold提出ルーターの[ソース準備](../artifacts/research/20261009-f005-online-last2-twofold-serving-router-preparation-v1/preparation.json)と[独立査読31項目](../artifacts/research/20261009-f005-online-last2-twofold-serving-router-independent-v1/review.json)が完了し、既存37件の数値ソースはバイト同一です。** [単fold0の実CPU・64/256資格47項目](../artifacts/research/20261009-f005-online-last2-actual-CPU64-256-root-review-v1/review.json)は二foldへ転用せず、二fold自身の実CPU→64→256資格は未成立です。fold1の弱教師学習は20:56の観測で3/12epochを完了し、固定12epochを継続中です。自己Public最良は0.951、P013は最後の20:33照会でstatus未記載・SDK既定PENDING・スコア未返却でした。親全体の時計・GPU清掃・最終CSVの統合は現在ソース準備中で、二foldの実資格取得とは別工程です。ソース準備の合格を精度改善や提出資格とは扱いません。

**10月9日 20:48 JST：単一foldの自作モデルはCPU・64件・256件の実診断47項目を合格しました。** [実行資格と次の設計の記録](../experiments/f005-online-last2-engineering-and-serving-plan-20261009.json)。この資格はfold0だけで、2fold統合や提出版の資格には転用しません。2fold設計V2はfold1の封印先だけを短い `b1` に修正し、既存215ファイル名の最長パス247文字と7つのCLI定義を静的に確認しました。旧設計V1と独立25項目の査読は保持し、統合推論のルーティングは実装中・未完成です。

自己Public最良は公式 **0.951** を維持し、0.955・厳密な0.960超えは未達です。P013は20:33の観測でraw status省略・SDK既定PENDING、スコア未返却です。fold1の時刻付きスナップショットは 2/12epoch（weak BCE 0.340666・AUC12 0.771251）で、固定12epochを継続中です。公開S6は学習コードと4重みの変更を確認しましたが、その新しい推論コード本文・互換性は未確認で、実行可能な新候補とは扱いません。教師なし64件の予測差は確認しましたが、学習から独立であることを確認していない単一foldのcohortであり、精度改善やPublic0.951との補完性の証明ではありません。


**10月9日20:17 JST：自己Public最良は0.951に更新しました。** [公式採点・次の提出・自作モデル進捗の記録](../experiments/p012-own951-p013-submitted-native-progress-20261009.json)。P012の保存Version 1は公式COMPLETE、Public **0.951**、エラーなし（ref56998356）。作者報告だけでなく、自分の提出で確認した値です。元p005の0.950とNotebook14は対照として保持します。目標0.955、厳密な0.960超えはまだ未達です。

P013は、基準p005へS5予測を固定20%で加える別候補です。実64件・256件の診断後、提出版の全3件・12所見、代替予測0、GPU解放、基準融合差0、提出CSVのバイト一致を独立20項目で確認し、同じ保存Version 1から1回提出が受理されました（ref57006968）。20:15の照会ではraw status省略、SDK既定PENDING、スコア・エラー未返却です。Input順序のAPI返却だけを取得ツールが厳しすぎる条件で比較していた不具合は、全22セルと環境の一致を確認して修正しました。モデル・入力の集合・予測計算は変えていません。

自作DINOv2最終2層モデルはfold0の固定12epoch学習とCPU再構築を完了し、新しいモデル自身の実64件・256件診断もそれぞれ10項目合格しました（320／1,280配列SHA、全12予測有限、両T4、代替0、約178／701秒）。256件の実COMPLETEを20:16に確認し、両GPUの観測最大は563MiBです。以前の凍結モデルの資格は流用していません。Serving・Public、将来の2fold合成モデルの資格はまだ未確認です。fold1は同条件の12epoch学習を継続し、最新保存スナップショットは1/12epoch（weak BCE 0.368269、weak AUC12 0.711189）。weak値をPublicへ換算せず、12epoch完了後に凍結対照と比較します。自分のGoldラベルの学習・選択利用は0、公開重みのGold/事前分布利用は別監査です。以下は過去の観測履歴です。


**10月9日19:35 JSTの公式観測を含む進捗：last2 fold0の固定12epoch学習とCPU再構築が完了し、fold1の比較を開始しました。** [学習・配布・診断の最新記録](../experiments/f005-last2-fold0-complete-CPU-privateInput-fold1-progress-20261009.json)。fold0の選択weak BCE/AUC12は凍結対照0.323206/0.794448からlast2の0.273758/0.877128へ改善し、最小BCEのepoch8を選択しました。12epochで実2,543更新・AMP skip1、選択checkpointのCPU厳格再構築8項目に成功。これは弱教師検証の結果で、Public精度への換算や患者独立性の保証ではありません。fold1の凍結対照も12epochを完了し、選択epoch10のweak BCE/AUC12は0.324100/0.800933。last2 fold1は同じ固定条件で開始済みで、途中checkpointによる採用判断は行いません。

last2 fold0の新しい非公開Input V1は19:38のKaggle CPU診断で215 memberの全SHAと厳格なモデル再構築に成功しました。専用の元online producerを使う実CPU8項目の証跡を拘束し、以前の凍結モデルの資格は転用していません。CPU診断はMRI読取・モデルforward0で、続くnative64/256・提出推論は未資格。p013は実64件を受理済み、256件は19:35の公式観測でRUNNING。p012提出ref56998356は19:27のAPIでも状態・スコア・エラーが未返却でした。自己Public最良0.950を維持し、0.955・厳密な0.960超えは未達です。以下の記述は各観測時点の履歴です。


**10月09日18:56 JST：p013の実64件診断を受け入れ、同じ修正版の256件を実行中です。** [実64件・修正・次の診断の根拠](../experiments/p013-cleanup-repair-and-last2-runtime-progress-20261009.json)。旧64件はS5開始前のCUDA清掃guardで停止、数値ログがなく当時の原因は未確定。新V3はcuBLAS作業領域の解放後に従来の清掃を行い、両T4のallocated/reservedが0、free各15.49GBとなりました。閾値と元14の7code／S5 core2／80:20を保持。実64件は全体942.90秒（S5追加区間716.54秒）、独立38項目成功、Reader選択384読取の失敗0、S5代替0、現在実行の元融合と固定20%の独立再計算一致。別実行ではReader CSV byte同一、CoAt6・基準融合10の順位セル差があり、完全な数値/精度同等性は断定しません。新256件はV1/kernel137854919として保存受理、18:52 JSTにRUNNING。256・hidden・精度の資格は未取得、p013競技提出0。

RTX4090のlast2 fold0は11/12epoch、weak BCE0.311269／AUC12 0.877300、実2,332更新・AMP skip0。12epoch完了前にcheckpointを使わず、最小weak BCEの選択ルールを維持。last2専用CPU入口・封印・転送・2fold driver、およびProfileに依存しない提出用清掃のソース査読は完了しましたが、実last2 CPU/64/256へ資格を転用しません。提出済みp012 ref56998356は18:53 JSTでもスコア・エラー未返却。自己最良0.950、0.955／厳密な0.960超えは未達。以下は観測履歴です。


**10月09日17:03 JST：次の候補はソース準備の段階です。** [次候補の準備と現在の観測](../experiments/f005-last2-and-fixed-S5-addition-source-preparation-20261009.json)。提出済みref56998356は16:56 JSTの公式読取で採点・エラー未返却、自己最良0.950を維持。RTX4090のlast2は7/12epoch、weak BCE 0.279831、AUC12 0.870949、条件変更・AMP skipなし。完了12epoch前にcheckpointを使いません。last2専用CPU入口V2は元8ソースの直接固定を含む独立8項目を通過し、実モデル再構成・推論は未実施。p013は元14の7codeとS5 core2を保持し、80:20と最終順位化を固定した未資格候補です。6人工CSV/構造テストの成功は実T4・64/256件・hidden・精度の成功を意味しません。Reader依存環境の一致、silent decode omissionの観測範囲、全体時間/資源を別途確認します。S5子は親開始から8.3時間の残時間を使用しますが、元14の非同期強制停止は未実装。追加の限定一次資料調査で新しい0.955以上の完全再現レシピは未確認です。0.955・厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:22 JST：S5 V9の非公開固定再現は可視実行・独立検査を完了し、公式API提出が受理されました（ref56998356）。** [実完了・検査・提出の記録](../experiments/p012-private-S5-fixed-V1-visible-complete-submission-accepted-20261009.json)。承認を受けた自分の非公開snapshot V1を接続し、Notebook24の原版5code/70:30/guardを保持。全3件・12所見を両T4で推論し、両モデルfallback0、約174秒、独立9項目・全36float32順位融合一致。保存V1から1回提出、初回APIではstatusが省略されSDK既定PENDING（採点・新自己Public未確認）。旧承認待ち／画面でのV9固定作業は不要になりました。RTX4090の独立last2は6/12epoch、BCE0.285281/AUC12 0.851437で継続。自己基準0.950を保持し、0.955／厳密な0.960超えは未達。以下は観測履歴です。

**10月09日16:12 JST：具体的なユーザー承認を受け、S5 V9の非公開固定Inputを作成し、原版5コードセルの再現を保存実行しました。** [承認・実作成・保存実行の記録](../experiments/p012-private-S5-snapshot-approved-and-run-started-20261009.json)。`rsraki/rsna-knee-compact-v1`はAPIで非公開Version1・8ファイル名/サイズ・元manifest byte一致を確認。新規Notebook24はInput参照と説明だけを変更し、元5code/70:30/guardを保持。新private SaveはV1でRUNNING、可視出力検査・採点はまだ。過去のアップロード承認待ちは解消しました。独立last2は6/12epoch、weak BCE0.285281・AUC12 0.851437。固定12epochの学習を維持。自己Public最良0.950、0.955／厳密な0.960超え未達。以下は観測履歴です。

**10月09日16:00 JST：S5 Version9の使用中6ファイルと元manifestを取得・ハッシュ一致まで確認しました。** [実取得と審査停止の記録](../experiments/p012-S5-private-snapshot-ready-review-rejected-20261009.json)。原版5コードセルのまま使える約885MBの非公開入力を準備。新規private Datasetへのアップロードは自動承認審査が具体的な送信内容の承認・宛先所有権確認不足として拒否したため、実アップロード0・Dataset作成0・再試行0。ユーザーへこの具体的なprivateアップロードの承認を質問済み。Kaggle画面で元Input V9をPinして保存実行する別経路も回答待ち。元last2学習は実プロセスをpollし継続、5/12epoch・weak BCE0.301530/AUC12 0.848946。固定条件を保持し、自己Public最良0.950／厳密な0.960超え未達。以下は観測履歴です。

**10月9日15:26 JST：実64件・256件の入力一致診断は成功。Helios作者0.951原版の再現は、APIによる旧Input版固定が反映されず停止しました。** [実行・調査の最新記録](../experiments/public951-native256-and-next-review-20261009.json)。64件は320配列SHA一致・204秒、256件は1,280配列SHA一致・541秒、両T4のsampled peak各563MiB、代替0。同じ凍結対照headの動作検証で、精度/hidden/最後2層モデルの資格とは区別します。Heliosに合うcompactはV9（外a5/使用部分c7）で、原版guardは正しいです。しかし標準Save APIの`datasetDataSources`に`/9`を付けても実際は最新V10（209a…）が接続され、約74秒でguardが停止。Kaggle画面でV9をPinし保存実行した版の確認が次です。23は予測コード未変更、自己スコアなし・競技提出0。Ryan V70は推論モデルでなく順位表（5,374チーム、0.955以上56・0.960超え7・最大0.964）。RTX4090最後2層は4/12epoch・実848更新・AMP skip0、weak BCE0.302986/AUC12 0.835976で継続中。自己基準0.950を保持し、0.955・厳密な0.960超えは未達。過去段落は観測履歴です。

**10月9日14:36 JST：新しい64件診断は入力パス指定の不備で停止。原因を実ログで確認しました。** [実エラー記録](../experiments/f005-native64-actual-mount-path-error-20261009.json)。旧flat競技rootを固定したためtrain_series.csvが見つからず、MRI処理・モデル読込前に約21秒で停止。競技Inputの実mount場所はこのログでは未観測です。構造CSV／画像ディレクトリの存在と固定CSV SHAで一意にrootを選ぶ新wrapperを準備し、同じモデル・64件で再診断します。CPUでの183ファイル一致・厳密再構築成功は保持、64／256件の資格は未取得。last2 fold0の固定12epoch学習は変更せず進行中。元p005自己0.950を保持、0.955／厳密な0.960超は未達。以下の結果待ちは過去の観測時点の履歴です。

**10月9日14:25 JST追記：元last2 producerの2/12epoch・実424更新／AMP skip0を確認。** [観測JSON](../experiments/f005-last2-fold0-actual-epoch2-observation-20261009.json)にrun snapshot SHA、epoch2と同epoch凍結対照のweak値を記録。実CPU診断は[独立21項目](../artifacts/research/20261009-f005-cached-native-CPU-independent-review-v1/review.json)も成功、実GPU64/256・Public改善は未確認。

**10月9日14:20 JST更新：Kaggleでの実モデル配布・CPU読込を確認しました。** [root根拠](../experiments/f005-last2-firstepoch-privateCPU-native64-root-progress-20261009.json)。cached adapter独立28、transport/wrapper V3独立28をbind。実非公開Dataset12459214/V1と予定2ファイルをAPIで確認後、CPU V1/kernel137810273がCOMPLETE。取得した実JSON2個から183 members全SHA、同bundle manifest、厳密CPU再構築・generic loader1・MRI0／forward0・教師/Report参照0を確認しました。Input内部IDはAPI未返却、要求版を明示し実byte一致で拘束、架空IDなし。native64 Saveは受理済み、実GPU結果・256・隠し提出は未確認。last2 epoch1の実212更新／AMP skip0も確認、元14/22・学習producer SHA保持、ownGold0、自己Public0.950・目標未達。以下は過去の観測履歴です。

**10月9日13:15 JST更新：全件feature受入・実凍結学習・選択checkpoint検査を完了しました。** [root記録](../experiments/f005-frozen0-complete-last2-actual-start-root-review-20261009.json)が全4207 cache QA9項目、完了run独立14項目、選択weak CSVからの全AUC/support一致とFP64 BCE再構成差4.12e-10、元CPU loader一回による厳密再構築をbindします。head全FP32／generic prefix不変／Native CLS表現・新producer4を確認。実12epoch・2,544更新・AMP skip0、ownGoldラベル再評価0。元last2 fold0はpreflight成功後にrunningとして作成済み、まだ対照結果・実native Kaggle runtime・Public改善は未確認です。元14/22 SHA保持。以下は過去の観測履歴です。

**10月9日12:39 JST更新：p011は自己Public 0.950で採点成功、精度改善なし。** [公式COMPLETEと採点記録](../experiments/p011-scored-review-20261009.json)にbindし、隠し詳細ログ／項目別AUC／非丸めAUCを取得したとは扱いません。p005を維持。凍結CLS cacheの[実人工CUDA資格・抽出開始](../experiments/f005-frozen-FP32-actual-qualification-and-cache-start-20261009.json)は144窓とsparse mask、16+3の人工観測群で各経路2 optimizer更新を実行し、feature反復／dropout RNG・logits／gradients・optimizer・scaler／検証／人工checkpoint互換・本番拒否が一致。元8 sourceと新4 producerは区別して固定しています。実4207 feature抽出は進行中、全件cache QA・正式fit・real accuracyはまだ未完です。ownGold fit・checkpoint選択0。

**10月9日12:25 JST更新：既存全件index builder完了、CUDA preflight成功。** [root11](../experiments/native392-full-index-and-frozen0-preflight-root-review-20261009.json)はindexの全UID／配列数／窓数／plan・manifest・source・24 summary・stage receipt SHA・archive sizeを照合し、元の学習CLIのCUDA preflightで3,386 train／821 valid／4,207 cacheを確認しました。32分の全配列読み出し検証は繰り返さず、その完了indexと小さなmetadataを受け入れています。元DICOM取得0・正式fit0・ownGold学習／選択0。凍結FP32cacheの数学的一致、cached producer、新checkpoint lineageはまだ未検証です。p011 [3回目公式照会](../artifacts/research/20261009-p011-scoring-status-tools-v1/observation-v3/review.json)もscore／raw status不明で、隠し完走や精度改善の証拠にはなりません。

**10月9日11:43 JST更新：全24 shardの取得・個別QA・canonical配置完了。** 23/24も有限Rangeの実ZIP SHA、各175件／875配列の[実13項目](../experiments/native392-shard23-completed-cache-final2-review-20261009.json)・[実13項目](../experiments/native392-shard24-completed-cache-final2-review-20261009.json)成功。2 job専用のsource71 gateで既存numeric／hardlink機構を保ち、[root全24受入8項目](../experiments/native392-all24-canonical-root-acceptance-20261009.json)で全4,207 UIDの重複なし・summary／receipt／stat／inode／mtime／nlinkを照合。元DICOM取得0、cacheに教師／Report／Gold学習0。全件indexを作成中、正式fit0、各foldの比較器は準備済みです。

**10月9日11:30 JST：競技提出受理／採点待ち**。[p011受理記録](../experiments/p011-submission-accepted-20261009.json)はexact private V1/kernel137784585/output submission.csvと実root12／peer33の証拠SHAに拘束。fresh GetSubmissionLimits1で残5、fresh V1 status1でraw/typed COMPLETE・failure空、CreateCodeSubmission1でref56986152を確認しました。再保存・再送0、Public／隠し成功はまだ未返却。自己0.950を置き換える根拠はまだありません。

**10月9日11:28 JST更新**：認証復旧後にp011のprivate Serving V1を保存実行し、公式COMPLETEと[保存ソース17項目](../experiments/p011-serving-savedV1-binding-review-20261009.json)を確認。7 JSON＋5 CSVの実12 body／48,372 bytesを取得し、[独立33項目](../experiments/p011-serving-visible-output-independent-review-20261009.json)と[root NumPy12項目](../experiments/p011-serving-visible-output-root-review-20261009.json)は成功しました。可視3 UID・13列・有限値、元融合差0、追加90:10融合差0、実submission CSV byte一致、基準seal不変を確認。追加モデルのstrict486 tensors・224解像度・3 forward／282窓・例外／fallback0を記録。原main／Readerの今回実forward数やcheckpoint SHAは観測しておらず、保存ソースと既256件の証拠を用います。隠し全件成功・精度・自己Public改善はまだ未確認です。

cache20/21/22は全Rangeの実ZIP SHAと各175検査／875配列の13項目を確認、既01〜19に続いてcanonical配置しました。PC計3,857検査／19,285配列、残350件です。23/24も公式COMPLETE・保存71項目・要約各28項目は成功。2 jobの保存証拠を3 job用100項目工具へ渡す前段不適合は、取得前のローカル停止でありNotebook実行失敗ではありません。旧工具を保った専用入口を準備中、全件index／正式fitは未完。以下は各確認時点の履歴です。

**10月9日04:07 JST：認証待ち／目標未達**。第三連続goal turnでも[公式API観測](../artifacts/research/20261009-kaggle-connectivity-probe-v1/observation-v3/receipt.json)はHTTP401。[停止判定の監査](../artifacts/research/20261009-goal-authentication-hold-audit-v3/receipt.json)で同一障害3回・独立確認・準備済み入口・実保存/提出0を照合し、再認証または外部認証状態の復旧が必要と判断しました。自動継続をblockedへ更新する条件を満たしています。自己最良0.950、0.955/strict>.960未達、元14/候補22V3のSHA不変。remote CPU jobsの停止や完了は追加確認できておらず、再起動していません。認証復旧後、fresh quota→保存実行→可視CSV確認→fresh提出枠→提出で再開します。ledger150→151の旧byte prefixを保持しました。以下は各確認時点の履歴です。

**10月9日04:03 JST再確認**：[公式APIの新観測](../artifacts/research/20261009-kaggle-connectivity-probe-v1/observation-v2/receipt.json)もHTTP401、Kaggle API再認証待ちです。保存実行/提出は0、自己最良0.950を保持。cache20/22の既metadataにも再利用できる署名URLはなく、取得を進められていません。[地域モデル接続前の不足](../experiments/regional-triplet-training-connection-review-20261009.json)として、CLS同形状の重みを混同しない専用schema/表現識別をsave/load/apply/factoryで検査する必要を確認。現factoryはNative固定で地域モデル未接続、既学習器への変更/実fit0です。ledger148→150は[旧byte prefix保持](../artifacts/research/20261009-goal-authentication-hold-audit-v2/receipt.json)。

**10月9日03:50 JST更新**：p011実256件はCOMPLETE、[root NumPy24](../experiments/p011-profile256-completed-root-review-20261009.json)・[独立47](../experiments/p011-profile256-actual-results-independent-review-20261009.json)成功。13 compact bodies/2,861,806bytes・status1/metadata22・source0（既ソース17再利用）。main512/Reader192/追加256×94窓、strict486 tensors/実checkpoint SHA、256 worker JSON SHA、二度decode記録SHA、4CSV/3,072予測のfloat32 byte SHA、元融合差0・CPU baseline byte不変・固定90:10再構成SHAを照合。3phase計977.913秒/sample GPU最大8,893MiB。独立MRI array再hash・精度・全件隠し成功ではありません。

- 提出22V3は[ソース24](../experiments/p011-serving-handoff-independent-source-review-20261009.json)・[qualified request11](../experiments/p011-serving-qualified-request-independent-review-20261009.json)でboth256 evidence/SHA、元7 code、21cell18code/11package、dynamicUID、OriginalT4OFF/Inputs1/1を確認。nonce requestはローカル確定のみ、ServingSave0/CompSubmit0。次は保存ソースと可視CSVを別検査し、fresh公式日次枠が1以上の時だけ提出します。
- [公式API probe](../artifacts/research/20261009-kaggle-connectivity-probe-v1/receipt.json)はHTTP401、初期default GetKernelはHTTP403を観測。認証原因の詳細は未確定。自動承認拒否0・credential body読取/表示0、ユーザーAPI再認証待ちです。20/22取得は0Range/0bytesで保持、23/24先頭GetKernelも途中記録を保持し、新snapshot入口を用意しました。
- PC cache01〜19計3,332件/16,660配列は保持。remote20/22各COMPLETE・summary28、21最終RUNNING、23/24 savedV1受理/kernel137723666/137723691、保存ソース/実QC未確認。全24 jobは起動済み、full4207 index/fit0。[4つの学習コマンド](../artifacts/research/20261009-f005-fast16-root-commands-v1/commands.json)も全24 QAを先に要求します。
- [新regional部品20項目](../experiments/regional-triplet-component-preparation-20261009.json)：CLS＋固定4×4 patch平均をheadへ渡し17instances/窓、head83,352params据置。人工CPU8/Ruff、既汎用DINOの4090人工2 logical windows・backward1・optimizer0。既Native/head/trainer/config/14 SHA不変、MRI/teacher/newweights/API0。full144資源/精度/専用checkpoint schema/学習接続は未完。
- ledger141→148は[旧byte prefix保持](../artifacts/research/20261009-p011256-auth-hold-progress-record-v1/receipt.json)、自己Public0.950・0.955/strict>.960未達を維持。

以下は各確認時点の履歴です。

**10月9日03:19 JST更新**：自己最良0.950、0.955/strict>.960未達。p011 private64 V1は明示COMPLETE、13小ファイル731,288bytes/13 GET、source RPC0（既正本再利用）/status1/metadata6。[root24項目](../experiments/p011-profile64-completed-root-review-20261009.json)は独立NumPyで元融合diff0・固定90:10候補CSV再構成SHA、64×12 raw値/F32 byte SHA、64 worker receipt SHA、二度decodeの記録hash、main128/reader48/expert64 forwardとstrict486 tensorの実重みを確認。[独立45項目](../experiments/p011-profile64-actual-results-independent-review-20261009.json)の標準ライブラリ計算でも一致。全3 phase計230.089秒（queue/import等除外）、sampled GPU最大8,893MiB。RSS15,547,322,368bytesは共有頁重複を含み厳密peakではありません。

- 追加前後のCPU基準CSV byte同一、main/Reader終了後のGPU allocated各9.128MiB・reserved404MiBを観測。zero解放とは扱わない。64×94窓/各GPU32件、decoder/initializer/fallback0、transport完了。候補CSV本体は診断末尾で削除、元4CSVから再構成したhashだけを照合。MRI配列bodyの独立再hashはしておらず、精度/256件/全件隠し完走は未証明。
- 初回getterはSDK `.get()`不在のローカル例外でRPC/body0停止し、旧失敗を保持してrequests Session新V2へ。root checkerのfull native metadata対hash-only inputと空mirror list対整数countの仮定も実schemaへ補正。[補正receipt](../artifacts/research/20261009-p011-profile64-root-review-v1/checker-correction-receipt.json)。runtime/予測/重み/閾値は不変。
- 256版は[local18](../experiments/p011-profile256-artificial-validation-20261009.json)・[独立22](../experiments/p011-profile256-independent-prelaunch-review-20261009.json)、launcher[21](../experiments/p011-profile256-root-launcher-preparation-20261009.json)・[独立24](../experiments/p011-profile256-root-launcher-independent-review-20261009.json)成功。rootのdefault preflight/CPU4/SDK DTO2も成功。元7code/9writefile/6helpers/90:10不変、64 prefix保持256件/255 groups。03:12:34 JST private保存V1/kernel137720781受理、quota1/Save1、競技提出0、結果待ち。 [実保存ソース17項目](../experiments/p011-profile256-savedV1-binding-review-20261009.json)も成功、03:14のraw status省略/SDK既定QUEUEDから実worker開始を断定しません。
- cache14〜19は各175 NPZ/875配列の13実検査・canonical配置成功、PC01〜19計3,332件/16,660配列。[root19集約9項目](../experiments/native392-shards01-19-canonical-stage-root-review-20261009.json)と[既16の独立264](../experiments/native392-shards01-16-canonical-stage-independent-review-20261009.json)を照合。今回は全19の実inode/size/mtime/nlink・summary byte一致、新17〜19の配列QA13 SHAを確認し、大ZIP再読をしていません。新3 ZIPは各36回の128MiB以下厳密206・全ZIP SHA・全5配列SHAを実検査済み。20〜22[保存100](../experiments/native392-shards20-22-saved-binding-review-20261009.json)成功/03:07観測RUNNING、残2未起動/full4207 index/正式fit0。ledger131→141の[旧prefix保持receipt](../artifacts/research/20261009-p01164-cache19-progress-record-v1/receipt.json)。
- [f005入口14項目](../experiments/f005-fast16-fullcache-fit-entry-review-r2-20261009.json)は両config/現資源gate/CUDAを受理。旧reviewの`ensure_cuda()`成功時NoneをFalseとする解釈だけをr2で補正。今回は教師/NPZ/重みpayload読取・fit0、全24成功後に正式preflight・同条件fold0/1学習へ。

以下は各確認時点の履歴です。

**10月9日01:56 JST追記**：13も全ZIP/member/5配列のSHA・shape・dtype・mask・coverage・padding等[13実項目](../experiments/native392-shard13-completed-cache-review-20261009.json)成功、canonical hardlink/SUMMARY byte同一配置を完了。PC01〜13計2,282検査/11,410配列、全4,207件index/正式fitは未完。ledger130→131は[旧byte prefix保持](../artifacts/research/20261009-native392-shards11-13-progress-record-v1/receipt.json)。以下の実検査件数は、その時点の履歴です。

**10月9日01:54 JST追記**：p010の実32件/10phase/40compact bodiesを[root17項目](../experiments/p010-AA-completed-compact-root-review-v4-r2-20261009.json)で検証。getter author10＋peer12の人工SDK、checker人工9もrootで成功、raw6比較は別のNumPy読込でも独立再計算しました。全10入力SHA、実重みSHA/strict/noGrad、16CSV、6 raw/ULP/rank/tie/GPU strataが一致。原main通常A/Aは最大差0.00032717・LM順位2セル変化、固定同GPU32/32でもbase0.00017780/stock0.00021404の差が残りました。固定反復は順位0、全6旧raw1e-7 gate不成立を保持、cause/変動上限/数学的同等性/精度は未証明。実Reader/融合0、候補gate false。GPU sampled最大8,895MiB、RSS最大18,024,181,760bytesは厳密peak/専有量ではありません。

remote native392は01〜13 COMPLETE・各summary28成功、PCは[11の175件/875配列](../experiments/native392-shard11-completed-cache-review-20261009.json)と[12](../experiments/native392-shard12-completed-cache-review-20261009.json)も13項目成功。全配列検査・canonical配置01〜12は計2,107件/10,535配列、[root14項目](../experiments/native392-shards01-12-canonical-stage-root-review-20261009.json)。13は取得済み/配列検査中、14〜16は[保存100項目](../experiments/native392-shards14-16-saved-binding-review-20261009.json)成功・観測RUNNING、残8未起動/full4207 index/正式fit未完。

新p011は設計レビューを経てローカル実装中、実64/256候補推論と提出は未実施。限定追加公開調査は[14項目](../experiments/post950-public955-delta-primary-research-v2-20261009.json)成功、0.955以上未発見と不在の証明を区別します。ledger124→130は[旧prefix保持](../artifacts/research/20261009-AA-and-cache13-progress-record-v1/receipt.json)。以下の01:24以前の記載は確認時点の履歴です。

更新日 2026年10月9日01:24 JST。自己最良はp005の公式0.950。p006/p007/p008も公式0.950、p009は公式0.948で全て非採用。p010 A/A診断savedV4は保存source36項目成功、01:16公式raw/typed COMPLETE・実phase出力分析は未完。PC cache01〜10は全配列検査・canonical配置成功、計1,757検査/8,785配列。11〜13は保存source100項目成功・01:16観測RUNNING、残11未起動。f005 fast16は新入口47項目・processed115資源検査13項目成功、全4,207件index/正式fitは未完。00:10公式日次5件/残0・更新時刻未確認、0.955/strict>.960未達。以下の採点待ちや未開始表示は各作業時点の履歴。

- [A/A診断savedV4の36項目](../experiments/p010-AA32-savedV4-binding-review-20261009.json)では全21セル/18code/8helpers、private/Original/T4/OFF、入力pin request /1を照合。source/status各1RPCのみ、01:16:57 JSTの明示COMPLETEは一回の観測。rawの省略空repeated fields対typed []のchecker不整合は新r2で補正し、旧失敗checkerを保持。実GPU・各phase raw値・候補gate・精度は別に確認します。
- 加工済みcache05〜07各176件・08〜10各175件の6 ZIPを最大128MiBの厳密206で取得し全体SHA、各NPZ/member、5配列のSHA/shape/dtype/mask/positions/planes/flags/coverage/paddingをroot検査、各13項目成功。[05](../experiments/native392-shard05-completed-cache-review-20261009.json)、[08](../experiments/native392-shard08-completed-cache-review-20261009.json)、[10](../experiments/native392-shard10-completed-cache-review-20261009.json)。PC01〜10で1,757件/8,785配列、全4,207件ではありません。
- [canonical配置準備22項目](../experiments/native392-index-stage-preparation-20261009.json)・人工16CPUをrootでも実行して成功。rootがchecked workspace内の親を明示作成し、one CLI/one shardで01〜10を実配置。[配置root8項目](../experiments/native392-shards01-10-canonical-stage-root-review-20261009.json)。旧ZIP全SHA/mtime前後一致、samefile・nlink+1、SUMMARY byte同一、全partial markerなし。hardlinkは同一inodeを共有し入力を不変扱い。ZIP byteコピー/展開、API、元DICOM/教師/モデル取得、fit0。
- [11〜13の保存100項目](../experiments/native392-shards11-13-saved-binding-review-20261009.json)は6read RPC、各175 whole-group、CPU/OFF/Original/decoder2、全埋込みsource/plan/Gold除外が一致。01:16観測raw/typed RUNNING、内部Input版IDは未観測、実出力QCは別gate。
- [人工full144速度計測84項目](../experiments/f005-window-throughput-review-20261009.json)・[root独立28項目](../experiments/f005-window-throughput-root-review-20261009.json)、peer source15項目成功。chunk2→16のfrozen6.27倍/last2 7.80倍は人工forward/backwardの計測、全epoch/I/O/optimizer時間の測定ではありません。last2最大予約1.115GiB、eval最大差1.78814e-5、gradient同値性未測定。実教師/optimizer更新0。
- [fast16新入口47項目](../experiments/f005-native-fast16-preparation-validation-20261009.json)・CPU28、人工full144 CUDA6/6を成功。旧config/source/handoffを保持し、新両armだけchunk16/source-specific schema/lineageを適用。rootの[processed115と資源13項目](../experiments/f005-native-fast16-processed115-root-review-20261009.json)では加工済み入力のラベルなしforward/backward・厳密reload0・凍結SHA不変・finite gradientが成功し、real resource validatorは両configを受理。正式fit0、全4,207件index必須。
- ledger118→124の[記録receipt](../artifacts/research/20261009-native392-fast16-progress-record-v1/receipt.json)では旧byte prefix保持と根拠hashを確認。自己最良0.950・目標未達を維持します。

## 自己Public0.950確定と改善計画（10月8日）

- **10月9日00:35更新**：[p010V3実失敗14項目](../experiments/p010-profile32-third-failure-review-20261009.json)と[9CSV数値監査12項目](../experiments/p010-profile32-third-failure-CSV-review-sanitized-r2-20261009.json)成功は、候補合格ではなく失敗内容の監査成功。bounded基準のLM2件順位入替、original/shared基準・融合byte同一、224 raw最大差0.0005064458と全12rank差0を確認。rawbase/UID→GPU/backend/資源数値は欠落、精度未測定。[同条件A/A設計](../experiments/p010-AA-precision-diagnostic-plan-20261009.json)で各phase receiptをguard前保存し、原因未確定を保持。旧ordinal checkerは未到達のfloat32解釈不整合があり、別修正で扱う。
- [Range独立24項目/人工23固有契約](../experiments/native392-remaining-range-independent-review-20261009.json)を経て02/03の旧partial stat/SHAをrootで取得、新ファイルへコピーし厳密206の16/7 Rangeで復旧。最終ZIP SHAと各176 NPZ/880配列まで[02](../experiments/native392-shard02-completed-cache-review-20261008.json)・[03](../experiments/native392-shard03-completed-cache-review-20261008.json)各13項目成功。旧prefixは前後SHA同一、rawDICOM/teacher/実fit0。PC検証済み704、全4207の完了ではない。
- CPU05/06/07は明示COMPLETE・source100と各summary28成功、06/07も[05同様の有限206 total/summary検査](../experiments/native392-shard05-summary-review-20261009.json)。追加08/09/10は各175・新nonce refs savedV1を受理、[全source/env100項目](../experiments/native392-shards08-10-saved-binding-review-20261009.json)・人工approval11成功。取得時RUNNING/内部Input版ID欠落、実output QCは未成立。CPU最大3進行を維持、残14。
- 05〜07 fresh Range入口は[準備17項目](../experiments/native392-full-range-preparation-20261009.json)と[root独立13項目](../experiments/native392-full-range-root-review-20261009.json)成功、rootでactualSDK人工16呼出/固有9契約とRuffを実行。既監査coreをSHA固定再利用、zero-prefix・最大128MiB厳密206・自動retry0・全体SHA/metadata/disk制約を保持し取得開始。実NPZ検査は取得後の別gate。ledger108→118を旧byte prefix保持で10事象追記。[receipt](../artifacts/research/20261009-native392-p010-progress-record-v1/receipt.json)。
- **10月9日00:04更新**：p009 ref56956158は10月8日23:49:34の公式raw/typed COMPLETE・自己Public0.948・errorなし。[採点10項目](../experiments/p009-scored-review-20261008.json)成功、改善しなかったので非採用。原14自己0.950保持、0.955/strict>.960未達。ledger107→108は旧byte prefix保持。
- p010新V3は二次停止の古いreplay aliasだけを専用p010 command/cache/3checkpointへ固定し、独立34項目/CPU39検査成功。23:51:38 JST private savedV3受理、[保存35項目](../experiments/p010-profile32-savedV3-binding-review-20261008.json)成功。[compact tools独立22項目](../experiments/p010-profile32-results-tools-independent-review-v3-20261008.json)後の一回取得はsource一致・明示ERRORでbody0停止。ログは4controls/全224等の内部検査を通過した後、original/bounded main CSV byte不一致でguard停止。完全SUMMARY未保存なので数値差・完整資源gateを成立したと扱わず、9CSVの限定取得・独立数値比較へ。新tolerance/精度/提出gateは未承認のままです。
- native392分割02/03/04は公式COMPLETE・要約各28項目成功、actual GET200/ZIP206総長一致・原DICOM取得0。04 processedZIP4,774,783,538bytes/SHA一致、[176NPZ/880配列の13実検査](../experiments/native392-shard04-completed-cache-review-20261008.json)成功、valid53〜124/16210、local直読解凍hash検査61.517秒。PC fullarray QA済みは01＋04計352検査。02/03のfullstreamはIncompleteReadで停止、旧partial実保存size2,599,419,904/3,994,025,984bytesを保持。転送カウンタとの差を落とさず、stat＋prefix SHAを新Range取得の位置に使う準備中。05/06/07はprivate CPU受理/保存100項目成功・取得時RUNNING、残17/fullindex待ち、fit0。
- [設計監査](../experiments/f005-native-contrast-design-review-v2-20261008.json)と[公式公開13候補調査](../experiments/post950-official-keyword-public-audit-20261008.json)を追記。既存所見別attentionの再実装を候補にせず、初回last2対照のconfig/seed/fold/teacherは保持。調査最大0.950、タイトル0.957版の実公式0.939、限定範囲の未発見と公開レシピ不存在を区別。

以下の10月8日23:34以前の記載は各確認時点の履歴です。

- **23:34追記**：p010診断savedV2は23:31 exactV2 read-only一回照会でraw/typed ERROR、failureMessage空。[status証拠](../artifacts/research/20261008-p010-profile32-status-v2-observation-v1/status.json)。実ログ原因未確認、候補gate falseのまま。CPU分割02/03/04は23:34 exactV1各一回の[照会](../artifacts/research/20261008-native392-shards02-04-status-observation-v1)でも全3件明示RUNNING、出力QC未確認。下記23:28のRUNNINGは履歴です。

- **23:28更新**：p010 private診断savedV1は実ERROR。original32完走後、観測用Reader子プロセスの`p007_metrics` importが一次原因と[14項目](../experiments/p010-profile32-first-failure-review-20261008.json)で確認。offline decoder pip警告や跨cell stdout probeを原因と混同しない。新V2は観測helperのhash/解決先guardと、同じReaderディレクトリ/PYTHONPATH条件の実子Python preflightだけを追加。[独立30項目・CPU34検査](../experiments/p010-profile32-independent-prelaunch-review-v2-20261008.json)成功、予測literal/他17cells/14/19/Serving21/失敗V1不変。23:25:06 JST API savedV2/kernel137683716受理、[保存版34項目](../experiments/p010-profile32-savedV2-binding-review-20261008.json)成功、23:26 raw/typed RUNNING。実CoAt/T4/追加224/精度・資源gateは未確認、競技提出0。
- 残キャッシュのうち分割02/03/04は128bit nonceのprivate CPU savedV1で起動済み。[保存版100項目](../experiments/native392-shards02-04-saved-binding-review-20261008.json)で正本V2/全cells/埋込ZIP10members/各176UID/whole-group/Gold除外/CPU/OFF/Original/decoder要求一致、取得時全3件raw/typed RUNNING。内部Input版IDと実出力QCは未確認。先頭176以外のcache body取得0、残20未起動、全4207 index/実教師fit待ち。公式競技枠は直近残0、p009 ref56956158は23:25 score/error/rawstatus未返却。0.955/厳密な0.960超を達成したという記録はありません。

以下の22:46以前の記載は各確認時点の履歴です。

- **22:46更新**：p009 private servingは[31保存項目](../experiments/p009-serving-saved-binding-review-20261008.json)と[15可視CSV項目](../experiments/p009-serving-visible-output-review-20261008.json)成功、3UID/13columns/全finite/固定rank融合差0/no fallback。22:36:51 JST savedV1を競技提出、ref56956158・[独立22binding](../experiments/p009-submission-binding-review-20261008.json)成功。初回score/rawstatus未返却、22:42公式日次5件/残0。
- native392 first176は[実SUMMARY25項目](../experiments/native392-shard01-summary-independent-review-range-v1-20261008.json)、[取得processed176/880配列の13独立項目](../experiments/native392-shard01-completed-cache-review-20261008.json)成功。実ZIP4,741,650,198bytes/SHA一致、176unique safeSTORED NPZ/inner6members rawsize cap/各memberと5array SHA/shape/dtype/fingerprint/mask/位置/plane/flags/coverage/zero padding成功。valid53..124/15994、localreaddecompSHAQC59.999秒、Kaggle1627.635秒/peakRSS1,508,470,784B。元DICOM/教師/実fit0、全cache未取得。残23のsameV2 CPU準備へ。[ledger98→101](../artifacts/research/20261008-p009-submit-native176-record-v1/receipt.json)はbyte prefix保持。
- [22:42公式Public上位100件](../experiments/current-public-leaderboard-head-20261008.json)で0.964首位/0.955以上73/strict>.96013、recipe/weights/自己現在順位は未取得。p010は人工CUDAでbase/stock224 raw差0だがTinyEncoderでありactualCoAt/T4/実MRI/精度の証拠ではない。独立peerがcancel時delivery task accountingを検出、remote前に補正中。candidate gateは閉じています。

- **22:24:13 JST、p008 ref56950968もraw/typed COMPLETE・自己Public0.950・errorなし。** [10採点項目](../experiments/p008-scored-review-20261008.json)。p006/p007同様表示改善未確認・非採用、p005を維持。未丸めAUC/個別AUC/隠し実行時間は不明。
- p009 corrected savedV2は公式COMPLETE、[実32件72項目](../experiments/p009-profile32-completed-review-v3-20261008.json)成功。main各64forward/320px/gradFalse/mask差0、reader3model/24forward/B4、2rank融合再計算差0、実Linux fork3task成功。可視log2行を2実行と数えた旧checkerを保持し、source/hook/phase/CSVで補正したv3。[root17項目](../experiments/p009-serving-prelaunch-root-review-20261008.json)から正本19をprivate savedV1/kernel137675752としてAPI起動、可視QA前で競技未提出。3イベントを[ledger95→98](../artifacts/research/20261008-p008-p009-progress-record-v1/receipt.json)に追加。
- native392 shard01のraw COMPLETE/source一致後、取得clientのDTO/一意basename mapping/untrustedAPI size問題を修正。HEAD byte metadataはhold、別one-offでbounded SUMMARY GET1068503bytesとZIP headers-only Range206 `bytes 0-0/4741650198`を確認、ZIP body消費0。JSON報告176/176・1627.635秒・QC514/予算skip14・一時0・archive4,741,650,198bytes。独立source/plan/全archive/176NPZ数値QC前で、残23本/正式fitは未開始。これは提出worker失敗と別の取得client問題です。

- **21:42:29 JST、p007 ref56951892はraw/typed COMPLETE・自己Public0.950・errorなし。** [10項目の採点証拠](../experiments/p007-scored-review-20261008.json)。p006と同じく表示改善未確認で非採用、p005 plain/k16を維持。未丸めAUC/個別AUC/hiddenruntime/fallback率は未取得。[ledger94→95](../artifacts/research/20261008-p007-scored-record-v1/receipt.json)。p008同時照会にはscore/error/rawstatusなし、再提出せず結果待ち。
- native392全24shard V2は[16準備項目/人工9tests](../experiments/native392-full-shards-preparation-validation-v2-20261008.json)、[独立31項目](../experiments/native392-full-shards-independent-prelaunch-review-v2-20261008.json)、[root16項目](../experiments/native392-shard01-root-prelaunch-review-20261008.json)成功。全24 sourceをgeneratorで再構築・canonical/ZIP/plan一致、4207 unique/4102wholegroup/Gold除外、176x7+175x17、8h finalelapsed・exclusive partial・サイズ/予算照合の修正を確認。先頭176/140groupをprivate CPU savedV1/kernel137670873で起動し[保存52項目](../experiments/native392-shard01-saved-binding-review-20261008.json)一致、metadata+decoderV2+competitionのみ。[ledger92→94](../artifacts/research/20261008-native392-shard01-progress-record-v1/receipt.json)。実archive/array QC・資源・decoder・cleanupは未取得、残23本/fullcache取得/fitは未開始。
- p009 corrected savedV2も[全13cells/3helper/Original/T4/OFF/Inputsの54項目](../experiments/p009-profile32-savedV2-independent-binding-review-20261008.json)一致。read-only selector auditでv1/v2別sourceを取得でき、v2とversion指定なしのstatusはraw未返却、SDK既定QUEUED。unsupported形式の試行はHTTPErrorで実job例外とは別。実fork/32件/AUCは未確認。以下は各時点の履歴。
- [SWA Input V1追加重み監査22項目](../experiments/publication-swa-v1-additional-checkpoint-audit-20261008.json)は既存metadata/sourceのみで実施。別seedなし、同seed42の224版は作者linked Public0.945を確認、自己再現ではない。384readerがbyte同一でも224crop/arch/AMP/strictloader/同環境GPU・Gold58の選択再利用は別監査。追加ensemble案はcandidate_gate=falseで、実weights取得/forward/score/servingは未実施。基準320の予測を保った共有volumeの逐次224枝を別p010として設計中。

- **21:19 JST現在**、p005自己0.950が採点済み最良、p006同0.950は非採用、p007/p008は21:16のraw APIにもscore/error/status未返却。再提出せず、worker段階/実時間を推定しない。0.955と厳密な0.960超は未達成。
- p009診断remoteV1は公式ERROR、[実ログの失敗原因](../experiments/p009-profile32-first-failure-review-20261008.json)は最初140mainのPoolで別exec名前空間の`_pre_one`をpickle解決できないこと。160/reader/fusionは未開始。保存ソース/設定[46項目](../experiments/p009-profile32-saved-binding-review-v2-20261008.json)成功は実完走を保証していなかった。修正版19a/v2はModuleType登録をPool/thread終了まで保持してfinally解除、実関数のForkingPickler identityを確認。人工9tests中8成功/Windowsfork1skipとRuff、[root18項目](../experiments/p009-profile32-prelaunch-root-review-v2-20261008.json)成功。Kaggle実fork人工3task＋実32件を必須gateにし、予測literal/19/14は不変、失敗V1を保持。今回はprofile wrapperの修正であり、過去hidden失敗の原因とは断定しない。
- [f005学習CLI/実processed115QA](../experiments/f005-native-training-preparation-validation-20261008.json)を準備し、[独立47項目/人工CPU10tests](../experiments/f005-native-training-independent-review-20261008.json)成功。v2 ZIP25files/config/source/hash-bound gate、累積16と末尾3の総観測教師分母、AMP skip更新数、Gold除外前の教師読取拒否、正epoch/strictsuffixhead/prefixSHA、CUDAfirst guard、全4207 completeindexを確認。実processed最大115窓を一検査ずつ使い、freeze/last2 peak reserved196/704MiB、保存再読込logits差0。実label読取0/optimizerstep0/実fit0で、実精度や同時B16容量の検査ではない。
- [native16追加独立26項目](../experiments/native392-pixel16-result-independent-review-20261008.json)成功。配列形式は維持し、新規moduleで24 whole-group shard/indexを準備中。全件cacheのKaggle実export/PC転送/real fitは未開始。上記失敗と準備/QAの5事象をledgerへ追記し86→91行、[旧byte prefix保持receipt](../artifacts/research/20261008-f005-native-progress-record-v1/receipt.json)。以下は各確認時点の履歴。
- p009修正profileは[独立35項目](../experiments/p009-profile32-independent-prelaunch-review-v2-20261008.json)も通過し、同refのsavedV2/kernel137665364をAPI起動。fresh unreserved GPU17.568h・公式V1ERRORの再照会・source/request hashを保存し[ledger91→92](../artifacts/research/20261008-p009-corrected-profile-record-v1/receipt.json)。初期raw statusなし/SDKQUEUED、実fork/実32件/精度は未確認。キャッシュ24分割は独立監査でfinalelapsed/exclusivepartial/index予算確認の不足を見つけ、未実行V1を保持し別V2を作成中。全件jobはその再監査後に限る。

- 20:35:26 JST、p006 ref56949480のraw/typed COMPLETE・自己Public文字列0.950・errorなし・379703bytesを公式GetSubmissionで確認し[10項目の採点監査](../experiments/p006-scored-review-20261008.json)成功。表示精度改善未確認でk24非採用、未丸め同値/個別AUC/hidden時間/fallback率は不明。旧pending履歴を保持して採点済み1行を追記、ledger79行。
- 20:38:57 JST、p007 visibleCOMPLETEを[11CSV/log検査](../experiments/p007-serving-visible-output-review-20261008.json)で確認し、同じsavedV1を競技提出、ref56951892/直前残2。[提出binding19項目](../experiments/p007-submission-binding-review-20261008.json)成功。servingの実mirror回数/grad/maskはsource-bound推論で、別real32 profileの直接観測と区別。20:42:32のGetSubmissionはrawscore/error/status未返却、SDK既定PENDING、worker不明。
- native392 CPU16は公式COMPLETE。元reader対bounded reader byteQC46成功/大native2skip、162.109秒、一時0、train_series metadata CSVのみ。processed pilotだけ397707087bytesを取得しSHA一致。[独立archive QA](../experiments/native392-pixel16-completed-review-20261008.json)で16 unique STORED NPZ・80配列hash・392/3x48 shape/dtype/fingerprint/no-pickle・mask/位置/plane/flags/zero padding成功。valid53..115/sum1406、local直読解凍hashQC4.720秒、Kaggle peakRSS1,482,244,096bytes。379.283MiBを全4207へ単純平均すると97.390GiB・CPU約11.84hだが、sample外の容量/時間上限ではない。元DICOM再decode/取得・教師学習・fullcacheは実施していない。
- [last2独立静的25項目](../experiments/dinov2-last2-independent-static-review-20261008.json)も成功。one-study144の容量証拠をbatch16の保証と扱わず、両対照をmicrobatch1/effective16へ揃えるCLIを準備。累積lossはeffectivebatch内の総観測教師数で正規化し、欠損教師を0にしない。実fit開始前にactualpixel workloadとstrict save/loadを確認する。

- 20:30 JST現在の更新。[p007 real32結果50/50](../experiments/p007-profile32-completed-review-20261008.json)は全6slot/B4・actual24/48forward・mirror8回・全gradFalse・固定融合差0・peak8,071MiBを確認。診断は提出しない。このgateから修正版16 localv2をprivate serving `rsraki/p007-convnext-mirror` remoteV1/kernel137659997へAPI起動し、[root21項目](../experiments/p007-serving-prelaunch-root-review-20261008.json)、[保存48項目](../experiments/p007-serving-saved-binding-review-20261008.json)を通過。初回status raw未返却/SDKQUEUEDと実worker状態は区別し、可視結果監査後に競技提出する。
- [18a pixel16人工8検査](../experiments/native392-pixel16-preparation-validation-20261008.json)、[独立38項目](../experiments/native392-pixel16-independent-prelaunch-review-v3-20261008.json)、[root15項目](../experiments/native392-pixel16-root-launch-review-20261008.json)が成功。移植helperのshutil importを隔離namespaceで直接呼ぶ回帰で確認し、未実行旧v1/失敗buildv2を保持。正本v3をprivate CPU `rsraki/native392-pixel16-diagnostic` remoteV1/kernel137660730へ起動。same392/3x48・wholegroup16・fold0/Gold除外・教師CSVguard・<=128MiB native基準とのbyteQC最低1回・NPZ往復・1GiB上限・一時領域emptyのruntimegateを維持。実pixelcacheの完了/QC/取得/実fitはまだ未確認。
- [DINO last2実装検証](../experiments/dinov2-last2-implementation-validation-20261008.json)は人工CUDA19契約、14 unittest、6files Ruff成功。generic ZIPから許可された重み/sourceのみを新QA runに抽出し元provenanceとhash一致。3x48全144窓・prefix/norm固定・last2勾配有限非zero・欠損loss非負例化・元CLS/head等価。checkpoint OFF/ONでlogits/grad差0、reserved4,270→806MiB。optimizer0・実ラベル/実MRI fitting0。純計算55.2min/epoch換算はfull144・3,386study条件でI/O/optimizer/validationを除き、valid6の2.60min条件とは混同しない。
- p008提出bindingは[28/28項目](../experiments/p008-submission-binding-review-20261008.json)成功。20:22:29 p006/p008のrawscore/error/status未返却、hidden worker・時間不明。SDKdefaultPENDINGやtotalBytes0から異常/空CSVを推定せず、同じ版を再提出しない。ledgerは既存70行のbyte prefixを維持して8イベント追記、78行。[追記receipt](../artifacts/research/20261008-p007-p008-progress-record-v1/receipt.json)。自己最良0.950・厳密な>0.960は未達成。

以下の待機・未開始表示は、その行に記載した確認時点の履歴です。

- 20:15:16 JST、p008保存V1をAPI code提出しref56950968を取得、直前の公式日次残3。[受理証拠](../experiments/p008-submission-accepted-20261008.json)。[56保存版項目](../experiments/p008-serving-saved-binding-review-v2-20261008.json)、[44実出力項目](../experiments/p008-serving-visible-output-review-20261008.json)が合格。stock2 strict FP32/233 state/20forward・specialist_used=true、元Apex/追加固定rank融合の独立CSV maxdiff0・他10stringsとatomic SHA保持。全12配列/maskはhash-bound実行時検査であり外部数値再計算ではない。[取得ツールunittest8項目](../experiments/p008-serving-results-tools-validation-20261008.json)はpagination・許可list・tie・rank維持・string drift/NaN等の拒否を確認。20:15:39 rawscore/error/status未返却、SDK既定PENDING、hidden段階不明。
- p007 profile32は20:11:37 JST公式COMPLETE/no failure、[独立保存50項目](../experiments/p007-profile32-saved-binding-review-20261008.json)成功。remoteV1の13cells/10code、allsource、helper、正しいlocalv2/両runno_grad、k16/bs4/workers4、同32UID/gold除外/private/Original/T4/offline/pinsを照合。実結果の数値監査は別gateで、profileは提出しない。
- [DINO最終2block微調整の準備監査20項目](../experiments/dinov2-last2-finetune-feasibility-20261008.json)。392px native tripletの保存CLS・192cache情報限界、generic initialization ZIP central-directory名前/size、mask/group/fold/Gold/欠損教師の共通契約、shard予算を確認。16件raw0.989GiB・4207上限260.09GiB、圧縮率は仮定せず、archive＋全展開の二重保持を避ける。新pixelcacheの実native QC/全件転送/実label fit/精度は未確認。人工CUDA＋少数pixelcacheから次の実行条件を測る。

- 19:55 JST、p008 private servingのsaved Version1/kernel137655738をAPI起動。[26静的整合・38 unittest](../experiments/p008-apex-menis10-validation-20261008.json)と[root25項目](../experiments/p008-prelaunch-root-review-20261008.json)成功。全11cells/元7codecell/3helperZIP/再計算源SHA、original k16/no_grad、実64gateとbindingを照合。19:59:39公式QUEUED/no failureで、実併用結果は未取得、競技未提出。rootがsource・実specialist_used=true・stock forward/メモリ/CSVを照合するまで提出gateはfalse。
- p007 v2は[16unittest/人工CUDA実run](../experiments/p007-convnext-mirror-validation-v2-20261008.json)成功。caller grad=True、trainable original head、外側inference_mode無しで12forward全grad=False、caller状態復元、cat→numpyと提出CSV成功。旧v1のdecorator脱落を再現すると同回帰が拒否。[20root項目](../experiments/p007-profile32-prelaunch-root-review-20261008.json)で14/15保護・修正AST・同32UID・既存profile全code除tail一致・埋込みplain/mirror源を直接比較。private診断 `rsraki/p007-profile-32-plain-mirror` remote V1/kernel137656226を初起動。local v2とremote V1を混同しない。

- 19:36 JST、実32/256診断・serving保存ソース/可視CSV監査を経て、APIで `rsraki/p006-convnext-k24` 保存Version1の `submission.csv` を競技提出。ref56949480、直前の公式日次残4。[受理証拠](../experiments/p006-submission-accepted-20261008.json)。19:36:43のGetSubmissionではraw status/Public/error未返却、SDK既定PENDINGのためhidden worker段階は不明。採点済み最良0.950、厳密な0.960超目標は未達成。
- profile256のcompact6出力を全6page/532ファイルのmetadataから取得し、MRI/モデル/coverage512個をローカルへ取得しない。公式COMPLETE/no failure、76独立項目成功、同一256 UID・13列・hash/receipt・6slot・batch4・融合再計算maxdiff0。[256結果監査](../experiments/p006-profile256-completed-review-20261008.json)。main313.74秒、reader16/24 234.96/273.52秒、k24 sampled device peak11,561MiB、RSS合計約10.81GiB。固定順・install/cache差・RSS共有ページ重複を含み、隠し時間や改善AUCの証明ではない。
- serving15の実保存V1は[44項目のソース/環境照合](../experiments/p006-serving-saved-binding-review-20261008.json)、実可視3件は[22項目のCSV/ログ照合](../experiments/p006-serving-visible-output-review-20261008.json)成功。実行infer.pyはk24一行変更版とbyte一致、3checkpoint/reader CSV/Fusion SUCCESS・例外fallback0、最終所見係数rank式の独立再計算maxdiff0。可視完走とhidden採点結果を区別する。
- Meniscus10 stock専門枝の実64件は公式COMPLETE、[32項目の独立監査](../experiments/public0033-profile64-result-review-20261008.json)成功。実9payload SHA/bytes・233 strict tensors・各T4 batch32×10・raw2/full12 CSV・mask再構成を照合。298/384slot有効、選択3557frame decode error0、計219.25秒、device sampled peak7,511MiB/個・allocator reserved約7.2GiB/個。実DICOMは全て非圧縮ExplicitVRLEで、JPEG/JPEG2000/RLEは未実証。公開全weakfitは独立OOFではなく、モデル追加のAUC利得は未測定。N<2/入力欠損時の明示契約と、予期しない例外を隠さない別提出版を準備する。
- 19:43 JST、p007の追加監査でhelper挿入による元runの `@torch.no_grad()` 脱落を発見し、実Kaggle起動を保留。外側inference_modeを使った人工forwardだけでは検出できなかった。builderでdecoratorとsignatureを一単位として扱い、AST＋実run経路のgrad禁止テストを加えた別v2へ修正中。旧未実行package/JSONの合格は当時の検査範囲の履歴で、実行可能な候補とは扱わない。14/p006はこの挿入の対象外。
- p006実提出は[独立46項目](../experiments/p006-submission-binding-review-20261008.json)成功。fresh allowance4、保存Version1のcode提出、受理ref、14/15SHA、事前44/76/22項目gateとfresh COMPLETEを照合。API rawstatus欠損時のSDK既定PENDINGを `ApiSubmission.from_dict` で再現。totalBytes0を空CSVの証拠と扱わず、hidden queue/worker/実行時間は不明とする。

- 18:43 JSTからAPIでp006実行開始。ユーザーがAPI実行・提出を明示許可しており、自己Public>0.960の継続目標は未達成。14基準不変、15のprediction code deltaはindex8のdefault `k=16→24`一行だけ。他6codecells/重み/Input版/主系/融合係数/decoder固定。8 unittest/Ruffと元Dataset/head＋random小encoderの人工RTX4090契約が成功し、実重み/MRIによる精度は未確認。[実装監査](../experiments/p006-convnext-density-implementation-20261008.json)、[人工検証](../experiments/p006-convnext-density-validation-20261008.json)。
- [15a](../notebooks/public/15a_profile_p006_k16_k24_32.ipynb)は固定gold除外32 UIDを教師/Report無しで実行する診断。4 unittest/Ruff成功、observerが元tensor tupleを同一objectで返すこと/失敗後のmethod復元/cohort32が256のprefixであること/モデルsource保持を確認。2回目fusion前のCoAt entry復元、stale own CSV削除、親stdoutのSUCCESS必須・EXCEPTION/FALLBACK禁止、独立rank式再計算を採用して偽成功を防ぐ。全codecellsを実行せずcompile済み。診断submission.csvは最後に削除する。
- APIのprivate profile32 SaveKernelはVersion1/kernelId137644181を受理、18:44 status QUEUED。初回ローカルresponse.ref照合は`/code/` prefix差で例外になったがremote保存は成功しており、再launchしていない。SDK helperをstrict owner/slugのままliteral prefixだけ受理するよう修正し8契約tests成功。実MRI診断完了、256検査、servingの保存/競技提出・スコアはまだ未確認。
- GPU quota元HTTPはallowance108000秒、used42840.065秒、reserved0、残65159.935秒（18.10h）、refresh10/10 09:00JST。SDKのduration.to_dictは丸1日を除外するため6hという誤表示が生じたが、元rawから枯渇していないと訂正。以下の手動計画/未開始は前ターン時点の履歴で、今回の実行許可と区別する。
- 19:00 JSTまでにprofile32公式COMPLETE/summary passedを確認し、小さいsummary＋CSV6を取得。root独立標準ライブラリ再計算37項目成功、同一UID/13列/finite/range/非定数/hashと元係数の最終rank融合のmaxdiff0を確認。[実MRI32監査](../experiments/p006-profile32-completed-review-20261008.json)。全32検査で6slot、batch4最大24slotを実際に処理。sampleGPU peakはk16 8069MiB/k24 11561MiB、CoAt56.52秒、reader16/24 61.29/62.25秒。観測wrapperとinstall/OScache効果があり、純forward速度比やAUC改善とは扱わない。
- 32gate通過後、[15b](../notebooks/public/15b_profile_p006_k16_k24_256.ipynb)を19:00 JSTにAPI保存・実行。private profile256 Version1、19:01 QUEUED、実結果はまだ待機。診断metadata/source/Inputを保持し、profileを競技へ提出せず、serving15の保存/採点へ進む条件を分離する。
- [p007の鏡映候補](../experiments/p007-convnext-mirror-implementation-20261008.json)はbase14/k16でinfercellのみ2view等率平均を追加。他6codecell/主系/weights/係数不変。Sag0/3にuint8channel逆順＋valid position1-p、Cor/Axは幅反転で、mask/所見名保持。7tests＋人工CUDAで二重鏡映画素exact、position誤差2.98e-8、allmissing予測差0、TTA対称差0。実重み/精度/資源未検証。[人工検証](../experiments/p007-convnext-mirror-validation-20261008.json)。native3..64/K16・24の反射centreを全確認し、別K3での丸め差を候補固有の失敗と誤記しない。
- [Meniscus10 adapter監査v4](../experiments/meniscus10-integration-feasibility-v4-20261008.json)は13tests/Ruff、作者cache処理との8人工case byte一致、stock validator、mockDINO＋作者head CPU/CUDAなど12項目成功。Apex volumeの流用でなく独立130mm/336px/12slice/1-99percentile/6slots/lateralityを使う。fullweak4349fitでOOF不可、固定90:10は2Meniscus列だけの未採点仮説、残10列/元CSV保持。実decoder/重み/stockbatch32/T4x2は未検証で64件診断を準備中。

- 18:03:45 JSTの公式GetSubmission(ref56943135)でraw/typed COMPLETE・Public文字列0.950・errorなし。既存29項目の保存source/環境/CSV監査とscript356323020のbindingを維持し、8/8項目成功。[採点監査](../experiments/p005-scored-review-20261008.json)。271位はユーザー報告でAPI直接観測でなく、未丸め値/個別Public AUC/Private/hidden時間/fallback率は未取得。採点成功を旧p003/f002原因の解明と扱わない。
- 最新公式本文/SDK制約も取得。日次5提出・最終選択2、最終10月23日08:59 JST、CPU/GPU9h・Internet OFF、12所見等重みmacro ROC-AUC。teamの残4枠は18:03のUTC当日観測で、今後の枠を保証しない。[条件監査](../experiments/competition-constraints-20261008.json)。Rules本文SHAは今朝と同一。
- [構成監査](../experiments/p005-growth-architecture-review-20261008.json)でCoAtNet96slice/K94/SWA/解剖mirror2view、ConvNeXt3foldEMA・k16/slot・所見別rank融合を確認。native320→384はnominal FOV116.7→140mmを戻す変更で細部sampling密度だけの変更ではない。384モデル再構築/relative-position/strictload/finiteGPU未検証なので、次の窓数変更と一緒に導入しない。
- [最新公開候補監査](../experiments/post950-public-research-20261008.json)はscore順100件・先頭17公式版・4source/log・3Dataset/小来歴5件、13/13項目とRuff成功。確認範囲の最高best0.950、公開0.955以上の採点済み版は未確認。タイトル/ログ保証と公式0.948等を分けた。Gold-Gated/比較版の自分によるgold係数選択は採用しない。
- 公開Meniscus10 V2はDINOv2-Baseの別表現候補だが、contractにallweak4349fit/fold_safe_oof=false、T4x2FP32/厳密336px cacheを確認。既存cache/独立weakCVへの直接接続は不可、単独Public/追加利得は未確認。全体stack0.945を枝の単独値と扱わない。モデル/MRI payloadを取得していない。
- [次段階の仕様](../experiments/p005-improvement-roadmap-20261008.json)はp006のConvNeXt k16→24だけを最初の1候補とし、人工variableK契約→32/256件資源/coverage比較→手動保存/限定採点という準備計画。Conv forward1.5倍は全体時間/VRAM倍率でなく、9h完走・0.955達成は未証明。新Notebook/新実MRI/新学習/外部提出をこの方針相談で開始していない。
- p0050.950の元14と過去採点/研究記録を保持し、PROJ/README/ロードマップ/設計を更新。0.955/0.960は目標とし、Publicの微調整探索・gold選択・公開fullfitをOOFとする評価を除外。自作f002の弱点をApexのPublic弱点と断定しない。
- root最終確認は新規23JSON構文・8文書localリンク381件・自己COMPLETE/score/8checks・研究13checks・制約/目標差の算術・14SHA不変を照合。台帳は旧58行のbyteを保持して採点/調査/計画3イベント追記、14列61一意ID。[整合性記録](../artifacts/research/20261008-post950-strategy-v1/verification.json)。新推論コード/GPUコードを変更していないため、人工forwardや既存学習テスト群の再実行は行っていない。

## 待機中の3作業の実行と判断（10月8日）

- 今回は手順整備でなくユーザーの実行依頼として、取得済み凍結特徴を使うローカルRTX4090のf004両fold実学習を実施。CUDA、計画config/16source/固定入力hash、元f003run/best/predictionhash、新規run未存在を照合。各24epoch、両CLI exit0、head fitting計767.040秒・CLI全体796.26秒。前者は起動/cache検証を除き、後者も事前監査・解析・元MRI特徴抽出/隠し推論を含まない。
- [完成比較](../experiments/f003-f004-fold01-review-20261008.json)の236項目が成功。4run×24epochのCSVhash/13列/UID順/全metrics/再計算masked BCE/更新数、4run×best/lastの計8CPUcheckpoint契約/finite tensor、固定教師・source・過去run不変を確認。GPUコードは変更しておらず、同じ人工forward群を繰り返していない。
- 事前の最小masked weak BCE選択を維持、epoch15/14。fold0 BCE−0.000329/AUC12−0.007871/補助11−0.002223、fold1 BCE+0.003235/AUC12−0.000738/補助11−0.000543。差はf004−f003。両fold利得が揃わず、f004へ差し替えない。各3000回group bootstrapの補助11差95%区間は両方0をまたぎ、改善も一律悪化も統計的確定とは扱わない。
- PF OAの点改善は両fold、MCLは方向不一致、Fractureは小さい改善も区間0跨ぎ。Synovitis fold0は陰性1で12平均bootstrapの1061/3000が未定義となり、定義可能drawだけの区間を全体の確証には使わない。供給group/既使用foldでの診断で、患者独立性・選択不確実性・Public改善は未証明。
- rootの標準ライブラリ独立検証は選択4CSVのmask/所見順/UID集合/有限確率/hash/最小BCE選択を確認し、確率から再計算したBCEが2e−6以内、AUC12が1e−12以内で一致。[独立結果](../artifacts/research/20261008-waiting-work-execution-v1/selected-prediction-verification.json)。初回の検証用header仮定を既存12項目へ修正してから成功し、学習・推論には変更していない。
- [教師17セル二次レビュー](../experiments/training-report-second-pass-review-20261008.json)は20項目と独立整合確認/Ruffが成功。閾値未満2・明示的正常1・直接言及なし3・未確定11。整理6は既存0または欠損と整合し、教師訂正採用0件。原文はprivate新規runに保存し、元packet/固定教師/gold bytes不変、gold/保護fold0・1除外、proposed_label空欄を確認。Report不足の0化・画像GT扱い・目的標本から母集団誤り率推定をしていない。
- p005は17:32と17:45:41 JSTに公式GetSubmissionを各1回read-only照会。raw status省略とSDK既定値PENDING=0を分けて保存し、スコア/エラー未返却、queue/worker段階は未観測。[最後の状態](../experiments/p005-status-final-20261008.json)。約50分のwall clockを隠し推論時間やtimeoutと扱わず、旧submitted/status監査を保持した。
- root最終監査17項目が成功。source16/固定入力/元f003/教師private出力/旧提出監査/14Notebook/3000回bootstrap hashを確認し、独立CSV再計算と比較値が一致。台帳は元53行のbyteを保持して5イベント追記、14列58一意ID、自己Public/gold評価欄は空欄。[全体実行記録](../experiments/waiting-work-execution-review-20261008.json)
- 新しいGPUコード・元DICOM/モデル取得・MRIdecode・教師変更・外部upload/実行/submitを追加していない。実行済みのf004出力を再作成せず、p005の採点結果が返った後に元保存版と実際の結果を分けて判断する。

## p005の実提出と採点待ちの確認（10月8日）

- 公式SDK08:02:55Z（17:02 JST）に最新提出ref56943135/`rsraki/rsna-knee-apex-grandmaster-stack` Version1/script356323020を取得。提出時刻07:55:21Z（16:55 JST）。API応答にstatus/publicScore/errorDescriptionがなく、成功・失敗・worker実行中をいずれも断定しない。[新規監査](../experiments/p005-submitted-review-20261008.json)
- 作者採点済みV1と全7 code cell・全source cell同一、14ともcode同一。rawNotebook SHAの差はmetadata/serializationで、推論差とは扱わない。Original digest・GPU有効/T4×2・Internet OFF・3 Input参照名も一致。公開2DatasetはcurrentV1/availableV1のみ。保存Input内部IDs/competition bundleはSDKに出ず、認証済みprivateUIも404のため、内部版の直接照合済みとは書かない。
- 可視COMPLETE、3checkpoint検出・reader CSV書込み・Apex Fusion SUCCESS、EXCEPTION/FALLBACKなし。小さい3CSVを独立検査して13列/3UID順/36有限[0,1]・非定数・所見別rank融合式maxdiff0等29項目が成功。SWA hashもログで原版一致。89.467秒は可視ログの最終CSV書込みまでで、隠しworker時間ではない。作者と同じlibjpeg導入エラーの後に融合は成功したが実圧縮系列の網羅性は未測定。
- 元source/ログと小CSVのみread-only取得。重み/MRIの追加取得・MRIdecode・学習・外部upload/実行/submitなし。過去の準備・提出失敗・成功の記録を保持した。
- ユーザーがreader/SWAの両InputともVersion1と回答し、ユーザー確認として監査へ追記。API内部版IDの直接観測とは区別する。
- f004は両output directoryが未作成。現CLI `train --help`で既存手順の全引数を確認し、doctorでPython3.12.13/torch2.10+cu128/CUDA利用可を確認。CUDA必須CLIは維持、学習は開始していない。既存f003の約356/358秒はhead fittingのみで、次のf004の速度保証ではない。
- 待ち時間の教師作業として元30セルから未確定17だけのprivate packetを新規作成。MCL5/PF OA7/Effusion2/Synovitis3、17unique studies。元CSVの列/値を維持し、proposed_labelとsecond_pass9欄は空欄。元manifest/source/入力/出力hash、gold/保護fold除外、既存component/Report hash、原値保持、書込み後の不変等13assertが成功。新しい医学的判断・翻訳・教師変更・学習は行っていない。[引渡し集計](../experiments/training-report-second-pass-handoff-20261008.json)
- 提出結果に依存しない作業の優先順位・f004採用条件・結果後の分岐を[待機中の計画](../experiments/p005-waiting-work-plan-20261008.json)へ保存。公開3件から精度・補完性を推測せず、公開重みの既知訓練画像を独立OOFと扱わず、他foldの訓練済みheadを平均して当該foldの改善と扱わない。13は任意のf002診断で、p005採点の前提ではない。
- rootで新規JSON・private資料hash・17組一意/記入欄空欄・f004設定hash/未実行・更新6文書のlocalリンク326件を再検査。台帳は元50行のbyteを保持して今回3イベントを追記、自己Public欄は空欄。10項目成功。[最終確認](../artifacts/research/20261008-p005-waiting-work-v1/verification.json)

## 10の可視完走と隠し提出失敗の再監査（10月8日）

- 公式SDK06:46:29Z（15:46 JST）、提出ref56904146/script355977821/Version1/`rsraki/notebook428d1e17f8`。全cellは配布10と一致、GPU ON・Internet OFF・Competition/feature/decoderの3 Inputs、旧head Datasetなし。可視COMPLETE、添付とroot summaryがbyte同一。一般例外・Publicなしであり、SDKのCOMPLETEだけを採点成功と扱わない。[22項目の監査](../experiments/f002-embedded-head-hidden-failure-review-20261008.json)
- 保存CSVを独立に確認：3 UID/order、正確な13列、36個のfiniteな[0,1]確率、hash、receipt、固定checkpoint/featurefingerprint一致。元07の完了exportと入力CSV hash/UID順も一致。処理20.300秒、CPU peak RSS4.179GiB、PyTorch peak reserved364MiB、最後の空きdisk19.501GiB。可視3件の推論セル計測で、Notebook全体・queue・隠しworker時間ではない。
- 4圧縮syntaxesの人工画像はKaggleでdefault/pylibjpeg画素完全一致。実MRIの先頭9系列headerは非圧縮。GDCM missing表示だけからdecoder欠落を断定しない。追加JPEG/JPEG-LS/RLEのlibrary availabilityは既存Windows同pin環境で確認したが、追加形式のKaggle画素/実MRI網羅性とは区別する。
- 独立人工1,300検査・5,842系列metadata・実random CPU headでは9提出不変条件が成功、UID順/全15,600確率/Reportとtrain不要/NPZなし/skipなしを確認。MRI/encoder/decoderはmockなので時間5.664秒を実推論速度と扱わない。[独立監査](../experiments/serving-contract-independent-review-20261008.json)
- spacing欠落・物理位置重複・一定強度で選択系列が検査全体を停止する人工ケースを再現。native形状・スライス長に応じてraw/stack/percentile/float64全体正規化配列が共存する。既存4,207件exportでは全12,621選択系列が正常に処理済みで、private真因は未確定。nonfinite header contextがstrict JSONを二重に失敗させ診断を隠す経路も観測した。これをvalid系列の失敗原因とは扱わない。
- `frozen_test_input.py`は実test.csv/test_series.csv/test_seriesの存在を満たすrootを選び、aliasを同一視し、複数の実rootは拒否。既存Notebookのexistsだけのroot選択に弱点を人工再現した。5 unittest成功・Windows symlink1件skip。メモリ変更の診断には同時導入せず一要因を保つ。

## 公開候補の採点版監査とp004の準備（10月8日）

- 公式版別view-modelのlinked submission ID/sourceScriptVersionId/scoreを照合し、0.950/0.949/0.946/0.945/0.944を確認。タイトル.950だけで実採点.809の例は候補から除外。自己スコアは全候補未測定。[primary audit](../experiments/public-candidates-20261008.json)
- p004はgoodpjw2008採点済みBest Version2/script355300588/submission56837602/0.944。元sourceSHA`582f4a1c768cb4b347483797cfff4dc2a3e7648f0bb603d8e878eeb9ccc2695f`、29 code cell・19公開Inputs・Original31430/digest`37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`/Python3.12を固定。元CP311/CP312 wheelsを最新CP313で代用しない。推論・係数・元fallbackを変更しない12とmanual manifestを作成。Copy & EditのV2 fork保証は未確認なので保存後にcode/Input/containerを照合してから手動提出する。
- .946/.945は必須private Inputs3つで完全再現不能。.949/.950はOAI2,399knees外部教師とGold58選択の作者記載を確認。直接データアクセス条件と公開派生checkpoint利用可否は分けて監査し、違反と断定しない。reader DatasetのApache2.0をNotebook全体へ転記しない。
- Kaggleでのp004実MRI/隠し完走・自己Public、f002メモリ診断の実行は未検証。実モデル/MRIの取得・学習・外部upload/submitは実施していない。

## f002メモリ配送だけの変更と13の検証（10月8日）

- 新`bounded_feature_reader.py`とadapterを追加。nativeとquantile scratchをtemporary float32 memmapへ置き、percentileは元全volumeのlinear規則、必要スライスのみ同じfloat64算術で整数化。physical順/spacing/letterbox/triplet/系列選択/encoder/headを維持。geometry不適合のskip・架空尺度・確率代替は追加しない。16core source/runtime/featurefingerprintは不変だがruntime reader bindingは変更しており、新reader/adapter hashを明記する。[検証記録](../experiments/f002-memory-reader-review-20261008.json)
- 人工DICOMのoblique/anisotropic/signed/負RescaleSlope/MONOCHROME1/59→48triplet/single-slice等で入力windows・位置/coverage・全5特徴配列がbyte同一。既存f002学習済みfold0 epoch11 headでRTX4090の12確率差0、提出契約valid。実DINO encoder/MRIは使用せず、実学習なし。
- reader10＋profile adapter5の15 unittestがNumPy2.5.3とKaggle同版2.1.3の既存Windows環境で成功。人工GPU streaming5件で成功/失敗summary、reader binding復元、owned temporary cleanup、既存診断保持、test mode拒否を確認。Ruff5files成功。KaggleLinuxで新readerの実MRIは未実行。
- 13は519,132byte/SHA`c5b0f73497a64fba20a785c10ade5951a40da2f744412782611542bf3a167515`、profile_train/seed20261007/limit1300/fixedweakmanifesthash。元10のstartup/head/runtime/decoderセルを維持、root+nestedsummaryに追加reader provenanceと資源events。既存train/weak CSVは全列を読むが、UID以外のReport/target値はfit/推論/評価へ消費しない。profile CSVは提出不可。head含有の13は個別Git除外。
- 一時disk予算はnative float32×2＋256MiB余裕、コピーchunk≤8MiB、success/errorでworkspace掃除。file-backed percentile partitionのRSS増加やdisk/IOは残るためconstantRAM/OOM解消を保証しない。無作為固定1300は全4207/隠しprotocol/最大nativeサイズの網羅ではない。root選択helper/geometry代替/診断JSONのNaN修正を同時導入せず、一要因を保つ。

## 公開pretrained例外の再監査と次のp005（10月8日）

- 最新公式Rulesの本文を再取得し10月4日取得と一致、本文SHA`9105f8f81f96c02aca988197c175d699052be8c06419b21131dccdd426bc404d`。外部無料公開modelの許可と事前学習入力のwinner-license例外を確認。HostのOAI回答は元datasetアクセスの条件で、公開checkpoint個別の禁止/承認は記されていない。SWA配布cardの大会利用許可と合わせ、元OAIを取得しない公開checkpoint固定推論として条件付き採用へ更新した。[再監査](../experiments/public-pretrained-eligibility-20261008.json)
- 作者0.950のApex V1/script356192954/submission56930357をp005固定対照14へ準備。元7 code cell、3公開Input内部版、Original31481/Python3.13/digest`2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`を保持。元codeのdecoder pip失敗後のcontinuation/series fallbackは変更しない。作者の採点済み成績と、可視ログでのcodec網羅性は別。自己採点/隠し完走は未確認。[引渡し監査](../experiments/p005-public-control-handoff-20261008.json)
- 14は133,344byte/SHA`75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546`。新5 unittest・Ruffが成功し、利用条件JSONのdecision/source/score/Input snapshotへのbinding、source tamper拒否、7セル保持とwritefile構文、12 hash/size（p004不変含む）を検査。原作者の可視ログにreader3checkpoint検出・`_own.csv`書込み・Apex Fusion SUCCESSを確認し、EXCEPTION/FALLBACKはなし。Kaggle実行前の静的検証であり、自分の実MRI完走ではない。
- `.950を使える`はモデル公開条件に基づく記録された判断で、Host個別承認、作者元OAI取得の適格保証、自己Gold未利用モデルという主張ではない。独自f002/f003のgold非利用と公開重みgold選択履歴を区別し、帰属・配布条件を維持する。
- root独立14項目で16core/元runtime/採点済みp002/失敗10の不変、12/14全code cellの原版一致、13のstartup/head/decoder不変、全新Notebookの空outputs/hash、追加したlocal document links35件を確認。台帳の既存44行をbyte prefixとして保持して6行を追記し、自己Public欄へ作者scoreは入れていない。過去run・初期prep packageも保存。[root独立記録](../artifacts/research/20261008-f002-submit-failure/final-workspace-review.json)

## 09の起動前接続失敗の再発と全体状況の再監査（10月7日）

- 公式SDKの05:58:56Z（14:58 JST）で、ユーザー指定`rsraki/notebook6372fe6490` Version1/script355963229はERROR。全source cellは配布09と一致、GPU有効・Internet OFF・Competitionと3Datasetsの4 Inputsが揃う。同じhead Dataset12402506/内部版20385564のKaggle管理下`/tmp/kglt/192.168.5.2/...`でmkdirがRead-only file systemとなり、Output0・visiblelog`[]`。Python・decoder・画像・モデル処理は未開始。[21項目の確認](../experiments/f002-recurrent-mount-review-20261007.json)
- 08失敗355951711→CPU確認355955526→GPU256完走355957807→09失敗355963229の経過を保存metadata/source/status/logから照合。今回初めて取得したCPU確認の保存sourceは配布08aと一致したが、実設定はCPU・全4 Inputs・Internet ONだった。従来の添付JSONのファイル内容監査は有効だが、head単独/Internet OFFの切り分けを実施したという証拠にはならない。古い添付監査は変更せず新記録で補足した。
- 公式提出一覧でp002 ref56840796・Public0.937を再確認し、成功保存版sourceも維持。p003の2提出と旧f002 ref56897486はerrorDescriptionあり・Publicなし。一般例外を9時間超過や共通原因と断定しない。DatasetのAPIreadyは各runtimeのmount成功を保証せず、exact backend欠陥とKaggle全体障害の公式根拠は未確認。[一次資料の調査](../experiments/kaggle-runtime-primary-research-20261007.json)
- 元head340,745byte/SHA256、16core moduleを再検査し不変。今回の取得は保存source/log/metadataのみで、重み・MRIの追加取得、MRI処理、実学習、外部アップロード/提出は行っていない。元台帳を新規research directoryへ保存し、以前の実験/失敗/完走記録を保持した。

## head Datasetを接続せずに復元する10の準備・検証（10月7日）

- [10_submit_f002_embedded_head.ipynb](../notebooks/public/10_submit_f002_embedded_head.ipynb)と生成スクリプト/stdlib復元helperを新規作成。元head ZIP3ファイルをbase64で埋め込み、全件のallowlist・サイズ・SHA256を検証して新規workingディレクトリへ復元。497,221 UTF-8 bytes、Notebook SHA256`8ca99ca50a5ddd60eef67ce7ec5a1c606d23194686b515e94f481b97680dc8ba`。1,000,000byte未満の保守的生成guardを置いた。ソース1MB制約は実サーバーエラーの一次体験報告であり、現在の公式一律上限確認とは区別する。[準備監査](../experiments/f002-embedded-head-handoff-20261007.json)
- 09のsetup/runtime/decoder cellは不変。run_streamingの引数はhead pathのbindingのみ変更し、summaryへdelivery情報を追加。固定fold0 epoch11/checkpointSHAと16core source、featurefingerprintを維持。手動Inputs manifestはCompetition＋元encoder/code＋元decoderの3つ。旧head Datasetを接続しない手順をNotebookとdocsに明記した。metadataだけではUIのInput参照が削除されない点も確認。
- 新規unittest6件が成功。復元byte一致、base64/hash/サイズ破損で書込み前に拒否、path allowlist、既存成果物保護、Notebook構文/空outputs/元cell不変/固定引数、core変更拒否、実checkpointを使った人工GPU提出契約を検査。RTX4090・人工17×3×48×384特徴で元headと復元headの確率最大差0、17×12・finite[0,1]・target順一致。MRIなし・実学習なし。Ruff3ファイル成功。
- rootの独立静的監査16項目も成功。Notebookのコピー/hash/サイズ/3Inputs、oldhead codebindingなし、固定call/cell/runtime/core、空outputs、検証記録を照合。元08/09は変更せず、新規`artifacts/kaggle/f002-embedded-head-v1-20261007/`へmanifest/検証記録を保存。head含有10は`.gitignore`へ個別登録してGit除外を確認。GPU演算は変更せず、旧21テスト群の再実行は行っていない。
- Kaggleの10可視/隠し実行と自己Publicは未確認。他Inputsの起動障害や旧07隠し例外の真因はこの配送変更では確定しない。[現在の操作](frozen-features.md#head接続の再発後は10を使う)。今回assistantは実モデル/MRI取得・実MRI処理・実学習・外部upload/submitを開始していない。

## p003の規模/失敗経路監査と新11の準備（10月7日）

- 旧64件のDINOcache520,224,768byteはRAM経路のみ。既定1GiB境界からDINO133件、Rad4slot669件/3slot892件でmemmapへ切り替わることをsourceと整数算定で確認した。1300件ならDINO9.841GiB、最大Rad2layout重複3.888GiBだがstageは逐次で、合計を全体peakや20GiB超過の証拠にしない。CoAt ready bagの有界payloadも全process RSS上限ではない。[静的監査](../experiments/p003-scale-audit-20261007.json)
- 元作者cell27はCoAt失敗を捕捉し残りで継続し得る。追加guardは全4系統/fallback0・不良eventを拒否するため、失敗時の完全再現ではない。正常な欠損layoutは既に許容し、予測代替を発生させるpreparation/inference例外とは区別した。この差が自分の隠し例外の真因とは未確定。過去Input版・作者採点CSVの同一性も未確認。
- [11_profile_p003_stress_256.ipynb](../notebooks/public/11_profile_p003_stress_256.ipynb)を新規作成、398,454byte・SHA256`d6ea88692faa5957712efa812ce86eb5505ace3947361e36b4ed1ff266e77308`。05の全22推論セル・14Inputs・モデル・前処理・AMP・microbatch・混合式・Raptor順は保持し、seed20261005/256件コホート・観測だけを変更。reference再計算と提出cell29は除き、profileのみ。元08/09/05/vendor/coreを変更せず`artifacts/kaggle/p003-stress256-v2-20261007/`へ保存。
- 2秒の親RSS/HWM・cgroup current/limit/kernelpeak/OOMevents・disk初終/最小空き・FS種類、array storage/range/SHA、child成否/例外tail、元event分類を記録。cgroupv1/v2/missingを分け、観測不能ならnull/availability/measurement_status=incompleteを保存して、完走モデルを観測不足だけで例外にしない。親CUDA統計はchildを含まず、worker receiptもGPU全体監視ではない。
- 新規人工unittest11件と3ファイルRuff check/formatが成功。縮小した元allocator132/133bytesでRAM/memmap/ゼロ画素、pixel hash/range非変更、tuple/mask/child例外identity、typed source event、cgroupv1/v2/missing・観測欠損時の完走summary、元推論cell/非提出/referenceOFFを検査。GPU演算コードを変更せず、実MRI/モデルを使うforwardやstressは実施していない。rootもNotebook/copy/hash/22cell一致/空outputs/構文を独立確認。
- 256件ではDINO1.938GiB memmapは通るがRad669/892境界・約1300件全体・隠しgeometry/compressionを保証しない。全cachehashの診断負荷を実測高速化率にしない。11は10の前提ではなく独立したp003診断で、出力CSVを提出しない。実MRI取得/処理・実学習・外部upload/submitは未実施。
- 今回の台帳追加は09再発/10準備/11準備の3eventで、14列44一意ID。作業前41行はbyte prefix/CSV内容とも保持。PROJECT/README/手順と検証範囲を更新し、旧手順を履歴表示・現在10へのリンクに整理した。

## 08・256件診断の完走と添付summary/CSV監査（10月7日）

- 添付`F002_STREAMING_SUMMARY.json`を新規 `artifacts/research/20261007-f002-streaming-profile-review/`へbyte同一で保存。SHA256 `a1d5ebc7151cf8bcb11bc2f7c0c4a8d0723962e23591d0a7774456d76299a786`。公式SDKの05:35:20Z（14:35 JST）確認で`rsraki/notebook245423d388` Version2・COMPLETE・GPU有効・Internet OFF、全source cellが配布08と一致。保存Output5件のobject版ID355957807、添付とroot保存summaryのbyte一致を確認。saved source/metadata/logと小さい4 JSON/CSVだけを取得し、MRI/重みは取得していない。[32項目の監査](../experiments/f002-streaming-profile-review-20261007.json)
- Summaryはprofile_complete_not_for_submission/complete、profile_train、studies=processed=256、seed20261007、skip/fallback/intermediate feature file0、Report不要、trainingfalse/Publicnull。f002 fold0 epoch11 checkpointSHA・featurefingerprintと学習記録が一致、ローカル16core sourceも維持。root summaryとnested summaryはdecoder_installの追加以外同一。元Version1のmount失敗は今回GPU保存実行で再現していないが、詳細な一時障害原因は未確定。
- decoder manifestSHAと全3wheel metadataがローカル固定版に一致。報告されたtorch2.11.0+cu128/NumPy2.1.3/pydicom3.0.2が導入前後で維持され、4形式のavailable/pylibjpeg利用とdefault/pylibjpeg両経路の人工画素完全一致が成功。Kaggle Linuxで導入・人工圧縮decodeを実行した保存結果を確認できた。別pluginのGDCM不足表示は利用可能decoder不足を意味しない。
- 保存`profile-studies.csv`と`profile_predictions.csv`の実byte SHAを再計算しsummaryに一致。前者SHA`42e4d2577f8ca2e06b29ff957606b8949406781823e93361586e0553b2435a93`、後者SHA`997878342b38ef4418beedbe74afd82fe43c10d177a39622e0d53c085c8be86d`。固定gold除外weakリストのseed20261007 SHA順先頭256 UIDを再構成し、両CSVのUID/順序と一致。12所見header・3,072有限[0,1]確率、全所見非定数、独立validate_submission receipt={studies256,targets12,validtrue}が成功。Outputにsubmission.csvはなく、profile CSVは提出しない。
- Tesla T4、計測748.383202601秒、平均2.923373852秒/検査。CPU peak RSS4.426090GiB、選択GPU PyTorch peak reserved364MiB、最後のdiskfree19.500645GiB。run_streamingの計測はencoder読込み/画像処理/head/診断IOを含み、前段code/decoder installとqueueを含まない。elapsedとperstudyは別時点のtimerで約0.0005秒ずれることを実装から確認し、監査は0.1秒以内の読み取り差を許容した。全GPU/driver/processメモリや旧版比のspeedupとは扱わない。
- transfer_syntax_countsは各selected series先頭headerの集計で、768件すべてExplicitVRLittleEndian（非圧縮）。圧縮decoderの動作確認は人工数値画像であり、今回の256件で実圧縮MRIを読んだとは主張しない。weak trainの256件はhidden分布/規模・Public accuracyの検証ではなく、09可視実行・隠し完走と旧07例外解消は未確認。
- 台帳へ新しいVersion2完走eventを追加し14列41一意ID。元失敗版・head単独確認・前台帳・添付を保持。Notebook/前処理/GPU/学習コードを変更していないため、既存21 unittest/人工GPUforwardは繰り返していない。assistantは既存source/logと小さいJSON/CSVのread-only取得・監査・記録更新のみで、実MRI decode・実学習・モデル取得・外部アップロード/実行/提出を開始していない。次工程は同じInputs/重みで09を新規private Notebookに保存実行し、CSV契約を確認後にユーザーが手動提出すること。

## 添付head単独接続チェックの成功確認（10月7日）

- Downloadsの`F002_HEAD_MOUNT_CHECK.json`を新規`artifacts/research/20261007-f002-head-mount-success/`へbyte同一で保存し、添付SHAを記録。passed/complete、元head slug、期待mount path、model_loaded=false/training_performed=false、元ZIP SHA、全3ファイルのcoverage/size/SHA、添付copy一致の10項目を確認した。[監査](../experiments/f002-head-mount-success-review-20261007.json)。probe保存版URL/Versionと実ログは未取得で、出所はユーザー添付JSONと区別する。
- best.pt340745 bytes/SHA`189ecab25e0d1c029e974ee23f3c46239e5eceef290b5dca33f6831dcb8e1b03`、config.json855 bytes、checkpoint-provenance.json5959 bytesの全SHAを、ローカルの変更していない元ZIP payloadから独立に再計算して一致。matches=trueを読むだけにせず元bytesのhashと照合した。
- この報告された確認実行では元head Inputを接続し全3ファイルを読めた。前のmountエラーはこの実行で再現しなかったが、一時障害の詳細原因や全GPU環境の復旧を証明しない。head再アップロード・再学習は不要として、同じ4 Inputs/新しいGPUセッション/Internet OFFで08へ戻る手順に更新した。
- Python3.13.15は以前のKaggle runtime版と同じだが、今回model load・decoder・MRI推論・256件処理は行っていない。08全Inputs接続、09完走、隠し例外解消、自己Publicは未確認。assistantは添付JSONと既存archiveの検査のみで、実データ/モデルの取得・MRI処理・学習・アップロード・Kaggle実行/提出を行っていない。Notebook/GPU/学習コードを変更せず、既存テストを繰り返していない。台帳は14列40一意ID、元の失敗記録と前台帳を保持した。

## 08のhead Dataset接続失敗と復旧準備（10月7日）

- ユーザー提供URL `rsraki/notebook245423d388`の保存Version1を公式SDKでread-only取得。04:53:09Z（13:53 JST）にERROR・failureMessage mount data: ERRORED_MOUNTING_DATASET、Output0、実行logは`[]`（2 bytes）を確認。failure pathのscript355951711、Dataset12402506/内部版20385564はユーザーの貼付と一致した。[監査](../experiments/f002-input-mount-failure-review-20261007.json)
- 保存source全cellは配布08と一致、GPU有効・Internet OFF、Competitionとencoder/code/head/decoderの4 Inputsが揃っている。Kaggle管理下の`/tmp/kglt/.../datasets/12402506/20385564`にmkdirする処理がRead-only file systemで停止。Notebook Python・重みロード・decoder導入・MRI処理・予測に未到達なので、コード内try/exceptでもsummaryは生成できない。07の隠し推論例外と共通原因とは断定しない。
- head Inputの公式API状態はready/current_version_number1。metadataのDataset ID12402506、private・owner rsrakiを確認し、best.pt340745 bytes、checkpoint-provenance.json5959 bytes、config.json855 bytesの3-file metadata取得が成功。これでremote payload hashや全runtime replicaのmountを確認したとは扱わない。Datasetや重みは追加取得していない。
- 既存head ZIP317253 bytes、SHA256 `3cce03fda576e4af3d6e36b9dee99ab541afc6949dd807f81f7bdf5b44dad24a`のCRC・allowlist・全payload hashとAPIのファイル名/サイズ一致を確認。best.pt SHA256は従来f002 fold0 epoch11の`189ecab25e0d1c029e974ee23f3c46239e5eceef290b5dca33f6831dcb8e1b03`。元ZIPを保持し、必要時だけ同一内容の別private Datasetを手動作成する回避案を記載した。
- 新規[08a_check_f002_head_mount.ipynb](../notebooks/public/08a_check_f002_head_mount.ipynb)はCPU/Internet OFF、head Input一つのみ。標準ライブラリで接続・3ファイルサイズ/hashを確認し、F002_HEAD_MOUNT_CHECK.jsonを保存する。重みロード・MRI・学習・推論・提出は行わない。Notebook outputsなし・execution_count空・code compile成功、immutable copyとhashを保存。Kaggle上では未実行。
- 前処理/GPU/学習コードと元08/09、run/checkpointは変更していないので、既存人工GPUテストや21 unittestを繰り返していない。新規研究dir `artifacts/research/20261007-f002-input-mount-failure/`へsaved source/metadata/log/API観測、既存ZIP確認、新規probe copy、変更前台帳を保存した。台帳は14列39一意ID。実データ/モデル取得・MRI decode・実学習・再実行・外部アップロード/提出・サポート連絡の送信はassistantが行っていない。
- [POSIX mkdirのEROFS説明](https://pubs.opengroup.org/onlinepubs/009695299/functions/mkdir.html)と[Kaggle Notebook文書](https://www.kaggle.com/docs/notebooks)を確認。現在の同一mount障害のKaggle全体発生を裏付ける公式情報は得ていない。内部host/cache/storageの詳細原因、fresh session/再追加/別名Datasetで復旧するかは未確認。復旧後に08の256件診断、さらに09で隠し成否を確認する必要がある。

## f002隠し例外の調査・08/09準備・f003完成・教師根拠監査（10月7日）

- 公式SDKの04:04:22Z（13:04 JST）照会で、提出ref56897486・scriptVersionId355923367・file_name=submission.csv、COMPLETEだがerrorDescriptionあり・Public nullを確認。04:08Zに保存Version2のsource/metadata/logを新規取得し、全source cellが監査済みV2と一致、GPU有効・Internet OFF、Output8件・公開3検査完走を確認。ログ13930 bytes、SHA-256 `bbb917685a7ebca46b4387c086a7d83380e36a41754d43d7cc02e2fea3335aeb`。private traceback/正味時間は取得できず、原因は未確定。提出から確認の約1時間48分はqueue込みで、9時間超過の証拠ではない。[失敗と対処の監査](../experiments/f002-hidden-failure-review-20261007.json)
- 元推論を静的に調べ、Report必須・公開3検査固定は認めず、圧縮decoderの網羅的事前確認がないこと、strict geometry等の例外が全体停止となること、NPZ stagingと全件metadataの繰返し書込みを確認。これらは調査候補で、隠し原因を確定していない。Kaggleの[公式エラー分類](https://www.kaggle.com/code-competition-debugging)、[大会データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)、[pydicom対応表](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html)を10月7日に確認した。
- 新規[scripts/frozen_streaming_runtime.py](../scripts/frozen_streaming_runtime.py)と[Notebook作成script](../scripts/build_frozen_serving_diagnostics.py)を追加。元16 core moduleとencoder/feature契約・f002 fold0 epoch11 checkpoint SHAを保持し、検査単位の特徴→head予測、decoder preflight、CSV全件coverage、phase/header/traceback/resources記録を実装。Report不要、NPZ中間ファイル0、skip/偽予測なし。08は固定weakからseed20261007・256検査のprofileで提出CSVを作らず、09はtest全件。元の訓練CLI CUDA guardを維持する。
- [08](../notebooks/public/08_profile_f002_serving_256.ipynb)と[09](../notebooks/public/09_submit_f002_streaming.ipynb)は未実行・outputsなし、全code cell compile成功、profileのcompetition-rerun guard、固定checkpoint/weak/source hash、root summary保存を確認。[最新Notebook梱包v3](../artifacts/kaggle/f002-streaming-v3-20261007/package.json)を保存し、過去の梱包v1/v2と元07を上書きしていない。
- 新しい[decoder ZIP](../artifacts/kaggle/dicom-decoders-py313-v2-20261007/rsna-dicom-decoders-py313-v1.zip)は5,331,559 bytes。公式PyPI Linux x86_64/Python3.13 wheel（pylibjpeg2.1.0、libjpeg2.4.0、openjpeg2.6.0）のpublisher SHA256とlicense filesを確認し、`--no-index --no-deps`でtorch/NumPy/pydicomを維持する。依存wheelだけを取得し、実データ/重みを取得していない。元NumPy2.5.3/torch2.10.0の訓練venvは変更せず、Windowsの人工decoder確認はNumPy2.1.3/pydicom3.0.2を入れた別venvで実行した。
- 自作16×16・uint16の数値画像をJPEG Lossless2形式とJPEG2000表記2形式に包み、Windows Python3.12の対応pluginでdefault/pylibjpegの両読み込みが元画素に完全一致した。患者画像ではない。[確認記録](../artifacts/research/20261007-f002-hidden-failure/windows-default-and-plugin-phantom-verification.json)。Kaggle Linux binary実行は未検証で、08/09の最初に同じ人工画素チェックを必須にした。実MRI画素/feature parityや全圧縮表現の正しさをこの4例で保証しない。
- `python -m unittest discover -s tests -p 'test_frozen*.py' -v`は**21件成功**。新規6件で、17人工検査のCUDA streaming/バッチ予測・Reportなし提出CSV、missing decoder/geometry/wrong checkpointで最終CSVを出さずfailure summaryを残すこと、profile subsetと非提出契約、Notebook compileを確認した。既存15件も成功し、含まれる学習は人工データの1 epochのみ。Ruff check/formatは新規3ファイルで成功。
- 既存**実f002 checkpoint**をGPUへ読み込み、人工17検査×3系列×48中心×384次元・可変maskを従来batch16+1と検査ごとのheadで比較。最大確率差**1.1920928955e-7**、絶対許容2e-6、finite/[0,1]が成功。RTX4090/torch2.10.0+cu128/NumPy2.5.3/Python3.12.13。[実重み・人工特徴の記録](../artifacts/research/20261007-f002-hidden-failure/actual-head-streaming-artificial-verification.json)。encoder/MRI/Kaggle/privateの実行やspeedup測定ではない。
- ユーザーが完成したf003 fold0/1のrun.jsonを別保存し、config一要因・環境/入力/source/fold一致、48 epoch CSVのhash/UID/順序/metrics/masked BCE、最初の12 epochの予測byte一致、BCE最低/最初tie選択、best/last CPU checkpointのfinite/契約等**136項目**を確認。fold0 selectedepoch15、BCE0.316232・AUC12 0.823060・aux11 0.820611。fold1 selectedepoch12で元f002の予測/model tensorと同一、BCE0.317469・AUC12 0.815430・aux11 0.808136。goldは学習/選択/評価に不使用。[完成比較](../experiments/f002-f003-fold01-review-20261007.json)
- fold0の供給group paired bootstrap3,000回、seed20261007でaux11差+0.008839677の95%区間[+0.001516712,+0.016844273]、全抽出で定義可能。AUC12は1,061抽出未定義、1,939定義可能への条件付き区間なので頑健性を断定しない。fold1はselected予測が同一なので同じbootstrapを繰り返していない。両foldで延長利得が揃わないと判断し、09はf002を維持する。[bootstrap](../experiments/f002-f003-fold0-bootstrap-20261007.json)
- 現行CSVだけから、goldと保護fold0/1・同一Report/供給groupの推移的連結を先に除き、学習側150検査/1,800セルを実抽出。gold/holdout混入0、原queue出力hashを保持。重点30セル（MCL9、PF OA7、Effusion7、Synovitis7）の原文部分文字列と一次判断を別private CSVに保存し、proposed_label全空欄・教師変更なし。[集計](../experiments/training-report-evidence-review-20261007.json)。17セルは未確定で誤ラベル数ではない。[Host定義](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)の低度MCL/古い変化、OA程度＋範囲、Effusionの量等を確認し、言語/閾値を再確認してから教師だけの比較とする。
- [f004設定](../configs/experiments/f004-dinov2-attention24-dropout40.json)は完成f003に対してreasonとhead.dropout0.2→0.4だけ変更、validatorとfeature契約一致が成功。24 epoch/seed/教師/入力/source/LR/fold/BCE選択を固定し、両foldの新規runが未作成と確認。[未実行計画](../experiments/f004-dropout40-plan-20261007.json)へbaseline checkpoint/予測/環境/hashを固定した。
- 最終handoffは9項目が成功：16 core hash、Notebook生成内容/未実行状態/梱包copy/hash、decoder ZIPのCRCと全wheel/phantom hash、元教師queue hashと30提案空欄、台帳14列38一意ID、変更6資料の全ローカルファイルリンク、git diff --check、f004両run未作成。[検査記録](../artifacts/research/20261007-f002-hidden-failure/handoff-verification.json)
- 既存run/checkpoint/元教師/元queue/過去evidenceを上書きせず、新規研究dir `artifacts/research/20261007-f002-hidden-failure/`にraw SDK source/log/状態、監査script、変更前台帳を保存。今回assistantは実MRI decode・実学習・gold評価・元DICOM/モデル追加取得・アップロード・Kaggle提出を行っていない。Kaggle08/09・隠し完走・自己Public・実測高速化は未確認で、追加ログがprivate実行から取得できるとは約束しない。

## f002提出の確認とf003・24 epoch比較の準備（10月7日）

- 公式SDKのread-only状態照会で提出ref56897486、2026-10-07T02:16:41.160Z、監査済みVersion2/scriptVersionId355923367を確認。02:20:31Z（11:20 JST）時点でPENDING・Publicなし・errorDescriptionなし。queue/正味実行時間やhidden完走は分からない。スナップショットは`artifacts/research/20261007-f002-waiting-plan/submission-status.json`に新規保存した。
- [f003 config](../configs/experiments/f003-dinov2-attention24.json)はf002とのdiffが`reason`と`train.epochs`の12→24だけで、既存validatorが成功。complete/非limited/4,207件のexport metadataと前処理/encoder契約、両baselineのconfig、12 epoch/fold、全16実装hash、現行torch/numpy版、export/manifest/weak/gold/fold-auditのbyte hashが一致した。goldはhashのみ照合し、学習/選択/評価に用いない。両baselineのbest.ptとselected予測CSVのhash、gold非使用宣言、新規run保存先の不存在も確認した。[12項目と比較計画](../experiments/f003-attention24-plan-20261007.json)
- 現行CLIはresume非対応なので、新しいfold0/1のrunへ24 epochを最初から学習する手順を記した。CUDA必須とBCE最低/同値時最初のcheckpoint選択は変更しない。実行後は最初の12 epochの一致を確認してから同一foldのBCE/12平均/補助11平均/所見別supportを比較する。selected BCE単独や既使用foldのbootstrapを独立した汎化証明に使わない。既存ラベル監査ツールは未実施のまま、fold0/1とgoldを保護する手順を維持した。
- 実装/GPUコード・Notebook・既存checkpoint/実験runは変更せず、実MRI decode・モデル/実データの追加取得・実学習・gold評価・外部アップロード・代理提出は行っていない。全NPZの再走査や人工GPU forward/全unittestはconfig/手順だけの変更に対して繰り返していない。台帳の更新前コピーと準備検査scriptは上記の新規artifact runへ保存した。

## f002のVersion2完走と、添付summary/CSVの監査（10月7日）

- ユーザーが設定修正後に添付したsummaryを新しいrunへbyte同一で保存。SHA-256 `d419437450ce8a6fa4b09d6e9b821abf8eb6a42e1c9bb2ba8f398ed1a99d7b0c`。公式SDKで同じNotebookのVersion2がCOMPLETE、GPU有効・Internet OFF、期待する3 Inputs、全5 source cellが配布07と一致と確認した。全Output8件とオブジェクト版ID355923367を取得し、summary/receipt/CSV/exportの小さい4ファイルだけをダウンロードした。添付summaryはこの保存版とbyte同一だった。[19項目の監査](../experiments/f002-visible-inference-review-20261007.json)
- Summaryはcandidate=f002-fold0-single-v1、passed/complete、training_performed=false、公開test3検査、cache_complete=true、cached_studies=3、receipt.valid=true、Tesla T4・torch2.11.0+cu128・Python3.13.15・CUDA12.8。checkpoint SHA-256 `189ecab25e0d1c029e974ee23f3c46239e5eceef290b5dca33f6831dcb8e1b03`はローカルfold0 BCE選択epoch11のbest.ptと一致し、feature contract/fingerprint/encoder出所/前処理実装も学習時と一致した。
- CSV本体を独立に検査し、StudyInstanceUIDと所定12所見の順、3 UIDの重複なし・export coverageとの集合/順一致、36確率すべてfinite・[0,1]、所見ごとに非定数、receiptとsummaryの一致を確認した。CSV SHA-256 `f7b7fb3ca6c2f5cf17764838993f9fd0a1e5779becac2fda71ddcf4794883449`はreceiptと一致。exportはunrestricted test全3検査・9系列で物理順を報告している。NPZ本体は取得せず、元test/sample CSVとの一致は同じsourceによる実行時receiptが根拠であり、今回の独立UID照合はexport coverageに対して行った。
- 推論セルのタイマーはpreflight0.1434秒・extract31.2499秒・head/validation0.3927秒・total31.7866秒。encoder読込みと特徴抽出を含むが、前段code import/setupやKaggle queue/起動を含むNotebook全体時間ではない。可視3検査から隠しtest時間へ外挿して完走を保証しない。自己Publicはnullで、手動提出後に測る。
- 今回は既存保存版source/metadata/log・JSON/CSVのread-only取得とローカル監査、手順/台帳更新。GPU/学習/推論コードとNotebook、元checkpointは変更せず、MRI decode/encoder/head forward・追加学習・gold評価・モデル/NPZ取得・外部アップロード/提出をassistantは行っていない。変更が記録/手順だけなので過去のunittest/人工GPU forwardは再実行していない。生ログ/添付コピーと監査scriptは`artifacts/research/20261007-f002-summary-review/`へ保存した。

## 07のOutput0件とGPU無効の原因確認（10月7日・Version1の履歴）

- ユーザー指定`rsraki/notebookdd86797040` Version1を公式SDKで取得。ERROR、enableGpu=false、machineShape=None、Internet ON、Output一覧0件。GPU guard前までのcode読込みは進み、ログはIn[1]の`assert torch.cuda.is_available(), 'CUDA GPU required'`でAssertionError。ログの停止行timestampは16.2488秒で、推論時間ではない。後段推論セルのsummary/CSV作成へ到達していなかった。[10項目の監査](../experiments/f002-saved-run-failure-review-20261007.json)
- source全5 cellは設定済み07と一致、Input参照もCompetitionと既存encoder/codeと新headの3件。モデル/特徴の問題やp003隠し例外と混同せず、CPU fallback/CUDAチェック削除を加えず、実行設定のGPU有効・Internet OFFで再保存する方針とした。標準SDK OAuth更新がsandboxの保存権限で一度停止したが、許可されたSDK通常認証の再実行で成功した。認証情報の値を直接読んだりコピー/表示したりしていない。
- raw source/metadata/約4.9KBのlogを新規runへ保存し、同じfailed runから生成されていないsummary/CSVを取得できるとは説明していない。次のVersion2で可視停止が解消したことを上記に記録し、Version1 evidenceを上書きしていない。

## f001/f002のfold1完成runと、07単独推論候補の確認（10月6日）

- ユーザー提供の二つのrun.jsonと全24 epochの保存予測、best/last checkpointをread-only監査し121項目が成功。status complete、12 epoch・各208 optimizer steps、seed20261002、fold1、学習3,316/検証891、RTX4090、weak BCE選択、encoder凍結、gold学習/選択/評価なし。config差はpoolingと仮説文だけで、source/教師/特徴/固定fold監査/環境が一致した。fold0の原runとhash、fold間のconfig/source/入力一致も確認した。[比較監査](../experiments/f001-f002-fold1-review-20261006.json)
- 全CSVのhash・891 UID/order・所見別AUC/supportを再計算して一致。欠損を除く6,207観測セルのBCEは予測確率から再計算し差1e-6未満、clippingなし。best.ptのhashは選択記録と一致し、CPU上で両checkpointのconfig/fold/epoch/source契約・全tensor finiteを確認。両方BCE最良epoch12でbest/lastも12、AUCも12で最大だった。監査scriptの初回はcheckpoint hashをhistoryにあると想定、次はconfigをrun.json内にあると想定して停止した。実際のselected/config.jsonを参照する検査へ修正して完了し、学習実装・元成果物は変更していない。
- f001はweak BCE0.334234596、12所見AUC0.792570858、補助11平均0.783085734、head fitting190.744秒。f002はBCE0.317468683、12平均0.815430270、補助11平均0.808135740、198.853秒。9/12所見で改善、Synovitisは差-0.001237113。MCLとPF OAは両foldで低下し、ACL・Medial Meniscus・Baker's等8所見で改善が続いた。epoch12は固定日程末尾で、収束完了やepoch延長の妥当性は未検証。
- 固定selected epoch12予測に、861 supplied groups・PCG64 seed20261006・3,000回のpaired bootstrapを適用。全抽出で12/11平均が定義され、12平均差+0.022859412の95%区間[+0.010083240,+0.035736371]、補助11平均差+0.025050005の区間[+0.012249187,+0.037048488]。固定予測の記述的不確実性で、同じfoldでのcheckpoint選択や患者独立性、Publicへの一般化は対象外。[bootstrap記録](../experiments/f001-f002-fold1-bootstrap-20261006.json)
- f002を単独Kaggle評価へ進める。初回はfold0基準のBCE選択epoch11を固定し、異なる検証集合のAUCの大小でfoldを選ばない。既存encoder/code Inputと同じ実装を使う設定済み07と317,253 bytesのprivate head ZIPを新規handoffへ準備。ZIPはbest.pt/config.json/checkpoint-provenance.jsonのみ、全bytesとcheckpoint hash一致を確認。所見別係数・fold混合・p002混合・教師/学習変更は加えず、ラベルの帰属/版/条件と採用理由を記録した。
- 設定済み07は学習/推論実装を変えず、Inputパス・checkpoint/source hash・fold/epochの固定と、phase/時間/例外tracebackを小さい`F002_INFERENCE_SUMMARY.json`へ保存する処理を追加した。実行前の全セル構文と空outputを確認。実Inputのcodeとprovenanceだけを写した人工環境で3 code cellを実行し、既存f002 headのRTX4090 forwardから、Reportのない人工2検査の12列CSV/UID/order/finite確率/receipt/hash検査が成功。違うcheckpoint hashの拒否と人工抽出例外の再送出も確認し、failed phase/tracebackを保存して代替CSVを生成しない。[handoff検証](../experiments/f002-fold0-single-handoff-20261006.json)
- 今回は既存成果物/weak CSVの監査・CPU bootstrap・梱包・人工特徴でのhead GPU推論・文書/台帳更新のみ。実MRIのdecode/encoder forward、モデル/実データ取得、追加学習、gold評価、外部アップロード/提出はしていない。実Kaggle T4/MRIの時間と自己Public/隠し完走は未確認。元runを上書きせず、原JSONコピー・監査/梱包/人工検査scriptを`artifacts/research/20261006-f001-f002-fold1-review/`へ保存した。

## f001/f002のfold0完成runと予測・checkpointの監査（10月6日）

- ユーザー提供の二つのrun.jsonと同じrun内の保存成果物をread-only監査。両方status complete、12 epoch、各epoch 212 optimizer steps、seed20261002、fold0、学習3,386/検証821、RTX4090、BCE選択、汎用encoder凍結、gold学習/評価/選択なし。固定CSV/fold-audit/特徴export/source/runtime/hashが一致し、config差はhead poolingと仮説文のみだった。[117項目の監査記録](../experiments/f001-f002-fold0-review-20261006.json)
- 全24 epochのCSV hash・821 UID/order・所見別AUC/supportを再計算して一致。欠損を除いた5,727観測セルのBCEも予測確率から再計算し、各run記録との差は1e-6未満、確率0/1のclippingなし。best.ptのhashは選択記録と一致し、CPU上でbest/lastのconfig/fold/epoch/source契約・全tensor finite値を確認した。bestは両方epoch11、lastはepoch12。BCE最低/同値時最初の選択規則を確認し、AUC最良epoch12へ後付け変更していない。
- f001はweak BCE0.332474872、12所見AUC0.770865231、補助11平均0.800034798、学習164.681秒。f002はBCE0.319130235、12所見AUC0.813290207、補助11平均0.811771135、179.893秒。9/12所見でAUC改善。12平均差+0.042424976の74.64%はSynovitis差+0.38（陰性1/陽性100）由来。MCLは陽性11、差-0.097808442、PF OAとFractureも小幅低下した。
- 既存CSV-only bootstrapをselected epoch11の固定予測に適用。812 supplied groups・同一group多重度で両モデルをpaired抽出、PCG64 seed20261006・3,000回。補助11平均差+0.011736338の95%区間[-0.002762089,+0.024256004]は0を跨いだ。12平均は1,127回が未定義で、区間[+0.027034037,+0.056546934]は定義可能な1,873回に条件付けられる。checkpoint選択の不確実性や患者独立性、画像由来正解/Publicへの一般化を含まない。[bootstrap記録](../experiments/f001-f002-fold0-bootstrap-20261006.json)
- 暫定候補はf002、次は同じconfig/seed/12 epoch/特徴/教師/BCE選択のfold1 paired比較とした。fold1学習3,316/検証891、Synovitis陽性97/陰性25を既存weak CSVから確認した。歴史的e009/rank50との比較は同じweak manifest上の記述比較で、異なるencoder/解像度/学習/選択方式の一要因効果とは呼ばない。
- 今回は既存runとweak予測の監査、CPU checkpoint検査、CSV bootstrap、実験台帳と手順の更新のみ。追加学習・モデル/実データ取得・MRI decode・画像からのforward・gold評価・外部アップロード/提出は行っていない。学習/GPUコードを変更していないため過去のunittest/人工GPU forwardは再実行していない。既存の学習runは上書きせず、原run.jsonコピーと検査scriptは`artifacts/research/20261006-f001-f002-fold0-review/`へ保存した。

## 05の再提出失敗と06全件Outputの取得・検証（10月6日）

- 公式SDKで提出ref `56870280`、2026-10-06T04:31:47.710Z、scriptVersionId `355633504`、errorDescriptionあり・Public nullを確認。保存`rsraki/notebook59f34ce4e6` Version 1の全セルは配布05と一致、出力オブジェクトの版IDも提出と一致した。READY passed・可視3件256.538秒、4 CoAt完了・fallback 0、可視CSVの報告hashは旧失敗02と同じ。可視実行ログ約21KBも保存した。[2回目の監査](../experiments/p003-second-failure-review-20261006.json)
- 元ソースを文字列として抽出して調べ、8時間全体予算と、その残り時間を子処理waitへ渡す構造、48件超でCoAtを逐次実行する分岐を再確認。4時間の専用timeoutは確認していない。取得ログは保存時の可視実行であり、隠しtraceback/時間は未取得。CPU/GPU資源、データ依存の前処理/fallback、子処理失敗などは原因候補で、修正・9時間超過を確定していない。
- ユーザー指定の`kaggle kernels pull rsraki/notebookbd6efc1cf7`でコードを取得し、Outputは別途公式SDKで取得。Version 2・出力scriptVersionId `355640726`、GPU有効・Internet OFF、保存版sourceは梱包06からPILOT=Falseのみ変更。保存実行status COMPLETE、約406KBのログでFeatures 4207/4207とcomplete=Trueを確認した。Inputはcompetitionとprivate feature Inputの二つで、Datasetの歴史的版番号はmetadataから確定していない。
- 全件exportのSHA-256 `aa164be29b14813b9793c51b9924ef2e4019f4f01c2aa1ec35d17fdf536e1f5a`。metadata 26項目が成功し、4,207 weak UID・固定group/fold・元CSV・汎用encoder出所・前処理/config/実装hash・系列選択が一致。gold UID/group混入なし。12,621系列・386,196中心、全系列物理順、元slice数12～320、特徴作成12,548.213秒、NPZ 281,589,228 bytes。[全件監査](../experiments/f001-full-feature-review-20261006.json)
- `data/features/f001-weak-sv355640726/`へ新規転送し、4,207 NPZとstudies.csvの全サイズ/SHA-256を照合。続いて既存verify_feature_cacheで全NPZのshape/dtype/finite値・mask・位置/plane/flag・fingerprintをCPU検証した。f001/f002の共通feature契約と、pooling以外の学習条件一致、fold0学習3,386/検証821も確認した。検証記録は`artifacts/research/20261006-p003-second-failure/full-payload-verification.json`。
- ローカルdoctorはPython 3.12.13、torch 2.10.0+cu128、必要な画像/学習moduleあり、CUDA trueで成功。f001/f002の既定outputは未作成で、手順には検証済みcacheの実パスを反映した。実学習は開始していない。
- 今回はユーザーが許可した保存済み特徴・JSON/CSV・コード・ログの取得と検査を実施。元DICOM/モデル重みの追加取得・MRI decode・実学習・外部アップロード・提出は行っていない。GPU/学習/推論コードは変更せず、過去のunittestや人工GPU forwardは再実行していない。実学習時間・weak AUC・独自Publicは未測定。公式CLIの[pull/outputの区別](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)も確認した。

## 06・32検査特徴pilotのユーザーOutput確認（10月6日）

- 添付`export.json`を新しいrunへbyte同一で保存し、SHA-256 `fdbeb8cd4b340969e97bc2070bce7aefb45e82273d56935a8ad53219f9e837b9`を記録した。[監査](../experiments/f001-pilot-review-20261006.json)の21項目が成功。complete/limited/32件、config/実装hash、特徴fingerprint、ローカル梱包済みInputのencoder出所・manifest hash、元train/series CSV hashが一致した。
- 既存の監査済みweak CSVからseed 20261002・fold0学習側の32件を再構成し、UID集合・構造hash・pilot CSVとstudies.csvのhashを再計算して一致。fold0とgold/groupは含まず、pilotのfold内訳は1:7、2:11、3:9、4:5。教師値は変更していない。
- 32検査すべてが3方向を含み、96系列・2,953中心。全系列physical_position順、PixelSpacingは0.1367～0.7031mmで正の有限値、中心数16～48、元slice数16～128、中心は全範囲を覆う。32 NPZの記録サイズ合計は2,152,567 bytes。NPZ本体は未提供で、shape/finite値とpayload hashはこのレビューで独立に再検証していない。現行コードは各特徴の生成時にfinite値を検査し、export保存後にverifyする構成。
- Tesla T4、torch 2.11.0+cu128、numpy 2.1.3、pydicom 3.0.2、Pillow 12.3.0。特徴作成83.0177秒、検査ごとの最小/中央値/最大は1.071/1.989/8.815秒。4,207件への単純換算は3.032時間・NPZ約283MBで、初期化などを除く目安であり全件実測・上限ではない。
- ユーザーは末尾の画像について「明らかな崩れは見当たらない」と回答。現行Notebookの表示対象は最初の3検査の各先頭系列であり、96系列全部の目視検証ではない。assistant自身は画像を見ていない。Kaggle保存Notebook/Input版・全実行ログも未取得で、complete=trueだけから最後の検証・表示セルの完了を断定しない。
- 全件特徴へ進む判断とし、同じInput、PILOT=False・1 shard・新しい保存実行を案内した。実学習は全件cache転送・検証後。今回実行したのは添付JSONとローカルCSV/metadataの監査だけで、モデル取得・MRI処理・学習・外部アップロード/提出は行っていない。推論/学習コードは変更せず、過去のunittest/GPU検査は再実行していない。

## 06開始手順とローカルInput梱包（10月6日）

- 公式DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`、ViT-S/14 registerなしの配布リンク、公式README/MODEL_CARDの出所説明を確認した。GitHub APIのWeb取得は失敗したが、公式commitページから完全なSHAを確認できた。source/重みの実payloadは取得していない。
- `package_frozen_feature_inputs.py`は手動取得した公式ZIP/重みを受け取り、source-tree/重みhashと出所JSON、固定コード、監査済みmanifest四ファイルを一つのprivate Input ZIPへ梱包する。元Notebookは変更せず、Inputパスだけ設定した06のコピーを新規runへ出力する。source import・モデル読込み・ネット通信・自動アップロードはない。出所の自己申告とローカルhashを独立した配布元認証とは扱わない。
- 標準ライブラリの人工fixture 5テストが成功。梱包後に展開してsource/重みhash一致、manifest allowlist、推論/QCセルの不変性、PILOT設定、構文・未実行状態、既存出力/古いfold監査/変更コードZIP/不正archiveの拒否を確認した。実画像・実重み・GPUを使わない検査である。
- 手順を公式2ファイルの取得→ローカル梱包→private Dataset→設定済み06の32検査保存実行→export.json確認へ具体化した。今回は手順整備と人工テストだけで、実モデル取得・実MRI処理・学習・外部書込みは未実施。全件特徴作成・学習はpilotの結果確認後に進める。

## 提出待ち中の独自モデル準備の確認（10月6日）

- 既存コードbundleの存在とSHA-256 `6c8731cfc34489c5ce1cccb60c163829d39e84c482dac64cffcaf0db882a6cce`、manifest四ファイルの存在を確認した。所定パスの汎用DINOv2 source・重み・記入済provenance・全件特徴cacheは存在しなかった。これは既知パスの確認で、PC全体やKaggle Inputの検索ではない。
- 06のInputパスはひな形、`PILOT=True`、`SHARD_COUNT=1`であることと現行手順を照合した。05の採点待ちにInput準備を進め、その後32検査pilot→全件特徴→4090学習比較と進む順序を追記した。公式DINOv2 READMEで汎用ViT-S/14と凍結特徴利用の説明も再確認した。
- 今回は準備状況の確認と文書整理のみ。元DICOM/モデル取得、MRI処理、実学習、外部アップロード・提出は開始していない。コード・Notebook・実験台帳は変更せず、過去のunittest/GPU検査は再実行していない。

## 新05・64検査診断のユーザーOutput確認（10月6日）

- ユーザー提供の`P003_SPEED_SUMMARY.json`を新しいrunへbyte同一で保存し、SHA-256 `a8c1868d007b6ad00de1b018b590a6bd472ac357d36b0a5e2cd41c32d50c1974`を記録した。[監査](../experiments/p003-speed-profile-review-20261006.json)の32項目が成功。ローカルmanifestとのsource hash・contract一致、実行順、seed/64 UID集合hash、366系列、29予測shape、全5 phase完了、4 CoAt系統のfallback 0・子処理終了0を確認した。最初の監査スクリプトではRadのshapeを二次元と誤って想定したため停止し、既存仕様どおりの5-fold×64×12へ確認条件を修正して完了した。推論コードの変更はない。
- T4×2、Python 3.13.15、torch 2.11.0+cu128、CUDA 12.8は旧04と一致。変更したRaptor 2系統は入力hash・生予測が完全一致、最大差0・順位差0。DINO20の共有経路差も0。最終診断CSVの報告hashは旧04と一致するが、実CSV/NPZや個別tensorから今回再計算したわけではない。添付内のCoAt receiptは確認したが、子ログ・PREFLIGHT本体・Kaggle保存Notebook/Input版は未取得。
- 総時間1,832.3586秒、追加reference 423.7238秒。単純差引き1,408.6348秒（23分28.6秒）は旧04の1,427.7626秒とほぼ同程度。cell27も差引き966.4636秒に対し旧948.1448秒で、大幅高速化の根拠はない。差引きは並行処理・cache・診断hash計算の影響を含む概算で、提出版の実測ではない。referenceはprefetchなしのため候補との比も公平な高速化率にしない。
- native処理のPyTorch最大予約メモリ1.55GiB、CoAt各worker最大2.72GiB。今回64検査の完走は確認できたが、GPU全体やホストメモリの最大値、隠しtestの時間内完走は保証しない。`raptor.status=started`は未更新の初期値、`speedup_verified=false`は固定値と現行コードで確認した。
- 次は追加学習なしで提出用05の保存実行へ進み、READY/SUMMARY確認後に手動提出する。自己Public・隠し時間・元例外の解消は未確認。今回はJSON監査・記録更新のみで、MRI処理・学習・ネットワークアクセス・アップロード・提出は行っていない。過去のunittest/GPU検査は再実行していない。

## 05診断実行後の手順整理（10月6日）

- ユーザーによる新05診断の実行報告を記録した。今回のOutputは未提供で、完走・実MRIでの一致・速度・メモリは未確認のままとした。
- 現行の速度guard・runtime・Notebook生成コードと照合し、最初に確認するファイルを`P003_SPEED_SUMMARY.json`へ絞った。[手順](p003-speed.md#05の診断を実行した後にすること)に、診断結果確認→提出用05の保存実行→手動提出→採点確認の順を明記した。06/07と独自学習は今回の提出の前提ではない。
- 今回は文書のみの変更。コード・Notebook・実験結果は変更せず、学習・MRI処理・Kaggleアクセスや提出は実施していない。過去のテスト結果を今回再実行したとは扱わない。

## p003例外監査・速度改善・新しい学習経路（10月6日）

- 公式SDKのread-only照会で提出56855132はPublic null・errorDescriptionあり。ユーザー画面もNotebook Threw Exception。SDK COMPLETEを採点成功と解釈せず、9時間超過は未確定と記録した。保存版1の全セルが配布02と一致し、Outputの版ID355483987も提出と対応。可視3検査のP003_READYはpassed・332.005950176秒。取得は保存コード・metadata・2つの小さいJSONのみで、隠しtracebackや隠し実時間は不明。[監査](../experiments/p003-failure-review-20261006.json)
- 新05はRaptor GPU1のnative384dense/native384を検査単位で交互実行。元02/04/vendor/guardを変更せず、全モデル・前処理・元forward・数値精度・microbatch・混合式・8時間内部予算を保持する。元のFP32再試行とイベントも同じ関数を通る。profileで全64検査の旧経路入力hash・生予測・順位差を比較し、phase時間・2モデル常駐のGPUメモリ・例外tracebackを記録する。reference再計算の時間は本番候補と区別する。
- 新05の人工CUDA試験は凍結元の`_ke_infer_input`関数を抽出し、小さいTinyRaptorだけを代替した。実行順変更前後で入力hash・生予測が完全一致。これは実公開重み・T4×2・実MRIの同等性/速度/VRAMの確認ではない。新規7テストは全成功。p003関連30件は29成功・既存Windows symlink権限の1件skip。Notebook再生成、構文、空出力、変更セルの限定を検査した。
- 新しいf001/f002は汎用DINOv2 ViT-S/14の凍結特徴とmean/attention head。元DICOM→392px・物理順・PixelSpacingを反映したletterbox・隣接3枚→384次元FP16特徴を共通経路とした。CUDAなしの学習を拒否し、Reportなし提出契約、欠損mask、無効slice不変性、encoder出所と特徴hashを確認する。checkpointはBCEで選択し、所見別weak AUC/supportを併記する。gold非使用とgroup分割は監査済みCSVのhashに結び付ける。
- 独立レビューで、特徴cacheが教師CSV全体hashに結び付いて教師だけの変更を拒否する点、head実装hash未固定、既import DINOv2の混入可能性を発見。UID/group/foldの構造hashと教師履歴を分離し、head実装hashの保存/照合とローカルencoder sourceの整合検査を追加した。検証DataLoaderは専用Generatorを使う。汎用事前学習画像との重複を独立排除したとは主張しない。
- ラベル根拠キューは学習側のみ・保護fold0/1・gold/推移的group除外・1group1検査・希少層優先・seed付きで作成する。N/B/U/Mと欠損を維持し、原Report/UIDをGit除外runへ保存する。新規10件と既存ラベル監査12件のstdlib unittestが成功。実レポート抽出や教師の変更はまだ行っていない。
- 最終の全204 unittestは**203成功・失敗0・既存Windows symlinkの1件skip（14.769秒）**。凍結特徴の新規10件には人工GPU1 epoch→checkpoint→Reportなし提出、shard結合と重複/欠落拒否、教師だけ変更した際のcache identity、古いfold監査とhead hash不一致の拒否を含む。Ruff checkと全90 Pythonファイルのformat checkが成功。新4 Notebookの構文・未実行状態、05のbyte同一再生成、f001/f002のpoolingだけの差、台帳19行、ローカルリンク165件を確認した。元p002/02/04/vendorおよび既存学習・画像・モデルの9ファイルはHEADと不変。
- 添付原文をhash付きで保存し、Kaggle公式条件、Meta DINOv2/DINOv3、KneePreMの一次資料を調査した。[判断と出典](research/p003-improvement-20261006.md)。元p003の自己0.943、速度改善幅、学習精度を記録していない。実データ/モデルの追加取得・実MRI decode・実学習・外部アップロード・代理提出は未実施。

実行手順は[05速度診断](p003-speed.md)、[凍結特徴学習](frozen-features.md)、[教師監査](label-review.md)。検査の集計は[今回の検証記録](../experiments/p003-improvement-verification-20261006.json)に残す。

## p003・64検査診断のユーザーOutput確認（10月5日23:37 JST）

- 添付`P003_PROFILE.json`を読み、status=`profile_complete_not_for_submission`、seed 20261005、64検査・366系列、T4×2を確認しました。添付は新しい調査runへbyte同一で保存し、元ファイルhash・実行環境・判断を[監査記録](../experiments/p003-profile-review-20261005.json)に残しました。Kaggleへの新たなアクセスや実画像処理は行っていません。
- UIDの重複・集合hash、元Notebookと全元セルのhash、cell6だけを変更した派生セルhash、cell29除外、実行22セルの順序をassertで照合しました。DINO20の人工forward比較は全てmax_abs=0、29種類の予測記録のshapeが整合し、4つのcache完了イベントは全slotが充足しています。degraded／再試行イベントはありません。
- 完了JSONを書き出す実装ではDINO20・A5全5fold・Rad校正器・Raptor全UID・追加CoAt4系統・fallback拒否・確率統合の再計算を要求します。この完了記録を次の保存実行へ進む根拠とします。ただし添付は要約JSONであり、個別receipt／子ログ／CSV・NPZ実体／P003_PREFLIGHTを今回独立に再検査したわけではありません。Kaggle保存版も未確認です。
- 総時間は事前検査込み1,427.763秒（23分47.8秒）。主なセルはDINO145.24秒、A5推論45.44秒、Rad62.21秒、Raptor＋追加CoAt948.14秒。全セル計1,209.465秒との差218.297秒は事前検査等です。仮に1,300件として総時間を単純比例すると約8時間3分、セル外時間を固定すると約6時間53分ですが、どちらも粗い条件付き計算で、予測区間や上限ではありません。隠しtestの画像構成・I/O・cache挙動により変わり、[公式9時間制限](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)内の完走を保証しません。
- 判断は、係数・計算精度・Inputsを変えずに02_submit_p003の保存実行へ進むことです。A5の精度診断は先行必須にしません。P003_READYとsubmission.csvの確認後にユーザーが手動提出し、自己Publicと本番時間を記録します。今回は記録・説明のみの変更で、過去の177 unittestを再実行したとは扱いません。JSON／台帳／リンク・差分の検査を実施しました。

## p002 Public 0.937の確定とp003の準備（10月5日）

- Input追加の補足：Kaggle公式のInput種類別説明とTU DelftのURL貼り付け手順を確認し、操作手順へ追記しました。配布p003の`metadata.kaggle.dataSources`が空であることをローカルJSONで確認。ユーザーのKaggle画面でのURL検索・Input追加・Import時の挙動は未実測です。今回は説明だけの変更で、Notebook・モデル・係数・Input契約を変更せず、過去の177 unittestを再実行したとは扱いません。
- 公式SDKのread-only確認でp002はCOMPLETE／Public 0.937。保存版1の16セルが前回handoffと一致し、可視実行の47資産hashと各receiptを照合しました。可視3検査164.699秒とユーザー報告の約8時間を区別し、隠しtest時間は不明と記録しました。[採点監査](../experiments/p002-scored-20261005.json)。
- 成功版p002と元V32をGit追跡先へ固定。生成器のGit除外調査フォルダ依存を解消し、隔離ディレクトリでも成功handoffと同一byteを再生成する検査が成功しています。旧runは変更していません。
- haideptry V2の公式SDK取得ソースを固定し、元23コードセルを保持するp003を作成しました。実効14 Inputs・公開148ファイルは期待hashを宣言し、大会CSV4件は実行時hashを記録します。公開重みpayloadをローカルで実照合したという意味ではありません。[Input契約](research/p003-input-contract-20261005.json)、[調査](research/p003-candidate-review-20261005.json)。
- p003 guardは全4 CoAt系統・DINO20・A5 5fold・Rad校正器・Raptor準備UIDを確認し、系統省略・定数埋め・dense前処理fallback・確率欠落を拒否します。確率平均後の全体rankと最終係数を再計算し、CSV再読時の浮動小数丸めを維持した照合を行います。独立レビューで子処理logだけに出るfallbackとCSV roundtripを発見し、検査に反映しました。
- A5診断はp002の元定義を抽出し、Kaggleの固定64検査で同じcache・5重み・3精度を比較する未実行Notebookです。RTX 4090では人工入力と小さい代替encoderでBF16／FP16／FP32の実行経路を検査しました。これはA5本体の全architectureやT4上の公開重み／実MRIの精度・速度確認ではありません。
- 64検査のp003診断も、train画像をラベルなしで測るための準備です。最終提出セルを除外し、提出用CSVは生成しません。train診断を独立OOFやPublic改善の証拠にはしません。
- 全177 unittestは176件成功、失敗0。実symlink作成の1件はWindows権限不足でskipし、模擬リンクの解除と元target保持は成功しました。Ruff check／format、Notebook再生成・構文・空出力・hash、ローカルリンク、差分の空白を確認しました。[検査記録](../experiments/p003-verification-20261005.json)。実MRI、新たな公開重み取得、実学習、アップロード、代理提出は行っていません。新候補の実Public／本番時間は未測定です。[次の操作](p003-next-step.md)。

## p002提出後の待機状態確認（10月5日19:29 JST）

- ユーザーの「約7時間後もNotebook running」を受け、公式SDKのListSubmissions/GetSubmissionをread-onlyで確認。提出ref 56840796は12:07:34 JST、最終ファイル名submission.csv、Publicはnull、SDKの型付きstatusはPENDING。JSONはdefault enumを省略するため、statusキー欠落を状態不明やエラーと解釈せず型付き属性を確認した。[一覧](../artifacts/research/p002-status-20261005T102826Z/submissions.json)、[詳細](../artifacts/research/p002-status-20261005T102826Z/submission-detail.json)。
- 提出URLは`rsraki/notebook3beaf8c48b?scriptVersionId=355324155`。今回の状態照会では保存ソースとhandoffのbyte同一性や隠しtestログを取得していない。PENDINGは待機・実行を区別できず、経過26,530.96秒は実推論時間ではない。成功・失敗・正常な進捗のいずれも断定しない。
- [公式要件](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)の実行上限9時間と、[公式データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)の約1,300 test検査を再確認。保存時の少数testと隠しtestの処理量は違う。元コードにはDINO段の8時間予算があるが、後続モデルもあるため全体8時間以内の保証ではない。実測の本番所要時間は未確定。
- コード・係数・提出を変更せず、停止・再提出は行わなかった。実データ・モデル・予測取得、学習や新たな推論も未実施。変更は状態記録のみで、141 unittestの過去結果を再実行したとは扱わない。

## p002の手動保存実行へ渡す準備（10月5日、完了）

- ユーザーが提出を担当する方針に従い、[handoff-v2 Notebook](../artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)と[操作手順](public-baselines.md#ユーザーが行う操作)を作成した。状態は`prepared_for_manual_saved_run_not_executed`で、Kaggle実行成功や採点準備の検査通過を意味しない。元source-only候補とhandoff-v1は上書きせず履歴として保持する。[今回の記録](../experiments/p002-handoff-20261005.json)。
- 元19 Input参照を元V32の実ファイル解決まで追跡し、Tonylica Dataset 5、Renta Dataset 2、Meta DINOv2-small Model 1、Mattia fold Dataset 2のtimm wheel、公式競技の5 Inputsへ固定した。[対応表](research/p002-effective-inputs-20261005.json)には公開資産47件の期待bytes/hashと除外理由がある。配布manifest等のhash宣言であり、未取得の重みpayloadをローカルで検証したとは扱わない。公開した小さいコード・manifest等だけ取得し、モデル・MRI・予測payloadは取得しなかった。
- 固定したRenta runtimeも静的確認し、T4 x2・FP32・2partitionを要求することを確認した。元Notebookの数値処理とBF16設定を保持し、timm 1.0.22だけhash検査済み添付wheelから`--no-index --no-deps`で固定する構成。他ライブラリはimport/APIを検査して実版を保存する。[環境・利用条件](research/p002-environment-20261005.json)。実際のKaggleイメージとの互換性は未確認。
- [生成スクリプト](../scripts/build_public_candidate.py)はネット接続や公開推論ソースの実行を行わず、元10セルのソースをそのまま文字列として埋め込み、[追加検査](../scripts/public_candidate_guard.py)から共通globalsで順番に実行する。開始・完了の2セルを加え、計12 code cells。NotebookをASTから逆に読んで元ソース10本との完全一致、埋込みguard/contractの一致、全コードの構文、未実行状態を確認した。再生成Notebookのbyte一致と既存ディレクトリ上書き拒否も確認。[静的QA](../artifacts/qa/20261005-p002-handoff-v2/checks.json)。
- 追加検査はfresh working、指定Inputのhash、余分なInput拒否、test mount、DINO 20 member・A5 5 fold・Raptor 4 armの完走とraw値、fallback拒否、固定校正器の適用、4 receipt、UID/所見順/値域、親・control・V6出力のhash連鎖を確認する。元のCSVを途中で生成しても、必須分岐が欠ければ成功としない。失敗時はこの実行で作ったCSVを`.disabled`へ退避し、既存実験の成果物は保持する。指定mount内のMRI treeを追加のmount検査で走査しないことも人工テストで確認した。
- 新規44件を含む**全141 unittestが成功（13.080秒）**。新規テストは人工CSV・配列・ソースstubだけで、公開モデルは実行していない。既存成果物保持、hash/パス/順序の拒否、member不足・NaN/Inf・fallback・重複イベント、receipt型/形状/UID/hash、変更禁止列を検査した。追加3 PythonファイルのRuff check/formatとgit diff --checkも成功。
- 最終NotebookのSHA-256は`f5075c34fd389277d2aa6acb3dd3f87275485d60a8796a91d22060a46c4d70b4`。元V32 SHAは`ba9491ac1e214ba55ca181de55118c9a793fe7f6163fc221fe1e68d596259dca`で不変。GPUモデルの実装・ローカル学習コード・既存run/cacheを変更していない。実MRI処理・公開モデルforward・学習・アップロード・代理提出は行っていない。
- 残る確認はユーザーのKaggle保存実行での全分岐完走と、手動Submit後の隠しtest時間・実測Public。`P002_READY.json`は保存実行の検査結果であり、隠しtestの完走や作者0.937再現、p001 0.924超えの証拠ではない。自己Publicはnull、台帳のpublic_lbは空欄のまま記録する。

## 公開モデルへの主軸変更と候補準備（10月5日）

- 添付分析をHEAD `ea27fbdc37f4b4119332bb83af22fce0b31b15bc`の実装・台帳・公開監査と照合し、[原文](research/user-analysis-public-model-20261005.txt)を保存した。[現行計画](research/public-model-strategy-20261005.md)に、既存0.924保持→公開候補一つ→元方式との同値性→互換cache→一要因改善の順序と採否条件を記録した。BNの別fold追試、長期化、192px crop全件化は当面保留。既存run/cache/重みを変更していない。
- 公式SDKのread-only要求でp001 Public 0.924（ref 56766978、scriptVersionId 354569007）とr001 Public 0.749（ref 56826389、scriptVersionId 355188174）をCOMPLETEとして確認。r001提出は10月4日23:27:19 JST。台帳へ追記し、[監査記録](../experiments/submission-audit-20261005.json)に詳細を保存した。weak 0.830249とPublic 0.749は別の評価である。
- p001 V1とr001 V3を明示指定してソースを新規保存した。版別Output URLの識別子も提出URLのglobal IDと一致するが、型付きの直接対応フィールドではないことを明示。p001成功ソースのhashは`3ec3a18fd97746e4ee44bf424775e902362ea45a25483861bbb78767c9d4b68e`。140mm/336px/64枚/42窓を確認し、384pxとCoAtNet型はcheckpoint情報で上書きされ得るfallbackとして記録した。歴史的Input版、配布重みhash、隠しtest時間は未確認で、取得済みのように扱わない。
- p001ソースのgoldによる過去のarm選択・比較の記述は作者の沿革自己申告として保存した。公開重みの競技利用と独立評価を分け、自分のgold学習・checkpoint/混合係数選択は禁止を維持。後付けfold、encoder凍結、pseudo-labelでも既学習検査への露出は消えない。
- [NTejas一次資料](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/tree/1c386ac71385ba7683f550b87a465386ba6c36a8)を固定commitで確認し、0.940/0.941は作者報告と記録。参照コードの実混合比と末尾summaryの不一致、失敗分岐継続、保存版/Input不明を確認した。DINOsaur V32は既存保存sourceと再取得SHA-256が一致し、最初の候補p002に選定。[候補監査](research/public-candidate-review-20261005.json)。
- [未実行候補Notebook](../artifacts/research/20261005-public-candidate/dinosaur-v32-final-candidate.ipynb)は元の10 code cellが完全一致し、変更は出力履歴・実行番号の消去とJSON再保存のみ。最終`submission.csv`はV6 overlay候補に固定し、保存0.937 controlと区別した。[manifest](../artifacts/research/20261005-public-candidate/candidate-manifest.json)に19 Input参照と未確認版、必要receipt、GPU metadata矛盾を記録。`ready_for_scoring=false`、自己Publicはnull。作者が採点したCSVと歴史的Input版が未確定なので、0.937再現済みとはしない。[実験計画](../experiments/p002-dinosaur-v32-plan-20261005.json)。
- V32を静的に確認した範囲では、実行時のgold評価・係数fit・checkpoint選択・学習はなく固定係数を適用する。容量計画にはtrain.csv全列を読んだ件数だけを使い、Report列は参照しない。test_series不存在時のtrain側CSV fallbackを確認し、test mountと実効runtimeのhash確認を次の条件に追加した。外部Inputの版未固定のため、全依存の実行時挙動を監査完了とはしていない。
- [現在のInput版一覧](research/public-current-inputs-20261005.json)を追加取得。14 Dataset・3 Notebook Outputは取得前後の版が一致し、Model V1を含む195ファイルの名前・サイズを確認した。payloadは未取得。DINOsaur V5 trainは前日V15→V17で、live参照とV32の実効依存を同一視しない。歴史版・有効ファイルhash・実互換性は未確認のまま明記した。
- 今回の検査はソースhash、構文、候補code不変性、未実行状態、JSON/CSV、資料リンク・差分の確認。結果は [今回の検査記録](../experiments/public-strategy-verification-20261005.json)。調査JSONの日本語破損をレビューで発見してUTF-8で修正し、ソースhash不変を確認した。GPUコードを変更していないため97 unittestや人工CUDAを再実行せず、10月4日の成功履歴と区別する。
- 認証情報の手動読取り/コピー、実画像・ラベル・モデル・予測payloadの取得、実MRI decode、学習、公開コード実行、アップロード、代理提出は行っていない。取得はNotebook/公開コード・metadataのみ。今後の実行操作は [公開モデル手順](public-baselines.md) を参照する。

## Kaggle Version 3の表示用test実行（10月4日、完了）

- ユーザーの提示ログを構造化して候補asset-manifestと照合した。表示用test 3検査の前処理はwarnings=0・debug_limit=None。通常BN／固定BNのCUDA推論は各3検査×12所見、最終rank ensembleもvalid=true。両checkpoint・全source・rank helperファイル/関数・pixel fingerprintのhashが一致し、RANK_MODULEはInput直下のrank50.pyを参照した。[確認記録](../experiments/r001-kaggle-v3-review-20261004.json)。
- 公式SDKのread-only確認でもNotebook `rsraki/rsna-knee-r001-rank50` はVersion 3・COMPLETE・GPU有効・Internet OFF。Output一覧に最終submission.csv（852 bytes）、単体予測2 CSV（各885 bytes）、submission.meta.json（871 bytes）を確認。Output本体は取得せず、CSVのvalidは実行ログの検査結果に基づく。Notebook内計測47.9549秒、提示ログは起動/変換を含み68.2秒まで。Debugger・mistune・nbconvertの警告は完走を妨げていない。
- [公式データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)のtest.csvは3検査の例で、Submit時に本番testへ差し替わる。現時点の提出履歴は旧公開モデルのPublic 0.924の1件だけで、r001の本番採点・全体時間・Publicは未確認。Version 3のOutputからsubmission.csvをユーザーがSubmitし、その結果を別途記録する。今回のレビューではMRI/重み/ラベル取得・decode・forward・学習・外部書込・代理提出は行っていない。

## Kaggleのrank helper参照先の修正（10月4日、再依頼）

- ユーザーの再実行ログは最初のcode cellのrank helper確認で停止し、直前の2 checkpointの存在/SHA検査は通過した。公式SDKの通常認証・read-only要求で実際の保存済みNotebook `rsraki/rsna-knee-r001-rank50` Version 2とInputを確認した。GPU有効・Internet OFF・競技Inputは設定済み。保存NotebookのInput参照は得られるが、厳密な添付版番号はAPI metadataに含まれない。[原因・根拠](../experiments/r001-kaggle-pathfix-20261004.json)。
- 現在のコードDataset Version 1から小さいcode/manifestだけ75,459 bytesを取得し、全15ファイルのSHAが候補資産と一致。ZIPの展開13ファイルと別添rank50.pyは正しい。保存コードのCODE_ROOTは`.../rsna-knee-code`だが、元の条件分岐はhelperもその中にあると仮定する。実際はInput直下のrank50.pyなので、ZIPを更新しても参照先は直らない。Debugger・nbconvertの警告は停止原因ではない。
- [修正版Notebook](../artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) はRANK_MODULEを`CODE_ZIP.parent / "rank50.py"`へ変更し、欠損とhash不一致を別メッセージで示す。実保存版のCODE_ROOT・後続code cell・全期待hash/provenance・前処理・重み・rank係数は保持。元のローカル資産を上書きせず、修正済み未実行Notebookを新フォルダに保存した。
- `artifacts/qa/20261004-r001-pathfix-v1/checks.json`で取得した実コードInputを人工mountへ配置し、元Notebookの同じ参照先エラーを再現。修正Notebookの初め2 cellを実行し、全source/hash確認・frozenコード配置・共通helper importが成功した。4 code cellのcompileと空output、実2 checkpointのepoch/config/所見順/共通fingerprintをCPUで照合。共通rank関数の対向順位が厳密に0.5になることも確認。
- CUDA初期化・forward・実MRI decode・追加学習・gold/ラベル読取・新しい重み取得・代理アップロード/保存実行/提出は行っていない。検査環境はローカルPython 3.12.13で、KaggleログのPython 3.13による全MRI処理と実採点は修正後の実行で確認する。元の97 unittest/GPU契約は以前の結果で、今回は設定・配置変更の検査を追加した。

## 続行依頼後の保存・更新計測と長期実験（10月4日、完了）

- 推論互換schema 1のbest_bce・best_auc・last・5/10/15/20 milestone保存を実装し、best.ptと選択予測CSVを設定した規則へのaliasにした。checkpoint indexにepoch・BCE/AUC・予測/metrics/重みhashを対応付ける。旧configのBCE規則、diagnosticのBCE強制、AUC未定義の非採用、gold非選択を維持する。lastはoptimizer/scaler/RNGを含まず、resume非対応を明示。
- optimizer post-step hookでAMP skipを除いた実更新を測定。機会・実更新・skipをepoch別と累積で記録する。500 microbatch毎、検証前、epoch保存後、完了時にprogress.jsonを更新する。追加forwardや学習lossの変更はない。
- checkpoint 7件・config 3件・extended集計8件・weak bootstrap 8件を含む全97 unittest成功（14.462秒）。CPU/CUDAで追加保存前後のRNG・sample順・BN・optimizer・scalerが完全一致し、AMP overflow skip、Reportなし推論、人工cache CUDA学習全経路を確認。Ruff check/formatは38対象ファイル、git diff --checkも成功。CSV比較のAUCは独立の人工5,400ケースでもpairwise定義と一致した。
- `artifacts/qa/20261004-extended-pipeline-v2/checks.json`で人工ノイズCUDAのAMP/backward/save/mask/提出3×12契約と旧e002 checkpoint互換を確認。QA 3.25秒、最大torch割当246,000,128 bytes。v1はlegacy専用補助チェックへImageNet正規化Cを渡す引数誤りで停止し、正しい旧e002を指定したv2で成功。productionコードはこの修正で変更していない。
- `artifacts/kaggle/20261004-extended-source-v1/rsna-knee-code.zip`を凍結（SHA-256 `9c85e2f120ba2eb9f0b3fc44b99ad328783f78026e46a52e8073b377d9bd496f`）。zip全ファイルが学習開始前の現sourceと一致し、imaging.pyのhashと既存cache fingerprintは不変。実学習中のsrc変更を行わない。
- [事前計画](../experiments/e006-e009-plan-20261004.json)に選択規則とconfig/input/source hashを保存して、fold 1のB/C各5 epoch、fold 0のC20 epochとBN統計固定20 epochを新規runで完了した。[完成集計](../experiments/e006-e009-extended-summary-20261004.json) は全4本valid=true。config・開始/終了/現source・入力hash、各epoch予測/mask/metrics、checkpoint/indexと選択、実更新数を照合し、CPUの実checkpoint読み込みでも内部epoch/fold/config/所見順/fingerprintを確認した。全体時間合計10,025.703秒、既存run上書きなし、gold cache/評価は無効、自作Public未測定。
- 実更新/機会/AMP skipはe006が4,142/4,145/3、e007が4,143/4,145/2、e008が16,932/16,940/8、e009が16,930/16,940/10。e008の最初5 epochのBCE・AUC・予測CSV hashは旧e005と全て完全一致した。BN固定モデルの20 BN module・60 running bufferは初期状態と完全一致し、affine 40/40・encoder 60/60のparameter tensorは学習で変化した。
- fold 1の事前学習差は12平均+0.093513、95%区間[+0.060727,+0.127135]。通常BNの5→8 epoch差は+0.009802、区間[−0.004061,+0.024398]。通常BN8→固定BN9は+0.019204、区間[−0.006118,+0.041423]で、平均優位は未確定。固定BNの5→9 epoch差は+0.024807、区間[+0.008788,+0.042459]。両20 epochとも終点は選択epochを下回った。[詳細と所見別の入れ替わり](controlled-experiments.md#続行後の実行結果e006e009実行完了)。
- weak CSV・供給groupだけのpaired bootstrapはPCG64、seed 20261004、3,000反復。fold 0の12平均はSynovitis陰性1groupの不在により1,124反復未定義で、区間は1,876反復に条件付き。補助11平均は全3,000反復で定義される。同foldのepoch/config選択・患者独立性・画像ラベルの正確さは保証しない。入力hashの解析前後一致を確認し、新規共有JSONには検査/group/Report識別子を保存しない。
- e008/e009の凍結コード・実重みから人工192px/24窓/3検査×12所見のCPU提出契約が成功。初期ImageNetファイル、Report、URL取得、CUDA初期化をCPU検査で禁止しても推論可能。実GPUでも同じ人工cacheから契約を確認し、CPUとの差は最大2.8313e-5/1.9849e-5。`artifacts/qa/20261004-extended-candidates-cuda-v2/checks.json` がvalid=true。最初のGPU検査は両forward成功後、CPU比較CSVのファイル名を誤って停止した。既存GPU出力の比較先だけを修正し、追加forwardやproduction変更なしでv2記録を作成、v1失敗記録は保持した。
- 所見別の入れ替わりを見た後の固定50:50 rank診断は12平均0.830249307、補助11平均0.815726517。gold・画像・モデルを読まず、係数/所見weight探索はしていない。全821検査を所見別に順位化し、欠損maskは評価時だけ適用。exact tieの二倍平均順位を整数のまま加算してから一度だけfloat化し、数学的な同順位を保った。[診断記録](../experiments/e008-e009-rank50-review-20261004.json) の共通helperと確定CSV全9,852値が完全一致、独立人工34拒否ケースと行順/左右交換/順位不変性も成功。固定BNとの差は補助11平均+0.029057、区間[+0.015129,+0.043443]だが、後付け候補選択の偏りを区間へ織り込んだものではない。順位値のBCEを確率校正の改善とは扱わない。
- [r001提出資産](../artifacts/kaggle/r001-bn-rank50-fold0/) は実選択2重み・凍結zip・診断とbyte同一の共通helper・未実行Notebookを含む。`artifacts/qa/20261004-rank50-contract/checks.json` で標準ライブラリ6契約チェックが成功し、保存済みCPU/GPU人工出力を各3×12で合わせたCSV契約もvalid=true。既存人工UIDだけを新QA内でnumericの架空UIDへ同じ対応で置換し、scoreと元CSVを保持した。モデルforwardやtorch importは行っていない。Notebookは2 checkpoint/source/helper hash、共通pixel fingerprint、cache1個・推論2回・全test一括rank・提出契約を確認する構成で、全code cellのcompileと未実行状態を検査した。
- 固定BN単体の [e009 v2資産](../artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) は元bundleに残った説明文のepoch誤記だけを修正した。`artifacts/qa/20261004-e009-bundle-v2-metadata/checks.json` でcode cell完全一致、重み/zip/index/config同一、旧6資産hash保持を確認した。追加モデルforwardは行わず、元Notebookと履歴を保持する。
- 公開方式の追加監査は [再現契約](research/public-reproduction-contract-20261004.json)。取得は公開code・metadata・小さい説明文書だけで、重み・MRI・ラベル表・予測配列取得、公開コード実行、外部書き込みはしていない。
- SDK版指定を`version_label="v32"`へ修正し、DINOsaur V32ソースを取得・hash固定した。[V32追補](research/public-reproduction-v32-followup-20261004.json)。表示0.937は作者報告で、歴史的Inputの厳密版と採点CSVは未確認。物理cropは [入力契約](research/physical-crop-contract-20261004.json) を調査しただけで、元MRIの追加取得・実decode・crop cache再生成はしていない。新しいKaggleアップロード・実test実行・提出も未実施。
- [完了時の検証記録](../experiments/e006-e009-verification-20261004.json) にテスト・各QA・主担当のrank9,852値再照合・Notebook/asset hash・JSON/文書リンク/台帳の確認をまとめた。[判断記録](../experiments/e006-e009-decision-20261004.json) は実施済みと次の提案を区別する。新しい台帳5行のgold/Public列は空欄で、r001へ学習時間を割り当てていない。

## 追加の添付分析と優先順位の再監査（10月4日）

- 新しい添付分析をHEAD `05b0710`、runtime/model/imaging/config、保存済み集計/epoch評価・予測CSVと照合した。原文を [user-analysis-review-20261004.txt](research/user-analysis-review-20261004.txt)、識別子を含まない数値/入力hash/調査根拠を [strategy-audit-20261004.json](research/strategy-audit-20261004.json) に新規保存した。
- C epoch 2→5は12所見AUC+0.007713、Synovitisは0.98→0.96、補助11平均+0.010232。7所見改善/5所見悪化で、Medial MeniscusはAUC−0.048177。同所見の悪化が全セルBCE増加の約77.8%を占める。添付の「Synovitis以外も改善」は支持するが、全所見の改善や長期化の成功を証明しない。
- goldを読まず、weak fold 0の812供給group/821検査を単位にpaired bootstrap（3,000回、seed20261004）。補助11所見C2−B1差+0.120629、percentile 95%区間[+0.078707,+0.162513]。C5−C2差+0.010232、区間[−0.005521,+0.025498]。12平均はSynovitis陰性1groupの不在で1,124回未定義となり、有効標本だけの区間を全体の確証に使わない。同じfoldでのepoch/config選択バイアス・患者独立性・教師誤差はこの区間の対象外。
- 現実装はBCE改善時のbest.ptだけ保存し、optimizer/scaler/RNG/Generatorの再開状態は保存しないことを確認。Cのepoch 5モデルも未保存。蓄積末尾はgroup_sizeで正しく補正され、fold 0の847回は更新機会でありAMPの実更新数の実測ではない。batch変更でmasked lossの検査重みも変わり得ること、config seedとmanifest seedの分離が未対応であることを確認した。
- 公式Overviewの検索索引でmacro12 AUC・Internet OFF/9時間・最終10月22日23:59 UTC（10月23日08:59 JST）を再確認。Host733826の索引本文で画像由来の正解とReportの不一致を確認。DINOsaur V4は表示Best 0.937 V32/最新V35、V32の厳密Inputは未取得。CoAtNet Training V8の未配布softラベルとgold選択は既存監査済みsourceを再読して確認し、gold選択を移植しない判断とした。
- 次の提案はAUC/BCE/lastモデル保存→fold 1のB/C各5 epoch→Cの20 epoch→BN固定だけの20 epoch比較。公開方式監査と自作Kaggle動作確認を並行させる。詳細は [対照実験の追加分析](controlled-experiments.md#追加分析に基づく優先順位10月4日)。
- 今回は調査とCSV/JSON再集計、文書更新のみ。src/config/script/test/Notebook/実験台帳を変更せず、学習・MRIデコード・GPU推論・データ/重み取得・外部アップロード・提出は実施していない。新しい提案を実装済み/学習済みと扱わない。
- 主担当も同じweak CSVと固定予測からbootstrapを独立に再計算し、上記区間を1e-12以内で再現した。調査JSONの厳密parse、保存source/添付hash、変更資料のローカルリンク40件とgit diff --checkが成功。GPUコード変更がないためunittest/人工CUDA検証は再実行せず、以前の71件成功と今回の調査検査を区別する。

## 対照実験の実装と診断（10月4日、再分析後）

- 共通tensor正規化、明示ローカルImageNet初期化、BN running statistics固定、epoch別weak CSV・所見別AUC/BCE/観測数/予測分布、固定train subsetのeval、gold監査無効化を実装。旧config/checkpointはrandom＋legacyを維持し、推論が初期重みを要求する経路はない。
- 追加評価でglobal Python/NumPy/torch CPU/CUDA RNG、DataLoader/sampler Generator、mixed train/eval modeを復元。人工CPU/CUDAでログ有無のsample順、model state、BN buffers、AdamW state/step、乱数状態が完全一致。詳細loss集計を付けても元のcanonicalセル平均BCEが完全一致した。
- 全71 unittestが成功。以前の31件にcache互換性13、model/input12、epoch診断3、対照集計7、実行記録5を追加。Ruff check/format（49 Pythonファイル）と差分の空白確認も成功。テストでは人工DICOMとノイズを使う。
- `scripts/check_synthetic_gpu.py`を追加。新規private QAに24架空weak＋予約2goldを用意し、gold cacheを作らず全cache readでgold UIDがないことをassert。両正規化でAMP/backward/optimizer/save、欠損mask、無効windowの予測不変、Reportなし3×12提出契約、メタデータ不整合の拒否を確認。実e002 checkpointも独立人工192px入力で契約がvalid=true。最大torch割当246,000,128 bytes、QA関数2.86秒で、実コンペ時間・性能とは区別する。
- 保存exportの完全性とpixel互換性を別検査するよう変更。実4,407検査・4,423ファイルのhash/CSV/UID/保存src/configが一致し、現repoのモデル・ログ変更を差分として明示しながら既存画像を再利用できた。保存exportを変更せず、imaging.pyのhashは従来と同じ。
- 公式URLからResNet18 ImageNet1K V1だけを明示取得。46,830,571 bytes、SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。公式prefixを照合しfull hashをローカル計算。コードlicenseと重みの条件を区別して記録した。実公式重みのfc込みstrict load、head/CPU RNG保持、人工CPU forwardを確認。元DICOMや新ラベルを取得していない。
- q001は原manifestを変えずfold 0学習側から16検査を選択し、dropout=0・100 epochで診断。初期同一subset eval BCE 0.694520、最良0.124106（epoch 32）、最終train-mode BCE 0.000188／eval BCE 0.179478、146.78秒。goldはpreflight/評価とも無効。定義可能AUCは7所見だけでmacro12は未定義。記録は [q001](../experiments/q001-train-fit.json)。これは少数trainへの適合診断で、独立CV・画像ラベルの正確さ・Publicの証明ではない。
- 公式SDKの通常認証とread-only要求でRules全文、Hostの外部LLM案内と全コメントを取得。認証値の表示・手動読み出し・コピー、外部投稿は行っていない。HostはreportからのLLMラベル抽出を条件付き許可し、NC制限だけでデータを禁止せず、賞金受領だけで商用とは扱わないと説明。公開READMEの想定用途とCC本文も照合し、現在の研究・学習目的と帰属等を維持して通常ライセンスで採用を進める判断に更新した。作者への追加許可を一律必須とはしない。確認UTC・URL・短い引用・根拠hashは [利用条件監査](research/kaggle-source-eligibility-20261004.json)。入賞時のWinner公開条文の不整合はその段階で確認する。
- [weak state監査](../experiments/weak-state-audit-20261004.json) でP/N/B/U/Mをweak全体・train・各foldに集計。Synovitisの全weak陰性は49件、fold別は1/25/7/10/6件で、全体の不足とfold 0の偏りを区別した。Effusion陰性2,674件はN 1,230＋B 1,444。N/Bは現行で同じ0でも意味が異なる。gold値・UID・group・reportは共有集計へ保存していない。既存分割とラベルは変更していない。
- e003/e004/e005を同じ192px・fold 0・seed・5 epochで新規実行。A→Bは正規化、B→Cはencoder初期化だけを変更し、MIL head初期hashも全て一致。Aの全5 BCEとbest予測CSVは旧e002と完全一致した。BCE選択の結果はA epoch 1: 0.430481/AUC 0.557023、B epoch 1: 0.411992/0.644652、C epoch 2: 0.371395/0.766895。cache事前検査込み・import除外の所要時間は1,104.29/1,043.59/1,039.36秒。gold cacheのpreflightと評価を無効にし、Publicは未測定。
- CSV/JSONのみの集計器でsource/config/manifest/input/subset/hash、全epochのUID集合・所見順・観測数・AUC/分布・historyとのBCE整合、first minimum-BCE選択とbest予測hashを監査しvalid=true。実checkpointのepoch/config/foldは別のCPU読み込みでも照合した。[比較集計](../experiments/e003-e005-controlled-summary-20261004.json)。CはBより12/12所見AUCが改善し、補助11所見平均も0.626893→0.747522。Cを暫定比較基準としたが、C epoch 5のAUC最大値0.774608を後付け採用していない。[判断記録](../experiments/e005-decision-20261004.json)。追加fold 1のB/C学習は未実行。
- 凍結code zipを新しいPythonプロセスで読み込み、実A/B/C checkpointと既存の独立人工192px・24窓cacheでReportなし3検査×12所見の提出契約が全てvalid=true。初期化関数とtorch.hub取得関数を呼べないようpatchし、Cの初期化pathを不存在に変えたQAコピーでもCSV hash一致。code zip SHA-256 `5f4f7a016225beec5e30d1696030f4ee879e7db1ceb600575396c436c6cacd78`、C checkpoint `677f5e793ebc9cfbb8eb76ab267582d7916b23034056af0bc3ac7c68bd448084`。[提出契約集計](../experiments/controlled-submit-contract-20261004.json)。実testのDICOM decode・Kaggle実行時間・新規採点を検証した結果ではない。
- `artifacts/kaggle/e005-pretrained-imagenet-fold0/` に上記code zipとbest.pt、hash/帰属記録、未実行の専用Notebookを新規保存。Input名は提案名で実mountは手動追加後に確認する。元MRI・レポート・ラベルCSV・認証値はbundleに含めず、アップロード・提出はしていない。過去e002/run/cacheを上書きしていない。
- C終了後、wrapperのKeyboardInterrupt/SystemExit記録とsource変更時のstatusを修正し、mockで中断/通常失敗/不一致/成功/再実行拒否/CUDA guardを確認。学習処理は変更せず、旧runの記録も書き換えない。今後は終了hashもcompleted保存前に確認し、対照集計器は記録済みの場合だけ追加照合する。
- 最終確認で共有JSON 21ファイル、Markdown 12ファイルのローカルリンク104件、PowerShellブロック22件を検査。台帳7行のID重複がなく、A/B/Cのgold/Public列は空欄。現sourceと実run、C bundleのコード・重みhash、専用Notebookの未実行状態と全コードcellのcompileが一致した。Gitで追跡されるdata/artifacts配下は従来の各READMEのみ。

## 添付分析の照合と次工程の調査（10月4日）

- 添付原文を [user-analysis-20261004.txt](research/user-analysis-20261004.txt) へ保存。SHA-256は `ee844f85bf1281db9d8e3a1a1f0261249ea887414c472dfd97a60e927fcb9f97`。判断は [次の実験計画](research/next-experiments-20261004.md)、出典・既存記録hash・容量計算は [調査記録](research/reanalysis-20261004.json)。
- 今回は実データ・重み取得、実MRIデコード、GPU推論、追加学習、Kaggle実行・アップロード・提出を行っていない。src・config・Notebook・script・test・実験台帳に変更はない。新しい学習結果を記録していない。
- runtime/model/imaging/contractsと現行設定を照合。所見別Attentionは既存、ResNet18はrandom、画像正規化は[-1,1]。batch 1でもencoderは同じ検査の24 windowを処理する。batch変更はBNだけでなく欠損mask付きlossの検査寄与も変える。学習と検証のloss集計方法も異なり、train-valid差を純粋な過学習量と解釈しない。
- installed PyTorch 2.10のDataLoader/RandomSamplerソースで、shuffle=Falseでもiterator生成がGeneratorを消費することを確認。train/valid共有Generatorに追加評価を入れると次epochの順序が変わり得る。専用評価GeneratorとRNG・module状態の復元、人工データでの学習不変性の確認は次工程の設計であり、まだ実装していない。
- export検査器が現在のsrc全体のhash一致を要求し、画像fingerprintがimaging.pyとpreprocess設定から決まることを確認。画像を変えないログ・モデル変更でのcache再利用には検査の分離が必要。既存hash保護を緩めていない。
- Notebook generatorの未圧縮容量式を再計算。4,407検査×3系列×8 window×3 channelで192/224/256/288/336pxはそれぞれ11,697,094,656／15,921,045,504／20,794,834,944／26,318,462,976／35,822,352,384 bytes。現行18,000,000,000 bytes未満の事前ガードは256px以上を止める。この値はrepoの保守的予算で、現在のKaggle公式枠や最大RAM使用量ではない。LIMITはprefix制限で全件shard処理ではない。
- Torchvision 0.25とPyTorch 2.10の公式資料、作者のsoft/hard比較、EfficientNet入力比較、既存CoAtNet model card、別Notebookのスコア表示、2024腰椎優勝者の記述を照合した。作者報告を自分の再現値と区別し、別条件の改善幅を現在の期待値へ転用しない。別NotebookのPublic 0.942は表示確認のみで、内部処理・Input・重みの来歴は未監査。
- 公式Overviewの索引本文で12所見平均AUC、Internet OFF・9時間、期限を再確認。最終期限は10月23日08:59 JST。CC公式資料とHostの別データへの回答から、提供元の条件と大会の条件を分けて確認する計画とした。Rules・外部LLM案内本文は取得できず、採用ラベルの許可・禁止は未確定。外部への問い合わせは送っていない。
- goldは既に監査値を見ているため完全未観測の最終holdoutとは呼ばない。学習・checkpoint選択・prompt・ensemble係数の調整には使わない。公開重みを後付けfoldで評価しても独立OOFとは主張しない。
- 調査JSONのparse、添付原文・保存コピー・既存2記録のSHA-256、全解像度の容量式、Markdown 11ファイルのローカルリンク、差分の空白を確認。実装・台帳に差分がないことも確認した。コード変更がないため31 unittestやGPU契約を再実行しておらず、10月3日の成功結果と今回の文書検査を区別する。Gitで追跡されるdata/artifacts配下は従来のREADMEだけ。

## e002の提出判断のための追加診断（10月3日）

- 今回の質問には保存済みCSVの再集計で対応し、GPU推論・実MRIデコード・追加学習・モデル取得・アップロード・提出は行っていない。学習や推論コードも変更していない。
- fold 0の学習3,386件で観測された所見別陽性率を定数予測として計算。検証やgoldから定数をfitしていない。同じ検証821件・観測5,727セルに欠損maskを適用したBCEは0.415326862。e002の保存予測からのBCEは0.430481425で、記録されたtorch集計値との差は1.2e-9未満。定数予測のBCEが低いことから、BCE単独を画像学習の有効性の証拠とはしない。
- 観測されたbinary weakラベルのmacro AUCは自作モデル0.557022616、定数予測0.5。BCEとAUCは異なる性質を測るため、「画像信号を全く学習していない」とは断定しない。特にSynovitisは陽性100・陰性1であり、weak macro AUCも不安定な指標を含む。gold 58件のAUCを隠しtestの予測値へ読み替えない。
- e001/e002のweak予測CSV同士、gold予測CSV同士のSHA-256がそれぞれ一致。5 epoch実験から提出するbest.ptはepoch 1が選ばれており、新しい予測改善はない。
- 識別子やレポートを含まない集計・元ファイルhash・計算方法・判断は `experiments/e002-submission-review-20261003.json`。学習側だけの陽性率計算、UID集合一致、BCEの再現、定数予測AUC=0.5をassertで確認した。
- 公式Overviewの検索索引で評価が12所見平均ROC-AUC、Internet OFF・9時間制限を再確認。Rules本文と採用ラベルのページ本文は今回のWeb取得でも開けず、大会利用の許可・禁止を確定していない。汎用事前学習候補について [Torchvision ResNet18の公式資料](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) を確認。実装・重み取得・効果検証は次工程。
- 診断JSONの構造と比較結果、ローカルリンク49件、差分の空白を確認。学習・推論・Notebook・設定・既存チェックの実装には変更がなく、全31 unittestは前回の成功結果を維持している。今回のCSV診断には上記assertを使用し、GPUテストは再実行していない。

## ラベル監査・固定分割・ローカル研究（10月3日）

- 認証済み公式APIから自分の提出履歴を読み、既存の1提出がCOMPLETE・Public 0.924と確認。提出日時は10月2日05:57:22 UTC、提出URLのscriptVersionId=354569007。該当Notebookの最新版metadataはVersion 1で、Inputはraptor-knee-widedense、Internet OFF。APIが返すInput参照は版なしなので、保存Input版・実行時間・global IDと表示版の直接照合は未確認。新規提出はしていない。結果はp001として台帳へ保存。
- 公開7候補のDataset版、metadata、ライセンス、生成方法とgold利用を公式CLI/SDKで監査。作者の学習コードはDreaddevelopment Version 8、Pilkwang Version 15を取得して読み、実行していない。前者のgold AUCによるcheckpoint選択、後者の学習分岐での公式ラベル混合を確認した。公開重みの厳密な履歴まで証明したものではない。
- vmohitrao Dataset Version 3を非商用研究候補として取得。元train.csv hash、weak_labels・weights・states・provenanceの4つの公開hash、4,349行のUID集合、全セルのstate/value/maskが一致。goldのUIDは含まれない。作者READMEが58検査を抽出・prompt開発から除外したと宣言しており、その根拠のhashを保存した。開発ログを独立監査してはいない。CC BY-NC 4.0の大会提出への利用可否は未確認。
- prepareでweak 4,207検査・gold 58検査を作成。全欠損138件、goldと同一レポート・groupで連結される4件を除外。供給report_groupを保った5-foldは821・891・888・778・829件。fold 0は学習3,386／検証821件。元入力hash、元ラベルとの一致、gold予約、group連結、fold別・所見別分布を追加監査した。
- fold 0のSynovitisは観測101件のうち陽性100／陰性1、欠損率87.7%。CSV整合性の成功はラベルの臨床的正確さやクラス分布の妥当性を保証しない。患者独立性も未確認。
- 新規の標準ライブラリunittest 12件でUNK→欠損、0とsoft labelの保持、出所hash・state/mask不一致、gold宣言不足、改ざんgroup、除外行を橋渡しにするgold漏洩、未観測クラス等を検査。既存を含む31件が成功。srcは変更しておらず、exportのsource hashと前処理fingerprintを維持。
- ユーザーはKaggleで元画像とcacheを目視比較し、見た範囲では問題なしと回答。件数とUIDは未報告であり、全画像の品質確認や医学的判定として扱わない。
- 人工ノイズ22検査とtest 3検査を独立したQAディレクトリに生成。192px・24 windowの設定でCUDA/AMP/backward/optimizer、1 epoch完走、checkpoint保存・再読込、Reportなしの提出契約を確認。全3行×12所見の契約がvalid=true。最大torch割当467,144,192 bytes。人工データの精度・時間・VRAMを実コンペ性能として記録しない。結果は `artifacts/qa/20261003-synthetic-cuda/checks.json`。
- 実キャッシュでe001-smoke-fold0が完了。weak検証BCE 0.4304814244、選択後のgold macro AUC 0.4871779862。checkpointは44,830,657 bytesで保存。historyの205.33秒はepoch内の経過時間で、cache事前検査と最後のgold評価を除く。コマンド全体の所要時間は未計測。
- e002-baseline-fold0は同じラベル・fold・seedでepoch数だけ5へ変更し、新規runで完了。weak検証BCEはepoch順に0.430481、0.430906、0.447424、0.502899、0.448503。最小のepoch 1を選択し、その後にgoldを評価した。gold macro AUCはe001と同じ0.487178。train関数は事前検査・最後のgold評価を含め1,082.83秒、Python importは計測外。epoch部分は1,034.21秒。実行環境・入力hash・config・全history・checkpoint hashはrunと共有可能な実験JSONへ保存した。Public LBは未測定。
- `artifacts/kaggle/e002-baseline-fold0/` を新規作成。code zipは許可されたsrc 9ファイルとbaseline.jsonだけで、SHA-256は9fd958c4585d3bced5552cb1c949a103ba6209fc4ddaa3e18b6d07219d49545f。別ファイルで置いたbest.ptのSHA-256は903cce571904a8e74ff54d3242e3cb6293db747b293475be378578767eb3f29b。展開zipのsource hashと学習記録が一致した。
- 新しいPythonプロセスで展開zipのsrcを読み込み、実e002 checkpointと人工test cacheで予測。Reportなしの3行×12所見について、UID集合・順序・列・有限値・0〜1範囲の提出契約がvalid=true。結果は `artifacts/qa/20261003-e002-submission/checks.json`。実MRI/testのdecodeとKaggle実行は未検証。bundleにRESEARCH-ONLY.jsonを追加し、アップロード・提出はしていない。
- 最終資料のローカルリンク48件、PowerShell手順ブロック15個、実験JSONと台帳3行の構造、保存run configと元configの一致を確認。run configのhashと元ファイルのhashは記録上区別した。Ruff・全31 unittest・差分の空白検査が成功。data/・artifacts/の追跡対象は各READMEだけで、ラベル・画像・分割・予測・重みはGit除外を確認した。

## 実学習へ移る前の確認（10月3日、以下は当時の状態）

- 取得済みのcoverage.jsonを集計。4,407検査すべて24 valid windows、選択された13,221シリーズすべてphysical_position順であり、記録された読み取りエラーとinstance_number fallbackは0。元画像やnpzの画像内容をこの確認でデコード・目視していないため、品質の確認とはしない。
- UIDやレポート本文を含まない集計を、Git除外の `data/exports/rsraki-rsna-knee-sv354838181/train-coverage-summary.json` へ新規保存した。
- 専用venvからKneeMILをevalモードでRTX 4090へ配置。人工入力1×4×3×32×32、有効mask 2枚でCUDA forwardを実行し、出力1×12と全値が有限であることを確認。公開重みを使わず、実学習、AMP、backward、optimizerの検証は行っていない。
- data/labelsは未作成。次の優先順位をラベル監査、画像目視、固定fold、1 epoch実行確認、基準実験、自作モデル提出へ更新した。実ラベル取得、実学習、外部アップロード・提出は実行していない。

## Kaggle Outputのファイル単位取得（10月3日）

- 指定URLはWeb閲覧ツールでは開けなかった。ユーザーのブラウザでのCLIログイン後、公式SDKで認証済みアカウントrsraki、private Notebook `rsraki/rsna-knee`、最新版Version 1を確認した。URLのscriptVersionId=354838181とVersion 1の対応はユーザー報告として記録する。
- Kaggle CLI 2.2.4をGit除外の `artifacts/tools/kaggle-venv/` に導入し、学習用venvから分離した。認証値は表示・コピーしていない。SDKの通常の認証を使用した。
- CLI 2.2.4の実装ではOutputの版suffixが要求へ渡されておらず、SDKのversion_label=1を明示した要求も対象Notebookでは404だった。版を省略したOutput要求は成功したため、取得前後に最新版番号を検査する取得器を追加した。将来最新版が変わった場合の古い版取得は未対応であり、異なる版を黙って取得しない。
- export.jsonを実取得し、complete=true、4,407検査、4,423ファイル、7,785,521,448 bytes、現在のsrcのhashとの一致を確認。保存Outputの全ページにmanifestの必要ファイルが揃うことを確認し、全ファイルを取得した。
- 取得器はSHA-256が一致する完成済みファイルを再利用し、不一致の再取得、`.part`からの置換、ページング、保存先の範囲、取得許可するファイル種別、空き容量を確認する。署名付きURLを記録へ保存しない。転送記録は画像exportの外側へ置く。
- 転送中にWindowsの長いパスがextended-length形式（`\\?\`）になり、通常表記のrootとの照合で停止した。取得済みファイルを保ち、取得器と既存export検査器の比較処理を共通化して再開。Windowsの通常・extended-length・UNC表記を比較時だけ揃え、保存先外のパスは引き続き拒否する。srcは変更しておらず、cache fingerprintは維持される。
- 新規の標準ライブラリunittest 7件で、Windowsの危険なパスとextended-length表記、同じサイズの破損ファイル、再開、hash不一致時の既存ファイル保持、debug/不正export、全ページ取得、取得中の版変更の拒否を検査した。既存テストも含めて19件が全て成功した。ネットワークを使うテストは追加していない。
- ローカルに取得したCSVを純粋なCSV監査CLIで再監査し、4,407検査・24,371シリーズ、58検査の全12項目gold、シリーズなし・空レポート0、同一レポート54groupを確認。患者独立性の確認ではない。集計は `data/exports/rsraki-rsna-knee-sv354838181/csv-audit.json` に保存した。
- 取得後のexport検査が `valid=true, studies=4407, files_verified=4423` で成功。inventory、全ファイルのSHA-256・サイズ、元CSV、npzのUID集合・件数、現在のsrc、前処理fingerprintが一致した。再開時は3,219ファイルを再利用し、残り1,204ファイルを取得。親ディレクトリのtransfer.jsonにtransfer_complete=trueと検査結果を保存した。
- 専用venvはtorch 2.10.0+cu128・torchvision 0.25.0+cu128になっており、torch.cuda.is_available()=trueとRTX 4090を確認。今回実行したforwardは既存unittestの人工画像CPU経路で、実学習のAMP/backwardを確認したわけではない。
- `ruff check .`、`ruff format --check .`、`git diff --check` が成功。ローカルリンク45件とPowerShellの手順ブロック7個も確認。取得データ・Output・CLI環境はGit除外で、data/とartifacts/の追跡対象はREADME.mdだけである。

## 全件キャッシュ実行のユーザー報告（10月3日）

ユーザーからKaggleでLIMIT=Noneの実行が終了したと報告を受けた。最後のexportセルの成功、complete=true、保存したNotebook版、Outputのファイル集合・件数・hashはこの作業では確認していない。保存版の閲覧画面からOutputを一括取得する方法と、private Dataset経由の代替方法を調査して案内した。実データのダウンロードやMRIデコードは実行していない。

[KaggleスタッフのOutput保存・取得用Notebook](https://www.kaggle.com/code/paultimothymooney/how-to-save-a-file-to-the-notebook-output-folder/output) の索引で「Download notebook output」を確認し、[公式Dataset資料](https://www.kaggle.com/docs/datasets) でOutputからDatasetを作成する経路を確認した。認証後のユーザー画面と現在のボタン配置は未確認。PROJECTの進捗を更新し、今回の資料差分をgit diff --checkで確認した。

## 初回提出のユーザー報告

ユーザーからNotebookの提出受付成功の報告を受けた。Kaggle画面、採点結果、Publicスコア、Notebook版はこの作業では確認していない。公開モデルの採点成功や自作モデルの学習成功とはまだ扱わず、次は採点結果の確認と10検査のキャッシュ準備へ進む。PROJECTの状態だけを更新し、未確認の数値を実験台帳へ追加していない。

## キャッシュ作成後の手順4〜9の確認

- [after-cache.md](after-cache.md) を追加し、README、PROJECT、GPU開始手順、roadmap、提出手順から参照できるようにした。変更対象は資料のみで、src、config、Notebookの実装は変更していない。
- 記載したprepare・train・転送検査・梱包コマンドの引数を現行CLIのhelpと照合し、PowerShellのコードブロック5個を構文解析した。実データに対するコマンド実行はしていない。
- README、PROJECT、docs内のローカルファイルリンク42件を確認。ラベルCSVの13列が提出契約と一致し、epoch数だけの変更ではcacheの前処理fingerprintが変わらないことを確認した。
- 専用venvで既存unittestを再実行し、10項目成功・2項目skip。人工DICOMを使用し、torch依存のforward・loss/CUDAガードとcacheテスト内の推論分岐は未実行。`doctor` でもtorch/torchvision未導入を再確認した。
- `ruff check .`、`ruff format --check .`、`git diff --check` が成功。
- 公式の索引から提出条件と日程を再確認した。直接openではKaggle本文を取得できなかったため、Rules細目やアカウント別上限を確認済みとはしない。引用元は手順書に記載した。

資料の確認は、実ラベルの監査、実MRIの品質、CUDA学習、Kaggleでの提出成功の代わりにはならない。これらの結果はまだない。実験台帳へ未実施のスコアやrunを追加していない。

## 10月2日のデスクトップでの確認

- `nvidia-smi` でRTX 4090、VRAM 24,564MiB、ドライバ591.86を確認。Cドライブ空きは484,710,727,680 bytes。RAM容量はOSの読み取りが拒否され未確認。PyTorchのCUDA利用は未確認。
- Windowsの `python` エイリアスが動かず `.venv` も存在しなかったため、Git除外の `.python/` と `.venv/` へPython 3.12.13を用意。numpy 2.5.3、pydicom 3.0.2、Pillow 12.3.0、ruff 0.16.10と本パッケージを導入した。torch/torchvisionは未導入。
- この環境でunittestを実行し、**10項目成功、2項目skip**。skipはtorch/torchvisionが必要なモデルforward・loss/CUDAガードのテスト。画像cacheテスト内のtorch推論分岐も実行していない。実DICOMではなく人工の非圧縮single-frame DICOMを使った。
- 新規 `00_prepare_cache.ipynb` の展開・cache作成・export記録セルを人工4検査で実行し、全件exportの転送検査が成功した。ファイル破損、debug export、不完全なhash一覧の拒否も確認。Kaggleパス解決セルと画像viewerはこの確認の実行対象外。
- 手順1〜3の説明時にコードInputの展開形式を補正。両Notebookのzip・展開済みディレクトリからのコード読み込みを確認し、ディレクトリ形式で無関係なファイルをコピーしないことを検査した。キャッシュNotebookは両形式とも人工1検査からexport検査まで成功。提出Notebookの依存確認はmockで省略し、torch推論は実行していない。記録は `artifacts/qa/20261002-code-input-formats/checks.json`。
- 両Notebookのコードセルを構文検査し、保存された出力と実行回数が空であることを確認。generatorからの再生成が一致するよう、既存提出Notebookのimportと書式も整理した。
- `ruff check .` と `ruff format --check .` が成功。
- `artifacts/kaggle/cache-v1/rsna-knee-code.zip` を生成し、srcのPythonファイルとbaseline.jsonだけの許可リストを確認。データ・重みを含めていない。アップロードしていない。

人工exportの検査記録はGit除外の `artifacts/qa/20261002-cache-export/checks.json`。Notebookは未実行テンプレートとして保存しており、Kaggleセッションでの動作、圧縮MRIのdecoder、実キャッシュサイズ・実行時間、CUDA/AMP/backward、学習精度・採点成功は未検証。

## 以前のCPU環境で確認済み

- unittest 12項目が全て成功。CSVの列順・UID・欠損・有限値、AUCのtieと未定義クラス、gold除外、同一レポートと追加groupの連結、空欄ラベル・出所不足の拒否を確認した。
- 人工のsingle-frame非圧縮DICOMを読み、ファイル名とInstanceNumberが物理順序と違う場合も、物理座標順にcacheを作れることを確認した。
- cacheの再開、設定不一致の拒否、全シリーズ欠落時の停止を確認した。
- maskされた画像を変えても予測が変わらないこと、欠損ラベルのloss除外、CUDAなしで学習が開始されないことを確認した。
- ランダム初期化checkpointを保存・読み込みし、CPUで人工画像の推論から提出CSV検査まで実行した。これは精度の検証ではない。
- ruff checkとformat checkが成功した。
- 提出Notebookのコードセルを構文検査。セル出力と実行回数は空で、未実行テンプレートとして保存した。
- コードzipの許可リストを検査。srcのPythonファイルとbaseline.jsonのみで、データ・重み・認証情報を含めない。
- プロジェクトのローカルリンク、Git差分の空白を確認した。

完全な12項目の実行には、既存のPython 3.10.11環境のtorch 2.7.0+cpu、torchvision 0.22.0+cpu、Pillow 11.2.1と、Git除外のQAフォルダへ取得したpydicom 3.0.2を使用した。他プロジェクトの環境やファイルは変更していない。

対応対象のPython 3.12.10でも軽量ツールを確認した。画像依存を用意した環境では10項目成功、torchなしの2項目をskip。このリポジトリ専用の `.venv` には基本パッケージだけを導入し、7項目成功、画像/GPU依存の5項目をskipした。専用venvへGPUライブラリは導入していない。

## KaggleとローカルGPUで行う確認

- 実CSVの列・件数・欠損・表記と、ラベルの出所・版・利用条件。
- 実MRIの圧縮形式、decoder、方向、順序、PixelSpacing、左右、両膝、multiframeや不正画像。
- 元画像とcacheの目視比較、所見が残ること、各クラスの陽性・陰性分布、患者・施設・画像重複の漏洩。
- CUDA/AMP、実際のbackward・optimizer・checkpoint選択、GPUでの学習時間とVRAM。
- 大きな検査を含む前処理込みの推論時間、Kaggleのオフライン依存、隠しtestでの採点成功。
- 公開Notebookの指定版・入力版の再現、実測Public LB。

初期版には学習再開、分散学習、事前学習encoder、ensemble、物理尺度cropを含めていない。競争力のある公開モデルの再現を優先してから、比較に必要な機能を追加する。
