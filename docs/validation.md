# 準備内容の検証範囲

更新日 2026年10月8日 JST。18:03 JST公式APIでp005 COMPLETE・自己Public0.950を確認。0.955/0.960超への調査と実験計画を準備。f004両fold実学習・教師17二次レビューは完了、差し替え/教師訂正は採用しない。以下の採点待ちや未開始表示は各作業時点の履歴。

## 自己Public0.950確定と改善計画（10月8日）

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
