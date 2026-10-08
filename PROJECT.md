# RSNA Knee プロジェクトの現在地

更新日 2026年10月8日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、**自己採点済みp005のPublic 0.950を基準に、次の0.955・最終0.960超を目指す**ことです。p002自己0.937と過去runも保持し、公開レシピ改善と独立した自作学習の評価を分けます。

## 最新：自己Public0.950確定、基準を固定して一要因の改善へ

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
