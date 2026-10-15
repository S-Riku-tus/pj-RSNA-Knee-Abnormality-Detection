# Public 0.960に向けた実行方針

**10月14日14:59 JST：自己最良Public 0.956を維持し、S6＋F006両fold E12・固定20％の正式提出57169291を受理確認しました。目標0.960以上は未達・継続中です。** [最新の記録](../experiments/F006-bothfold-formal-Submit57169291-Public956-checkpoint-20261014.json)。可視3検査・20出力を再計算してRoot受理後、Submitを1回実行。V22Rは公式COMPLETE・自己0.956、P028・P024・F006両foldはPublic結果待ちで、旧提出は再送しません。本日4件・残枠1・累計27件は前回公式値と今回受理1件からの導出で、提出後の公式枠再照会値ではありません。

F006の可視3検査成功はhidden完走やPublic精度の確認とは別です。F007も可視20出力はRoot受理済みで正式Submitは0。次は同じF006重み・同じ固定20％で、親モデルだけをS6から自己0.956のV22へ切り替える候補を検証します。V22作者報告0.956と自己値は区別し、Docker差37c64…／2757e…、公開重みのGold選択履歴の不確かさを保持します。MRI-COREは条件付き調査候補で、重み取得・実装・追加学習は開始していません。以下は過去時点の記録です。


**10月14日14:33 JST：自己再現V22Rの公式Public 0.956を確認し、現在の最良値として採用しました。従来0.954から+0.002、目標0.960以上は未達で継続中です。** [採用記録](../experiments/V22R-official-Public956-Root-adoption-20261014.json)。Submit57164195を自己アカウントの公式Get・ListともにCOMPLETE・0.956・エラーなしで確認。作者V22の報告0.956と自己値0.956は別記録です。P028・P024はPublic未返却で再提出せず、本日3件・残枠2・累計26件です。

V22は元コード6セル・Input3固定版・推論式を保持しましたが、Dockerは作者要求37c64…から実保存2757e…へ変わっており、R7単独の改善や非公開精度・公開重みのGold cleanを断定しません。次はS6＋F006両foldの固定20％を正式対照として検証し、同じF006重み・同じ係数でV22へ親モデルだけ切り替えて比較します。F006の新native64・内容SHAとF007可視20出力はRoot受理済み。MRI-COREは条件付き調査候補に留め、重み取得・実装・学習は開始していません。以下は過去時点の記録です。


**10月14日14:21 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [最新の記録](../experiments/F007-visible-pass-F006-bothfold-native64-pass-20261014.json)。14:13 JSTの公式Get／ListではP028・V22R・P024のraw状態とPublicは未返却、エラーなし。本日3件・残枠2・累計26件です。

F007は可視3件・20出力の検証を通過しRoot受理、正式Submitは0です。F006両fold E12も64 MRI・256 forwardと新重みの内容SHA確認を通過しRoot受理しました。F006の提出推論・Publicはこれから検証します。次はF006を優先し、V22との組み合わせは公式スコアを見て判断します。ローカルMRIデコードと追加学習は0、CPU復元とKaggle MRI推論は実施済み。旧エラーを含む以下の記録は過去時点のものです。


**10月14日13:45 JST：確認済み自己最高Public 0.954。目標0.960以上は未達で、正式提出3件の採点待ちです。** [今回の正本](../experiments/F006-bothfold-CPU-private-ready-F007-finalmask-error-20261014.json)。13:35 JSTの公式Get／ListはP028・V22R・P024のraw状態／Public未返却、エラーなし。本日3件・残枠2・累計26件です。

F006両fold E12の新4モデルは実CPU復元と独立検証を通過しRoot受理。CLS0・REG0・REG1の実指紋は保持し、CLS1だけを置き換えました。fold1モデルの非公開Dataset12573715/V1はREADY・2ファイル名／サイズ一致を読み取りで確認済み。Create応答guardの停止を保存し再送せず解決しました。新しい推論ソースへ伝播中で、Kaggle上の内容SHA・新native64・Serving・Publicはまだ未検証です。

F007のWindowsパス修正job138608301/V1はS6側の可視3件を完了後、最終mask処理でKeyError serving_shasとなりERROR。Submit0のまま、最終CSVまでのメタデータ契約を人工データで検証しています。元の数値コード・旧エラー・native64記録は保持。ローカルMRIデコードと追加学習は0、必要なCPU復元・Kaggle MRI推論は実施済み。以下は過去時点の記録です。


2026-10-14 JST。目標は**自分の正式提出によるPublicスコア0.960以上**。指標は12所見のmacro ROC AUCであり、正解率ではない。自己最高は0.954、追加候補P026は0.953。作者値・weak検証・自己Publicを混同しない。

13時台の[実行記録](../experiments/three-submitted-two-native64-pass-F007-visible-error-20261014.json)：P028・V22R・P024の3件を正式提出済みで、スコア未返却・本日残枠2。P024とF007は各64検査／256 forward・内容SHA・FP32集約・資源と終了処理の再検証を通過しRoot受理済み。P024は可視3件／20出力も通過して正式提出57166365を受理。F007の提出用job138605901/V1はコード・Input9・設定の一致を確認したがERROR終了し、Submit0のまま原因を調査中。弱差・64検査・可視テストの成功をPublicの改善と混同しない。

公式の最終提出期限は2026-10-22 23:59 UTC（**日本時間10月23日08:59**）。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)を2026-10-14に再確認した。提出推論はInternet OFF、9時間以内、`submission.csv`、12所見の確率が必要。

## 現状からの判断

- F006の学習率変更、F007のCLSを保つ局所残差、F008の残差学習24epoch化は両foldとも完了した。文書の旧「学習中／待機」は履歴であり、現在の状態ではない。[全96epochのcheckpoint・CSV監査](../experiments/goal-state-audit-20261014.json)はPASS。
- F006のBCE主選択は旧モデルとの混合で改善を示さなかった。一方、固定10所見AUC副選択のE12はfold0 +0.004010、fold1 +0.002311で、供給groupの条件付き95%区間も正だった。[比較記録](../experiments/f006-completed-twofold-selection-comparison-20261014.json)。元BCE bestを保持し、副選択を別候補にする。
- F008の24epoch化はF007からfold0 +0.001515、fold1 +0.000329。fold1の区間は0を跨ぐ。[比較記録](../experiments/f008-twofold-completed-weak-comparison-20261014.json)。さらに48epochへ延長する優先度は低い。
- 以上は元REGとの固定50:50のweak比較。選択後の同じ検証データでの診断なので、未知データの改善保証やPublicの推定値に使わない。fixed10はMCL・Synovitisを除いた診断で、提出指標の全12を置き換えない。全12のbootstrapで未定義になった反復も保存する。
- [F006とF007の直接比較](../artifacts/research/20261014-f006-vs-f007-direct-weak-complementarity-v1/summary.json)では、同じREGとの固定混合でF006 E12が両foldのAUCで上回った。事前固定CLS50:50の追加混合はBCEを改善したがAUCは下げ、誤りの相関も高かった。比率探索はせず、F006 fold1 E12も導入する候補を優先する。現在の旧f005 fold1 E9と、f006自身のBCE best E9は別checkpointとして保持する。
- [Versia7 V22](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7?scriptVersionId=358019432)の作者報告は0.956。元6コードセル・3Input版を保った自己候補は可視推論とCSVの再計算を通過し、正式提出57164195が受理された。自己Publicは未返却。Kaggleが保存した実Dockerは要求した作者Dockerと異なるため、その差を記録して同じrecipeの実保存環境での再現として扱う。新R7重み、stacker、50:50混合、失敗時の順位処理などが変わる。[版・Input・gold利用・manifest rawとwrapperの別監査](../experiments/public-research-096-20261014-v2.json)を参照。

## 段階的な実行

1. **提出推論を成立させる。** S6 control 0.954を固定し、P028の親guard修正をKaggle可視推論、正式提出のhidden再実行の順に検証する。可視3件のPASSだけで全件完走やスコア改善を主張しない。
2. **最新版の再現値を取る。** V22の元6コードセルと3Inputの版を固定し、自己アカウントの非公開Notebookで再現する。数値処理を保った提出契約確認だけを加える。作者0.956との差が出たら、追加モデルを混ぜる前に前処理・Input・Docker・fallbackの差を確認する。要求Dockerは`37c64f7d…`、実保存は`2757e0c7…`と実照会で確認済みで、作者環境の厳密再現とは主張しない。
3. **完了済みモデルを一候補ずつ試す。** 先にP024のF006 fold0 E12を、既存の最高S6 controlと事前固定Native20%で比較する。fold1 E12は別のCPU復元・輸送・Kaggle推論確認後に追加する。F007/F008はこの比較結果と補完性を見て次の候補にする。
4. **補完性を確認して次の混合へ進む。** UID・fold・入力が一致する予測だけで所見別相関と誤りを確認し、候補と混合比を提出前に固定する。V22が自己再現できた場合は、V22を新しい比較基準にする。異なるレポートラベルと画像ラベルの関係は未検証として残す。
5. **追加学習は不足が見えた後。** まず完了したモデルと公開recipeのPublicを取る。次に必要なら局所crop、スライス密度、補助教師を一要因ずつ比較する。曖昧なレポートラベルの改修は訓練partition内で行い、goldやPublicによるcheckpoint選択をしない。

## 採用と停止の条件

自己Publicが現基準を上回れば採用候補とする。同値・低下は改善と扱わない。0.960以上を公式Get／Listで確認し、エラーなしの完了状態と受理済み提出の系譜を束縛するまでGoalを完了しない。Publicとprivateの一致・患者独立性は未確認で断定しない。

学習・CSV比較・保存・推論・採点は別の段階として記録する。モデル・設定・seed・fold・入力SHA・環境・採用理由を新しいrun／観測ディレクトリに残す。旧run、BCE best、審査拒否、失敗した送信attemptを上書きしない。

ユーザーは2026-10-14に、今回と今後の目標に必要な自己`rsraki`への非公開送信・実行・手動提出を明示承認した。[承認範囲](../experiments/user-authorized-private-kaggle-scope-20261014.json)。MRI、Report本文、教師行、認証情報は送信対象に含めない。送信先・実ファイルSHA・内容を各操作前に確認し、結果不明のCreate／Submitを再送しない。自動アップロード・自動提出のコードは追加しない。
