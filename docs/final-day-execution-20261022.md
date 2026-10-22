# 2026-10-22 最終日の実行記録

## 16:18 JST更新

公式Get/List/limitsは正式45件・正常採点34・例外10・未返却1、最高.958・残4件で一致。P033/P048のCOMPLETE/.958、P050 ref57512319/script361027273のPublic未返却を再照合した。[観測](../artifacts/research/20261022-final-live-state-v2/observation-20261022T071806958844Z/summary.json)。P049/P052は可視RUNNINGを継続し、P049の公式live logを最大25秒の読取限定で一度取得したがHTTP応答前にReadTimeoutだった。[証拠](../artifacts/research/20261022-P049-live-stage-readonly-v3/receipt.json)。再Save/再Submitは行っていない。

Tiny両foldはE3で、16:18実測はVRAM4,259MiB、物理空16.05GiB・commit空18.33GiB。時間・資源停止条件を維持する。新CNN接続の静的点検では、旧routerが持つgeneric factory回数1の記録を新CNNへ流用しないよう、元DINO 1回＋CNN 2回と現model classを明示して独立検算するよう修正した。関連stdlib6試験・Ruff PASS。実E12重みがまだないため、この変更は現GPU/MRI資格の証拠ではない。学習source16やP049/P052保存sourceは不変。

公開情報は既調査との差分だけ確認した。[採点待ちDiscussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751502)には14:40:57 JSTにJackが6.5時間後のtimeoutを報告し、[Kernel Stuck](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751576)には14:56:55 JSTに参加者が再実行を提案していた。Kaggle運営の原因説明や完了保証ではなく、自分のrunの停止・再実行を正当化する証拠にも扱わない。公式SDK本文/返信rawと取得時刻は `artifacts/research/20261022-{scoring,kernel}-wait-public-delta-v1/` に保存。外部書込0。

## 15:52 JST更新

**Tinyの両fold E2が完了し、事前固定した時間・メモリ条件を満たすため固定E12を継続する。** fold0のE1/E2は2022.687／2003.891秒、fold1は2018.818／1997.595秒。中間精度は採否に使用していない。遅い直近2epochの速度を使う保守的E12完了見込みは21:30頃で、00:30の作業目標内。以後もrolling速度・物理空6GiB/commit空10GiBの余裕を監視し、超過時は自分のrunだけを停止する。[判断と実更新数](../artifacts/research/20261022-F016-first-two-epochs-time-review-v1/decision.json)。全12epochの完走・精度・CNN実servingは未確認のまま。

15:43公式Get/List/limitsもP050未返却・最高.958・残4。P049/P052は15:47にも可視RUNNING。追加のlive log読取はSDK request routeが400、SDK CLIと同じGET routeは読取timeoutで内部段階未確認。再Save/再Submitはしていない。read-only watcher v2は両fold E2完了eventで終了した（session14714）。学習supervisor session38818は継続中。

## 15:22 JST更新

F016の両foldはE1を完了しE2へ進んだ。E1はfold0 2022.687秒、fold1 2018.818秒（各約33.7分）、実更新212/212と208/208、AMP skip 0。中間精度を採否に使用していない。同速を仮定すればE12は21:30頃だが、正式な時間継続判定はE2完了後の遅い速度で行う。CNNの実E12 CPU資格器とP049からCLS2だけを置換するP053生成器も準備し、未資格packageを受理しない接続境界をstdlib6試験で確認した。実CNN package/Notebookはまだ生成していない。

本人から最終選択に関する回答としてP050 URL（script361027273）を受領し、正式ref57512319へ対応付けた。SDKでは選択状態自体を確認できず、選択済み件数・他の選択は不明。最終選択操作はしていない。15:16の再照会もP050 Public未返却・最高.958・残4。P049/P052は15:20の可視statusがRUNNING。待機と未提出を区別し、二重実行/二重提出を避ける。

[引継ぎ用の固定記録v2](../experiments/final-day-execution-20261022-v2.json)には現source/コード/証拠hashを保存した。v1はPowerShellのパイプ入力が既定ASCIIだったため日本語keyが`?`へ置換され、key衝突もある無効記録として訂正証拠ディレクトリの`invalid-final-day-execution-v1.json`へ保存した。今回追加した台帳5行の説明だけを日本語へ訂正し、それ以前の行はbyte保持。P050の公式提出説明にも置換があるが、ファイルから送ったNotebook sourceとCSVは検算済みhashと一致している。公式rawは変更せず、今後は`$OutputEncoding`もUTF-8指定する。[訂正証拠](../artifacts/research/20261022-Japanese-record-encoding-correction-v1/correction.json)。

15分を上限とする読取専用watcher（session21017）は候補の終端、公式結果、両fold E2完了のeventでrootへ戻す。外部書込・学習開始・停止操作は持たない。学習の資源/時間停止supervisor（session38818）は別途継続中。

## 15:11 JST更新

**P050を正式ref57512319として15:03 JSTに受理。scriptVersion361027273・kernel140159065/V1。Publicは未返却で、最高0.958・目標未達・残4件です。** 15:04のGet/List/limitsは45件、正常採点34、例外10、未返却1。提出前に今回の可視3検査・全枝成功・保存コード/版・Input/Docker・所見別36値の独立FP32検算を確認しました。証拠は `artifacts/research/20261022-P050-formal-submit-v1/` と `artifacts/research/20261022-final-live-state-v1/observation-20261022T060358803400Z/`。採点待ちを0や改善成功と扱いません。

P049修正版（kernel140158980/V1）は可視RUNNING継続。保存した4コードセル・Input正規化ラベル・Docker・非公開/offline/T4は読み戻し一致。P052（kernel140160594/V1）を15:02:38 JSTに手動Save受理し可視RUNNING。P051は `artifacts/kaggle/p051-native10-reader-20261022-v1/` へsource準備のみで、Save/Submitなし。P049の4セルとP041のReader本体・preflightをbyte一致で保持し、Reader失敗時の親CSV復旧を成功と数えない確認を追加しました。P049が公式.957以下なら優先から外す条件を維持します。

F016の両foldはE1学習を終えつつあり、完全epochはまだ未返却。E1途中の精度は採否に読みません。新しい [固定E12監査器](../scripts/check_f016_fixed12.py)、[条件付きpackage生成器](../scripts/prepare_f016_serving_package.py)、[CNN専用constructor接続runtime](../scripts/f016_cnn_serving_runtime.py)を準備しました。未完の新runは数値評価前に拒否します。旧F014の順序policyやpreviewは変更していません。

`artifacts/research/20261022-F016-new-namespace-CPU-qa-v1/result.json` は、新しい推論namespaceのsource16から元generic全重みを初期化し、既存E0人工checkpointだけをCPU復元する12確認にPASS。prefix/全selected tensor/全state不変、4窓3plane有限12出力、padding不変、全欠損拒否、CPU RNG不変、CUDA未初期化を確認。GPUの重複試験・臨床画像/ラベル・学習はこのCPU確認で使っていません。実F016 E12・REG結合・T4・MRIの資格ではありません。別途stdlib5試験で未合格package拒否、元REG/pair保存、P051の元source保持とReader失敗拒否を確認し、今回の追加コードはRuff PASSです。

## 14:52 JST時点

引継ぎ時にGit差分なし。14:40 JSTの本人SDK Get/List/limitsは正式44件、正常採点34件、例外10件、未返却0件、最高0.958、残5件で前回と一致した。正常採点はPublic非null・errorDescriptionなしで分類する。P048のref57464683とscript360631920を再照合。0.959は未達。観測は `artifacts/research/20261022-final-readonly-state-v1/observation-20261022T054012911817Z/`。締切は10/23 08:59 JST。

14:41実測はRTX4090 877/24564MiB・2%、物理空17.65GiB・commit空30.56GiB。WDDMのcompute-appsにもGUIが列挙されたため、PIDをGet-ProcessのOS名と照合した。学習Pythonなし、既存pythonwとGUIは停止していない。CIMはアクセス拒否のためGlobalMemoryStatusExとGet-Processを使用。

既存外部操作許可は `experiments/goal959-resumed-scope-and-P043-prospective-20261020.json` に手動提出を含む継続scopeとして記録されている。今回の指示に従い、この記録を確認してから限定した手動Saveを実施。自動アップロード・自動提出コードは追加していない。最終選択操作の既存明示許可は別途確認が必要。

## 最優先P049

- [生成器](../scripts/prepare_p049_native10.py)は正式P048 V4のsource SHAを固定。内包16ファイル中、blend本体2本と親extensionの3本だけを変更する。F013 CLS2・既存REG2・P033科学8セル・前処理・fold/家族50%・Input全版指定・Dockerは同じ。
- alpha、runtime引数、受理guard、formula、nominal receiptを20→10%へ変更。成功集合内の平均tie順位、乗算前FP32、別積2回と加算1回、親/最終の非再順位化、失敗セル・MCL/Synovitisの元文字列を維持。
- [標準ライブラリ試験](../tests/test_p049_native10.py)7件PASS。独立Fraction順位＋struct FP32、FP32衝突tie、逆UID順、成功部分集合、失敗・除外文字列、alpha不正/nonfinite拒否、receipt nominal係数、元restorerによる全16ファイル実展開を確認。Ruff PASS。
- 初回 `p049-f013-native10-20261022-v1` kernel140158616/V1はsource ZIP形式でモデル前にERROR。正式提出0。圧縮ZIPを元と同じ無圧縮ZIPへ戻した。元guardは変更していない。旧source/runを保存。
- 修正版 `p049-f013-native10-20261022-v2` kernel140158980/V1を14:48:58 JSTに手動Save受理。source SHA `465beec9a8ee9d7b013a1ca88545471ee749e6c7dfd970d140346b12f47ea5e0`。現在は可視推論待ち。正式提出はまだ0。

根拠のweak proxyは再利用809/855検査・単一除外fold R6で、P033全OOFや実native2fold平均ではない。10%最適・Public増分・非劣性を主張しない。可視出力の当該source/モデル/CSV検算後に正式提出を判断する。

## 独立係数候補

[生成器](../scripts/prepare_final_p033_coefficients.py)でP050とP052を準備。P033の7/8科学セルはbyte同一で、最後の係数セルだけ変更。全Input/Docker/GPU設定は元P033と同じ。構文確認済み。これはnative10より弱い根拠の探索である。

P050はLOA=.25、PF=.55、Synovitis=.75、Baker=.40、他=.50、A224=.35。`p050-p033-vector-reverse-20261022-v1` kernel140159065/V1を14:49:35 JSTに手動Save受理。source SHA `28da44616824dd78d6e8395fa4c6b970fe4c57465bffa4b0f8d86989f9e2d6e4`。元payloadに含まれたローカル照合欄がSDK fieldではないため、最初の手動呼出前に検出・除去した新規package-v2を使用。Save APIは一度のみ、正式提出0。

P052はA224だけ.35→.20、`artifacts/kaggle/p052-p033-a224-low-20261022-package-v2/` に準備済み。source SHA `98c6e7876408dd388a49c20addb918fda4920a947ba9db7ae8f53dcef2fbd55e`。まだSave/Submitなし。P051 Reader追加と最後1枠は保留。枠消費を目的にしない。

## 新scope F016 Tiny

14:42:28 JSTにTinyだけを初期重みからfold0/1各12epochで起動した。旧F014 E3から再開せず、旧run・旧fold0先行policyを維持。新規runは `artifacts/runs/f016-convnext-tiny-fold{0,1}-fresh-20261022-v1/`。起動・source/config/入力/generic重みhash・実行前資源・採否条件は、結果を見る前の `artifacts/research/20261022-F016-tiny-fresh-twofold-v1/policy.json` に固定した。launcher PID19420/9068、監視supervisorは同dirのlaunch.json。exec session38818。

既存16source/config/generic重み/人工GPU資格が一致することを起動時と元train preflightで照合。CUDAチェックは維持。学習窓/teacher/LR/head/gamma/seedは変更しない。現在E1で両foldとも768検査・48実更新/48機会まで記録、約415秒。完了epochの速度は未測定。

最初の完全2epoch以降、遅い方の直近2epoch速度から固定E12完了を外挿し、10/23 00:30 JSTを超えるなら自分の両runをSTOP_REQUESTEDで停止する。60秒監視で物理空6GiBまたはcommit空10GiB未満が3回続く場合・片fold例外・作業期限到達も停止。途中epochを提出へ救済しない。他processやOS設定を変更しない。

採用には両fold固定E12と工学監査の完了、F006対照、同じREGとの50%家族、実F013+REG50%家族対照、同じ809/855 R6+native10 proxyを確認する。CNN家族が実F013家族を両foldで改善し、proxyが両bank非悪化を要求。固定10所見・全12支持数併記・alpha=.10/家族=.50固定、後付け係数探索なし。新CNNのserving runtimeは未完成で、復元/現モデル人工GPU/MRI/失敗契約/CSVが別に必要。

元DICOM一括取得0、本人Gold数値のfit/checkpoint/係数選択0、患者独立性は未確認。公開重みの上流Gold接触は従来監査を継承し、本人非利用と同一視しない。
