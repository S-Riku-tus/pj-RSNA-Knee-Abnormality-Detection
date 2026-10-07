# p003の実行順改善候補

更新日2026年10月7日。元02と05は隠し再実行で例外になりました。9時間制限超過とは確定していません。[失敗監査](../experiments/p003-failure-review-20261006.json)

## 新たに確認した規模分岐と元作者版との差

64件診断では通らなかったcache分岐を静的確認しました。DINO画像cacheは最大1検査8,128,512byte（6×12×336²・uint8）、既定1GiBまでRAM、**133件以上ではmemmap**です。元のmemmap経路は全領域をゼロで初期化し、CPU/cgroup memoryとscratchの負荷が増える候補になります。1300件ならDINO約9.841GiB、Radの2layout重複は最大約3.888GiBという算定ですが、これだけでディスク上限超過やOOMが起きたとは証明できません。Rad4slot cacheの別境界669件は256件では通りません。実FS種別・free・全processのmemoryは診断で確認します。

元作者V2のcell27はCoAt子失敗を捕捉し、残りmemberで継続する箇所があります。こちらのP003GuardはそのeventとCoAt全4系統/fallback0の不成立を停止させます。つまり**通常時の予測式が一致しても、失敗時の挙動は元版と完全同一ではありません**。正常な欠損layoutは既に許容しますが、CoAt fallbackは実処理例外の予測代替であり、単なる不足planeではありません。これが自分の隠し例外の原因かはログがなく未確定です。失敗familyと対象入力を特定するための観測を追加します。

新しい[11_profile_p003_stress_256.ipynb](../notebooks/public/11_profile_p003_stress_256.ipynb)を用意しました。全モデル・05の推論セル・前処理・精度・microbatch・混合式・Raptor実行順は維持し、ラベル不要trainコホートをseed20261005のSHA順256件へ増やします。以前64件で比較済みのreference再計算はOFFにし、DINO memmap到達・cache range/hash・親RSS/cgroup/初終と最小diskfree・子処理成否・元eventの型を記録します。hash計算の追加負荷があるため公平な速度比較ではありません。実MRI実行・隠し完走・自己Publicは未確認です。

人工検査11件・Ruffが成功し、全22推論セルは旧05診断とbyte一致です。cgroupはv1/v2に対応し、取得不能な資源情報はnull/availabilityと`measurement_status=incomplete`で記録します。観測できなかったことをメモリ安全の証拠やモデル失敗とは扱いません。固定Input版・cache算定・元作者/追加guard差・Notebook SHAと未確認事項は[静的監査](../experiments/p003-scale-audit-20261007.json)に保存しました。

ユーザーの直近操作は[f002の10・3 Inputs](frozen-features.md#head接続の再発後は10を使う)です。11はp003の独立診断であり、10の前提ではありません。並行して行うなら、別private Notebookへ11をImportし、[従来14 Inputsの指定版](p003-next-step.md#p003でユーザーが行う操作)・**T4×2・Internet OFF・新しいセッション**でSave & Run Allします。f002のhead/decoder Inputsは追加しません。GPU割当て/quotaに余裕がなければ10の後に回します。

完走後は`P003_STRESS_SUMMARY.json`の`status=profile_complete_not_for_submission`・studies256・DINO memmap到達・CoAt全系統成功/fallback0を確認。失敗時は`P003_STRESS_FAILURE.json`と`P003_STRESS_EVENTS.jsonl`、該当CoAt logを使い、最後のphase/元例外/child returncodeと資源を照合します。`profile_predictions.csv`は提出しません。256件成功でも約1300件の全cache分岐・隠し入力の網羅ではなく、同じ05再提出へ自動で進む判断にはしません。[公式Dataの件数/入力差](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)

## 履歴：05の2回目の失敗と64件診断

**最新：05も隠し再実行の例外で失敗しました。同じ05の再提出は保留します。** 提出ref `56870280`、scriptVersionId `355633504`、自己Publicなし。保存版の全セルが配布05と一致し、可視3件はREADY passed・256.538秒、全4 CoAtのfallback 0でした。取得ログは保存時の可視実行で、隠しtracebackと正味時間は未取得です。ユーザー報告の約4時間だけから、9時間超過や8時間内部予算を原因とは決められません。[2回目の失敗監査](../experiments/p003-second-failure-review-20261006.json)

64件の診断結果は正しいものの、**予測一致と少数データの完走だけでは、隠し例外の解消を確認できませんでした。** 当時は64件を上回る規模診断と追加CoAt各系統の切り分けが未準備でした。10月7日に上記11を追加しました。失敗UID・前処理条件・子処理returncode・CPU/GPUメモリ・到達phaseを確認してから提出版を変更します。

10月7日、f002の08は256件完走後、09でhead接続が再発停止。現在は[head接続を外した10](frozen-features.md#head接続の再発後は10を使う)を使います。f003の24 epoch比較は両fold完了し、fold0改善・fold1予測不変で延長の利得は揃っていません。p002の採点済み0.937を保持します。f002とp003の共通原因は未確認で、10はp003の修正版ではありません。以下の05提出手順は再失敗より前の履歴で、今すぐの再提出案内ではありません。

提供JSONでは64検査・366系列の診断が完走し、変更した2系統の入力hash・生予測・順位が完全一致しました。追加CoAt 4系統のfallbackは0、最終診断CSVの報告hashも旧04と一致します。[確認記録](../experiments/p003-speed-profile-review-20261006.json)

所要時間は30分32.4秒で、照合用再計算が7分3.7秒含まれます。単純に差し引くと約23分28.6秒ですが、これは提出版の実測ではありません。旧04の23分47.8秒とほぼ同程度で、**大幅な高速化は確認できていません**。並行処理やcache、診断用hash計算の影響があるため、差分を厳密な高速化率にはしません。今回の予測一致と完走を根拠に提出候補の保存実行へ進み、隠しtestの成否・時間・自己Publicは別途確認します。

## 05の診断を実行した後にすること

**今回の高速化候補を提出するための追加学習は不要です。診断結果の確認 → 提出用Notebookの保存実行 → 手動提出 → 採点結果の確認、の順に進めます。** 既存の公開学習済みモデルを使います。

1. 診断NotebookのOutputから`P003_SPEED_SUMMARY.json`を取得します。この1ファイルに予測の一致・処理時間・GPUメモリ・全系統の完了記録がまとまっています。`status=completed_unscored`、`contract.mode=profile`、`raptor.parity_passed=true`を確認し、時間とメモリの記録も見て提出候補へ進む判断をします。こちらへ結果確認を依頼する場合は、このJSONを添付してください。ファイルがない、または実行が失敗した場合は`P003_SPEED_FAILURE.json`、それもなければ末尾のエラーログを確認します。
2. 診断結果を確認できたら、[05_submit_p003_speed.ipynb](../notebooks/public/05_submit_p003_speed.ipynb)を**別の新しいprivate Notebook**へImportします。同じ14 Inputs・T4×2・Internet OFFで保存実行します。診断用の`05_profile`や`profile_predictions.csv`は提出しません。
3. 提出版の`P003_READY.json`が`status=passed`、`P003_SPEED_SUMMARY.json`が`status=completed_unscored`で、`submission.csv`が生成されていることを確認し、その保存版から手動提出します。提出後の隠しtest再実行と採点結果まで確認して、初めて今回の成否・自己Publicが分かります。

`speedup_verified=false`は実装上の固定値で、診断失敗を意味しません。`raptor.status=started`も更新されない初期値です。完了は全体の`status`と`parity_passed`で判断します。また、診断は旧経路との照合を追加実行するため、診断全体の所要時間だけでは提出版の高速化率を判断できません。

06の特徴作成・ローカル4090での学習・07の独自モデル推論は、その後の精度改善用の別経路です。今回の05提出を進めるために先に実行する必要はありません。

## 最初に実行するNotebook

[05_profile_p003_speed_64.ipynb](../notebooks/public/05_profile_p003_speed_64.ipynb)を新しいprivate NotebookへImportし、[従来と同じ14 Inputs・指定版](p003-next-step.md#p003でユーザーが行う操作)、**T4×2・Internet OFF**で保存実行します。元04と同じseed・64検査を使用します。

今回変えるのはRaptorのGPU1の実行順です。native384denseとnative384を検査ごとに続けて処理し、同じ画素のdecode cacheを再利用しやすくします。全モデル・前処理・数値精度・microbatch・最終混合式は保持します。ただしGPU1に2モデルを同時常駐させるので、T4のメモリ余裕も確認が必要です。

診断版は変更した2系統を元の順でも再計算します。追加reference計算には時間がかかります。**診断全体の時間を、旧04の23分47.8秒と直接比較しないでください。** referenceはprefetchなしで数値差を確認する実装なので、candidate/referenceの比も公平な高速化率ではありません。実行時間の根拠は同条件のphase比較と、その後の提出候補の実測です。

| 出力 | 確認内容 |
| --- | --- |
| `P003_SPEED_SUMMARY.json` | `status=completed_unscored`、元の全系統完了検査、実行したsource hash |
| `P003_SPEED_RAPTOR.json` | `parity_passed=true`、両系統の入力hash一致、生予測完全一致、rank差0、GPU peak reserved memory |
| `P003_SPEED_PHASES.json` | Raptor、CoAt各系統の開始・終了。elapsed差を見て時間が集中する段階を特定 |
| 各CoAt receipt／子処理log | 系統別の処理・欠損・fallbackの有無 |
| `P003_SPEED_FAILURE.json` | 失敗したセル、traceback、到達phase。時間切れかデータ依存かを区別する手掛かり |

decode時間・prepare時間・GPU時間は重なります。各秒数を単純に足して総時間にしません。今回の実MRI 64検査では入力・予測の一致を確認し、native処理のPyTorch最大予約メモリは1.55GiB、CoAt各worker最大は2.72GiBでした。GPU全体使用量や隠しtestのメモリ上限の測定ではありません。元の人工CUDAテストと今回のユーザー実行結果は区別します。

## 提出候補

診断の一致・メモリ・時間を確認したら、別の新しいprivate Notebookに[05_submit_p003_speed.ipynb](../notebooks/public/05_submit_p003_speed.ipynb)をImportします。同じInputs・T4×2・Internet OFFで保存実行します。この版にはreference再計算がありません。

`P003_READY.json`と`P003_SPEED_SUMMARY.json`の完了、および`submission.csv`を確認してから手動Submitします。保存実行の3検査成功は隠しtest完走の保証ではありません。旧02の8時間内部予算と全モデル完了の検査は維持し、未完了モデルを省いたCSVは採用しません。

この変更は速度改善候補です。0.943は作者の値であり、自分の採点成功や高速化実測値はまだありません。隠し例外の解消も未確認です。余裕を確保できなければ、p002の採点済み0.937を維持して次の変更要因を検討します。

## 再生成

既存の出力先は上書きできません。以下はコードの生成だけであり、MRI処理やKaggleアクセスは開始しません。

```powershell
.\.venv\Scripts\python.exe scripts/build_p003_speed.py --mode profile --output-dir artifacts/kaggle/p003-speed-profile-next
.\.venv\Scripts\python.exe scripts/build_p003_speed.py --mode submission --output-dir artifacts/kaggle/p003-speed-submit-next
```

学習による精度改善は[凍結特徴の新経路](frozen-features.md)、教師改善は[ラベル根拠監査](label-review.md)、判断根拠は[調査記録](research/p003-improvement-20261006.md)へ分けて記録します。
