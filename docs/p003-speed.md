# p003の実行順改善候補

2026年10月6日。元02は隠し再実行で例外になりました。9時間制限超過とは確定していません。[失敗監査](../experiments/p003-failure-review-20261006.json)

**今回の結果確認は完了しました。次は下記の手順2、提出用05の保存実行です。** 提供JSONでは64検査・366系列の診断が完走し、変更した2系統の入力hash・生予測・順位が完全一致しました。追加CoAt 4系統のfallbackは0、最終診断CSVの報告hashも旧04と一致します。[確認記録](../experiments/p003-speed-profile-review-20261006.json)

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
