# p003の実行順改善候補

2026年10月6日。元02は隠し再実行で例外になりました。9時間制限超過とは確定していません。[失敗監査](../experiments/p003-failure-review-20261006.json)

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

decode時間・prepare時間・GPU時間は重なります。各秒数を単純に足して総時間にしません。実MRIでの同等性・速度・2モデル常駐のメモリは未確認です。人工CUDAテストは小さい代替モデルを使っており、実T4の負荷検証ではありません。

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
