# 公式0.959へ進む判断（2026-10-21）

13:10 JST時点の自己最高は0.958、0.959は未達です。競技の[公式取得記録](../artifacts/research/20261019-evening-official-P038-preflight-v1/observation-20261019T094220164959Z/competition.json)の指標名は「Roc Auc Score」です。ここでは正解率と混同せず、Publicスコアと呼びます。

0.957／0.958が続く結果は、試した変更から表示上の改善を確認できていないという証拠です。3桁の同点だけでは予測の同一性や微小な増減は分かりません。AUCは正例と負例の順位で変わるため、順位の小さな改善が表示境界を越えると0.001刻みに見えます。上位者の推移も、既存モデルの改良・別モデルとの合成など複数の説明と両立し、具体的手法を特定する証拠にはなりません。一要因ずつ比較する方針は妥当ですが、上位者の方法を再現できたとは扱いません。

| 比較 | 確認結果 | 判断 |
| --- | --- | --- |
| P033/P039/P041/P043/P044 | 各公式0.958 | 元bestを保持 |
| P040/P042 | 各公式0.957 | 不採用 |
| P045・DINO10%追加 | 公式0.956 | P046の20%案も無提出 |
| R6事前4案 | 両bankが正の案なし | P047無提出 |
| P048 V4 | ref57464683受理、採点待ち | 本日残4枠 |
| F014・新encoder | 固定E12のfold0学習中 | 精度は未確定 |

公式値は[提出履歴](../experiments/ledger.csv)、R6は[共通成功1664件の確定判断](../experiments/R6-four-prespecified-variants-nonpositive-no-P047-20261021.json)に基づきます。失敗した案を係数・所見・epochの選び直しで救済しません。

[公開調査13:00更新](../experiments/public-frontier-refresh-1300-20261021.json)ではscoreDESC上位100件・dateRun最新100件の範囲で作者Public最高0.958でした。タイトル「0.9586」を公式0.959や自分の再現値には数えません。これは調査報告の保存で、生応答の独立監査や全公開資料に高スコアが存在しないことの証明ではありません。同じ公開レシピへの再提出より、根拠のある変更に枠を使います。

[F013のnative家族50%平均](../experiments/P048-native50-paired-weak-positive-sourceV2-20261020.json)はfold0 +0.000701、fold1 +0.000370と微改善しました。ただし反復したholdoutで、実提出の2fold平均や全親OOFではありません。別の[単体gamma差のfold1区間](../experiments/F013-F009-matched-counter-fixed12-exploratory-positive-20261020.json)はゼロを跨ぎます。元AMP-HOLDを維持した別scopeの有限性・復元・実GPU資格と、精度の根拠を区別します。[P048 V4](../experiments/P048-V4-visible-qualified-formal57464683-pending-20261021.json)は現モデル・可視3件・CSV監査を通過して1回正式提出しましたが、Publicはまだnullです。

[F014](../experiments/F014-ConvNeXt-artificial-qualified-fold0-running-20261021.json)はConvNeXtへのencoder選択を比較します。構造・一般初期重み・幅・suffixが連動し、構造だけの効果ではありません。[source16の固定と実GPU確認](../experiments/F014-independent-static-training-review-20261021.json)は工学的資格であり、Public採点とは別です。固定E12のfold0が正かつ全guard合格ならfold1へ進み、両fold正ならserving QA・CSV検算後に手動Submitを判断します。F015 SmallもTinyの精度を見る前に独立条件を固定し、その後generic重み・人工GPU12項目・空き資源を確認してfold0を開始しました。Tinyへの影響が事前条件を超えたらSmallだけを停止します。両候補とも固定E12の精度はまだ未確定です。

自分のGold画像ラベルの学習・選択利用0、欠損mask、同一検査/report/groupのfold分離を維持します。正式0.959以上を確認できれば追加の正式提出は不要です。
