# p002 保存ソースの帰属

`romantamrazov-dinosaur-v32.ipynb` は Roman Tamrazov (`romantamrazov`) の
[RSNA Knee DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4)
の Version 32 を公式SDKで取得したソースです。2026年10月5日に固定しました。
版の正本は保存API metadataの `currentVersionNumber=32` と下記SHA-256です。

- 元Notebook: `https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4/versions/32`
- SHA-256: `ba9491ac1e214ba55ca181de55118c9a793fe7f6163fc221fe1e68d596259dca`
- ソースの変更: なし。元Markdownの帰属を含めてbyte単位で保持。出力・実行回数は元から空。
- 元Notebookが明示する主要な出典: Renta K. (`renta0426`) の
  [RSNA Knee 0.937 | Weak-Label DINOv2 Meniscus Resid](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid)。

`../01_submit_p002_scored.ipynb` はこのソースを無変更で実行するローカルguard付き
引渡し版の固定コピーです。2026年10月5日にユーザーが提出した
`rsraki/notebook3beaf8c48b` Version 1 と全16セルのsourceが一致し、
公式APIは提出 `56840796` の Public 0.937 / COMPLETE を返しました。
Notebookの保存APIはJSON書式を変更するため、ファイル全体のhashとセルsource一致を分けて記録しています。

原作者・上流資産の著作権と利用条件を保持し、このリポジトリ独自の許諾へ置き換えません。
関連する公開重みには Apache-2.0、CC BY-NC-SA 4.0、Meta DINOv3 等の条件が混在します。
ここには重み、画像、ラベル、予測CSVを含めていません。
既存の[環境・利用条件監査](../../../docs/research/p002-environment-20261005.json)と
[競技側の利用条件根拠](../../../docs/research/kaggle-source-eligibility-20261004.json)を参照してください。
上流全系譜の個別ライセンスやgold露出履歴を新たに確定したものではありません。

引渡し版の追加部分は `scripts/build_public_candidate.py` と
`scripts/public_candidate_guard.py` が生成します。前処理・モデル・係数・ラベル順は
元V32を保持し、Input hash・完走・CSV契約の検査を追加した変更です。
