# RSNA Knee Abnormality Detection

2026年のRSNA膝MRIコンペを進めるためのリポジトリです。**Kaggleで画像キャッシュを作成、このPCのRTX 4090で学習、Kaggleで提出推論**を行います。Cドライブの空き約485GBに対して公式元画像は約570GBのため、元画像は全取得せずキャッシュを移します。データ・重み・認証情報はGitに入れません。

全件画像キャッシュ（4,407検査、約7.79GB）の取得・検証、weakラベルの監査と固定分割、RTX 4090での基準学習を完了しています。**weak 4,207検査・gold 58検査、fold 0の学習3,386／検証821検査**です。10月3日の5 epoch基準実験は約18分、最良checkpointはepoch 1でした。公開モデルを使った既存提出の実測Public 0.924は公式APIで確認済みで、自作モデルのスコアとは区別します。

**最新（10月8日）：p005の自己Public0.950を公式APIで確認しました。** ref56943135/own Version1/script356323020がCOMPLETE・エラーなしで、今回の保存版は隠し再実行/採点に成功。271位はユーザー報告です。[採点監査](experiments/p005-scored-review-20261008.json)。この版を基準に、次0.955・最終0.960超へ[一要因の改善計画](experiments/p005-improvement-roadmap-20261008.json)を準備しました。最初はConvNeXt窓数16→24の限定比較、次に異なる表現の1専門枝を検討します。新Notebook/学習/推論は未開始。f004差し替え見送り・教師訂正採用0は維持します。[現在の判断](PROJECT.md)。旧p003/f002例外は未解決、p002自己0.937・14の固定コピーを保持します。

08 Version2の256件成功は保持します。人工1,300件の提出契約は成功しましたが、MRI/encoderはmockで隠し完走の証拠ではありません。[提出経路の独立監査](experiments/serving-contract-independent-review-20261008.json)。f002には元画像全体の配列重複と不適合系列で停止する経路を確認し、メモリ配送だけを変えた[13の診断](docs/frozen-features.md#10月8日以降のf002診断)を別に準備しました。p003/f002の自己Publicはなく、作者0.943/0.944を保証しません。p002自己Public0.937を保持。0.945〜0.950も版別採点・Input・外部重み条件を[監査](experiments/public-candidates-20261008.json)しました。生成10/13は重みを含むため個別Git除外です。

f003の24 epoch学習はユーザーが両foldを完了し、136項目の監査が成功しました。fold0 weak AUC12は0.813290→0.823060、fold1のBCE選択予測は元f002と同一で、延長の利得は揃っていません。[完成比較](experiments/f002-f003-fold01-review-20261007.json)。[f004のdropoutだけ0.2→0.4](configs/experiments/f004-dinov2-attention24-dropout40.json)を今回両foldで実行しましたが、両fold平均AUC低下で採用を見送りました。教師17セルの二次レビューは6セルの原文状態を整理、11セル未確定、訂正採用0です。[根拠監査](docs/label-review.md)。実データ/モデル追加取得・MRI処理・外部アップロード/提出は行っていません。

**提出の最新（10月6日）：05も隠し再実行で例外になりました。同じ05の再提出は保留します。** 提出ref `56870280`、scriptVersionId `355633504`、自己Publicなし。保存版の全セルは配布05と一致し、可視3件は256.538秒で完走しましたが、隠しtracebackは取得できず原因は未確定です。ユーザー報告の約4時間は隠し推論の実測時間ではなく、9時間超過とは断定しません。[2回目の失敗監査](experiments/p003-second-failure-review-20261006.json)、[診断の方針](docs/p003-speed.md)。自己採点済みの基準は[p002 Public 0.937](notebooks/public/01_submit_p002_scored.ipynb)。作者0.943は設計上の参考値で、自己再現値として扱いません。以下は以前の履歴を含みます。

10月4日に公式RulesとHost回答を取得し、vmohitrao Version 3ラベルを研究・学習目的とCC BY-NC 4.0の帰属等を守って採用する判断に更新しました。[利用条件の根拠](docs/research/kaggle-source-eligibility-20261004.json) に本文の確認日と出典を保存しています。

## 最初に読む資料

1. [PROJECT.md](PROJECT.md) — 現状、次にすること、完了条件。
2. [今後の参加計画](docs/roadmap.md) — 環境の判断、最初の二日間、締め切りまでの優先順位。
3. [GPU端末での開始手順](docs/gpu-start.md) — Windows環境とKaggleキャッシュを使った学習手順。
4. [キャッシュ作成後の手順4〜9](docs/after-cache.md) — 転送検査、ラベル監査、fold 0学習、採点、改善、最終選択の操作と完了条件。
5. [大会と調査結果](docs/competition.md) — 公式に確認した条件、添付分析との照合、未確認事項。
6. [公開モデルの再現手順](docs/public-baselines.md) — 最初の採点に使う候補と記録項目。
7. [モデルと検証の設計](docs/experiment-design.md) — ラベル、前処理、分割、改善の順序。
8. [Kaggleへの提出手順](docs/kaggle-submit.md) — オフライン推論の準備。

10月5日、固定50:50 rank候補r001の**Public 0.749を公式APIで確認**しました。weak AUC 0.830249とは対象・正解が異なります。今後は**Public 0.924の公開学習済みモデルを基準に、公開候補一つの推論比較を最優先**にします。次候補DINOsaur V32は指定5 Inputsを固定し、元の推論ソースに開始・完了検査を加えた[提出用Notebook](artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)を準備済みです。次はユーザーが[操作手順](docs/public-baselines.md#ユーザーが行う操作)に沿ってT4 x2・Internet OFFで保存実行し、成功後に手動提出します。実MRIでの完走と自己採点は未確認です。[段階別計画](docs/research/public-model-strategy-20261005.md)、[提出監査](experiments/submission-audit-20261005.json)。自作ResNet18の追加学習は補完根拠が得られるまで保留します。

10月4日までのA/B/C・fold 1対照・通常BN／固定BNの20 epochは完了済みです。当時の全97 unittest・Ruff・人工CPU/GPU提出契約の成功は [検証履歴](docs/validation.md)、weak評価の詳細は [対照実験](docs/controlled-experiments.md) に残します。公開重みのgold履歴を監査し、自分の学習・選択にgoldを使わない方針を維持します。

## 大会の要点

MRI検査ごとに12所見の連続値を予測し、12項目の平均ROC-AUCで評価されます。提出はKaggle Notebookで、インターネット無効、実行9時間以内、出力名は `submission.csv` です。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

参加・チーム統合は **2026年10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。期限は変更される可能性があるため、提出前に公式ページを確認してください。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

レポートは学習ラベルを作るための情報として扱い、推論コードはMRIと検査IDだけを使用します。提供ラベルの少なさ、レポート由来ラベルとの違いが今回の設計上の焦点です。

## このPCでできるチェック

Python 3.11〜3.13を使用してください。基本パッケージは標準ライブラリだけで動きます。この端末にはPython 3.12.13の `.venv` と画像依存があり、torch 2.10.0+cu128・torchvision 0.25.0+cu128でRTX 4090のCUDA利用可を確認しています。以下は準備済み環境で実行できます。

```powershell
.\.venv\Scripts\python.exe -m rsna_knee doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py --output-dir artifacts/kaggle/cache-v1
```

新しい端末でのvenv作成と、Windowsの `python` がアプリ実行エイリアスになる場合の対処は [gpu-start.md](docs/gpu-start.md) を参照してください。

## 構成

```text
configs/             再現可能なJSON設定
src/rsna_knee/       軽量なCSVツールとGPU用パイプライン
scripts/             提出コードの梱包
notebooks/           Kaggleキャッシュ作成・提出Notebookのひな形
docs/                調査、実行手順、設計判断
docs/research/       ユーザー提供分析の原文と出典一覧
experiments/         実験台帳と記入用テンプレート
tests/               人工データによる契約・分割・前処理チェック
data/                実データ、ラベル、キャッシュ、分割（Git対象外）
artifacts/           重み、予測、ログ、提出用zip（Git対象外）
```

自作モデルは **ResNet18と所見別Attention**。ランダム初期化を比較基準に、ImageNet事前学習を固定条件で検証しました。公開CoAtNet/DINOの提出再現と、自作モデルのweak検証は分けて記録します。
