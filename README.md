# RSNA Knee Abnormality Detection

2026年のRSNA膝MRIコンペを進めるためのリポジトリです。**Kaggleで画像キャッシュを作成、このPCのRTX 4090で学習、Kaggleで提出推論**を行います。Cドライブの空き約485GBに対して公式元画像は約570GBのため、元画像は全取得せずキャッシュを移します。データ・重み・認証情報はGitに入れません。

全件画像キャッシュ（4,407検査、約7.79GB）の取得・検証、weakラベルの監査と固定分割、RTX 4090での基準学習を完了しています。**weak 4,207検査・gold 58検査、fold 0の学習3,386／検証821検査**です。10月3日の5 epoch基準実験は約18分、最良checkpointはepoch 1でした。公開モデルを使った既存提出の実測Public 0.924は公式APIで確認済みで、自作モデルのスコアとは区別します。

**最新（10月6日）：新05の64検査診断が完了し、変更した2系統の入力・予測は完全一致しました。次は[提出用05](notebooks/public/05_submit_p003_speed.ipynb)の保存実行です。追加学習は不要です。** 診断時間は照合用再計算を含め30分32.4秒で、単純差引き約23分28.6秒は旧04とほぼ同程度。大幅高速化は確認できていません。[診断監査](experiments/p003-speed-profile-review-20261006.json)、[次の操作](docs/p003-speed.md)。元p003の隠し例外は時間超過とは未確定で、新05の隠し完走・Publicも未確認です。自己採点済みの基準は[p002 Public 0.937](notebooks/public/01_submit_p002_scored.ipynb)。公開0.943を設計上の基準に、[392px凍結特徴からの独自学習](docs/frozen-features.md)と[教師の根拠監査](docs/label-review.md)は別経路として準備済みで、実学習・精度改善は未検証です。以下は以前の履歴を含みます。

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
