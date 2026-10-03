# RSNA Knee Abnormality Detection

2026年のRSNA膝MRIコンペを進めるためのリポジトリです。**Kaggleで画像キャッシュを作成、このPCのRTX 4090で学習、Kaggleで提出推論**を行います。Cドライブの空き約485GBに対して公式元画像は約570GBのため、元画像は全取得せずキャッシュを移します。データ・重み・認証情報はGitに入れません。

調査資料、実験方針、CSV監査、ラベルの取り込みと検証分割、DICOMキャッシュ、画像モデルの学習・推論、提出Notebookのひな形を用意しています。2026年10月3日、Kaggleの全件画像キャッシュ（4,407検査、約7.79GB）をファイル単位でローカル取得し、全件のhash・export検査を完了しました。**weakラベルの監査・固定分割・実学習は未実施**で、初回提出の採点結果もまだ未確認です。

## 最初に読む資料

1. [PROJECT.md](PROJECT.md) — 現状、次にすること、完了条件。
2. [今後の参加計画](docs/roadmap.md) — 環境の判断、最初の二日間、締め切りまでの優先順位。
3. [GPU端末での開始手順](docs/gpu-start.md) — Windows環境とKaggleキャッシュを使った学習手順。
4. [キャッシュ作成後の手順4〜9](docs/after-cache.md) — 転送検査、ラベル監査、fold 0学習、採点、改善、最終選択の操作と完了条件。
5. [大会と調査結果](docs/competition.md) — 公式に確認した条件、添付分析との照合、未確認事項。
6. [公開モデルの再現手順](docs/public-baselines.md) — 最初の採点に使う候補と記録項目。
7. [モデルと検証の設計](docs/experiment-design.md) — ラベル、前処理、分割、改善の順序。
8. [Kaggleへの提出手順](docs/kaggle-submit.md) — オフライン推論の準備。

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

自作ベースラインは **ランダム初期化ResNet18と所見別Attention** です。公開CoAtNet/DINOモデルとは別実装で、公開スコアを再現するものではありません。最初の採点成功には公開Notebookの再現を優先し、自作コードは比較できる実験を行う土台に使います。
