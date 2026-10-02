# RSNA Knee Abnormality Detection

2026年のRSNA膝MRIコンペを進めるためのリポジトリです。**このPCでは編集・調査・軽量チェック、別のCUDA GPU端末ではデータ準備と学習、Kaggleでは提出推論**を行います。データ・重み・認証情報はGitに入れません。

現状は、調査資料、実験方針、CSV監査、ラベルの取り込みと検証分割、DICOMキャッシュ、画像モデルの学習・推論、提出Notebookのひな形まで用意しています。**実データの取得、実MRIでの検証、学習、Kaggle採点は未実施**です。

## 最初に読む資料

1. [PROJECT.md](PROJECT.md) — 現状、次にすること、完了条件。
2. [大会と調査結果](docs/competition.md) — 公式に確認した条件、添付分析との照合、未確認事項。
3. [GPU端末での開始手順](docs/gpu-start.md) — LinuxとWindowsでの環境準備、データ取得後のコマンド。
4. [公開モデルの再現手順](docs/public-baselines.md) — 最初の採点に使う候補と記録項目。
5. [モデルと検証の設計](docs/experiment-design.md) — ラベル、前処理、分割、改善の順序。
6. [Kaggleへの提出手順](docs/kaggle-submit.md) — オフライン推論の準備。

## 大会の要点

MRI検査ごとに12所見の連続値を予測し、12項目の平均ROC-AUCで評価されます。提出はKaggle Notebookで、インターネット無効、実行9時間以内、出力名は `submission.csv` です。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

参加・チーム統合は **2026年10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。期限は変更される可能性があるため、提出前に公式ページを確認してください。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

レポートは学習ラベルを作るための情報として扱い、推論コードはMRIと検査IDだけを使用します。提供ラベルの少なさ、レポート由来ラベルとの違いが今回の設計上の焦点です。

## このPCでできるチェック

Python 3.11〜3.13を使用してください。基本パッケージは標準ライブラリだけで動き、GPUライブラリの導入は不要です。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m rsna_knee doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py
```

`python` がpyenvの未設定エラーになる場合は、インストール済みPythonの実体を指定してvenvを作成します。GPU端末の準備は [gpu-start.md](docs/gpu-start.md) を参照してください。

## 構成

```text
configs/             再現可能なJSON設定
src/rsna_knee/       軽量なCSVツールとGPU用パイプライン
scripts/             提出コードの梱包
notebooks/           Kaggle提出Notebookのひな形
docs/                調査、実行手順、設計判断
docs/research/       ユーザー提供分析の原文と出典一覧
experiments/         実験台帳と記入用テンプレート
tests/               人工データによる契約・分割・前処理チェック
data/                実データ、ラベル、キャッシュ、分割（Git対象外）
artifacts/           重み、予測、ログ、提出用zip（Git対象外）
```

自作ベースラインは **ランダム初期化ResNet18と所見別Attention** です。公開CoAtNet/DINOモデルとは別実装で、公開スコアを再現するものではありません。最初の採点成功には公開Notebookの再現を優先し、自作コードは比較できる実験を行う土台に使います。
