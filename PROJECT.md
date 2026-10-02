# RSNA Knee プロジェクトの現在地

更新日 2026年10月2日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、採点に通る比較基準から段階的に改善することです。

## 現在の状態

- GitHubの既存リポジトリに、実データなしで開発できる構成を追加。
- 調査の正本は [docs/competition.md](docs/competition.md)。原文は [ユーザー提供分析](docs/research/user-analysis-20261002.txt)。
- 今後の判断と三週間の計画は [docs/roadmap.md](docs/roadmap.md)。ユーザーはKaggleでキャッシュを作成し、ローカルGPUで学習する経路を選択済み。
- データ取得・ラベル取得・重み取得・学習・提出は未実施。
- 自作コードの検証範囲は [validation.md](docs/validation.md) に記録。
- ベースライン設定は [configs/baseline.json](configs/baseline.json)。学習コマンドはCUDAがなければ停止する。
- 現在の端末はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブの空き約485GBに対して公式元画像は569.76GBのため、全取得しない。
- この端末にPython 3.12.13の `.venv` を新規準備。numpy 2.5.3、pydicom 3.0.2、Pillow 12.3.0、ruff 0.16.10を導入済み。torch/torchvisionは未導入で、CUDAのPython利用は未確認。
- [00_prepare_cache.ipynb](notebooks/00_prepare_cache.ipynb) と転送確認用 [verify_cache_export.py](scripts/verify_cache_export.py) を追加。Notebookは未実行テンプレート。

## 次にすること

1. 大会ルールに同意し、参加登録を済ませる。登録期限は10月16日08:59 JST。
2. [公開モデル再現](docs/public-baselines.md) に従い、Kaggle上で最初の採点成功を確保する。
3. この端末の専用venvへCUDA対応のtorch/torchvisionを導入し、[環境手順](docs/gpu-start.md)でCUDAと人工画像forwardを確認する。
4. KaggleでCSV監査と10検査のキャッシュ・目視確認を行い、問題がなければ全件を作る。
5. private Outputを手動でダウンロードし、検査数、ファイルhash、ソース、configを検査する。元DICOMは移さない。
6. 公開ラベルの版・ライセンス・方法・gold使用状況を記録してprepareし、ローカルのfold 0学習へ進む。
7. 学習後の重みと対応コードを手動でKaggleへ追加し、オフライン提出推論・採点を確認する。

## 次へ進む条件

- 初回提出：Notebook版、入力資産の版、採点成功、スコア、実行時間を台帳に記録。
- データ準備：件数、ラベル欠損、シリーズ失敗、前処理画像の目視、使用ラベルの出所が確認済み。
- 自作モデル：学習・weak検証・gold検証の役割を分け、goldをcheckpoint選択に使わず、漏洩監査済み。
- 改善：同じfoldとseedで一要因ずつ変更。12クラス全体と弱いクラス、速度を合わせて判断。
- 最終候補：Kaggleの隠しテストで採点成功し、十分な時間余裕を持つ。未採点の最終変更に依存しない。

## 作業場所

Kaggleは元MRIの前処理と最終提出、このPCはキャッシュからのGPU学習に使う。今回の作業は準備と人工データ検証まで。Gitへ戻すのはコード、設定、データを含まない記録のみ。レポート、Study UID一覧、分割CSV、画像キャッシュ、重みは `data/` と `artifacts/` 以下へ保存する。

`pj-kaggriculture/PROJECT.md` と `ptcc_pokemon_ai_buttle/experiments` の、実装・実験・生成物を分ける運用を参考にした。ゲーム用の提出形式や対戦評価はMRIコンペへ持ち込んでいない。
