# RSNA Knee プロジェクトの現在地

更新日 2026年10月3日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、採点に通る比較基準から段階的に改善することです。

## 現在の状態

- GitHubの既存リポジトリに、実データなしで開発できる構成を追加。
- 調査の正本は [docs/competition.md](docs/competition.md)。原文は [ユーザー提供分析](docs/research/user-analysis-20261002.txt)。
- 今後の判断と三週間の計画は [docs/roadmap.md](docs/roadmap.md)。ユーザーはKaggleでキャッシュを作成し、ローカルGPUで学習する経路を選択済み。
- ユーザー報告で初回Notebookの提出受付に成功。採点の完了・Publicスコア・Notebook版はまだ未確認。
- weakラベル取得とローカル学習は未実施。Kaggleで作成済みの自作パイプライン用全件画像キャッシュのローカル転送・検査は完了。
- 10月3日、ユーザーの依頼でprivate Notebook `rsraki/rsna-knee` のVersion 1（ユーザー提示のscriptVersionIdは354838181）へAPIでアクセス。全件exportのcomplete=true、4,407検査、4,423ファイル、7,785,521,448 bytes、現在のsrcとのhash一致を確認した。
- 大きいZIPを避ける [download_cache_output.py](scripts/download_cache_output.py) を追加。取得先は `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/`。版確認、4並列、ファイル単位のhash検証・再開、取得後のexport検査に対応する。
- 全4,423ファイルを取得し、4,407検査のexport検査がvalid=trueで成功。サイズ・hash・UID・元CSV・src・前処理が一致。取得記録は親ディレクトリのtransfer.jsonでtransfer_complete=true。元DICOM、モデルは取得せず、実学習も始めていない。
- 自作コードの検証範囲は [validation.md](docs/validation.md) に記録。
- ベースライン設定は [configs/baseline.json](configs/baseline.json)。学習コマンドはCUDAがなければ停止する。
- 現在の端末はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブの空き約485GBに対して公式元画像は569.76GBのため、全取得しない。
- Python 3.12.13の専用 `.venv` にtorch 2.10.0+cu128、torchvision 0.25.0+cu128が導入され、CUDA利用可・RTX 4090認識を10月3日に確認。学習環境とは別の `artifacts/tools/kaggle-venv/` にKaggle CLI 2.2.4を導入。人工データのunittest 19件が成功した。実学習は開始していない。
- [00_prepare_cache.ipynb](notebooks/00_prepare_cache.ipynb) と転送確認用 [verify_cache_export.py](scripts/verify_cache_export.py) を追加。Notebookは未実行テンプレート。
- 手順1〜3の具体的な操作を整理。両Notebookはコードzipと展開済みInputの両形式に対応し、GPU開始手順へ版を固定したPyTorch導入例を追記した。実データ処理・CUDA導入は未実施のまま。
- 続く [手順4〜9](docs/after-cache.md) にファイル単位転送の実行方法を追記。weakラベルの採用元は未確定であり、ラベル監査、固定fold、学習、採点はこれから行う。

## 次にすること

1. 大会ルールに同意し、参加登録を済ませる。登録期限は10月16日08:59 JST。
2. 受付済みの初回提出をMy Submissionsで確認し、採点成功、実測Publicスコア、Notebook版、Input版、実行時間を記録する。提出受付と採点成功を区別する。
3. CUDA利用は確認済み。[環境手順](docs/gpu-start.md)の人工画像CUDA forwardを終えていなければ確認する。実学習はラベルと分割の監査後に行う。
4. 全件exportの転送検査は完了。以後のprepare・trainでは取得済みの `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` を使用する。
5. 原画像とcacheの目視、coverageの失敗・fallbackを確認した記録を残す。今回の転送検査だけで画像品質が確認済みとはしない。
6. 公開ラベルの版・ライセンス・方法・gold使用状況を記録してprepareし、ローカルのfold 0学習へ進む。
7. 学習後の重みと対応コードを手動でKaggleへ追加し、オフライン提出推論・採点を確認する。

キャッシュ作成後の実行コマンドと段階ごとの完了条件は [after-cache.md](docs/after-cache.md) を使う。改善はfold 0の比較基準と自作提出の採点成功が揃ってから進め、10月20〜22日は最終候補の再実行・選択に充てる。

## 次へ進む条件

- 初回提出：Notebook版、入力資産の版、採点成功、スコア、実行時間を台帳に記録。
- データ準備：件数、ラベル欠損、シリーズ失敗、前処理画像の目視、使用ラベルの出所が確認済み。
- 自作モデル：学習・weak検証・gold検証の役割を分け、goldをcheckpoint選択に使わず、漏洩監査済み。
- 改善：同じfoldとseedで一要因ずつ変更。12クラス全体と弱いクラス、速度を合わせて判断。
- 最終候補：Kaggleの隠しテストで採点成功し、十分な時間余裕を持つ。未採点の最終変更に依存しない。

## 作業場所

Kaggleは元MRIの前処理と最終提出、このPCはキャッシュからのGPU学習に使う。今回の作業は準備と人工データ検証まで。Gitへ戻すのはコード、設定、データを含まない記録のみ。レポート、Study UID一覧、分割CSV、画像キャッシュ、重みは `data/` と `artifacts/` 以下へ保存する。

`pj-kaggriculture/PROJECT.md` と `ptcc_pokemon_ai_buttle/experiments` の、実装・実験・生成物を分ける運用を参考にした。ゲーム用の提出形式や対戦評価はMRIコンペへ持ち込んでいない。
