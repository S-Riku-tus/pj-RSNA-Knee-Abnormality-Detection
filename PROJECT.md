# RSNA Knee プロジェクトの現在地

更新日 2026年10月2日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、採点に通る比較基準から段階的に改善することです。

## 現在の状態

- GitHubの既存リポジトリに、実データなしで開発できる構成を追加。
- 調査の正本は [docs/competition.md](docs/competition.md)。原文は [ユーザー提供分析](docs/research/user-analysis-20261002.txt)。
- データ取得・ラベル取得・重み取得・学習・提出は未実施。
- 自作コードの検証範囲は [validation.md](docs/validation.md) に記録。
- ベースライン設定は [configs/baseline.json](configs/baseline.json)。学習コマンドはCUDAがなければ停止する。
- このPCの `.venv` はPython 3.12.10と基本パッケージだけを準備済み。`.venv/Scripts/python.exe -m rsna_knee doctor` で確認できる。

## GPU端末で最初にすること

1. 大会ルールに同意し、参加登録を済ませる。登録期限は10月16日08:59 JST。
2. [公開モデル再現](docs/public-baselines.md) に従い、Kaggle上で最初の採点成功を確保する。
3. 別のGPU端末へこのリポジトリをcloneし、[環境準備](docs/gpu-start.md)を実施する。
4. その端末で公式データを準備し、`audit` で実測件数と列構成を確認する。
5. 公開ラベルの版・ライセンス・方法・gold使用状況を記録して取り込む。
6. 小規模DICOMキャッシュを確認後、全体キャッシュ、fold 0学習、オフライン推論へ進む。

## 次へ進む条件

- 初回提出：Notebook版、入力資産の版、採点成功、スコア、実行時間を台帳に記録。
- データ準備：件数、ラベル欠損、シリーズ失敗、前処理画像の目視、使用ラベルの出所が確認済み。
- 自作モデル：学習・weak検証・gold検証の役割を分け、goldをcheckpoint選択に使わず、漏洩監査済み。
- 改善：同じfoldとseedで一要因ずつ変更。12クラス全体と弱いクラス、速度を合わせて判断。
- 最終候補：Kaggleの隠しテストで採点成功し、十分な時間余裕を持つ。未採点の最終変更に依存しない。

## 作業場所

このPCでのデータ取得・MRIデコード・学習は行わない。GPU端末からGitへ戻すのはコード、設定、データを含まない記録のみ。DICOM、レポート、Study UID一覧、分割CSV、画像キャッシュ、重みは各端末のGit除外ディレクトリへ保存する。

`pj-kaggriculture/PROJECT.md` と `ptcc_pokemon_ai_buttle/experiments` の、実装・実験・生成物を分ける運用を参考にした。ゲーム用の提出形式や対戦評価はMRIコンペへ持ち込んでいない。
