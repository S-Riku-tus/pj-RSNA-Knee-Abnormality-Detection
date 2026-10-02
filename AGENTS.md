# RSNA Knee 作業ガイド

このリポジトリ全体に適用する。最初に README.md、PROJECT.md、docs/validation.md、experiments/ledger.csv を読む。

- 会話と説明は日本語。現在の作業はCPUノートPCで行う。
- このPCでデータやモデルを取得したり、実MRIをデコードしたり、学習を始めたりしない。
- 別のGPU端末用のコードと実行手順を準備する。学習CLIのCUDAチェックを維持する。
- 学習・推論の前処理とラベル順を共通化する。推論にReportを要求しない。
- 欠損ラベルを0に置き換えない。gold/weak/public LBを区別して記録する。
- gold画像由来ラベルを学習やcheckpoint選択に使わない。公開重み・ラベルのgold利用は別途監査する。
- 同一検査・同一レポート・供給されたgroupをfold間で分けない。患者独立性を未確認で断定しない。
- 変更は一要因ずつ。config、seed、fold、入力ハッシュ、実行環境、採用理由を残す。
- 成果物は新しいrunディレクトリに保存し、以前の実験を上書きしない。
- data/とartifacts/はGit除外。認証情報や他プロジェクトのkaggle.jsonを読んだりコピーしたりしない。
- 公開スコアは作者報告と自分の再現値を区別する。調査時点とNotebook/Inputの版を記録する。
- 外部へのアップロード・Kaggle提出を自動で行うコードを追加しない。
- 純粋なCSVツールのチェックは標準ライブラリのunittestで行える。
- GPUコードの変更時は、利用可能なら人工データのforwardと提出契約を確認する。実学習はGPU端末で行う。
- タスク完了時にPROJECT.mdとdocs/validation.mdの現状を更新する。
