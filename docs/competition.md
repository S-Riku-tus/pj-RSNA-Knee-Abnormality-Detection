# RSNA Knee 大会概要と調査結果

調査基準日 2026年10月2日 JST。添付分析を出発点に公式概要、RSNA公式紹介、公開資産の配布ページを照合した。実データを取得していないため、配布CSVの件数は本リポジトリでの実測ではない。

## 競技の目的と評価

2026年のRSNA Knee Abnormality Detectionは、MRI検査から12の所見を予測する研究コードコンペ。複数所見が同時に存在する多ラベル分類であり、検査ごとの予測を出す。評価は各ラベルのROC-AUCを等しく平均する。AUCは順位付けの尺度で、診断正解率ではない。[Kaggle公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

提出列は `StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`。コード上の正本は `src/rsna_knee/contracts.py`。実際のsample_submission.csvも照合する。

主催者による所見の定義と図示は [Label Description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) を参照する。靱帯・半月板損傷、関節の変性、液貯留・滑膜炎、嚢胞、骨所見を一括して扱うため、病名の単純な有無だけでレポートをラベル化しない。

## 日程と提出環境

公式Timelineの各期限は23:59 UTC。JSTへの変換は9時間加算したもの。

- 開始：2026年7月30日。
- 参加登録・ルール同意、チーム統合：10月15日23:59 UTC → **10月16日08:59 JST**。
- 最終提出：10月22日23:59 UTC → **10月23日08:59 JST**。
- 受賞者の成果物提出：11月5日23:59 UTC → 11月6日08:59 JST。

提出NotebookはCPU/GPUとも9時間以内、インターネット無効、出力は `submission.csv`。公開され無料で利用できる外部データ・事前学習モデルは概要上許可される。通常順位と効率賞の順位は別枠。[公式OverviewとTimeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

## データの理解

RSNA公式紹介は、世界19施設、5,000件超のMRI、約12言語の読影レポートという全体規模を示す。[RSNA公式紹介](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

添付分析と公開モデル作者の説明では、学習検査4,407件、提供ラベル付き58件、レポートから教師信号を作る4,349件という内訳。配布データの実測値はGPU端末で `audit` を実行して確定する。58件をコードへ固定しない。[作者によるデータと学習の説明](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

想定するデータ構造は、`train.csv` に検査ID・Report・12ラベル、`train_series.csv` にシリーズID・断面・液体強調・脂肪抑制、`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm` に画像。`test.csv`、`test_series.csv`、`test_series/`、`sample_submission.csv` が推論側の入力。公開Notebookログではこの構造を確認できたが、CSV本文と実ファイルは未取得。[配布データページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)、[公開実行ログ](https://www.kaggle.com/code/vigneshguguloth/version1/log)

添付分析と複数の公開実装は、テスト時にReportがないこと、goldラベルがレポート抽出と一致しない場合があることを示している。今回、Dataページと主催者回答の本文はWeb取得で表示されなかったため、原文の再確認はGPU端末での初動項目に残す。実装はReportを推論入力にしない設計とし、後から依存を除く必要がないようにした。[主催者回答の確認先](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491)

## 添付分析から採用する戦略

以下は本プロジェクトの判断であり、公式な推奨解法ではない。

1. 最初に公開Notebookで採点成功を確保する。精度改善より先に提出経路を確認する。
2. 学習ラベルの出所と、MRIで判定された正解との違いを明示する。
3. 学習・推論の前処理を共通にし、シリーズ選択、スライス順、キャッシュを監査する。
4. 独立性の確認された検証を持ち、ラベル・画像・モデルの変更を一要因ずつ比較する。
5. 単体モデルの安定後に、間違い方の異なる少数モデルを組み合わせる。

CPUノートPCは編集端末、外部GPUは学習端末、Kaggleは最終推論端末とする。学習を提出Notebookに含める必要はない。

## まだ確定していない項目

- 最新のLeaderboard全行、参加チーム数、銅メダル境界、最新首位スコア。
- Public/Privateの比率、日次提出回数、最終選択数、チーム人数の最新制限。添付や他実装の数値を固定しない。
- 配布データの現在の実サイズ・件数・CSV本文、欠損や不正DICOMの件数。
- 特定の公開Notebook版のソース、完全な入力バージョンと再現性。
- 外部LLM・特定外部医療データの利用に関する最新ルール本文。

[Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)、[Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)、[Discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion) のライブ表示を参加・提出前に確認する。公開0.924や添付の0.94台は、メダル保証や自分の再現値ではない。
