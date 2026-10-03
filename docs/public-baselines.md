# 公開ベースラインを最初の比較基準にする

まずKaggle上で作者のNotebookを再現し、採点に通るモデルを確保する。このリポジトリの自作ResNetへ別アーキテクチャの重みを読み込むことはできない。

## 優先する単体モデル

[Knee MRI twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) と [重みの配布ページ](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) が第一候補。2026年10月2日のWeb調査では、作者は単体・TTAなしのPublic 0.924を報告していた。

配布説明では、`raptor_ft_coatnet_v4_full.pt` がその成績を出したcheckpoint。CoAtNet、384px、隣接3スライス、所見別Attention、レポート由来soft labelsの構成で、提供ラベル付き58件を学習から除外したと説明する。SWA版もあるが、作者は元のcheckpointを使用した。[作者のモデル説明](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

追加調査で取得できた配布ページは **Version 2、585.66MB、CC0** と表示していた。これは索引が持つ表示版で、現在のライブ版や再現するNotebookのInput版と一致する保証はない。Notebookの厳密な版は未確定のため、Kaggleで再現前に記録する。

ここまでの説明は作者の報告。10月3日、ユーザーが既に提出した `rsraki/knee-mri-twelve-findings-from-a-single-model` を公式APIで確認し、status=COMPLETE・実測Public 0.924を記録した。今回の作業で公開重み取得・再実行・新規提出は行っていない。保存Input版・実行時間・厳密な生成履歴は未確認。物理座標の切り出し、正規化、Attention実装、ラベル順が一致しないまま重みだけ使わない。

## 再現の手順

1. 公式競技へ参加後、上記NotebookをKaggleのCopy and Editで開く。
2. Best Scoreを出した版と現在の版を区別し、再現するNotebook版を決める。
3. 競技入力、checkpoint、必要な追加パッケージの入力を確認し、各Inputの版を記録する。
4. 元の前処理とモデル定義を保ち、GPUを選択して実行する。推論時のインターネットは無効にする。
5. 例示テストでの完走後、NotebookをSave and Run AllしてSubmitする。
6. 採点成功・実測Public LB・実行時間・Notebook/Inputの版を `experiments/ledger.csv` に記録する。
7. 複製したNotebookのソースをGPU端末で保存し、変更の比較元として固定する。データや重みはGitへ入れない。

作者版で動かない場合はInputs、環境の版、checkpointとラベル順から確認する。先にモデルや前処理を変更すると再現の原因を切り分けられない。

## その他の候補

- [Pilkwang RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) と [Inputs](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1/input)：DINO系と公開ラベルをたどる入口。添付分析には0.891とあるが、今回の本文取得ではその版のスコアを再確認できなかった。
- [単体モデルの学習Notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)：再現後、画像の見せ方と学習条件を調べる。
- [DINOsaur train](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v3-train)：別系統の学習設計の比較対象。
- [Full 4 arm ensemble](https://www.kaggle.com/code/nishantkharga/rsna-knee-full-4-arm-ensemble-v55)：単体モデルの理解後に比較。添付の成績や版は未再確認。
- [CPU pixel cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache)：デコードの再利用を考える資料。画質・スライス削減は情報損失を伴う。

公開重みの利用条件、学習対象、goldの使用、ラベル作成器のgoldへの調整も記録する。公開重みを後からランダムに分割したデータで評価して、独立した検証と呼ばない。

## 公開ラベルと学習コードの採用監査（10月3日）

公式Kaggle CLI/SDKから公開metadata、ファイル一覧、版を取得し、作者の説明と確認できたCSVの事実を分けた。記録は [label-sources-20261003.json](research/label-sources-20261003.json)。以下の版はDataset版であり、CSV名に含まれるv2/v4とは別である。

| 候補 | 確認した版・ライセンス | gold利用と今回の判断 |
|---|---|---|
| [Steven](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) | Dataset 6、CC0 | v2はgold比較からSynovitisをEffusionで補正。v4はそのラベルをblend。独立gold評価には使わない。元v1の開発履歴も未確認 |
| [Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels/versions/1) | Dataset 1、CC0 | YES/NO/UNKとconfidenceを持つ。配布生成器は未配布のllm_labelerをimportし、prompt・scoreの完全な監査ができないため未採用 |
| [lixin73](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56/versions/1) | Dataset 1、CC0 | APIのdescriptionが空で、生成方法・未言及・goldの扱いは未確認 |
| [Yehezkiel](https://www.kaggle.com/datasets/yehezkielhaganta/psuedo-labeling-knee/versions/7) | Dataset 7、MIT | 1/0/-1の定義を確認。-1は未言及であり欠損として扱う必要がある。goldへの調整は未確認 |
| [Laymond](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/versions/2) | Dataset 2、CC0 | gold結果を踏まえたprompt改訂と共起補正を説明。raw版でもpromptの独立性を保証しない |
| [nartaa](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets/versions/9) | Dataset 9、CC0 | perlabel版はgoldで所見ごとに選択。v4系は調整済みラベルを継承。gemlow単独版の開発履歴は未確認 |
| [vmohitrao](https://www.kaggle.com/datasets/vmohitrao/rsna-knee-report-labels/versions/3) | Dataset 3、CC BY-NC 4.0 | 作者READMEはgold 58件を抽出・prompt開発から除外したと明記。元train hashと4つのCSVの公開hashが一致。非商用の研究用候補として取り込み |

vmohitraoのgold非使用は作者の宣言であり、非公開の開発ログまで監査した事実ではない。生成ラベルは臨床的な正解ラベルではない。4,349行のラベルは公式goldを含まず、P=1、N/B=0、U/M=空欄、weightは観測maskである。全セルのstate・value・mask一致を検査した。

全欠損138件とgoldの同一レポート/groupにつながる4件を除外して、研究用manifestはweak 4,207件・gold 58件。供給されたreport_groupを保ち、fold 0〜4は821・891・888・778・829件。患者独立性は未確認である。

fold 0のSynovitisは観測101件のうち陽性100・陰性1、欠損率87.7%。欠損を陰性へ変えず、この偏りを実験の制約として記録する。weak BCEの良さだけで全12所見の画像性能が良いと判断しない。

CC BY-NC 4.0の利用条件を記録し、今回の採用範囲は非商用のローカル研究とする。現時点で大会Rules本文との完全な照合はできておらず、この候補のKaggle提出利用・受賞時の権利処理を確認済みとはしない。ラベル・レポート・重みはGitへ入れず、提出も行っていない。

### 公開学習コードをそのまま移植しない理由

取得したDreaddevelopmentの学習コードはNotebook Version 8。goldを学習から除外する一方、各epochのgold AUCでbest checkpointとtop-kを選ぶ。配布checkpointの厳密な生成履歴との一致は未確認だが、この選択方法を本リポジトリへ移植しない。本実装ではweak検証masked BCEでcheckpointを選び、goldは選択後に評価する。

Pilkwang baseline Version 15の学習分岐にも公式ラベルを高いweightで混ぜる処理がある。公開Notebookや重みのスコアを、自分の独立したholdout成績として記録しない。
