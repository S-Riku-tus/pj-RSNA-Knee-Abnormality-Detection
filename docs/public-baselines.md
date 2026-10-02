# 公開ベースラインを最初の比較基準にする

まずKaggle上で作者のNotebookを再現し、採点に通るモデルを確保する。このリポジトリの自作ResNetへ別アーキテクチャの重みを読み込むことはできない。

## 優先する単体モデル

[Knee MRI twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) と [重みの配布ページ](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) が第一候補。2026年10月2日のWeb調査では、作者は単体・TTAなしのPublic 0.924を報告していた。

配布説明では、`raptor_ft_coatnet_v4_full.pt` がその成績を出したcheckpoint。CoAtNet、384px、隣接3スライス、所見別Attention、レポート由来soft labelsの構成で、提供ラベル付き58件を学習から除外したと説明する。SWA版もあるが、作者は元のcheckpointを使用した。[作者のモデル説明](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

追加調査で取得できた配布ページは **Version 2、585.66MB、CC0** と表示していた。これは索引が持つ表示版で、現在のライブ版や再現するNotebookのInput版と一致する保証はない。Notebookの厳密な版は未確定のため、Kaggleで再現前に記録する。

ここまでが作者の報告。本プロジェクトでは重みの取得・ソース再実行・採点をしていない。物理座標の切り出し、正規化、Attention実装、ラベル順が一致しないまま重みだけ使わない。

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
