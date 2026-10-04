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

10月3日時点ではCC BY-NC 4.0の利用条件を記録し、採用範囲を非商用のローカル研究としていた。10月4日に公式Rules・Host回答と照合し、研究・学習目的と帰属等を維持して大会用途へ採用を進める判断に更新した。根拠は [利用条件監査](research/kaggle-source-eligibility-20261004.json)。入賞時の権利処理や商用化は、その目的が発生した段階で確認する。ラベル・レポート・重みはGitへ入れない。

### 公開学習コードをそのまま移植しない理由

取得したDreaddevelopmentの学習コードはNotebook Version 8。goldを学習から除外する一方、各epochのgold AUCでbest checkpointとtop-kを選ぶ。配布checkpointの厳密な生成履歴との一致は未確認だが、この選択方法を本リポジトリへ移植しない。既存A/B/Cはweak検証masked BCEで選択した。以後もrun開始前にweak検証の選択規則を固定し、goldをcheckpoint選択に使わない。

Pilkwang baseline Version 15の学習分岐にも公式ラベルを高いweightで混ぜる処理がある。公開Notebookや重みのスコアを、自分の独立したholdout成績として記録しない。

## DINOsaurの版・依存・再現契約の追加監査（10月4日）

公式SDKの通常認証で、公開Notebookのソース5件、Datasetのmetadataとファイル一覧15件、小さいソース・帰属・実行契約文書13件を取得した。実MRI・ラベル表・学習済み重み・予測配列は取得せず、公開コードの実行・アップロード・提出も行っていない。出典URL、版、ソースhash、確認できた事実と未確認事項は [public-reproduction-contract-20261004.json](research/public-reproduction-contract-20261004.json) に保存した。

[DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4/output) の表示Bestは作者報告0.937のV32で、最初に取得した最新ソースはV35だった。追加調査でSDKの `version_label="v32"` を指定し、応答の版番号32とソースhashを固定できた。[取得経路と追補](research/public-reproduction-v32-followup-20261004.json)。保存InputのDataset／Notebook版、提出されたCSVと採点の対応、自分の再現スコアは未確認である。取得できたV35は、0.937 controlを別CSVへ保存した後、PublicDual・Synovitis・CoAt残差を追加する。最終 `submission.csv` はPublicCoAtのmain（内部alpha 0.16）であり、0.937 controlとは別候補になる。最新をコピーしただけでV32を再現したことにしない。

V32の最終 `submission.csv` も保存controlとは異なる。controlへV6 five-targetの外側weight（ACL 0.80、Lateral OA／PFOA／Synovitis 0.35、Baker's Cyst 0.375）を重ねる。coreのcode cell 2〜10はV35とbyte一致するが、その後の分岐が異なる。作者がどの出力CSVを採点したかはソースだけでは確定しない。V32指定のOutput一覧9ファイルを確認したが、予測payloadは取得していない。

V35のlive metadataには14 Dataset、3 Notebook Output、DINOv2-smallのModel V1と競技入力がある。Notebookに埋め込まれたInput IDには継承されたmetadataがあり、Notebook Outputの数もlive metadataと一致しない。IDを配列の順番でInput名へ対応させず、保存版のInput画面と有効ファイルのhashから固定する。

今回観測したInputの最新Dataset版は次のとおり。**この表はV32の保存Input版ではない。**

| Input | 今回観測したDataset版 | 契約上の注意 |
|---|---:|---|
| Dread maxspan／native384／native384dense | 各1 | v5 SWA／v8 SWA／v10の3重み。前処理が異なる |
| Mattia knee-mri-fold-weights | 2 | DINOv3の5 fold重みとtimm wheel |
| Tonylica consolidated assets | 5 | 現在は別公開方式の41 checkpointと校正表をまとめた資産。元V4のrecipeと同一とは限らない |
| Renta public0033 Meniscus bag | 2 | safetensors、runtime、manifest、run contractを持つ |
| Antoine E9／E11 Rad heads | 3／1 | RadImageNet由来。gold学習・係数選択の履歴を持つ |
| Mattia residual CoAt | 3 | e4/e6/e8はepoch ensemble。5-foldとは別 |
| Dread widedense | 2 | 既存0.924のv4重み候補。V35の3 Raptorとは別 |

V35はDINO／DINOv3／Rad／Raptorの推論とrank変換を重ね、途中CSVを複数回書く。DINO側は全public-frontier票の完成を要求する一方、CoAt分岐には失敗時に先行CSVを維持する処理がある。提出CSVの形式検査だけでは、意図した全分岐の完走を確認できない。member数・fingerprint・Input hash gate・分岐のreceipt・最終CSVの選択をログで確認する。表示される数分の実行時間は例示testの記録であり、hidden test全体の見積もりに使わない。

### gold履歴と利用条件

- Renta Dataset V2の配布run contractは、weak 4,349件、公式ラベル付き58件を除外、gold値・gold metricsを読まず、seed 2027で30 epochの最終重みだけを選んだと記録する。文書とsidecarのhashは確認したが、非公開実行ログや教師Pilkwang `report_labels_v2` の生成独立性まで確認したわけではない。`fold_safe_oof=false` が明記されている。
- Antoine E9 Dataset V3のREADMEは公式goldでweakを上書きして学習すると明記し、E10のblend係数選択もgold 58件を使うと説明する。配布学習ソースにもgold代入と係数探索がある。latest DINOsaur V5 train V15にもgoldを学習targetへ入れ、gold検証AUCでepochを選ぶ処理がある。historical V32重みとの厳密な生成対応は未確認で、これらを本リポジトリのgold除外学習へ移植しない。
- Tonylica V5の `SOURCE_AND_LICENSES.md` は一括CC0ではなく、CC0／Apache／CC BY-NC-SA／DINOv3の元条件等が混在すると説明する。E13の再配布licenseは過去監査で別途確定していない。公開Datasetの表示が「Other」であることを、条件なしの再配布許可と解釈しない。公開NotebookをKaggle上で参照する経路と、第三者重みを自作bundleへ再配布する経路を区別する。

### 次に作る採点候補

まず、自作モデルの比較結果から選択規則が確定したcheckpointを、凍結code zip・提出Notebook・hash manifest・帰属記録とともに新しいbundleへ保存する。競技入力とprivate bundleをKaggleで手動追加し、Internet OFF・GPUでSave and Run All、例示testとログの確認、手動採点へ進む。既存の実測0.924提出を保持し、自作候補のPublicは採点後に記録する。

公開方式の次候補は、取得済みV32ソースと作者の保存Input画面を照合し、その版をCopy and Editして入力版と提出CSVを固定する経路を優先する。V32のscriptVersionId、Dataset／Notebook／Modelの版、runtime、実測Publicを揃えるまでは「0.937再現済み」としない。SDKの版指定metadataには14 Dataset／3 Notebookの参照があるが、それらの版番号がない。embedded metadataの少ないInput ID一覧から補完しない。V35を実行する場合は独立した新しい候補として記録し、保存0.937 controlと最新PublicCoAt mainのどちらを採点するかを先に固定する。今回、DINOsaurの採点可能な新規bundleを作ったとは扱わない。

CoAtNet推論の最新V7はソースとhashを固定でき、依存はnumpy・torch・timm・pandas・pydicom・cv2と `raptor_ft_coatnet_v4_full.pt`。汎用初期重みの取得をせず `pretrained=False` で構築する。140mm crop・336px画像・64スライス・6〜94% span・384px入力・42 windowがソースの契約である。作者が0.928と報告するmaxspan v5は2〜98% span・62 windowを要求するため、ファイル名だけ差し替える比較をしない。Training V8の未配布softラベル・gold選択は、この推論再現とは別の課題として残る。
