# RSNA Knee 次の実験計画 2026年10月4日

この文書は再分析時の提案を保存している。その後の実装・実行は [対照実験の手順](../controlled-experiments.md)、[PROJECT](../../PROJECT.md)、[検証記録](../validation.md)を参照する。以下の「今回」は分析時点の範囲で、後続の実験を未実施とする記述ではない。

今回の提案は、採点済みPublic 0.924の公開モデル再現を保持し、既存192pxキャッシュで学習動作と事前学習の効果を調べてから、ラベルと画像入力へ進むこと。最初に評価記録を整え、正規化と初期化を分けた比較を行う。現行モデルの50 epoch化や、大型モデル・複数fold・ensembleの同時導入は優先しない。

これは次工程の提案であり、改善を保証する設定ではない。今回は添付分析、実装、既存集計、公式資料と作者の公開実験を照合した。追加学習、重み取得、実MRIのデコード、新しいKaggle実行・提出は行っていない。添付原文は [user-analysis-20261004.txt](user-analysis-20261004.txt)、出典と容量計算は [reanalysis-20261004.json](reanalysis-20261004.json)。添付の外部成績は、自分の再現結果へ置き換えない。

## 現状から確実に言えること

| 項目 | 確認済みの結果 | 判断上の意味 |
|---|---|---|
| e002 | 5 epoch、約18分、bestはepoch 1 | epoch増加による採用予測の改善はなかった |
| e001とe002 | weak・gold予測CSVがそれぞれhash一致 | 5 epoch実験のbestを提出しても、1 epoch版と同じ予測になる |
| weak検証BCE | 自作0.430481、学習側陽性率の定数予測0.415327 | BCEの値だけでは画像学習の効果を示せない |
| weak macro AUC | 自作0.557023、定数0.5 | 順位付けが全くないと断定もしない。ただしレポートラベルの評価 |
| gold macro AUC | 58検査で0.487178 | 小さい画像評価集合の監査値で、隠しtestの予測値ではない |
| Synovitis fold 0 | 観測101、陽性100、陰性1、欠損87.7% | AUCが一つの陰性の順位に強く依存する |
| Public 0.924 | 10月3日に公式APIで確認した既存提出 | 公開CoAtNet再現の自分の実測値。自作モデルの成績ではない |

根拠は [e002](../../experiments/e002-baseline-fold0.json)、[保存予測の追加診断](../../experiments/e002-submission-review-20261003.json)、[既存提出](../../experiments/p001-public-model-reproduction.json)。epoch 2以降のweak AUCや予測CSVは保存されておらず、BCE悪化からAUC悪化まで断定できない。

## 添付分析を実装と照合した結果

| 指摘 | 実装で確認した事実と補足 |
|---|---|
| ランダム初期化 | `model.py` はResNet18のweights=None。所見別Attentionは既に実装済み |
| 学習と検証のBCE集計差 | 学習ログは各batchの観測セル平均の平均、検証は集合全体の観測セル平均。Dropout、BN、更新中の重みも異なり、差を純粋な過学習量と呼べない |
| batch 1とBN | encoderは1検査の有効24 windowをまとめて処理する。1画像だけを見るわけではない。勾配蓄積4回でもBNの統計は4検査をまとめた統計にならない |
| batch 4との比較 | 現行lossのままでは、BNに加えて観測ラベル数による検査の寄与も変わる。BNだけの効果を断定できない |
| 入力情報の削減 | 3系列・8 window・隣接3枚・192px。mm cropやモデルへの系列種別・位置入力はなく、全画像を正方形へresizeする |
| 他のbackbone | configだけでは切替できず、CLIはresnet18以外を拒否する。モデル生成と特徴次元の実装が必要 |
| ラベルの重み | 取得元のlabel_weightsは観測maskで、信頼度weightではない |
| goldの扱い | 添付の開発利用に関する一般論を、gold非学習・非選択という本repoの方針を変更する理由にしない |

[runtime.py](../../src/rsna_knee/runtime.py)、[model.py](../../src/rsna_knee/model.py)、[imaging.py](../../src/rsna_knee/imaging.py)、[contracts.py](../../src/rsna_knee/contracts.py)を読んで確認した。BNの動作は [PyTorch 2.10公式資料](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html) と一致する。BNや情報削減が現在の不振の原因であることは、まだ実験で確定していない。

## 最初に整える評価と互換性

各epochについてweak予測CSV、12所見のAUC・陽性数・陰性数・欠損数、所見別BCE、予測の平均・標準偏差・陽性陰性別分布をprivate runへ保存する。既存valid推論の出力から計算すれば追加forwardは不要。soft labelを0.5で二値化したAUCを公式評価として扱わず、未定義クラスを別に記録する。

同じcheckpointをevalにして、固定train部分集合とvalidのセル平均BCE・所見平均BCEを同じ方法で計算する。train部分集合のUID一覧はGitへ入れず、件数・seed・hashだけを共有する。これは更新中のtrain lossとの混同を解くための診断で、独立検証にはしない。追加評価の所要時間を既存train/validから分けて記録し、過去の約18分と同じ範囲で比較する。

ログ追加が学習を変えないことも確認する。現行trainとvalidはDataLoaderのGeneratorを共有し、shuffle=Falseでもiterator生成が乱数状態を消費する。追加train-eval loaderは専用Generatorを使い、必要なRNGとmoduleごとのtrain/eval状態を保存復元する。既存validのGeneratorを付け替える変更も、過去baselineとの再現条件が変わるため別途記録する。人工データでsample順・optimizer step数・state_dict・BN buffers・RNG状態がログ有無で一致することを確認する。[PyTorch 2.10 DataLoaderソース](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/data/dataloader.py)

キャッシュの検査にも整備が必要。現在のexport検査器はsrc全体を現在のrepoと照合するので、ログやモデルだけの変更でも不一致になる。一方、画像fingerprintはimaging.pyとpreprocess設定から決まる。保存export内のコード・ファイルの完全性と、現在コードの画像前処理互換性を別々に検査する。hash検査を単に削除しない。既存exportを上書きせず、変更したrunのコードhashとcheckpointの入力正規化modeを保存する。

## 既存キャッシュで行う最小比較

まず人工データのCUDA forward・backward・mask・旧checkpoint読込・Reportなし提出契約を確認する。その後、少数の学習側検査で固定画像に十分適合できるか診断する。goldとそのgroupを使わず、結果は精度比較に含めない。失敗したら初期化の比較へ進む前にUID、cache、mask、ラベル、gradient、optimizer更新を調べる。

正常なら、fold 0、seed 20261002、ラベル版、192px、3系列、8 window、Attention、5 epoch、学習率、損失、checkpoint選択規則を固定して、次の3条件を比較する。

| 条件 | Encoder初期化 | Tensorの正規化 | 比較の目的 |
|---|---|---|---|
| A | random | 現行の[-1,1] | e002を基準に、ログ追加後の再現を確認 |
| B | random | ImageNetのmean/std | Aと比較して正規化の影響を調べる |
| C | ImageNet事前学習 | Bと同じ | Bと比較して初期化の影響を調べる |

ImageNet正規化はuint8キャッシュをtensorへ変換する段階の変更なので、元DICOMからのキャッシュ再生成は不要。Torchvisionの224px center cropまで同時に導入しない。現行MRIの3channelは隣接sliceでありRGBではないため、ImageNet初期化の効果は仮説である。[Torchvision 0.25公式ソース](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/resnet.py)

初期重みは出所・版・license・hashを監査し、学習時は明示したローカル重みを読み込む。推論時は保存checkpointだけを読む構成にして、Internet OFFで暗黙に初期重みを取得しない。入力正規化はconfig/checkpointへ保存し、学習・推論で共通化し、既存checkpointには従来modeを明示的に適用する。

最初の比較ではbestを選ぶ規則も現行weak BCEに固定する。weak AUCを新たに観測しても、後から都合のよいepochだけを選び直さない。選択指標やlossを変更するなら、その変更自体を次の一要因の実験として事前に定める。

Cの学習挙動に疑問が残る場合は、まず事前学習済みBNのrunning statisticsを固定する条件をCと比較する。affine weightの学習条件はCと同じに保ち、毎epochのmodel.train()後にBNのmodeを維持する。実batch 4との比較は、その後にlossの検査重みをそろえて行う。ランダム初期化BNを無条件に固定することは初手にしない。

## ラベルと画像を改善する分岐

少数train診断に失敗したら、データ対応か学習処理の修正を優先する。診断に成功してCが改善したら、その条件を固定して追加seedかfoldで再確認する。診断に成功してもBとCが改善しなければ、BNの診断とラベル・入力の監査を優先し、大型encoderの探索へ直ちに進まない。weak AUCの改善が観測された場合も、レポートラベルへの適合であり画像goldへの転移が保証されたわけではない。

ラベルでは、Synovitisの陽性率をgoldの比率へ合わせず、明示的陰性・未言及・不確実・基準未満の意味を元stateと照合する。否定、重症度、部位、言語差、Effusionとの混同を、学習側のレポートと公開された所見定義に基づいて監査する。レポート本文や個別ラベルをチャット・Gitへ出さず、rawと根拠はprivateに保存する。現在のsourceはP=1、N/B=0、U/M=欠損。未知を一律0や0.5へ変えない。[Hostの所見定義](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)

報告されやすい症例だけが観測される偏りは、fold層化やclass weightだけでは解決しない。別ラベル版は新manifestへ保存し、同じ版で新しい比較基準を作る。手動監査集合はレポートの意味の評価であり、画像専門家のgoldの代替とはしない。必要なら所見平均BCEを一条件として試すが、陰性がほぼない所見へ大きい重みを付けても欠けた情報は増えない。

層化foldが必要なら旧foldを残し、同一レポート・供給groupの連結を保持した別版を作る。陽性だけでなく陰性・観測数も監査し、比較元も新分割で学習する。患者情報が確認できなければ患者独立とは呼ばない。

画像の次の最小比較は、元DICOMから作る224px全体像と、その同解像度での物理crop。192→224で解像度、224全体→224cropでcropを別々に調べる。mm cropの大きさ・中心・縦横比と、関節や周辺所見が切れていないかを確認する。公開モデルの140mmは候補であり、全所見に最適と証明された値ではない。series数・window密度・位置特徴の追加も、その後の別条件へ分ける。

## 高解像度化の容量条件

現行Notebookは全検査の未圧縮uint8配列量を計算して18,000,000,000 bytes未満を要求する。これは保存量の保守的ガードで、一括RAM使用量でもKaggleの最新公式枠そのものでもない。

| 解像度 | 4,407検査のraw画像量 十進GB | 現行全件Notebook |
|---|---:|---|
| 192 | 11.697 | 事前ガード通過 |
| 224 | 15.921 | 事前ガード通過。実容量・空き容量は別確認 |
| 256 | 20.795 | 事前ガードで停止 |
| 288 | 26.318 | 事前ガードで停止 |
| 336 | 35.822 | 事前ガードで停止 |

根拠は [Notebook generator](../../scripts/generate_cache_notebook.py)。現在の約7.79GBは圧縮npzとCSV・コードを含む実測export量なので、この表と異なる。圧縮率を面積比だけで外挿して保存可能と断定しない。

256px以上の全件化では、少数試作で圧縮量・decode時間を計測し、現行のKaggle保存枠とworking領域を確認してからshardを設計する。LIMITは先頭prefixを選ぶだけなので全件分割機能ではない。元train.csvは全件の同じhashを保ち、処理対象UIDだけを明示選択する。全shardの版・UID集合・fingerprint・ファイルhashを記録し、統合manifestで4,407件の重複・欠落なしを検査する。ガードを外して全件実行することは初手にしない。

## 公開手法から使える根拠

| 一次資料 | 作者が報告した結果 | 今回へ適用する範囲 |
|---|---|---|
| [soft/hardの比較](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105) | 3 seed、rank平均のgold差+0.0143、95%区間[-0.0041,+0.0330] | 未言及hard0/soft0.28で現行maskと違う。soft labelや改善幅をそのまま採用しない |
| [EfficientNet入力比較](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597) | 単一foldでslice選択差+0.018、60 epoch差+0.0008 | 4系列・288px・140mm・weighted BCE・scanner-aware split。入力を調べる参考で、現在の条件での再現値ではない |
| [既存CoAtNetのmodel card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) | 5系列slot、64画像、140mm crop、cache336→input384、所見別Attention | 前処理とモデルを一体で扱う。現行192px cacheへcheckpointだけ移植しない |

索引から確認したPublic 0.942の別Notebookは、内部処理・重みの来歴・Input版をまだ監査できていない。追加の再現候補として扱い、今回の学習計画を置き換える証拠にはしない。2024腰椎優勝の局所化は別競技で座標教師があり、今すぐ膝検出器を作る理由にはしない。詳しいURLと確認範囲は出典JSONに残した。

## 提出利用条件と評価の独立性

利用条件の確認はローカル研究と並行する。採用ラベルはCC BY-NC 4.0で、賞金付き参加や学習済み重みの公開を含む今回の利用について、作者側の条件と大会側の条件を確認する。Hostは別データについて、大会側の利用可能性と提供元の規約適合を区別している。これは採用ラベルの許可・禁止を決める回答ではない。[Host回答](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)、[CC公式FAQ](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses)

sourceはhosted APIとfree quota利用を作者が宣言している。無料だったという情報だけで元データ送信や大会利用の許可は証明できない一方、それだけで禁止とも断定できない。RulesとHostの外部LLM案内本文は今回も取得できなかった。確認する対象はExternal Data and Tools、Data Security、Winners Obligations、[外部LLM案内](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)。問い合わせが必要になった場合は質問文を準備するが、この調査では外部へ送信していない。

goldは既に結果を見ているため完全未観測の最終holdoutとは呼ばない。学習・checkpoint選択・prompt調整・ensemble係数合わせには使わず、事前に定めた節目の監査値とする。公開競技重みの学習範囲とgold選択履歴は厳密に未確認なので、後付けfoldの評価を独立OOFと呼ばない。自作モデルをPublic 0.924へ単純に半分混ぜる根拠も、まだない。

## 残り期間の進め方

以下は10月4日時点の提案。計画日であり所要時間や改善の保証ではない。利用条件を確認する間も、非商用研究範囲の診断とコード整備を進める。

| 期間 JST | 優先作業 | 次へ進む条件 |
|---|---|---|
| 10月4〜6日 | 評価記録・cache互換性整備・人工契約・少数train診断。利用条件を並行確認 | ログで学習経路が変わらず、入力と更新が正常 |
| 10月6〜9日 | A/B/C比較、必要な場合だけBN診断 | 初期化と正規化の寄与を別々に説明できる |
| 10月8〜10日を目安 | 利用可能な自作モデルをKaggleで手動実行・一回採点 | Internet OFF、実test decode、全体時間、提出契約、採点成功 |
| 10月10〜15日 | 判明した制約に応じてラベル版か224px/cropを別条件で比較 | 同じ条件の比較基準があり、観測数・時間・容量が適切 |
| 10月16〜19日 | 有望な少数条件だけ追加fold/seed、必要なら別encoder | 改善が一つのfoldや希少クラスだけに依存しない |
| 10月20〜22日 | コード・重み・依存を固定し、再実行・採点・最終選択 | 成功済み候補があり、最終の変更も採点済み |

初期比較が改善しなくても、利用条件を満たす現行モデルで提出経路を確認する価値はある。大規模な調整完了まで自作Notebookの動作確認を待たない。提出全体はInternet OFF・9時間以内で、ローカル18分の学習時間とは別に計測する。

参加・チーム統合期限は10月16日08:59 JST、最終提出は10月23日08:59 JST。10月22日中に最終選択を終える提案とする。現在の順位・メダル境界は未確認で、Publicの小さな差からメダルを予測しない。[公式OverviewとTimeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
