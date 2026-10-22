# GPU解放後の再評価 — 2026-10-22

**判断を更新する。GPUが空いた現在、ConvNeXt Tinyの2fold学習を条件付きの最終候補として再検討する価値がある。** 追加の人工測定では2processのGPU計算が各約2.8倍遅くなったため、並列化の効果は実epochの速度を測るまで未確定である。主力のnative20→10%比較を先送りせず、既存候補の提出準備と並行し、学習候補は最大1枠に限定する。GPUの空きを根拠にSmall・Tiny・DINO等を同時に大量実行する方針にはしない。

今回は追加調査・ログ/metadata監査・人工GPU計測までである。実データ学習、実MRIデコード、画像を使う新forward、モデル取得、外部アップロード、Kaggle Save/Submitは開始していない。前回の[全履歴・最終5枠分析](goal959-analysis-20261022.md)から、GPU可用性に依存した判断を更新する。

## GPUだけでなくホストメモリも回復した

10月22日14:25 JSTの実測はRTX 4090の24,564MiB中851MiB使用、GPU稼働率2%。物理RAM31.85GiB中17.71GiB空き、Windowsのcommit上限46.50GiB中30.74GiB空きだった。GUI等のGPUコンテキストは残るが、重い学習がGPUを占有している状態ではない。[今回の読取証拠](../artifacts/research/20261022-gpu-free-root-analysis-v1/evidence.json)。

昨日の失敗時はsystem commitが上限49.92GBに対して49.88GB近くまで達していた。Tiny自身のcommitは約5.25GB、Smallも約5.09GBで、イベントが列挙した他プロセス群のcommitも約20.8GBあった。Tinyはuint8画像配列63.3MiBの確保、SmallはキャッシュZIP読み込み中の確保で失敗した。**CUDAのVRAM不足が原因という証拠ではない。** GPUだけ空いてRAMの問題が残ったのではなく、現在はcommit余裕も約30GiBへ回復している。ただし再発しない保証ではない。[失敗時のOS記録](../artifacts/research/20261021-F015-MemoryError-root-diagnosis-v1/windows-resource-exhaustion-events-redacted.json)。

したがって、前回の「CNNは現有の完成モデルとして使えない」は今も正しいが、「新規CNNを締切までに作る道も外す」という結論までは維持しない。失敗したモデルの精度が低かったわけではなく、予定した評価まで到達していない。

## E3から厳密には再開できない

TinyのE3、F006/F013のE12をpickleの命令列と生成コードで監査した。保存物はencoder suffix/head等の推論重みで、optimizer・GradScaler・global CPU/CUDA乱数・shuffle Generatorの状態がない。checkpointのtensorはこの監査ではロードしていない。[checkpoint/所要時間監査](../artifacts/research/20261022-gpu-free-feasibility-v1/result.json)。

E3重みから学習すること自体は新しいwarm-start実験として可能だが、元runの残9epochを同条件で継続したことにはならない。Adamの状態をリセットし、学習率・乱数列を再設計する別条件になる。[PyTorchのcheckpoint説明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)も、学習再開にはmodel以外にoptimizer状態等が必要とする。

最終日の標準案は、旧runを保存し、同じ初期重みから新規runで固定12epochを学習すること。旧E3の精度を見て継続epochを選ぶことはしない。

## 学習案を提出までの総工数で比較

| 案 | 確認済みの材料 | 時間・未完部分 | 判断 |
| --- | --- | --- | --- |
| **Tinyのfold0/1を新規並列学習** | 手元generic重み、native392キャッシュ、コード、人工GPU資格、固定E12評価器あり | 1foldは過去約6.0–6.8h相当。2fold並列の実epoch時間は未測定。CNNのKaggle接続は未実装部分あり | **学習を1案追加するなら最優先** |
| Tinyを逐次2fold | 同上 | 学習だけ約12.1–13.6h相当で、夜明け前に接続・採点が集中する | 並列が遅い場合の安易な切替先にしない |
| Smallを新規2fold | generic重み・人工GPU資格あり | 完了epochがなく全epochの実測なし。Tinyより計算量が多く、追加効果も不明 | 今回は外す |
| F013の学習窓を最大48→24 | 推論は全窓のままにできる、公開train/eval減窓例あり | 実有効窓削減は23.7%。新sampling・RNG・検証が必要で、病変を落とす可能性 | 今回の主学習案にしない |
| F013の完成encoderを固定しheadのみ再学習 | 最終CLS特徴は小さい。過去のcached-head学習は12epoch約15分 | F013 encoder別の全件窓特徴は未作成。抽出・新headの仮説・serving接続が必要 | 次善の研究経路。15分で提出モデルが完成する案ではない |
| F006 fold2を作り封印OrthoDiff C=.01を評価 | fit/CSV評価コードと外側予測が準備済み | 対照1foldに約6–7h。その後の補完性が正か未知、実推論も別 | 有効な未完研究だが、今回の第一候補より優先しない |

ここで8–10時間等の数字を、学習から正式採点までの保証時間として使わない。モデル作成、弱検証、有限性・復元、Kaggle接続、可視推論、CSV算術、正式採点が別に必要になる。

## Tinyの並列案には条件を付ける

旧Tinyの完了epochは1,811／1,946／2,035秒、約30.2–33.9分だった。これは昨日の環境で実測した数字で、全12epochの実測ではない。12倍すると1fold約6.0–6.8時間、2fold逐次なら約12.1–13.6時間になる。

2つの独立processでfold0/1を同じGPUに載せ、1process当たりのepoch時間が旧値の1.2倍なら約7.2–8.1時間、1.5倍なら約9.1–10.2時間で両方終わる計算になる。2倍まで遅くなるなら逐次同様で、締切前の運用余裕が乏しい。これらは**競合率を置いたシナリオ計算**で、実学習の測定結果ではない。

旧運用はfold0が正ならfold1へ進む順序だった。今から並列にするなら、**精度を読む前に、両fold・seed・12epoch・比較対象・計算予算を固定する新しい実行計画**として記録する。旧policyとrunを上書きせず、両foldを先に計算する予算配分の変更として扱う。元のCUDAチェック、同一Report/group分離、欠損mask、前処理、ラベル順を維持する。

学習を進める場合の具体的な時間条件は、次を提案する。まだ学習を実行した事実ではない。

1. 10月22日15:00–15:30頃までに始められる場合、Tinyだけの2foldを新規runで並列実行する。Tiny＋Smallの並行探索や途中の高いepochへの切替はしない。
2. 最初の完全なepochと次のepochの速度から、遅い方の固定E12完了が10月23日00:30までに収まるかを計算する。45分/epochなら12epochで9時間。残りepochは直近の遅い方の速度で見積もる。見込みが越える場合、最終提出の必須経路から外し、途中epochを提出へ救済しない。
3. 起動前の物理RAM空き14GiB以上・commit空き20GiB以上を目安とし、実行中は物理空き6GiB・commit空き10GiBを割る状態が続くなら、自分の新規runだけを停止対象にする。これらは今回提案する余裕幅で、測定済みの安全上限ではない。GPU VRAM、processの増分、I/O待ちも記録する。OSのページファイルや他のprocessは変更しない。
4. 日中にCNN用serving接続を並行実装する。現在の[preview](../experiments/CNN-P039-disabled-source-preview-ready-20261021.json)は実行を拒否する設計で、完成済みruntimeではない。generic CNN復元、trained checkpoint結合、現在の全モデルのGPU確認、実MRI経路、失敗時の契約、CSV検算を完成させる必要がある。この実装が夜までに見通せない場合も、学習モデルを必須候補にしない。
5. 00:30までに2foldの固定E12・所定検証が揃えば、01:00–02:00頃の手動提出を目指す。これは独自の作業目標で、Kaggleの別締切や採点時間の保証ではない。公式締切は10月23日08:59 JSTのままである。

人工入力によるGPU競合計測を実施した。[scriptと全結果](../artifacts/research/20261022-gpu-free-root-analysis-v1/synthetic-concurrency-v1/summary.json)は、手元の固定generic Tiny重み・元config・元constructorで、1processと2process、それぞれ有効92窓と144窓、warm-up3回＋測定24回のforward/backwardを記録する。実画像/ラベル・optimizer更新は0で、全parameterが計測前後で不変だった。各processのpeak予約VRAMは92窓946MiB、144窓1,062MiB。測定用processはすべて終了した。

| 人工入力 | 単独の中央値 | 2process時の各中央値 | 各processの遅延率 |
| --- | ---: | ---: | ---: |
| 92有効窓 | 0.08351秒 | 0.23706／0.23700秒 | 約2.84倍 |
| 144有効窓 | 0.12818秒 | 0.36700／0.36659秒 | 約2.86倍 |

GPU計算だけを連続実行した総throughputは2processの方が単独の約70%で、**GPU計算の並列化自体は有利ではなかった**。92窓は全キャッシュ平均約91.8窓を参考にした人工例で、実系列構成/画像/勾配分布を代表する保証はない。finite-check等の同期も含み、1系列のように詰めたmaskも臨床の系列分布とは異なる。これをそのままepoch遅延率にはしない。

旧runの1検査全体は約0.41–0.49秒で、人工GPU部分だけの数字より長い。二つのprocessがキャッシュ読み込み待ちを隠せる可能性は残るが、CPU/ディスク競合のため逆に遅くなる可能性もある。105.7GiBの画像キャッシュ読み込み・解凍・hash・検証処理は人工測定に含まれない。したがって上記7–10時間のシナリオを実証済み見込みへ昇格せず、実epochの完了見積もりを条件にする。GPUの稼働率やVRAM余裕だけから、2foldが同じ時間で終わるとは推定しない。

## 学習時の窓数半減が、今回それほど速くない理由

公開S6の作者はtrain64窓を使い、取得済みV5のtrainerもtrain時だけ候補から非復元抽出、evalは全候補を使う。[S6作者説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)。CoAtNet作者もtrain12窓・最終infer94窓を記載する。[CoAtNet作者説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)。この二つは、学習時のbagを軽くする仮説の参考になる。ただし元モデルや前処理が異なり、当方での増分を示すものではない。

当方のキャッシュは3系列×最大48窓だが、実際の有効窓は全4,207検査で386,196窓。各系列を最大24にすると294,684窓となり、残る比率は76.30%、削減は23.70%である。半分にはならない。fold0/1の学習窓削減は23.68%／23.57%、全窓を使う検証も含めた1epochの単純forward件数は約81%へ減るだけである。[maskだけの独立集計](../artifacts/research/20261022-gpu-free-alternatives-v1/window24-metadata-analysis.json)、[キャッシュsummaryによる一致確認](../artifacts/research/20261022-gpu-free-feasibility-v1/bag24-metadata-only.json)。

現在の読み込みは約105.7GiBの圧縮キャッシュから毎検査63.3MiBのuint8配列を展開する。maskだけ減らしても、圧縮member全体の読み込み・解凍は減らない。samplingには病変の局在を落とすリスクがあり、公開S6コード自身も均等間隔のcoverageとrandom phaseを用いる理由を記している。単純な窓削減を無害な高速化として追加する根拠は弱い。

途中調査で得たTejasの例も再点検した。同コードはtrain8窓をrandom抽出し、validationも8窓をlinspace抽出する。**train8/eval16の例ではなかった**ので、そのような根拠には使わない。今回の新規学習へsamplingとencoder変更を同時に入れず、CNNを試すなら元の全窓・固定設定を優先する。

## 特徴キャッシュと短時間学習の現実的な範囲

手元にあるfold別CLS＋16局所特徴の2cacheは、F005の旧BCE選択encoder E8/E9に由来する。各4,207検査・約7.92GBで、F006/F013の完成encoder特徴ではない。これを現在のF013 cacheとして使うことはできない。過去F007のhead学習12epochは約902–908秒、F008の24epochは約1,826秒だったが、それだけで今の親へ追加価値を出す新モデルは得られない。

F013完成encoderの最終CLS特徴だけを一度抽出するなら、padded最大144窓×384次元×FP16×4,207検査で約0.465GB。局所16分割込みなら約7.91GB/encoderである。対して最終2block直前の全token特徴は、実有効386,196窓でも約232.8GB、paddingを含めれば約365.2GBになる。さらにF013はpixel gammaをprefix前に使うので、一つの固定prefix cacheで元の学習と同条件を再現できない。

[公開two-stage実装](https://github.com/mmatmath/rsna-knee-two-stage)も、encoder特徴を保存して小さなsequence modelを学習する構造を示す。しかし配布済みの自己0.959モデルではない。F013ではfoldごとにそのencoderで学習側・検証側を抽出し、同じ除外foldを維持する必要がある。encoderが見た検証検査を、headを変えただけで独立検証にしてはいけない。

なお、当方の現在のheadは既に所見別slice attentionと所見別series attentionを持つ。「attentionを追加すれば改善する」という一般論は新しい案にならない。JEV教師への置換/補正もF011/F012で基準未達だった。GPUが空いたこと自体は、これらの負結果を覆す根拠にはならない。

## 採用はCNN単体の弱AUCだけで決めない

新しいCNNの固定E12ができても、それだけで提出しない。事前に固定する比較は、元のF006対照のencoder選択比較に加え、実際に差し替えるF013 CLS＋既存REGの家族50%との比較である。CNNは無gamma、F013はgammaなので、後者は最終model置換の比較であり、純粋なarchitecture効果とは呼ばない。

両foldの固定10所見差、全12所見の値と支持数、既存REGとの組合せ、既存809/855検査でのR6＋native10% proxyを確認する。新しいalphaや所見別maskを後から探して救済しない。新CNNが両foldで実際の家族対照を改善し、同じR6 proxyでも両bankを悪化させないことを、今回提案する最終候補化の条件とする。これは完全な親OOF・患者独立性・Public改善を証明する条件ではない。

係数・比較集合・時刻・工学条件は新しい結果を読む前に固定する。旧F014のfold0→fold1順序・旧監査scopeは保存し、今回の並列予算/最終置換判断は別のscopeへ記録する。自分のGold画像ラベルによる学習・checkpoint・配合選択は行わない。

## 残り5提出は「既存案＋学習予約1枠」へ更新

前回第1候補の**P048 native20→10%**は維持する。既存モデルを使う第2〜4候補の準備と正式比較を、学習結果待ちで止めない。前回P053案として予約したA224=.50の枠は、条件を満たしたCNN候補ができればそちらへ使う。CNNが未完成・検証負・serving未完なら、元の予約案を含む既存候補から判断する。

私の推奨は、**既存モデルの10%案を最優先で進め、学習を追加する場合はTiny一案の2foldに予算を絞る二本立て**である。実epochで00:30までの完了見込みとメモリ余裕を満たさなければ、CNNを最終提出の必須経路から外す。GPUが空いたので学習の選択肢は戻ったが、学習を終えることと0.959を達成することは別である。新学習によるPublic増分や達成確率を数値で約束できる根拠はない。正式0.959以上を確認した時点で目標達成を記録する。
