# RSNA Knee プロジェクトの現在地

更新日 2026年10月6日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、**採点済みp002のPublic 0.937を保持し、公開0.943を設計上の基準に、その先の精度を目指す**ことです。

## 最新：p003の例外確認と速度・独自学習の改善

提出`56855132`（scriptVersionId `355483987`）は自己Public未測定で、公式SDKとユーザー画面の双方で隠し再実行の未処理例外を確認しました。**9時間超過とは未確定**です。SDKのCOMPLETEは処理終了を表し、errorDescriptionがある今回は採点成功を意味しません。保存版1の全セルは配布02と一致し、可視3検査のP003_READYはpassed・332.006秒。隠しtracebackと実時間は取得できていません。[失敗監査](experiments/p003-failure-review-20261006.json)

次は新しい[05・64検査診断](notebooks/public/05_profile_p003_speed_64.ipynb)です。Raptor GPU1の二つの入力方式を検査ごとに続けて処理してdecode cacheを再利用し、元経路の入力tensor・生予測との一致、GPUメモリ、処理時間を記録します。[提出候補05](notebooks/public/05_submit_p003_speed.ipynb)は別Notebookです。全モデル・係数・精度・失敗拒否・8時間内部予算を保持します。2モデル常駐のT4メモリと実効速度は未検証で、これだけで隠し例外を解消したとは扱いません。

精度改善は[凍結DINOv2特徴の学習経路](docs/frozen-features.md)を追加しました。Kaggleの元DICOMから392px・最大3系列×48枚の特徴を作り、4090で平均集約f001とAttention集約f002を一要因比較します。汎用encoderの出所・hash、特徴と前処理の共通契約、固定fold、gold非使用、欠損mask、CUDA必須、Report不要推論を維持します。新しい[ラベル監査キュー](docs/label-review.md)ではfold0/1とgold連結groupを保護して学習側の根拠を確認します。

今回は実装・一次資料調査・人工データの検査を実施し、実データ/モデルの追加取得、実MRI処理、実学習、外部アップロード、代理提出は実施していません。0.943は作者報告、0.937は自己実測です。[調査・比較順序・採否条件](docs/research/p003-improvement-20261006.md)。以下の「次は旧02提出」という記述は10月5日時点の履歴です。

## 最新：0.937の確定と次候補p003

23:37 JST追記：ユーザーが64検査の時間診断をKaggleで実行し、提供された`P003_PROFILE.json`は`profile_complete_not_for_submission`でした。T4×2、64検査・366系列、事前検査込み1,427.763秒（23分47.8秒）。DINO20の共有経路比較は全て差0、追加CoAt4系統の完了記録があり、元ソース／派生セルhashと実行順も一致しました。次は**同じ14 Inputs・T4×2・Internet OFFで提出用p003を別の新しいNotebookとして保存実行**し、P003_READY確認後に手動提出へ進めます。A5精度診断は先行必須ではありません。[今回の判断記録](experiments/p003-profile-review-20261005.json)。隠しtestの時間・Publicは未測定で、個別receiptや予測実体はこの添付から再検査していません。

Input追加で名前検索に出ない場合のため、[最新手順](docs/p003-next-step.md#add-inputの検索で見つからない場合)に基本URL貼り付けとDataset／Notebook Output／Model／Competitionの区別を追記しました。配布NotebookのImportだけでInputsが自動追加されるわけではありません。

公式SDKでp002の提出ref 56840796を **COMPLETE / Public 0.937** と確認しました。保存版1（scriptVersionId 355324155）の16セルは前回handoffと一致し、可視実行の全receiptもpassedです。約8時間はユーザー報告の経過時間であり、可視3検査の164.699秒や隠しtest正味時間とは区別します。[採点監査](experiments/p002-scored-20261005.json)、[固定成功版](notebooks/public/01_submit_p002_scored.ipynb)。

次候補は **haideptry V2のp003**。作者Public 0.943を確認した指定版ソースを取得し、実効14 Inputs、追加CoAtNet 4系統、確率平均後の全体順位化、失敗時の提出拒否を固定しました。[提出Notebook](notebooks/public/02_submit_p003.ipynb)、[実行手順・分析・0.955へ向けた順序](docs/p003-next-step.md)、[Input契約](docs/research/p003-input-contract-20261005.json)。自己Publicは未測定で、0.945到達を保証するものではありません。

先に[64検査の全体時間診断](notebooks/public/04_profile_p003_64.ipynb)で48件超の逐次CoAt分岐を確認し、その後p003を別のprivate Notebookで保存実行・手動提出します。[A5 BF16/FP16/FP32診断](notebooks/public/03_a5_precision_probe.ipynb)も用意しました。いずれもT4 x2・Internet OFFでユーザーが実行します。新候補の実MRI処理・重み取得・学習・アップロード・代理提出は今回行っていません。自作ResNet18の長期化は優先しません。

追加公開CoAtには作者のgold58によるcheckpoint選択履歴があります。凍結済み競技用モデルとして比較し、自分のgold利用や独立OOFとは分けます。0.955に向けた残差集約の新規学習は、特徴cacheと元学習除外UIDの対応を整えてから別実験にします。以降の0.924基準・p002未採点記述は、今回の更新前の履歴です。

19:29 JST追記：ユーザーがp002を提出し、約7時間後もNotebook runningと報告。公式SDKで提出ref 56840796、12:07:34 JST、`submission.csv`、status=PENDING、Public未確定を確認しました。経過約7時間22分は提出からの壁時計時間で、実際の隠しtest実行時間や進捗はAPIから取得できません。今回の多モデル構成で数時間かかることはあり得ますが、正常進行や9時間以内の完走を保証しません。現提出を維持し、結果確定後に採点と時間を記録します。[状態確認](artifacts/research/p002-status-20261005T102826Z/submission-detail.json)。下記の準備完了時点の未提出記述は履歴です。

## 今回の方針変更

**開発の主軸を公開学習済みモデルへ移します。** r001のPublic 0.749を公式APIで確認し、0.924の成功版ソースも保存しました。自作ResNet18は補完効果を調べる副系統として保持し、BNの別fold追試・50/100 epoch化・全fold展開・192px物理crop全件化を先に進めません。[提出監査](experiments/submission-audit-20261005.json)、[段階別計画と採否条件](docs/research/public-model-strategy-20261005.md)。

最初の新候補p002はDINOsaur V4のV32です。**第1段階の準備を完了し、ユーザーが第2段階のKaggle保存実行・手動提出へ進める状態にしました。** [提出用Notebook](artifacts/kaggle/p002-dinosaur-v32-handoff-v2/02_submit_p002.ipynb)を新しいprivate NotebookへImportし、[操作手順](docs/public-baselines.md#ユーザーが行う操作)に従って指定5 Inputs・T4 x2・Internet OFFでSave & Run Allします。実行成功と`P002_READY.json`を確認して最終`submission.csv`を手動Submitします。実MRIでの互換性・完走・本番所要時間・Publicは未確認です。

元19参照を有効5 Inputsへ整理し、公開資産47ファイルの期待サイズ・SHA-256を固定しました。元10セルの推論ソース・前処理・係数を保持し、ファイル検査・各モデルの完走・fallback拒否・最終CSV検査を追加しています。重みpayloadのhashはKaggleで実行時に照合します。[Input対応表](docs/research/p002-effective-inputs-20261005.json)、[引渡し・検査記録](experiments/p002-handoff-20261005.json)。歴史的Input版と作者の採点CSVは未確認で、今回固定した組合せを作者0.937の再現済みモデルとは扱いません。公開0.940/0.941系は保存コード等の不一致があり、参考候補に留めます。[候補選定](docs/research/public-candidate-review-20261005.json)。

公開重みのgold履歴は競技用候補の監査として記録し、自分の学習・checkpoint選択・係数調整にgoldを使わない方針は維持します。公開モデルの学習に露出した検査を後から分割しても独立OOFにはなりません。今回は調査・未実行Notebookの準備・人工データでの検査・手順更新を実施しました。実データ/モデル取得・実MRI処理・追加学習・アップロード・代理提出は行っていません。以下の10月4日までの未測定・次工程の記述は当時の履歴です。

## 現在の状態

- GitHubの既存リポジトリに、実データなしで開発できる構成を追加。
- 調査の正本は [docs/competition.md](docs/competition.md)。原文は [ユーザー提供分析](docs/research/user-analysis-20261002.txt)。
- 今後の判断と三週間の計画は [docs/roadmap.md](docs/roadmap.md)。ユーザーはKaggleでキャッシュを作成し、ローカルGPUで学習する経路を選択済み。
- 初回の既存提出を10月3日に公式APIで確認。status=COMPLETE、実測Public 0.924、提出日時は10月2日05:57:22 UTC。提出URLのscriptVersionIdは354569007、該当Notebookの現在のVersionは1。保存Input版と実行時間、global IDと表示版の直接照合は未確認。自作ResNetのスコアとは別に記録した。
- 全件画像キャッシュの転送・検査後、研究用weakラベルの取得・監査・固定分割を実施。1 epochの動作確認と5 epochの基準実験が別runで完了し、自作重みとコードをローカルで梱包した。
- 10月3日、ユーザーの依頼でprivate Notebook `rsraki/rsna-knee` のVersion 1（ユーザー提示のscriptVersionIdは354838181）へAPIでアクセス。全件exportのcomplete=true、4,407検査、4,423ファイル、7,785,521,448 bytes、現在のsrcとのhash一致を確認した。
- 大きいZIPを避ける [download_cache_output.py](scripts/download_cache_output.py) を追加。取得先は `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/`。版確認、4並列、ファイル単位のhash検証・再開、取得後のexport検査に対応する。
- 全4,423ファイルを取得し、4,407検査のexport検査がvalid=trueで成功。サイズ・hash・UID・元CSV・src・前処理が一致。取得記録は親ディレクトリのtransfer.jsonでtransfer_complete=true。元DICOM・公開モデルは取得していない。
- 自作コードの検証範囲は [validation.md](docs/validation.md) に記録。
- ベースライン設定は [configs/baseline.json](configs/baseline.json)。学習コマンドはCUDAがなければ停止する。
- 現在の端末はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブの空き約485GBに対して公式元画像は569.76GBのため、全取得しない。
- Python 3.12.13の専用 `.venv` にtorch 2.10.0+cu128、torchvision 0.25.0+cu128が導入され、CUDA利用可・RTX 4090認識を10月3日に確認。学習環境とは別の `artifacts/tools/kaggle-venv/` にKaggle CLI 2.2.4を導入。CSV監査ツールの新規12件を含むunittest 31件が成功。
- [00_prepare_cache.ipynb](notebooks/00_prepare_cache.ipynb) と転送確認用 [verify_cache_export.py](scripts/verify_cache_export.py) を追加。Notebookは未実行テンプレート。
- 両Notebookはコードzipと展開済みInputの両形式に対応。Kaggleでの全件キャッシュ作成と転送、CUDA版PyTorch導入、ローカルの初回実学習は完了。
- coverageでは4,407検査すべて24 window、選択された13,221シリーズすべてphysical_position順、読み取りエラー・fallbackは0。ユーザーがKaggleで元画像とcacheを目視比較し、確認した範囲では問題なしと回答。比較した件数・UIDは未記録。
- 人工ノイズ画像でAMP・backward・optimizer・checkpoint保存・Reportなしの提出CSVまでGPUパイプラインを確認。これは精度や実データの所要時間の検証ではない。
- 公開ラベル7候補を監査し、vmohitrao Dataset Version 3を研究用に採用。10月4日に公式RulesとHost回答を取得し、研究・学習目的とCC BY-NC 4.0の帰属等を守って大会用途へ採用を進める判断に更新した。外部LLM抽出は条件付き許可、NC制限だけで大会側が禁止するものではない。根拠は [利用条件監査](docs/research/kaggle-source-eligibility-20261004.json)。gold非使用は作者READMEの宣言に基づき、非公開の開発ログを独立監査した事実ではない。
- `data/labels/vmohitrao-v3/` と `data/manifests/v1-vmohitrao-research/` にラベル・出所・groups・固定分割を保存。元train hashと4つのCSV hash、全state/value/maskが一致。weak 4,207件、gold 58件、全欠損138件とgold連結4件を除外。5-foldは821・891・888・778・829件。患者独立性は未確認。
- e001-smoke-fold0が完了。学習3,386／検証821件、weak検証BCE 0.430481、選択後のgold 12クラスmacro AUC 0.487178、epoch時間205.33秒。checkpointはweak BCEで選択。これはPublic LBではない。
- e002-baseline-fold0の5 epoch基準実験が完了。変更はepoch数1→5だけであり、再開せず同じseed・foldで初期化から実行。最良はepoch 1で、weak検証BCE 0.430481、選択後のgold macro AUC 0.487178。事前検査と最後のgold評価を含むtrain関数の所要時間は1,082.83秒（約18分、Python importを除く）。epoch数を増やしてもweak検証は改善せず、現行ランダム初期化モデルは実装上の比較基準として保存する。
- `artifacts/kaggle/e002-baseline-fold0/` にbest.pt、コードzip、hash manifest、RESEARCH-ONLY.jsonを保存。展開したzipのsrcが学習時のhashと一致し、実checkpointを使った人工test 3検査×12所見の提出契約がvalid=true。Reportを使わず予測できる。実testのDICOM decode、Kaggle全体時間、新規アップロード・提出は未実施。
- 提出の優先順位を判断する追加CSV診断を実施。学習側の所見別陽性率だけを全検査へ出す定数予測は、同じ検証821件・観測5,727セルのBCEが0.415327で、自作モデルの0.430481より低い。一方、自作モデルのweak macro AUCは0.557023、定数予測は0.5であり、画像の順位付け信号が全くないとは断定しない。e001/e002のweak・gold予測CSVはそれぞれhashが完全一致。集計は [提出判断の追加診断](experiments/e002-submission-review-20261003.json)。
- 10月4日、追加ログの学習不変性、cache互換性、共通正規化、ImageNet初期化、BN統計固定を実装。全71 unittestと人工CUDA検証が成功。少数train診断q001は初期eval BCE 0.694520→最良0.124106（epoch 32）、最後のtrain-mode BCE 0.000188。適合は進むがevalとの差が残り、画像ラベルの正確さや汎化の証明ではない。
- A/B/Cの5 epoch対照実験が完了。Aは旧e002の全epoch BCE・採用予測を完全再現。正規化だけを変えたBはweak BCE 0.411992／AUC 0.644652、encoder初期化だけを変えたCはepoch 2で0.371395／0.766895。CはBより12/12所見のAUCが改善し、次のローカル比較基準に暫定採用した。各約17〜18分、gold監査は無効・Public未測定。[比較集計](experiments/e003-e005-controlled-summary-20261004.json)、[採用理由](experiments/e005-decision-20261004.json)。
- Cの `artifacts/kaggle/e005-pretrained-imagenet-fold0/` にcode zip・best.pt・hash/帰属記録・未実行提出Notebookを準備済み。凍結コード＋実A/B/C重みで人工3検査×12所見・Reportなしの契約が成功し、CのImageNet初期化ファイル不要も確認。Kaggleへの新規アップロード・実test decode・採点は未実施。
- 続行の依頼を受け、BCE/AUC/last/milestone保存、実optimizer更新・AMP skip、進捗記録を実装。全97 unittest・Ruff・凍結コードの人工CPU/GPU提出契約が成功。e006〜e009の4本を新規runで完了し、CSV・入力/保存hash・選択規則・実更新の監査がvalid=true。[完成集計](experiments/e006-e009-extended-summary-20261004.json)。実checkpoint内部も別途CPUで照合した。合計約2時間47分（cache事前検査込み、import除く）。モデル保存は推論専用で、正確な学習再開は未対応。
- fold 1でもB→CのAUCは0.638538→0.732050、10/12所見が改善。通常BNの20 epochはAUC最良epoch 8で0.784410、BN固定はepoch 9で0.803614。ただしBNの補助11平均差の95%区間は0を跨ぎ、ACL・PF OA・Lateral Meniscusの大きな悪化とMedial Meniscus等の改善が入れ替わる。[対照実験と判断範囲](docs/controlled-experiments.md)。通常BN e008の最初5 epochの旧C予測は全hash完全一致。両20 epochの終点は最良値を下回り、単純な長期化の継続を優先しない。
- 上記の所見別の入れ替わりを見た後、2モデルの固定50:50 rank平均を一度だけCSV診断。weak AUC 0.830249、補助11平均0.815727で両単体を上回った。[固定比率診断](experiments/e008-e009-rank50-review-20261004.json)。BN単体との差の補助11区間は[+0.015129,+0.043443]だが、同じfoldの選択済みモデルによる後付け診断で、別fold・画像ラベル・Publicの改善は未確認。係数/所見weightの探索とgold利用はしていない。
- [r001提出資産](artifacts/kaggle/r001-bn-rank50-fold0/) に2重み・凍結zip・共通rank helper・hash/帰属記録・未実行Notebookを保存。確定weak診断と同じ関数で保存済み人工CPU/GPU出力を合わせ、提出契約とrank専用6チェックが成功した。[提出手順](docs/kaggle-submit.md)。固定BN単体の [e009 v2](artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) も用意した。ユーザーがInputをアップロードし保存実行したが、rank helperの参照先が違って最初のcellで停止。自作Publicはまだ未測定。
- 再依頼で保存済み `rsraki/rsna-knee-r001-rank50` Version 2と現在のコードInputをread-only確認。コードInputの15ファイルは全hashが正しいが、NotebookがInput直下のrank50.pyを展開source内へ探していた。[原因・修正記録](experiments/r001-kaggle-pathfix-20261004.json)。[修正版Notebook](artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) を準備し、実Input配置を写した人工環境で元の失敗を再現、修正後の初め2 cell・CPU checkpoint内部検査が成功。最小修正はRANK_MODULEの1行で、ZIP/重みの再アップロードや再学習は不要。
- ユーザーが保存実行したVersion 3は公式APIでCOMPLETE、GPU有効・Internet OFF。提示ログは表示用3検査の前処理警告0、両GPU推論と最終3×12提出CSVがvalid=trueで、source/重み/helper/fingerprintも候補資産と一致。Outputにsubmission.csv（852 bytes）を確認した。cell内計測47.95秒、提示ログ最終時刻68.2秒。本番の隠しtest採点は未実施で、自作Publicは未測定。[Version 3確認記録](experiments/r001-kaggle-v3-review-20261004.json)。次は変更せずVersion 3のOutputから最終CSVを手動Submitする。
- 今回の採用範囲・不確実性・次の順序と資産hashは [判断記録](experiments/e006-e009-decision-20261004.json)、検査結果は [完了時の検証記録](experiments/e006-e009-verification-20261004.json)。台帳のr001は追加学習ではなくCSV診断として記録した。
- 公開方式の固定契約を追加監査。DINOsaur V35の最終出力はV32の0.937候補と異なり、一部公開資産にはgold学習/選択と混在ライセンスがある。現在のgold除外方針へそのまま移植しない。[再現契約](docs/research/public-reproduction-contract-20261004.json)。重み・MRI・新ラベルの取得や外部書き込みは行っていない。

## 次にすること

1. p002 Public 0.937と元p003を保持し、新05診断を同じ14 Inputs・T4×2・Internet OFFで実行する。数値一致、2モデル常駐メモリ、失敗traceback、phase時間を確認する。旧04の再実行を指すものではない。
2. 予測一致と速度・メモリを確認できた05だけを新規保存実行・手動提出する。一般的な例外を時間超過と決めつけず、モデル省略やfallback許容で完了扱いにしない。
3. 06で汎用DINOv2・392px特徴の小規模品質確認を行う。元コード/重みを手動で用意し出所とhashを固定する。確認後にweak全件特徴を作り、ローカルでexportを検証する。
4. f001 meanとf002 attentionを同一特徴・教師・seed・fold0で学習し、BCE選択・所見別AUC/supportを記録する。別runでfold1も比較する。実学習はまだ開始していない。
5. 教師監査は事前にfold0/1を保護した150検査キューから始める。規則を固定した別版教師との比較を、集約やencoder変更と混ぜない。
6. 新モデルの検証が成立してから、p003との補完性・事前固定した5%/10%統合候補・全体時間を比較する。Gold58や公開重みへの後付けfoldを独立評価とは呼ばない。

10月4日の添付分析の再監査は、続行依頼前の調査履歴である。HEAD 05b0710と保存済みweak評価を照合し、C2−B1の補助11所見差+0.120629の95%区間[+0.078707,+0.162513]、C5−C2差+0.010232の区間[−0.005521,+0.025498]を計算した。後者を追加実験の根拠とし、その後に上記4本の実学習へ進んだ。公式Hostは画像由来の正解とReportの不一致を認めており、weak AUCとKaggle採点を区別する。[当時の調査記録](docs/research/strategy-audit-20261004.json)。

ラベル利用条件の本文確認は完了。研究・学習目的と帰属等を維持し、出所・版・変更点を提出Inputにも記録する。現在の目的で作者への追加許可を新たな必須条件にしない。実際に入賞した場合はWinner公開条文の不整合を確認し、商用化へ目的を変える場合は利用条件を再検討する。

初回提出の採点成功と実測PublicはAPIで確認済み。My Submissionsと保存NotebookでInput版・実行時間を補完する。以後のprepare・trainには `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` を使用する。

キャッシュ作成後の既存コマンドは [after-cache.md](docs/after-cache.md)、現在の判断・比較設計は [10月5日の計画](docs/research/public-model-strategy-20261005.md) を使う。[10月4日の実験計画](docs/research/next-experiments-20261004.md) は履歴として保持する。現行全件Notebookの18GB容量ガードでは256px以上が事前停止する。224pxのraw画像量は約15.92GB、288pxは約26.32GBで、実測192px圧縮export約7.79GBとは区別する。

goldの結果は既に見ているため完全未観測の最終holdoutとは呼ばず、学習・checkpoint選択・prompt調整・ensemble係数合わせには使わない。10月20〜22日は最終候補の再実行・選択に充てる。

## 次へ進む条件

- 初回提出：Notebook版、入力資産の版、採点成功、スコア、実行時間を台帳に記録。
- データ準備：件数、ラベル欠損、シリーズ失敗、前処理画像の目視、使用ラベルの出所が確認済み。
- 自作モデル：学習・weak検証・gold検証の役割を分け、goldをcheckpoint選択に使わず、漏洩監査済み。
- 改善：同じfoldとseedで一要因ずつ変更。12クラス全体と弱いクラス、速度を合わせて判断。
- 最終候補：Kaggleの隠しテストで採点成功し、十分な時間余裕を持つ。未採点の最終変更に依存しない。

## 作業場所

Kaggleは元MRIの前処理と最終提出、このPCはキャッシュからのGPU学習に使う。ユーザーの「進められるところまで進める」依頼と目視確認を受け、準備から実データでのローカル研究へ移行した。Gitへ戻すのはコード、設定、集計と実験記録のみ。レポート、Study UID一覧、分割CSV、画像キャッシュ、重みは `data/` と `artifacts/` 以下へ保存する。

`pj-kaggriculture/PROJECT.md` と `ptcc_pokemon_ai_buttle/experiments` の、実装・実験・生成物を分ける運用を参考にした。ゲーム用の提出形式や対戦評価はMRIコンペへ持ち込んでいない。
