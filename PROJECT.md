# RSNA Knee プロジェクトの現在地

更新日 2026年10月4日 JST。目的は膝MRIの12所見を予測するKaggleコンペに参加し、採点に通る比較基準から段階的に改善することです。

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
- [r001提出資産](artifacts/kaggle/r001-bn-rank50-fold0/) に2重み・凍結zip・共通rank helper・hash/帰属記録・未実行Notebookを保存。確定weak診断と同じ関数で保存済み人工CPU/GPU出力を合わせ、提出契約とrank専用6チェックが成功した。[提出手順](docs/kaggle-submit.md)。固定BN単体の [e009 v2](artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) も用意した。元の資産を保持し、新規アップロード・Kaggle実行・採点は未実施。
- 今回の採用範囲・不確実性・次の順序と資産hashは [判断記録](experiments/e006-e009-decision-20261004.json)、検査結果は [完了時の検証記録](experiments/e006-e009-verification-20261004.json)。台帳のr001は追加学習ではなくCSV診断として記録した。
- 公開方式の固定契約を追加監査。DINOsaur V35の最終出力はV32の0.937候補と異なり、一部公開資産にはgold学習/選択と混在ライセンスがある。現在のgold除外方針へそのまま移植しない。[再現契約](docs/research/public-reproduction-contract-20261004.json)。重み・MRI・新ラベルの取得や外部書き込みは行っていない。

## 次にすること

1. 固定50:50 rank候補をKaggle Internet OFFで手動実行・採点し、自作weak評価と画像由来の採点がどれだけ対応するかを先に測る。[提出手順](docs/kaggle-submit.md)。実測Public 0.924の既存提出を保持し、Input/Notebook版・全体時間・採点値を別記録する。rank候補のBCEは未校正順位値の補助計算で、確率校正の改善とは扱わない。
2. 通常BN／固定BNの20 epochをfold 1でも同じ条件から対照実行し、固定50:50も再確認する。fold 0だけでBNやensembleの全面採用を決めず、goldを選択に使わない。旧fold 1の5 epochを再開せず新規runとし、最初5 epochの一致も確認する。先に追加学習時間だけを50/100 epochへ増やさない。
3. 192pxの全体像と140mm物理cropだけを変える入力比較を、Kaggleの学習側24〜32件QCから準備する。[入力・容量・互換性の契約](docs/research/physical-crop-contract-20261004.json)。PixelSpacingのない既存cacheから物理cropを後付けせず、元DICOMから別cacheを作る。旧前処理hashを維持する版設計を先に行う。
4. DINOsaur V4の作者報告Best 0.937はV32。SDKの版指定を修正してV32ソースを取得・hash固定できたが、保存Input版と採点された出力CSVは未確認。[追補](docs/research/public-reproduction-v32-followup-20261004.json)。CoAtNetの140mm crop・系列選択等を参考にするが、公開Training V8の未配布softラベルとgold選択をそのまま移植しない。新たな学習比較は汎用事前学習重みから行う。
5. 次は物理cropを同じ解像度で比較し、解像度・系列/window密度・scheduler・ラベルを一要因ずつ検討する。入力試作はKaggleで少数検査から始め、256px以上の全件化は容量試作とUID分割・統合検査を先に整える。学習seed比較は分割seedとの分離後に行う。10月20〜22日は採点成功済み候補の再実行・最終選択に充てる。

10月4日の添付分析の再監査は、続行依頼前の調査履歴である。HEAD 05b0710と保存済みweak評価を照合し、C2−B1の補助11所見差+0.120629の95%区間[+0.078707,+0.162513]、C5−C2差+0.010232の区間[−0.005521,+0.025498]を計算した。後者を追加実験の根拠とし、その後に上記4本の実学習へ進んだ。公式Hostは画像由来の正解とReportの不一致を認めており、weak AUCとKaggle採点を区別する。[当時の調査記録](docs/research/strategy-audit-20261004.json)。

ラベル利用条件の本文確認は完了。研究・学習目的と帰属等を維持し、出所・版・変更点を提出Inputにも記録する。現在の目的で作者への追加許可を新たな必須条件にしない。実際に入賞した場合はWinner公開条文の不整合を確認し、商用化へ目的を変える場合は利用条件を再検討する。

初回提出の採点成功と実測PublicはAPIで確認済み。My Submissionsと保存NotebookでInput版・実行時間を補完する。以後のprepare・trainには `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` を使用する。

キャッシュ作成後の既存コマンドは [after-cache.md](docs/after-cache.md)、添付分析を照合した判断・比較設計・容量条件は [次の実験計画](docs/research/next-experiments-20261004.md) を使う。10月4日の再分析は調査のみで、その後の依頼を受けて上記の実装・診断へ進んだ。現行全件Notebookの18GB容量ガードでは256px以上が事前停止する。224pxのraw画像量は約15.92GB、288pxは約26.32GBで、実測192px圧縮export約7.79GBとは区別する。

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
