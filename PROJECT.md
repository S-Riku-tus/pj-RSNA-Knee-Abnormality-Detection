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
- 公開ラベル7候補を監査し、vmohitrao Dataset Version 3を非商用の研究用候補として取り込んだ。CC BY-NC 4.0であり、Kaggle提出への採用可否は未確認。gold非使用は作者READMEの宣言に基づき、非公開の開発ログを独立監査した事実ではない。
- `data/labels/vmohitrao-v3/` と `data/manifests/v1-vmohitrao-research/` にラベル・出所・groups・固定分割を保存。元train hashと4つのCSV hash、全state/value/maskが一致。weak 4,207件、gold 58件、全欠損138件とgold連結4件を除外。5-foldは821・891・888・778・829件。患者独立性は未確認。
- e001-smoke-fold0が完了。学習3,386／検証821件、weak検証BCE 0.430481、選択後のgold 12クラスmacro AUC 0.487178、epoch時間205.33秒。checkpointはweak BCEで選択。これはPublic LBではない。
- e002-baseline-fold0の5 epoch基準実験が完了。変更はepoch数1→5だけであり、再開せず同じseed・foldで初期化から実行。最良はepoch 1で、weak検証BCE 0.430481、選択後のgold macro AUC 0.487178。事前検査と最後のgold評価を含むtrain関数の所要時間は1,082.83秒（約18分、Python importを除く）。epoch数を増やしてもweak検証は改善せず、現行ランダム初期化モデルは実装上の比較基準として保存する。
- `artifacts/kaggle/e002-baseline-fold0/` にbest.pt、コードzip、hash manifest、RESEARCH-ONLY.jsonを保存。展開したzipのsrcが学習時のhashと一致し、実checkpointを使った人工test 3検査×12所見の提出契約がvalid=true。Reportを使わず予測できる。実testのDICOM decode、Kaggle全体時間、新規アップロード・提出は未実施。
- 提出の優先順位を判断する追加CSV診断を実施。学習側の所見別陽性率だけを全検査へ出す定数予測は、同じ検証821件・観測5,727セルのBCEが0.415327で、自作モデルの0.430481より低い。一方、自作モデルのweak macro AUCは0.557023、定数予測は0.5であり、画像の順位付け信号が全くないとは断定しない。e001/e002のweak・gold予測CSVはそれぞれhashが完全一致。集計は [提出判断の追加診断](experiments/e002-submission-review-20261003.json)。

## 次にすること

1. epochごとのweak予測・所見別AUC・観測数・分布と、同じ集計方法のeval lossを記録できるようにする。追加評価がDataLoaderの乱数やBN状態を変えないことを人工データで検査する。exportの完全性と現在の画像前処理互換性を分けて検査し、コード変更時もhashの保護を維持する。
2. 人工入力の契約と少数の学習側検査への適合を確認後、同じ192pxキャッシュ・fold・seed・5 epochでA（random＋現行正規化）、B（random＋ImageNet正規化）、C（事前学習＋Bと同じ正規化）を比較する。weak BCEのcheckpoint選択を固定する。必要ならCからBN running statistics固定だけを変更し、affine学習条件はCと同じにする。batch変更はlossの検査重みをそろえてから比較する。
3. ラベルのCC BY-NC 4.0、Kaggle Rules、外部LLM利用、成果物公開の条件を並行確認する。RulesとHostの外部LLM案内本文は取得できておらず、許可・禁止は確定しない。条件が合わなければ、gold開発非使用と大会利用の条件を確認できる別のラベル版で新しい比較基準を作る。
4. 利用条件を確認した自作モデルで、早い段階にInternet OFFのKaggle実行・採点・全体時間を確認する。大規模な調整が終わるまで延期しない。改善がなくても、利用可能な現行モデルの一回の提出は提出経路の確認として価値がある。既存bundleは研究用として保持し、未確認を許可済みとしてアップロード・提出しない。
5. 診断に応じてラベル版か画像入力を一条件ずつ改善する。画像は224px全体像、その同解像度の物理cropの順。256px以上の全件化は容量試作とUID分割・統合検査が先に必要。有望な条件だけ追加fold・seedで確認し、Public 0.924の既存提出を候補として保持する。

初回提出の採点成功と実測PublicはAPIで確認済み。My Submissionsと保存NotebookでInput版・実行時間を補完する。以後のprepare・trainには `data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/` を使用する。

キャッシュ作成後の既存コマンドは [after-cache.md](docs/after-cache.md)、添付分析を照合した最新の判断・比較設計・容量条件は [次の実験計画](docs/research/next-experiments-20261004.md) を使う。10月4日の作業は調査と計画の更新のみで、上記の実装・学習は未実施。現行全件Notebookの18GB容量ガードでは256px以上が事前停止する。224pxのraw画像量は約15.92GB、288pxは約26.32GBで、実測192px圧縮export約7.79GBとは区別する。

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
