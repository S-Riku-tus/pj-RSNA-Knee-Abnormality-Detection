# 準備内容の検証範囲

更新日 2026年10月4日 JST。10月3日までに全件キャッシュ転送、研究用weakラベルの監査・分割とローカル実学習を実施した。10月4日は添付分析と実装・一次資料の照合、次工程の計画更新のみ。以下では各日の確認範囲を分ける。

## 添付分析の照合と次工程の調査（10月4日）

- 添付原文を [user-analysis-20261004.txt](research/user-analysis-20261004.txt) へ保存。SHA-256は `ee844f85bf1281db9d8e3a1a1f0261249ea887414c472dfd97a60e927fcb9f97`。判断は [次の実験計画](research/next-experiments-20261004.md)、出典・既存記録hash・容量計算は [調査記録](research/reanalysis-20261004.json)。
- 今回は実データ・重み取得、実MRIデコード、GPU推論、追加学習、Kaggle実行・アップロード・提出を行っていない。src・config・Notebook・script・test・実験台帳に変更はない。新しい学習結果を記録していない。
- runtime/model/imaging/contractsと現行設定を照合。所見別Attentionは既存、ResNet18はrandom、画像正規化は[-1,1]。batch 1でもencoderは同じ検査の24 windowを処理する。batch変更はBNだけでなく欠損mask付きlossの検査寄与も変える。学習と検証のloss集計方法も異なり、train-valid差を純粋な過学習量と解釈しない。
- installed PyTorch 2.10のDataLoader/RandomSamplerソースで、shuffle=Falseでもiterator生成がGeneratorを消費することを確認。train/valid共有Generatorに追加評価を入れると次epochの順序が変わり得る。専用評価GeneratorとRNG・module状態の復元、人工データでの学習不変性の確認は次工程の設計であり、まだ実装していない。
- export検査器が現在のsrc全体のhash一致を要求し、画像fingerprintがimaging.pyとpreprocess設定から決まることを確認。画像を変えないログ・モデル変更でのcache再利用には検査の分離が必要。既存hash保護を緩めていない。
- Notebook generatorの未圧縮容量式を再計算。4,407検査×3系列×8 window×3 channelで192/224/256/288/336pxはそれぞれ11,697,094,656／15,921,045,504／20,794,834,944／26,318,462,976／35,822,352,384 bytes。現行18,000,000,000 bytes未満の事前ガードは256px以上を止める。この値はrepoの保守的予算で、現在のKaggle公式枠や最大RAM使用量ではない。LIMITはprefix制限で全件shard処理ではない。
- Torchvision 0.25とPyTorch 2.10の公式資料、作者のsoft/hard比較、EfficientNet入力比較、既存CoAtNet model card、別Notebookのスコア表示、2024腰椎優勝者の記述を照合した。作者報告を自分の再現値と区別し、別条件の改善幅を現在の期待値へ転用しない。別NotebookのPublic 0.942は表示確認のみで、内部処理・Input・重みの来歴は未監査。
- 公式Overviewの索引本文で12所見平均AUC、Internet OFF・9時間、期限を再確認。最終期限は10月23日08:59 JST。CC公式資料とHostの別データへの回答から、提供元の条件と大会の条件を分けて確認する計画とした。Rules・外部LLM案内本文は取得できず、採用ラベルの許可・禁止は未確定。外部への問い合わせは送っていない。
- goldは既に監査値を見ているため完全未観測の最終holdoutとは呼ばない。学習・checkpoint選択・prompt・ensemble係数の調整には使わない。公開重みを後付けfoldで評価しても独立OOFとは主張しない。
- 調査JSONのparse、添付原文・保存コピー・既存2記録のSHA-256、全解像度の容量式、Markdown 11ファイルのローカルリンク、差分の空白を確認。実装・台帳に差分がないことも確認した。コード変更がないため31 unittestやGPU契約を再実行しておらず、10月3日の成功結果と今回の文書検査を区別する。Gitで追跡されるdata/artifacts配下は従来のREADMEだけ。

## e002の提出判断のための追加診断（10月3日）

- 今回の質問には保存済みCSVの再集計で対応し、GPU推論・実MRIデコード・追加学習・モデル取得・アップロード・提出は行っていない。学習や推論コードも変更していない。
- fold 0の学習3,386件で観測された所見別陽性率を定数予測として計算。検証やgoldから定数をfitしていない。同じ検証821件・観測5,727セルに欠損maskを適用したBCEは0.415326862。e002の保存予測からのBCEは0.430481425で、記録されたtorch集計値との差は1.2e-9未満。定数予測のBCEが低いことから、BCE単独を画像学習の有効性の証拠とはしない。
- 観測されたbinary weakラベルのmacro AUCは自作モデル0.557022616、定数予測0.5。BCEとAUCは異なる性質を測るため、「画像信号を全く学習していない」とは断定しない。特にSynovitisは陽性100・陰性1であり、weak macro AUCも不安定な指標を含む。gold 58件のAUCを隠しtestの予測値へ読み替えない。
- e001/e002のweak予測CSV同士、gold予測CSV同士のSHA-256がそれぞれ一致。5 epoch実験から提出するbest.ptはepoch 1が選ばれており、新しい予測改善はない。
- 識別子やレポートを含まない集計・元ファイルhash・計算方法・判断は `experiments/e002-submission-review-20261003.json`。学習側だけの陽性率計算、UID集合一致、BCEの再現、定数予測AUC=0.5をassertで確認した。
- 公式Overviewの検索索引で評価が12所見平均ROC-AUC、Internet OFF・9時間制限を再確認。Rules本文と採用ラベルのページ本文は今回のWeb取得でも開けず、大会利用の許可・禁止を確定していない。汎用事前学習候補について [Torchvision ResNet18の公式資料](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) を確認。実装・重み取得・効果検証は次工程。
- 診断JSONの構造と比較結果、ローカルリンク49件、差分の空白を確認。学習・推論・Notebook・設定・既存チェックの実装には変更がなく、全31 unittestは前回の成功結果を維持している。今回のCSV診断には上記assertを使用し、GPUテストは再実行していない。

## ラベル監査・固定分割・ローカル研究（10月3日）

- 認証済み公式APIから自分の提出履歴を読み、既存の1提出がCOMPLETE・Public 0.924と確認。提出日時は10月2日05:57:22 UTC、提出URLのscriptVersionId=354569007。該当Notebookの最新版metadataはVersion 1で、Inputはraptor-knee-widedense、Internet OFF。APIが返すInput参照は版なしなので、保存Input版・実行時間・global IDと表示版の直接照合は未確認。新規提出はしていない。結果はp001として台帳へ保存。
- 公開7候補のDataset版、metadata、ライセンス、生成方法とgold利用を公式CLI/SDKで監査。作者の学習コードはDreaddevelopment Version 8、Pilkwang Version 15を取得して読み、実行していない。前者のgold AUCによるcheckpoint選択、後者の学習分岐での公式ラベル混合を確認した。公開重みの厳密な履歴まで証明したものではない。
- vmohitrao Dataset Version 3を非商用研究候補として取得。元train.csv hash、weak_labels・weights・states・provenanceの4つの公開hash、4,349行のUID集合、全セルのstate/value/maskが一致。goldのUIDは含まれない。作者READMEが58検査を抽出・prompt開発から除外したと宣言しており、その根拠のhashを保存した。開発ログを独立監査してはいない。CC BY-NC 4.0の大会提出への利用可否は未確認。
- prepareでweak 4,207検査・gold 58検査を作成。全欠損138件、goldと同一レポート・groupで連結される4件を除外。供給report_groupを保った5-foldは821・891・888・778・829件。fold 0は学習3,386／検証821件。元入力hash、元ラベルとの一致、gold予約、group連結、fold別・所見別分布を追加監査した。
- fold 0のSynovitisは観測101件のうち陽性100／陰性1、欠損率87.7%。CSV整合性の成功はラベルの臨床的正確さやクラス分布の妥当性を保証しない。患者独立性も未確認。
- 新規の標準ライブラリunittest 12件でUNK→欠損、0とsoft labelの保持、出所hash・state/mask不一致、gold宣言不足、改ざんgroup、除外行を橋渡しにするgold漏洩、未観測クラス等を検査。既存を含む31件が成功。srcは変更しておらず、exportのsource hashと前処理fingerprintを維持。
- ユーザーはKaggleで元画像とcacheを目視比較し、見た範囲では問題なしと回答。件数とUIDは未報告であり、全画像の品質確認や医学的判定として扱わない。
- 人工ノイズ22検査とtest 3検査を独立したQAディレクトリに生成。192px・24 windowの設定でCUDA/AMP/backward/optimizer、1 epoch完走、checkpoint保存・再読込、Reportなしの提出契約を確認。全3行×12所見の契約がvalid=true。最大torch割当467,144,192 bytes。人工データの精度・時間・VRAMを実コンペ性能として記録しない。結果は `artifacts/qa/20261003-synthetic-cuda/checks.json`。
- 実キャッシュでe001-smoke-fold0が完了。weak検証BCE 0.4304814244、選択後のgold macro AUC 0.4871779862。checkpointは44,830,657 bytesで保存。historyの205.33秒はepoch内の経過時間で、cache事前検査と最後のgold評価を除く。コマンド全体の所要時間は未計測。
- e002-baseline-fold0は同じラベル・fold・seedでepoch数だけ5へ変更し、新規runで完了。weak検証BCEはepoch順に0.430481、0.430906、0.447424、0.502899、0.448503。最小のepoch 1を選択し、その後にgoldを評価した。gold macro AUCはe001と同じ0.487178。train関数は事前検査・最後のgold評価を含め1,082.83秒、Python importは計測外。epoch部分は1,034.21秒。実行環境・入力hash・config・全history・checkpoint hashはrunと共有可能な実験JSONへ保存した。Public LBは未測定。
- `artifacts/kaggle/e002-baseline-fold0/` を新規作成。code zipは許可されたsrc 9ファイルとbaseline.jsonだけで、SHA-256は9fd958c4585d3bced5552cb1c949a103ba6209fc4ddaa3e18b6d07219d49545f。別ファイルで置いたbest.ptのSHA-256は903cce571904a8e74ff54d3242e3cb6293db747b293475be378578767eb3f29b。展開zipのsource hashと学習記録が一致した。
- 新しいPythonプロセスで展開zipのsrcを読み込み、実e002 checkpointと人工test cacheで予測。Reportなしの3行×12所見について、UID集合・順序・列・有限値・0〜1範囲の提出契約がvalid=true。結果は `artifacts/qa/20261003-e002-submission/checks.json`。実MRI/testのdecodeとKaggle実行は未検証。bundleにRESEARCH-ONLY.jsonを追加し、アップロード・提出はしていない。
- 最終資料のローカルリンク48件、PowerShell手順ブロック15個、実験JSONと台帳3行の構造、保存run configと元configの一致を確認。run configのhashと元ファイルのhashは記録上区別した。Ruff・全31 unittest・差分の空白検査が成功。data/・artifacts/の追跡対象は各READMEだけで、ラベル・画像・分割・予測・重みはGit除外を確認した。

## 実学習へ移る前の確認（10月3日、以下は当時の状態）

- 取得済みのcoverage.jsonを集計。4,407検査すべて24 valid windows、選択された13,221シリーズすべてphysical_position順であり、記録された読み取りエラーとinstance_number fallbackは0。元画像やnpzの画像内容をこの確認でデコード・目視していないため、品質の確認とはしない。
- UIDやレポート本文を含まない集計を、Git除外の `data/exports/rsraki-rsna-knee-sv354838181/train-coverage-summary.json` へ新規保存した。
- 専用venvからKneeMILをevalモードでRTX 4090へ配置。人工入力1×4×3×32×32、有効mask 2枚でCUDA forwardを実行し、出力1×12と全値が有限であることを確認。公開重みを使わず、実学習、AMP、backward、optimizerの検証は行っていない。
- data/labelsは未作成。次の優先順位をラベル監査、画像目視、固定fold、1 epoch実行確認、基準実験、自作モデル提出へ更新した。実ラベル取得、実学習、外部アップロード・提出は実行していない。

## Kaggle Outputのファイル単位取得（10月3日）

- 指定URLはWeb閲覧ツールでは開けなかった。ユーザーのブラウザでのCLIログイン後、公式SDKで認証済みアカウントrsraki、private Notebook `rsraki/rsna-knee`、最新版Version 1を確認した。URLのscriptVersionId=354838181とVersion 1の対応はユーザー報告として記録する。
- Kaggle CLI 2.2.4をGit除外の `artifacts/tools/kaggle-venv/` に導入し、学習用venvから分離した。認証値は表示・コピーしていない。SDKの通常の認証を使用した。
- CLI 2.2.4の実装ではOutputの版suffixが要求へ渡されておらず、SDKのversion_label=1を明示した要求も対象Notebookでは404だった。版を省略したOutput要求は成功したため、取得前後に最新版番号を検査する取得器を追加した。将来最新版が変わった場合の古い版取得は未対応であり、異なる版を黙って取得しない。
- export.jsonを実取得し、complete=true、4,407検査、4,423ファイル、7,785,521,448 bytes、現在のsrcのhashとの一致を確認。保存Outputの全ページにmanifestの必要ファイルが揃うことを確認し、全ファイルを取得した。
- 取得器はSHA-256が一致する完成済みファイルを再利用し、不一致の再取得、`.part`からの置換、ページング、保存先の範囲、取得許可するファイル種別、空き容量を確認する。署名付きURLを記録へ保存しない。転送記録は画像exportの外側へ置く。
- 転送中にWindowsの長いパスがextended-length形式（`\\?\`）になり、通常表記のrootとの照合で停止した。取得済みファイルを保ち、取得器と既存export検査器の比較処理を共通化して再開。Windowsの通常・extended-length・UNC表記を比較時だけ揃え、保存先外のパスは引き続き拒否する。srcは変更しておらず、cache fingerprintは維持される。
- 新規の標準ライブラリunittest 7件で、Windowsの危険なパスとextended-length表記、同じサイズの破損ファイル、再開、hash不一致時の既存ファイル保持、debug/不正export、全ページ取得、取得中の版変更の拒否を検査した。既存テストも含めて19件が全て成功した。ネットワークを使うテストは追加していない。
- ローカルに取得したCSVを純粋なCSV監査CLIで再監査し、4,407検査・24,371シリーズ、58検査の全12項目gold、シリーズなし・空レポート0、同一レポート54groupを確認。患者独立性の確認ではない。集計は `data/exports/rsraki-rsna-knee-sv354838181/csv-audit.json` に保存した。
- 取得後のexport検査が `valid=true, studies=4407, files_verified=4423` で成功。inventory、全ファイルのSHA-256・サイズ、元CSV、npzのUID集合・件数、現在のsrc、前処理fingerprintが一致した。再開時は3,219ファイルを再利用し、残り1,204ファイルを取得。親ディレクトリのtransfer.jsonにtransfer_complete=trueと検査結果を保存した。
- 専用venvはtorch 2.10.0+cu128・torchvision 0.25.0+cu128になっており、torch.cuda.is_available()=trueとRTX 4090を確認。今回実行したforwardは既存unittestの人工画像CPU経路で、実学習のAMP/backwardを確認したわけではない。
- `ruff check .`、`ruff format --check .`、`git diff --check` が成功。ローカルリンク45件とPowerShellの手順ブロック7個も確認。取得データ・Output・CLI環境はGit除外で、data/とartifacts/の追跡対象はREADME.mdだけである。

## 全件キャッシュ実行のユーザー報告（10月3日）

ユーザーからKaggleでLIMIT=Noneの実行が終了したと報告を受けた。最後のexportセルの成功、complete=true、保存したNotebook版、Outputのファイル集合・件数・hashはこの作業では確認していない。保存版の閲覧画面からOutputを一括取得する方法と、private Dataset経由の代替方法を調査して案内した。実データのダウンロードやMRIデコードは実行していない。

[KaggleスタッフのOutput保存・取得用Notebook](https://www.kaggle.com/code/paultimothymooney/how-to-save-a-file-to-the-notebook-output-folder/output) の索引で「Download notebook output」を確認し、[公式Dataset資料](https://www.kaggle.com/docs/datasets) でOutputからDatasetを作成する経路を確認した。認証後のユーザー画面と現在のボタン配置は未確認。PROJECTの進捗を更新し、今回の資料差分をgit diff --checkで確認した。

## 初回提出のユーザー報告

ユーザーからNotebookの提出受付成功の報告を受けた。Kaggle画面、採点結果、Publicスコア、Notebook版はこの作業では確認していない。公開モデルの採点成功や自作モデルの学習成功とはまだ扱わず、次は採点結果の確認と10検査のキャッシュ準備へ進む。PROJECTの状態だけを更新し、未確認の数値を実験台帳へ追加していない。

## キャッシュ作成後の手順4〜9の確認

- [after-cache.md](after-cache.md) を追加し、README、PROJECT、GPU開始手順、roadmap、提出手順から参照できるようにした。変更対象は資料のみで、src、config、Notebookの実装は変更していない。
- 記載したprepare・train・転送検査・梱包コマンドの引数を現行CLIのhelpと照合し、PowerShellのコードブロック5個を構文解析した。実データに対するコマンド実行はしていない。
- README、PROJECT、docs内のローカルファイルリンク42件を確認。ラベルCSVの13列が提出契約と一致し、epoch数だけの変更ではcacheの前処理fingerprintが変わらないことを確認した。
- 専用venvで既存unittestを再実行し、10項目成功・2項目skip。人工DICOMを使用し、torch依存のforward・loss/CUDAガードとcacheテスト内の推論分岐は未実行。`doctor` でもtorch/torchvision未導入を再確認した。
- `ruff check .`、`ruff format --check .`、`git diff --check` が成功。
- 公式の索引から提出条件と日程を再確認した。直接openではKaggle本文を取得できなかったため、Rules細目やアカウント別上限を確認済みとはしない。引用元は手順書に記載した。

資料の確認は、実ラベルの監査、実MRIの品質、CUDA学習、Kaggleでの提出成功の代わりにはならない。これらの結果はまだない。実験台帳へ未実施のスコアやrunを追加していない。

## 10月2日のデスクトップでの確認

- `nvidia-smi` でRTX 4090、VRAM 24,564MiB、ドライバ591.86を確認。Cドライブ空きは484,710,727,680 bytes。RAM容量はOSの読み取りが拒否され未確認。PyTorchのCUDA利用は未確認。
- Windowsの `python` エイリアスが動かず `.venv` も存在しなかったため、Git除外の `.python/` と `.venv/` へPython 3.12.13を用意。numpy 2.5.3、pydicom 3.0.2、Pillow 12.3.0、ruff 0.16.10と本パッケージを導入した。torch/torchvisionは未導入。
- この環境でunittestを実行し、**10項目成功、2項目skip**。skipはtorch/torchvisionが必要なモデルforward・loss/CUDAガードのテスト。画像cacheテスト内のtorch推論分岐も実行していない。実DICOMではなく人工の非圧縮single-frame DICOMを使った。
- 新規 `00_prepare_cache.ipynb` の展開・cache作成・export記録セルを人工4検査で実行し、全件exportの転送検査が成功した。ファイル破損、debug export、不完全なhash一覧の拒否も確認。Kaggleパス解決セルと画像viewerはこの確認の実行対象外。
- 手順1〜3の説明時にコードInputの展開形式を補正。両Notebookのzip・展開済みディレクトリからのコード読み込みを確認し、ディレクトリ形式で無関係なファイルをコピーしないことを検査した。キャッシュNotebookは両形式とも人工1検査からexport検査まで成功。提出Notebookの依存確認はmockで省略し、torch推論は実行していない。記録は `artifacts/qa/20261002-code-input-formats/checks.json`。
- 両Notebookのコードセルを構文検査し、保存された出力と実行回数が空であることを確認。generatorからの再生成が一致するよう、既存提出Notebookのimportと書式も整理した。
- `ruff check .` と `ruff format --check .` が成功。
- `artifacts/kaggle/cache-v1/rsna-knee-code.zip` を生成し、srcのPythonファイルとbaseline.jsonだけの許可リストを確認。データ・重みを含めていない。アップロードしていない。

人工exportの検査記録はGit除外の `artifacts/qa/20261002-cache-export/checks.json`。Notebookは未実行テンプレートとして保存しており、Kaggleセッションでの動作、圧縮MRIのdecoder、実キャッシュサイズ・実行時間、CUDA/AMP/backward、学習精度・採点成功は未検証。

## 以前のCPU環境で確認済み

- unittest 12項目が全て成功。CSVの列順・UID・欠損・有限値、AUCのtieと未定義クラス、gold除外、同一レポートと追加groupの連結、空欄ラベル・出所不足の拒否を確認した。
- 人工のsingle-frame非圧縮DICOMを読み、ファイル名とInstanceNumberが物理順序と違う場合も、物理座標順にcacheを作れることを確認した。
- cacheの再開、設定不一致の拒否、全シリーズ欠落時の停止を確認した。
- maskされた画像を変えても予測が変わらないこと、欠損ラベルのloss除外、CUDAなしで学習が開始されないことを確認した。
- ランダム初期化checkpointを保存・読み込みし、CPUで人工画像の推論から提出CSV検査まで実行した。これは精度の検証ではない。
- ruff checkとformat checkが成功した。
- 提出Notebookのコードセルを構文検査。セル出力と実行回数は空で、未実行テンプレートとして保存した。
- コードzipの許可リストを検査。srcのPythonファイルとbaseline.jsonのみで、データ・重み・認証情報を含めない。
- プロジェクトのローカルリンク、Git差分の空白を確認した。

完全な12項目の実行には、既存のPython 3.10.11環境のtorch 2.7.0+cpu、torchvision 0.22.0+cpu、Pillow 11.2.1と、Git除外のQAフォルダへ取得したpydicom 3.0.2を使用した。他プロジェクトの環境やファイルは変更していない。

対応対象のPython 3.12.10でも軽量ツールを確認した。画像依存を用意した環境では10項目成功、torchなしの2項目をskip。このリポジトリ専用の `.venv` には基本パッケージだけを導入し、7項目成功、画像/GPU依存の5項目をskipした。専用venvへGPUライブラリは導入していない。

## KaggleとローカルGPUで行う確認

- 実CSVの列・件数・欠損・表記と、ラベルの出所・版・利用条件。
- 実MRIの圧縮形式、decoder、方向、順序、PixelSpacing、左右、両膝、multiframeや不正画像。
- 元画像とcacheの目視比較、所見が残ること、各クラスの陽性・陰性分布、患者・施設・画像重複の漏洩。
- CUDA/AMP、実際のbackward・optimizer・checkpoint選択、GPUでの学習時間とVRAM。
- 大きな検査を含む前処理込みの推論時間、Kaggleのオフライン依存、隠しtestでの採点成功。
- 公開Notebookの指定版・入力版の再現、実測Public LB。

初期版には学習再開、分散学習、事前学習encoder、ensemble、物理尺度cropを含めていない。競争力のある公開モデルの再現を優先してから、比較に必要な機能を追加する。
