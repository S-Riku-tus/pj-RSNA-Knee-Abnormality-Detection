# 準備内容の検証範囲

更新日 2026年10月3日 JST。今回はユーザーの明示的な取得依頼により、Kaggleの全件画像キャッシュのローカル取得・検査を完了した。元DICOM・学習済みモデルを取得したり、実MRIをデコードしたり、実学習を始めたりしていない。以下では以前の検証と今回の確認を分ける。

## 次工程に向けた確認（10月3日）

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
