# RSNA Knee の参加計画と次の作業

10月4日時点では、全件cache取得・CUDA環境・研究用ラベル監査・1/5 epoch実験・提出bundle検査は完了し、公開モデルの実測Public 0.924もAPI確認済み。現在の判断は [PROJECT.md](../PROJECT.md) と [次の実験計画](research/next-experiments-20261004.md) を参照する。以下の「最初の二日間」などは10月2日時点の準備計画であり、未着手の現状として読み替えない。

2026年10月2日 JST時点の判断。リポジトリの資料、ソース、設定、テスト、Notebook、実験台帳を確認し、公式競技説明と公開資産を再調査した。現在の端末はRTX 4090を搭載する。ユーザーの選択に従い、**Kaggleで画像キャッシュを作り、このデスクトップで学習し、Kaggleで提出推論する**。まず公開モデルで採点を通し、比較基準を確保する。

## この競技で行うこと

複数のMRIシリーズを持つ一つの検査から、靱帯・半月板損傷、変形性関節症など12所見の予測値を出す。評価は12項目の平均ROC-AUCで、陽性を陰性より高く順位付けする能力を測る。0.924を診断正解率92.4%と解釈しない。[公式概要](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

学習時の読影レポートから教師信号を作れるが、テスト時にはReportがない。提出Notebookは未知のMRIから予測を作る必要があり、手元で作ったCSVをそのまま提出する経路ではない。例示テストは3検査、本採点は約1,300検査である。[公式データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)

## 現在の準備と不足

| 対象 | 確認した状態 | 次に必要なこと |
|---|---|---|
| CSV契約とラベル準備 | 列順、欠損、gold除外、同一レポートと供給groupの連結を実装済み | 実CSVの件数・欠損・各foldの陽性数を監査する |
| MRI前処理 | 幾何情報で並べ、最大3シリーズから2.5D窓を作る | 実DICOMのdecoder、方向、左右、失敗例、所見の保持を確認する |
| 自作画像モデル | ランダム初期化ResNet18と所見別Attention | 工学的な比較基準として動作と速度を測る |
| 学習と推論 | CUDAチェック、weak BCEでのcheckpoint選択、Report不要の推論を実装済み | CUDA、AMP、backward、実学習と実採点を確認する |
| 公開モデル | CoAtNet単体モデルなどを調査済み | 特定Notebook版とInput版を固定して自分で採点する |
| 実験記録 | 台帳はヘッダーのみ | 初回採点から実測結果を記入する |

このリポジトリは実験の土台であり、精度が確認された解法はまだない。既存の12テスト成功の記録は旧環境での人工データ検証で、実MRI・CUDA学習・採点の成功を表すものではない。今回の確認は [validation.md](validation.md) に分けて記録する。

## RTX 4090 をどこで使うか

端末の読み取りによる確認値はRTX 4090、VRAM 24,564MiB、ドライバ591.86。Cドライブは約1.02TB、空き484,710,727,680 bytes、約484.7GBである。公式Data Explorerは569.76GB、819,640ファイルと表示する。現在のCドライブでは元画像の全展開が収まらず、圧縮アーカイブとキャッシュの保存領域も別に必要になる。[公式データページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)

| 場所 | 担当する作業 | 理由 |
|---|---|---|
| Kaggleの公開モデルNotebook | 初回提出と公開モデルの再現 | 元画像や重みをローカルへ移さず、採点経路を確認できる |
| Kaggleのキャッシュ作成Notebook | 競技Inputを直接読み、選択・縮小した画像を保存 | 約570GBの元画像をこのPCへ全取得する必要をなくす |
| このPCのRTX 4090 | キャッシュから学習、検証、実験比較 | 自分のGPUで反復できる。適切なバッチと所要時間は実測する |
| Kaggleの提出Notebook | 隠しtestの前処理と推論 | 競技が指定する環境で未知データを処理する |

Kaggle Notebookはクラウドの計算機であり、ブラウザをこのPCで開いてもRTX 4090は使われない。このGPUを使うにはローカルPythonから学習を実行する。Windows PowerShellで開始できるため、初回実験の前提としてWSLやLinuxへの移行は求めない。

提出の9時間制限は提出Notebookの実行条件であり、学習と提出を一つのNotebookへまとめる設計は必要ない。ただし、今回WebではRules本文を取得できなかった。参加時にローカル学習・競技由来キャッシュのprivate保存・利用資産の条件を競技画面で確認する。[Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)、[Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

## 最初の二日間にすること

1. **参加登録と提出環境を確認する。** ルールに同意し、アカウント画面でGPU利用可否、残り枠、日次提出回数、最終選択数を確認する。固定の無料GPU時間を前提にしない。
2. **公開単体CoAtNetモデルを複製して採点する。** [作者Notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) のスコアを出した版と現在版を区別し、Inputs、GPU、オフライン依存を合わせる。Save and Run All後に手動でSubmitし、採点成功まで確認する。[再現手順](public-baselines.md)
3. **採点記録を残す。** 自分のPublic LB、Notebook版、Input版、checkpoint、全体実行時間、変更点を記入する。作者報告0.924は自分の結果欄へ入れない。
4. **専用Python環境のCUDA対応を整える。** 今回はPythonと軽量チェック用依存だけを用意した。CUDA版torchとtorchvisionは未導入である。公式の対応する組を入れ、`doctor` と人工画像のforwardで確かめる。[環境手順](gpu-start.md)
5. **Kaggleで10検査のキャッシュを確認する。** [00_prepare_cache.ipynb](../notebooks/00_prepare_cache.ipynb) へ競技InputとコードzipをAttachし、入力版を記録する。coverageと原画像・縮小画像を比較してから全件へ進む。

初回提出の成功と、ローカル学習準備は独立して進められる。初回採点が動かなければInputsと版を直し、モデル変更を重ねない。

## キャッシュを移して学習する手順

全件キャッシュはKaggleのprivate Outputから手動でダウンロードし、`data/exports/` 以下の新しいディレクトリへ展開する。Notebookはcache、学習CSV、config、前処理ソース、SHA-256一覧を保存する。転送後にhash、完全性、前処理fingerprintを検査してから学習する。具体的なコマンドは [gpu-start.md のキャッシュ経路](gpu-start.md#kaggleのキャッシュでローカル学習する) を使う。

現行configの画像配列は、4,407検査と仮定すると `4407 × 3シリーズ × 8窓 × 3枚 × 192 × 192 × uint8`、約10.9GiBである。これはコードの形状からの計算で、実測ファイルサイズではない。npz圧縮後のサイズ、デコード時間、実際の検査件数は最初の実行で確認する。参加者にも約11GiBのキャッシュ化報告があるが、その6枠・224pxのキャッシュを本コードへそのまま読み込むことはできない。[作者のキャッシュ実験](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)

Kaggleの保存可能なOutputは公式ドキュメントに20GBと記載されるため、現行Notebookは18GBを目安に検査し、出力内に同容量のzipを追加作成しない。画面の現在の制限も確認する。実行時間がセッション制限を超える場合は、保存できたprivate OutputをInputにし、同じコード・configで`RESUME_CACHE`から再開する。必要なら対象検査を分割する機能を別途追加する。[Notebookの仕様](https://www.kaggle.com/docs/notebooks)、[OutputをDatasetにする手順](https://www.kaggle.com/docs/datasets)

弱教師ラベルは出所、版、利用条件、goldへの調整の有無を確認して用意する。現実装には公開ラベルの自動取得やLLMによる抽出器はない。空欄テンプレートを作るだけでは学習できない。`prepare`でgoldを予約し、同一レポートと供給groupをまとめてからfold 0をまず1epochで動かす。この1epochも全fold学習・検証検査を処理する。転送、ラベル監査、学習、提出と改善の具体的な操作は [手順4〜9](after-cache.md) を使う。

最初のキャッシュは自作ResNetの動作と検証用に使う。メダルを目指す主力の改善は、再現できた公開モデルの学習コード・前処理と、汎用事前学習encoderを使った独立のfold学習を候補にする。公開競技重みが全weak検査を学習済みなら、その重みを自分のfoldで評価・fine-tuneしても独立検証にはならない。公開重みを使った比較は提出再現として区別する。

## 三週間の優先順位

以下は10月2日開始の提案であり、公式日程や学習時間の保証ではない。遅れた場合は実験数を減らし、最終採点の余裕を残す。

| 期間 JST | 到達したい状態 | 次へ進む条件 |
|---|---|---|
| 10月2〜4日 | 公開モデルで初回採点、CUDA準備、10検査cache確認 | 成功した提出が一つある |
| 10月5〜8日 | 全件cache、ラベル監査、fold 0の比較基準 | 入力hash、分割、weak検証、速度が記録されている |
| 10月9〜15日 | 同じfoldとseedで少数の改善実験 | 改善した要因を説明でき、他クラスを悪化させていない |
| 10月16〜19日 | 有望な条件を追加foldで確認、必要なら2モデルを比較 | 検証の独立性と前処理込みの提出時間に問題がない |
| 10月20〜22日 | 採点成功済み候補を固定し、最終候補を選ぶ | オフライン再実行と隠しtestの採点が成功している |

参加・チーム統合期限は **10月16日08:59 JST**、最終提出は **10月23日08:59 JST**。本プロジェクトでは10月22日中に選択を終える。未採点の変更を最後の候補にしない。[公式Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)

改善順は、実画像の前処理監査、事前学習encoder、シリーズ・窓の見せ方、ラベルの誤り、少数ensembleを出発点にする。ボトルネックが計測できたら順序を変える。モデル・ラベル・cropを同時に変えず、変更理由を事前に一行書く。1epochの時間を測って候補runの予算を決め、最初から5foldすべてを回さない。

## 改善を判断する基準

- **weak検証BCE**でepochを選ぶ。レポート由来ラベルへの適合を測るため、公式AUCと同じ意味ではない。
- **goldのクラス別AUCと陽性数**で画像由来ラベルとの整合を確認する。学習やcheckpoint選択には使わず、何度も合わせ込んで小さな差を採用理由にしない。
- **Public LB**は外部評価として確認する。提出ごとに仮説を持ち、連続した微調整で公開部分に合わせ込まない。
- **前処理込みの時間と失敗率**も採用条件にする。約1,300件を9時間で割ると平均約25秒/検査だが、起動や長い検査も含まれるため上限ぎりぎりを狙わない。

12クラスは同じ重みで評価されるため、平均だけでなく弱いクラスも見る。soft labelを0.5で丸めたAUCを公式値と同一視しない。患者対応が未確認なら患者独立と呼ばない。金標準への露出が不明な公開ラベル・重みでは、独立性も未確認として扱う。

## 残る確認事項

現在の順位・メダル境界、Rulesの細目、公開Notebookの厳密な版と再現性は未確認。0.924や添付の0.94台からメダル獲得を予測しない。メモリ容量は端末の読み取りが拒否されたため未確認で、全cacheをRAMに載せる方式は前提にしない。現行Datasetは検査単位で読み出す。

最初に必要なのは、参加登録、公開モデルの採点成功、CUDA利用可否、実画像10検査の監査である。この四つが揃ってから長時間学習へ進む。調査の取得状況は [sources.json](research/sources.json)、実際に行ったチェックは [validation.md](validation.md) に保存する。
