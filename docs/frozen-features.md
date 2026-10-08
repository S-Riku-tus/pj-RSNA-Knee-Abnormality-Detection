# 392px凍結DINOv2から独自headを学習する

更新日：2026年10月8日 JST。**p005は18:03 JSTに自己Public0.950で採点成功を確認。待機中のf004両fold実学習・教師17セル二次レビューも完了しました。** [f004完成比較](../experiments/f003-f004-fold01-review-20261008.json)は両foldに揃う改善がなく差し替え見送り、[教師再確認](label-review.md#17セルの二次レビュー結果10月8日)は訂正採用0件です。新しい主軸は[0.950基盤の一要因改善計画](../experiments/p005-improvement-roadmap-20261008.json)。以下の[f004コマンド](#次の学習候補f004と教師監査)は実行履歴で、同じrunを再実行する必要はありません。10の再提出は保留、13は独自f002の任意診断です。

## 10月8日以降のf002診断

添付summaryと保存Outputはbyte同一。提出ref56904146/Version1/script355977821の全sourceは配布10と一致し、3 Inputs・GPU・Internetも正常です。可視3件とCSVは22項目で正常ですが、隠し一般例外・自己Publicなしでした。**f002は独自headであり、作者0.943を再現するp003とは別モデル**です。[今回の事実と限界](../experiments/f002-embedded-head-hidden-failure-review-20261008.json)

件数固定やReport依存は人工1,300検査で否定できた範囲があります。残る問題候補は、選択されたMRI系列の条件と資源です。元readerは全スライスのfloat32配列・stack・percentile作業配列・float64全体正規化を持ちます。既存exportの最大系列はfloat32一枚のvolumeだけで約605MiBでした。不適合なspacing/位置/強度で全体停止する経路も人工再現しました。ただし元readerは実4,207学習検査を既に完走しており、どちらも今回の隠し真因とは未確定です。欠損値・物理尺度・予測確率を作って回避しません。

[13_profile_f002_memory_1300.ipynb](../notebooks/public/13_profile_f002_memory_1300.ipynb)は、**メモリ配送だけを変更した診断**です。原head・encoder・decoder・系列選択・geometry条件・全volumeのpercentile・letterbox・12所見順を保ち、native/scratchをtemporary memmapにして必要スライスのみ正規化します。元16core file/runtimeは改変せず、新readerのruntime binding/source hashを明記します。人工DICOMの入力窓・位置・特徴配列はbyte一致、実学習済みheadのRTX4090人工予測差0。実MRI/Kaggleでは未実行です。memmap partitionのページがRSSを増やす可能性とdisk/IO増加は残ります。

1. p004の固定対照とは**別の新しいprivate Notebook**へ13をImportします。13は診断用で、提出用CSVは作りません。
2. Inputsは10と同じ3つ：Competition `rsna-knee-abnormality-detection`、`rsraki/rsna-frozen-feature-inputs-v1`、`rsraki/rsna-dicom-decoders-py313-v1`。旧head Datasetは追加しません。
3. 10と同じPython3.13系の環境・GPU T4×2・Internet OFFでSave & Run Allします。これはp004のPython3.12とは別の系統で、環境を混ぜません。
4. fixed gold除外weak4,207件からseed20261007で1,300 UIDを固定選択して処理します。教師値のfit・精度評価・checkpoint選択は行いません。無作為に固定したcohortなので、最大nativeサイズや全protocolを必ず含むとは扱いません。
5. root `F002_STREAMING_SUMMARY.json`、通常events、readerの資源events、完了時`profile_predictions.csv`を確認します。`profile_complete_not_for_submission`・`phase=complete`・`processed=studies=1300`・receipt valid・skip/fallback0、新readerのhashを確認します。失敗時は最後のphase/header/資源/tracebackを分類します。**profile CSVは提出しません。**

13が成功しても隠しtest固有の不適合系列・環境差を網羅したとは扱いません。次のf002提出候補はこの診断結果を確認してから一要因ずつ決めます。別に準備したroot選択helperや系列の代替policyを同じ13へ混ぜません。以下の「次は10/09」は以前の作業時点の履歴です。

## 履歴：head接続の再発後は10を使う

`rsraki/notebook6372fe6490` **Version1/script355963229**を公式SDKで確認し、配布09と全cell一致・必要4 Inputs・GPU有効・Internet OFFでした。同じhead Dataset12402506/内部版20385564を接続するKaggle管理処理が`mkdir ... Read-only file system`となり、Python開始前にERROR・Output0・log`[]`です。08 Version1の接続失敗と同じ箇所で、間にCPU確認/08 Version2が成功しても再発しています。操作違いやMRI処理中の例外とは区別します。[21項目の再発監査](../experiments/f002-recurrent-mount-review-20261007.json)

[10_submit_f002_embedded_head.ipynb](../notebooks/public/10_submit_f002_embedded_head.ipynb)は、元headのbest.pt/config/provenanceをコードから同一byteで復元し、サイズ/SHA256を検証します。モデルはf002 fold0 epoch11のまま、09の固定code・encoder・decoder・推論を引き継ぎます。旧head Datasetのmountを不要にする変更です。他Inputsの接続や旧07の隠し例外まで直ったとは判断しません。headを含むため生成Notebookは個別Git除外し、private Notebookとして扱います。

1. **新しいprivate Notebook**を作り、10をImportします。元09を残し、既存の08/07を改造しません。
2. Inputsを次の**3つだけ**にします。
   - Competition `rsna-knee-abnormality-detection`
   - 既存Dataset `rsraki/rsna-frozen-feature-inputs-v1`
   - 既存Dataset `rsraki/rsna-dicom-decoders-py313-v1`
3. **`rsna-f002-head-fold0-v1`をInputsに追加しません。** 既存Notebookをコピーした場合はInputsの参照から削除します。Python内のパス変更だけでは実行前の接続失敗は避けられません。新しいhead Dataset・ZIP再アップロード・06再実行・再学習は不要です。
4. **GPU T4×2・Internet OFF**でSave Version → **Save & Run All**します。元と同じcuda:0を使います。最初の復元セルでは3ファイルのhash一致を確認し、その後decoderの人工画素検査とtest全件推論へ進みます。
5. Outputの`F002_STREAMING_SUMMARY.json`で`status=passed`、`phase=complete`、`mode=test`、`processed=studies`、`receipt.valid=true`、`csv_name=submission.csv`、skip/fallback0を確認します。公開testの件数はtest.csvから読み取ります。
6. 成功したら、**その10の保存Versionからsubmission.csvを手動提出**します。採点終了後はPublicの有無とerrorDescriptionを確認します。可視完走だけでは提出成功ではありません。

失敗箇所ごとの次の操作も固定します。

| 停止した段階 | 次に確認する証拠・対応 |
| --- | --- |
| `mount data`・Output0 | 失敗したInput名を確認。旧headが表示されたらInputsに参照が残っています。別Inputならその接続問題で、学習/decoder調整では直りません。反復試行より保存URL/Version/全文を使ったKaggle側調査へ進みます |
| 復元/hash/decoderのセル | 表示されたエラーとNotebook版・Input内容を照合します。別モデルや数値を代用しません |
| 可視MRI推論中 | root summaryと`f002-streaming-diagnostics/events.jsonl`の最後のphase/traceback/header/資源を確認します |
| 可視成功後、提出だけ一般例外 | 提出ref/保存Versionと一般エラーを記録。隠しtracebackは取得できるとは限らず、同じCSV再提出や9時間超過の決め打ちはしません |

08の256件はskip/fallback0・12分28.4秒で完走し、decoderの4形式はKaggle Linuxで人工画素一致を確認済みです。その実MRIの系列先頭768headerは非圧縮で、実圧縮MRIの全条件は未検証。公式Dataの隠しtest約1,300件と可視3件は規模/入力が違います。[公式Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series)、[公式のエラー分類](https://www.kaggle.com/code-competition-debugging)。10はこの独自headのPublicを初めて測るための候補で、p003作者0.943の再現モデルではありません。

## 08の256件診断が成功した後の09手順

以下は09の接続再発前の操作です。現在は上記10・3 Inputsを使います。


`rsraki/notebook245423d388` Version2/script355957807は公式SDKでCOMPLETE、配布08と全source cell一致、添付summaryは保存Outputとbyte同一。256件をskip/fallbackなしで完了し、固定UID/orderと12所見・3,072確率・CSV hashを独立に検査しました。32項目の確認が成功しています。[完走監査](../experiments/f002-streaming-profile-review-20261007.json)

| 確認項目 | 結果 |
| --- | --- |
| Profile完了 | status=profile_complete_not_for_submission、phase=complete、processed=studies=256 |
| CSV検査 | receipt.valid=true、UID/所見順/hash一致 |
| 固定head | f002 fold0 epoch11、checkpoint/featurefingerprint一致 |
| 圧縮decoder | 4形式すべて利用可能、人工画素はdefault/pylibjpeg両方で完全一致 |
| 処理計測 | 748.38秒＝12分28.4秒、約2.92秒/検査 |
| CPU / GPU / disk | peakRSS4.43GiB / PyTorch peak reserved364MiB / 最後の空き19.50GiB |

計測はrun_streaming内で、前段setup/decoder install/queueを除きます。旧経路との公平な速度比較ではありません。実系列先頭のheader集計768件は非圧縮形式で、圧縮4形式は人工画像による確認です。この256件完走から隠し例外解消・隠し時間・自己Publicを保証しません。GDCMのmissing_dependencies表示があっても、pylibjpeg等がavailableなので追加導入は不要です。

1. [09_submit_f002_streaming.ipynb](../notebooks/public/09_submit_f002_streaming.ipynb)を**別の新しいprivate Notebook**へImportします。実行済み08を提出版へ改造せず、09を使います。
2. 同じ4 Inputs：Competition `rsna-knee-abnormality-detection`、`rsraki/rsna-frozen-feature-inputs-v1`、元`rsraki/rsna-f002-head-fold0-v1` Version1、`rsraki/rsna-dicom-decoders-py313-v1`を接続します。今回成功した内容を維持し、headの再アップロード・06再実行・追加学習は不要です。
3. **T4×2・Internet OFF・新しいkernel**でSave Version → **Save & Run All**します。GPUはsource上のcuda:0を使うため、summaryのTesla T4表記は正常です。
4. Outputの`F002_STREAMING_SUMMARY.json`が**status=passed、phase=complete、mode=test、processed=studies、receipt.valid=true、csv_name=submission.csv**であること、skip/fallback0とdecoder人工画素検査の成功を確認します。公開testの件数は動的に読み取るので256件ではありません。失敗ならsummary/events.jsonlを確認し、提出へ進みません。
5. 09の保存実行が成功したら、**その09保存版のsubmission.csvを手動提出**します。提出ref/Version/scoreを記録します。08のprofile_predictions.csvや旧07Version2を提出しません。自己Public・隠し成否はその採点結果で初めて確認できます。

09実行後のsummaryをレビューへ渡す場合は今回と同じ`F002_STREAMING_SUMMARY.json`です。mode/profileの違いと保存版Notebookを記録し、診断成功と提出成功を区別します。

<a id="head単独確認の成功後に行う操作"></a>

## 履歴：CPU確認の後に08へ戻した操作

以下は再発前の操作です。現在は[10・3 Inputs](#head接続の再発後は10を使う)へ進み、元headを再接続しません。後の保存版監査ではCPU確認の実Inputsは4つ・Internet ONでした。添付hash検査の成功とhead単独の分離試験を実施した証拠を区別します。

添付`F002_HEAD_MOUNT_CHECK.json`はpassed/complete、元head Inputを接続し、best.pt/config/provenanceのサイズとSHAがこちらの元ZIP payloadと全件一致しました。model_loaded=false/training_performed=falseで、接続と内容だけの確認です。[結果監査](../experiments/f002-head-mount-success-review-20261007.json)。**headの再アップロード・別名Dataset作成・再学習は不要**です。4 InputsでのGPU起動・decoder/MRI・隠し完走まで成功したとは判断しません。

1. CPU確認08aのセッションを停止し、[08のKaggle編集画面](https://www.kaggle.com/code/rsraki/notebook245423d388/edit)へ戻ります。Notebookは[配布08](../notebooks/public/08_profile_f002_serving_256.ipynb)のまま使い、08aを256件診断へ改造しません。
2. InputsをCompetition `rsna-knee-abnormality-detection`、`rsraki/rsna-frozen-feature-inputs-v1`、元の`rsraki/rsna-f002-head-fold0-v1` Version1、`rsraki/rsna-dicom-decoders-py313-v1`の4つにします。元headはそのまま再利用し、外れていれば再追加します。
3. 新しいGPUセッション、T4×2・Internet OFFで**新しいVersionとしてSave & Run All**します。CPU設定は08a用であり、08はGPU必須です。
4. Outputの`F002_STREAMING_SUMMARY.json`を確認します。`status=profile_complete_not_for_submission`、`phase=complete`、`processed=studies=256`、`receipt.valid=true`、`skipped_series=fallback_predictions=0`が成功条件です。decoderの人工画素チェックと資源も確認します。08はCompetitionへ提出しません。
5. 08の結果を確認できた後、以下の[09手順](#今行う08の診断と09の提出準備)へ進みます。08が失敗した場合はroot summaryとevents.jsonlを確認し、mount data/Output0なら起動前の接続エラーとして再び切り分けます。

<a id="08のinput接続エラーを復旧する"></a>

## 履歴：08のInput接続エラーを復旧する

以下は初回接続失敗時の再試行案です。成功後も09で再発したため、現在は[10でhead接続を不要にする](#head接続の再発後は10を使う)方針です。

`rsraki/notebook245423d388` Version1を10月7日13:53 JSTに公式SDKで確認しました。status ERROR、`mount data: ERRORED_MOUNTING_DATASET`、Output0。全source cellは配布08と一致し、GPU有効・Internet OFF、Competition＋encoder/code＋head＋decoderの4 Inputsが揃っています。head DatasetはAPIでready/Version1、best.pt・config.json・checkpoint-provenance.jsonの一覧とサイズを取得できました。[接続エラーの監査](../experiments/f002-input-mount-failure-review-20261007.json)

失敗は、Kaggleが`rsraki/rsna-f002-head-fold0-v1`（Dataset12402506、内部版20385564）を接続するため、管理下の`/tmp/kglt/...`へmkdirする段階です。そこで`Read-only file system`となっており、**NotebookのPythonはまだ動いていません**。正常なInputの読み取り専用属性とは別に、接続準備側の書込みが失敗しています。APIreadyでも全実行環境への接続を保証しません。Kaggle内部のhost/cache/storageのどこで起きたか、全体障害か一時的かは未確認です。[mkdirのEROFS定義](https://pubs.opengroup.org/onlinepubs/009695299/functions/mkdir.html)

最初はコード・重みを変えずに次を行います。

1. 08の編集画面で動いているセッションを停止します。Python kernelだけの再起動ではなく、セッション自体を停止し、新しい実行環境での再試行を狙います。別hostへの割当ては保証されません。
2. Inputsから**head Inputの参照だけ**を外し、`rsraki/rsna-f002-head-fold0-v1`をAdd Inputで再追加して**Dataset Version1**へ固定します。残る3 Inputsを維持します。
3. GPU T4（UIでT4×2の場合も可）・Internet OFFを維持し、同じ08を**新しいVersion2としてSave & Run All**します。今回の停止はGPU計算に到達しておらず、T4×2による計算失敗の証拠はありません。
4. 再実行がPythonまで進んだら、従来どおり`F002_STREAMING_SUMMARY.json`の256件完走を確認します。接続復旧だけで07の隠し例外が解消したとは判断しません。

同じmountエラーが続く場合は、[08a_check_f002_head_mount.ipynb](../notebooks/public/08a_check_f002_head_mount.ipynb)を**別の新しいprivate Notebook**へImportします。Inputはhead Dataset一つだけ、**Accelerator None（CPU）・Internet OFF**でSave & Run Allします。Competition/encoder/decoderのInputはこの確認には不要です。画像処理・重みロード・学習・推論・提出は行いません。

| 08aの結果 | 意味と次の操作 |
| --- | --- |
| `F002_HEAD_MOUNT_CHECK.json`がpassed/complete、3ファイルのmatches=true | このCPU実行では接続と監査済みbyte一致まで成功。08へ戻り、4 Inputs・新しいGPUセッションで再試行 |
| 再びmount dataで停止しOutput0 | Python開始前の接続問題が再現。Notebookのtry/exceptやsummary生成では修正できない |
| JSONはあるがfailed | Pythonは開始。記録されたfile/path/hashの問題を確認してから08へ戻る |

head Datasetだけの接続失敗が継続する場合、最後の回避案として[既存0.32MBのhead ZIP](../artifacts/kaggle/f002-fold0-single-v1-20261006/rsna-f002-head-fold0-v1.zip)を、別名のprivate Dataset **`rsna-f002-head-fold0-mount-v1`**へ手動アップロードできます。元ZIP SHAと中の全3ファイルを検査済みで、重みは同じf002 fold0 epoch11です。新しいDataset参照で経路を変える試みであり、Kaggle内部障害を必ず回避できる対策ではありません。元Datasetは保持します。

この回避案を使う場合、08aは`HEAD_SLUG`だけを新名に変え、08/09は末尾セルの`HEAD_INPUT`を設定する**2か所のslug**だけを変更します。

```python
HEAD_INPUT = Path('/kaggle/input/datasets/rsraki/rsna-f002-head-fold0-mount-v1')
if not HEAD_INPUT.is_dir():
    HEAD_INPUT = Path('/kaggle/input/rsna-f002-head-fold0-mount-v1')
```

08/09のhead Input参照は新しい方一つへ入れ替え、checkpoint/sourceのSHAチェックは維持します。新しい08/09保存版・Dataset版を記録します。接続が広く失敗する／再作成でも続く場合は、Kaggle SupportへNotebook URL・Version1/script355951711・Dataset12402506/内部版20385564・mountエラー全文・再試行結果を報告します。[報告文案](../experiments/f002-input-mount-failure-review-20261007.json)のsupport_messageを利用できます。assistantは再実行・アップロード・報告の送信を行っていません。

<a id="今行う08の診断と09の提出準備"></a>

## 履歴：08の診断と09の提出準備

以下は旧07の提出失敗直後に準備した手順です。08は256件完走済み、09は起動前接続が再発したため、現在は[10・3 Inputs](#head接続の再発後は10を使う)を使います。

提出ref56897486、Version2/scriptVersionId355923367は、公式SDKの13:04 JST確認でCOMPLETE・errorDescriptionあり・Publicなしでした。COMPLETEは処理終了を表し、この場合は採点成功ではありません。11:16の提出から確認まで約1時間48分ですが、queueを含む経過時間で、隠し実行時間ではありません。公開3検査の保存ログには例外がなく、private tracebackは取得できていません。[今回の失敗・対処・検証記録](../experiments/f002-hidden-failure-review-20261007.json)

Kaggleは提出後に隠しデータでNotebookを再実行します。`Notebook Threw Exception`は未処理例外で、timeoutとは別の分類です。同じ一般エラーでもp003と同じ原因とは限りません。隠しログが通常のOutput取得で見られるとは約束できません。[公式説明](https://www.kaggle.com/code-competition-debugging)

優先して調べる候補は、圧縮DICOMのdecoder不足、PixelSpacing等のデータ依存条件、CPU/GPU・ディスク等の資源です。元コードにtestを3件固定する処理やReport必須は見つかっていません。**原因はいずれも未確定**で、9時間超過を前提に再学習や同じVersion2の再提出をしません。圧縮画素には形式に対応した追加libraryが必要です。[pydicomの公式対応表](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html)

今回の新しいファイルは以下です。元07、以前のInput、f002 checkpoint、16個の学習・前処理moduleは保持しています。

| ファイル | 用途 |
| --- | --- |
| [decoder用ZIP](../artifacts/kaggle/dicom-decoders-py313-v2-20261007/rsna-dicom-decoders-py313-v1.zip) | 新規private Datasetにする約5.33MB。最新はディレクトリ名がv2のもの |
| [08_profile_f002_serving_256.ipynb](../notebooks/public/08_profile_f002_serving_256.ipynb) | 固定weakリストから256検査を診断。学習せず、提出CSVも作らない |
| [09_submit_f002_streaming.ipynb](../notebooks/public/09_submit_f002_streaming.ipynb) | test全件を検査ごとに予測し、最後に提出CSVを検査する候補 |
| [今回の梱包記録](../artifacts/kaggle/f002-streaming-v3-20261007/package.json) | Notebook/runtime/hashと未実行状態。最新のNotebook梱包はv3 |

1. KaggleのDatasetsで**新しいprivate Dataset `rsna-dicom-decoders-py313-v1`**を作り、上のdecoder用ZIPをアップロードします。既存encoder/headの再アップロードは不要です。Inputの直下に`decoder-manifest.json`、`wheels/`、`phantoms/`が見える配置にします。ZIP一つだけが残る場合は、ZIPの中身を展開してその構成でアップロードします。
2. **新しいprivate Notebook**へ08をImportします。Inputsは、Competition `rsna-knee-abnormality-detection`、既存`rsna-frozen-feature-inputs-v1`、既存`rsna-f002-head-fold0-v1`、新規`rsna-dicom-decoders-py313-v1`の**4つ**です。既存二つはVersion2で使った内容を維持します。GPU **T4**、Internet **OFF**、新しいkernelで**Save Version → Save & Run All**を実行します。
3. Outputの**`F002_STREAMING_SUMMARY.json`**を確認します。成功条件は`status=profile_complete_not_for_submission`、`phase=complete`、`processed=studies=256`、`receipt.valid=true`、`skipped_series=fallback_predictions=0`です。`decoder_install.synthetic_pixel_checks`の4項目がすべて`default_and_pylibjpeg_exact_pixels=true`、`decoders`の全`available=true`も確認します。別候補decoderのmissing_dependenciesが残っていても、利用可能pluginがあれば全候補の導入は不要です。**08は提出しません。**
4. 08が完走した後、**別の新しいprivate Notebook**へ09をImportし、同じ4 Inputs・T4・Internet OFFでSave & Run Allします。test件数は動的に読み取り、公開実行と隠し再実行で全対象を処理します。summaryの`status=passed`、`phase=complete`、`processed=studies`、`receipt.valid=true`を確認します。
5. 09の保存実行が成功した場合、その**保存版の`submission.csv`を手動提出**します。Notebookも提出先も08と混同せず、提出ref/Version/scoreを記録します。隠し成功と自己Publicはこの後に初めて確認できます。

08/09で例外になったら提出へ進まず、rootのsummaryと`f002-streaming-diagnostics/events.jsonl`で最後のphase・error/traceback・対象系列・圧縮形式・slice数・matrix・PixelSpacing・CPU/GPU/diskを確認します。例：decoder段階なら依存／runtime、`decode_and_preprocess_series`ならDICOM／geometry、`encode_series`ならencoder／資源、`validate_csv`なら提出契約を調べます。OSによる強制終了ではtracebackが保存されず、最後のイベントだけになる場合があります。

09は**特徴抽出→head予測→特徴配列を解放**し、検査ごとのNPZ保存と全件分へ膨らむmetadataの再書込みを除きました。失敗画像をskipしたり仮の確率で埋めたりせず、全件完了後だけ最終CSVを作ります。元経路もencoderは逐次処理なので、今回だけでencoderのピークメモリや隠し例外が直るとは断定しません。

decoder ZIPは保存Version2で確認した**Linux x86_64 / Python3.13 / NumPy2.1.3 / pydicom3.0.2**用です。PyPI公式wheelのhashとlicenseを保持し、NumPy/torch/pydicomを変更しないoffline導入にしました。Python版が変われば明示的に停止します。当時の準備ではWindowsの人工圧縮4形式の画素一致のみでしたが、その後08 Version2でKaggle Linuxの導入・人工画素一致・実MRI256件完走を確認しました。実圧縮MRIの全条件・旧経路との実画素/特徴同等性・隠し完走は未確認です。人工GPU17検査の実checkpoint比較の最大確率差は1.1921e-7、関連21 unittestも成功した履歴を保持します。

## 採点待ちに行うf003の比較

これは採点待ちに提案した比較で、**現在はユーザーの学習と結果監査まで完了**しています。f002との変更は上限12→24 epochだけ、入力/source/環境/seed/固定fold/教師は一致し、最初の12 epochの予測CSVは両foldともbyte同一でした。48 epoch分のCSV、BCE選択、best/last checkpoint等136項目を確認し、goldを学習・選択・評価に使っていません。[完成監査](../experiments/f002-f003-fold01-review-20261007.json)、[当時の準備計画](../experiments/f003-attention24-plan-20261007.json)

| fold | BCE選択epoch：f002→f003 | weak BCE：f002→f003 | weak AUC12：f002→f003 | 補助11平均：f002→f003 |
| --- | --- | --- | --- | --- |
| 0 | 11→15 | 0.319130→0.316232 | 0.813290→0.823060 | 0.811771→0.820611 |
| 1 | 12→12 | 0.317469→0.317469 | 0.815430→0.815430 | 0.808136→0.808136 |

head fittingはfold0 355.92秒、fold1 358.13秒（cache事前検査等を除く）。fold1のselected予測CSVとmodel tensorは元f002と同一ですが、config等のmetadataが違うためptファイル全体のhash同一とは言いません。fold0補助11平均差+0.008840のgroup paired bootstrap95%区間は[+0.001517,+0.016844]、3,000抽出すべて定義可能。ただし同じ検証foldでのcheckpoint選択・患者独立性・Public一般化を保証しません。12平均は陰性1のSynovitisにより1,061抽出未定義で、全体の頑健性に使いません。[bootstrap](../experiments/f002-f003-fold0-bootstrap-20261007.json)

24 epoch時点のtrain BCEは両fold約0.217～0.219まで低下し、weak BCEは0.3412/0.3428へ悪化しました。順位AUCの上昇と確率損失の悪化は両立し、過信・過学習と整合する傾向ですが、教師誤りの証明ではありません。**両foldで延長の利得が揃わないので、さらに長く学習したり最大AUCのepochへ後付け変更したりせず、09はf002 fold0 epoch11を維持**します。実施済みf003を同じoutputへ再実行しません。

## 次の学習候補f004と教師監査

[f004設定](../configs/experiments/f004-dinov2-attention24-dropout40.json)は完成f003に対して**dropoutだけ0.2→0.4**に変更し、24 epoch・特徴・教師・LR・seed・fold・BCE選択を固定して両foldの実学習を完了しました。事前の[run・input/source/環境・比較条件](../experiments/f004-dropout40-plan-20261007.json)を照合し、新規runへ保存。head fitting計767.04秒・CLI全体計796.26秒、事前検査/解析は別です。p005の学習・重みではなく独自f002/f003系の比較なので、提出例外の修正やApexへの追加を意味しません。

[完成比較](../experiments/f003-f004-fold01-review-20261008.json)：fold0 BCE0.316232→0.315904/AUC12 0.823060→0.815188、fold1 BCE0.317469→0.320704/AUC12 0.815430→0.814692。補助11差は−0.002223/−0.000543、3000回group bootstrapの95%区間は両方0をまたぎ、事前採用条件を満たさないので差し替えません。PF OAは点改善したものの、MCLはfold間で方向が異なり、全体を置き換える根拠にはしません。236項目で96epochのCSV/metrics/BCE/更新数、4run×best/lastの計8 CPU checkpoint、固定入力/source/過去run不変を確認。rootの選択4CSV独立再計算も一致しました。

以下のコマンドは今回の実行履歴です。fold0・1の同名runは完成済みで、繰り返し実行しません。

```powershell
.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest data/manifests/v1-vmohitrao-research `
  --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json `
  --cache data/features/f001-weak-sv355640726 `
  --config configs/experiments/f004-dinov2-attention24-dropout40.json `
  --fold 0 `
  --output artifacts/runs/f004-dinov2-attention24-dropout40-fold0

.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest data/manifests/v1-vmohitrao-research `
  --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json `
  --cache data/features/f001-weak-sv355640726 `
  --config configs/experiments/f004-dinov2-attention24-dropout40.json `
  --fold 1 `
  --output artifacts/runs/f004-dinov2-attention24-dropout40-fold1
```

両[fold0 run.json](../artifacts/runs/f004-dinov2-attention24-dropout40-fold0/run.json)・[fold1 run.json](../artifacts/runs/f004-dinov2-attention24-dropout40-fold1/run.json)とBCE選択予測を保存しています。最大AUC epochへの選び直しはせず、元教師のまま評価しました。goldを学習/選択せず、f002提出head/p005保存提出版も保持しています。

教師側は[17セルの二次レビュー](label-review.md#17セルの二次レビュー結果10月8日)まで進み、6セルの原文状態は既存0/欠損と整合、11未確定、教師訂正の採用0でした。程度/範囲/部位/言語の根拠不足を埋める教師版や教師だけの学習は作りません。goldと保護fold0/1の連結groupを読む前に除外し、欠損→0やEffusion→Synovitisを行っていません。

新encoderやpatch特徴は別実験で、現cacheのCLS特徴からは復元できず新たなKaggle画像処理と推論予算が必要です。まず提出経路の成功と教師の根拠を確認します。f002とp002の混合も補完性・追加時間を評価してから行います。fold0/1の検証予測を別foldのheadと平均すると相手が当該検査を学習済みなのでOOF改善の根拠にはできません。p003は[別の例外診断](p003-speed.md)として保持します。

## 07の完走確認と次の提出

以下はVersion2の手動提出前の確認と操作の履歴です。**その後、Version2は隠し例外となりました。現在は上記10・3 Inputsが次の手順で、Version2を再提出しません。**

`rsraki/notebookdd86797040` Version2（scriptVersionId355923367）は公式SDKでCOMPLETE・GPU有効・Internet OFF。配布07と全source cellが一致し、添付summaryは保存Outputとbyte同一でした。公開test3検査・9系列、cache完了、UID/12所見順/確率/CSV hash/学習時checkpointと特徴契約が一致し、19項目の監査が成功しています。[完走監査](../experiments/f002-visible-inference-review-20261007.json)

| 確認項目 | 結果 |
| --- | --- |
| Summary | status=passed、phase=complete |
| 特徴の件数 | studies=cached_studies=3、cache_complete=true |
| 提出CSV | 3検査×12所見、receipt.valid=true、実CSV hash一致 |
| 使用GPU | Tesla T4 |
| 特徴抽出 | 31.25秒 |
| headとCSV検査 | 0.39秒 |
| 推論セル全体 | 31.79秒 |

31.79秒は前段import/setupやKaggle起動/待ちを含まない推論セルの計測です。公開3件の完走は隠しtest時間/完走やPublic0.943を保証しません。今回の目的はこの単独モデルの自己Publicを測り、後でp002との補完性を検討することです。

1. [対象Notebook](https://www.kaggle.com/code/rsraki/notebookdd86797040)を開き、完走した**Version2**を表示します。コード/モデル/Inputsを変えず、この保存版を使います。
2. **Submit to Competition**を選び、`rsna-knee-abnormality-detection`へVersion2の**`submission.csv`**を手動提出します。提出説明は`f002 attention single / fold0 epoch11`など、p002/p003と識別できる名前にします。JSONや特徴NPZは提出ファイルではありません。
3. 提出一覧で結果が確定するのを待ちます。自己Publicが出たら**スコア、提出ref、保存版番号**を渡してください。例外の場合は画面の正確なエラー文面を渡します。保存時COMPLETEと、隠し再実行/採点の成功を区別します。
4. p002 Public0.937を保持し、f002単独の結果と時間を確認してから混合の効果・追加時間を検討します。追加fold混合、所見別係数、教師/encoder変更、epoch延長は別の比較として扱います。06/ローカル学習/07可視実行を今繰り返す必要はありません。

直前のVersion1はGPU無効のため最初のCUDA guardで止まり、summary/CSVを作る前にOutput0件となっていました。Version2ではGPU有効・Internet OFFでその可視停止を解消しました。[失敗版の記録](../experiments/f002-saved-run-failure-review-20261007.json)

## fold1の結果と採用判断

両方status complete、12 epoch、学習3,316/検証891、各epoch208更新、RTX4090、seed20261002。特徴/教師/実装/環境は一致し、集約方式だけを変えた比較として全24 epoch予測とbest/lastを照合しました。両方ともBCE最良epoch12を採用し、goldを学習/選択/評価へ使っていません。[121項目の監査](../experiments/f001-f002-fold1-review-20261006.json)

| fold1の指標 | f001：平均集約 | f002：Attention |
| --- | ---: | ---: |
| 採用epoch（BCE選択） | 12 | 12 |
| weak BCE：低いほど良い | 0.334235 | 0.317469 |
| weak 12所見平均AUC | 0.792571 | 0.815430 |
| 補助11平均AUC：Synovitis除外 | 0.783086 | 0.808136 |
| head学習時間 | 3分10.7秒 | 3分18.9秒 |

fold0に続き9/12所見で改善。fold1のSynovitisは陽性97/陰性25で、AUC0.896907→0.895670とほぼ同じです。今回はその所見に依存せず、ACLは0.711496→0.796678、内側半月板は0.707787→0.777750、Baker'sは0.811617→0.889877と改善しました。両foldで8所見の改善が続きます。

MCLは陽性16/陰性700、AUC0.691964→0.664554、PF OAは陽性78/陰性417、0.796471→0.772121と低下し、両fold共通の弱点です。少数陽性のMCLに所見別係数を合わせる変更は加えません。固定selected予測の861 supplied groupsを3,000回paired bootstrapした11平均差+0.025050の95%区間は[+0.012249,+0.037048]、12平均差の区間も[+0.010083,+0.035736]で正、全抽出が定義可能でした。患者独立性・checkpoint選択の不確実性・Publicへの一般化を示す区間ではありません。[bootstrap](../experiments/f001-f002-fold1-bootstrap-20261006.json)

**f002を単独推論の候補へ進めます。** epoch12は固定学習期間の末尾なので収束は未確認ですが、追加学習や新しい特徴作成を今の工程には加えません。初回推論は従来のfold0基準に揃え、f002 fold0のBCE選択epoch11を固定します。fold1は改善の安定性確認に使い、異なる検証集合のAUCの大小でcheckpointを選びません。weak AUCは自己Public0.943を予測する値ではなく、採点済みp002 Public0.937を保持します。

## 次に行うf002単独推論の操作

以下は今回のVersion2完走前に準備した操作の履歴です。現在はInput作成・Import・保存実行・手動提出まで完了しています。

今回使うのは以下の**設定済みコピー**です。元の`notebooks/public/07_predict_frozen_features.ipynb`は未設定のテンプレートのまま保持しています。06・ローカル学習の再実行と、汎用encoderの再取得/再アップロードは不要です。

| ローカルのファイル | 用途 |
| --- | --- |
| [rsna-f002-head-fold0-v1.zip](../artifacts/kaggle/f002-fold0-single-v1-20261006/rsna-f002-head-fold0-v1.zip) | 新しいprivate Datasetへアップロードする約0.32MBのhead |
| [07_predict_f002_fold0_single.ipynb](../artifacts/kaggle/f002-fold0-single-v1-20261006/07_predict_f002_fold0_single.ipynb) | KaggleへImportする設定済みNotebook |
| [package.json](../artifacts/kaggle/f002-fold0-single-v1-20261006/package.json) | 固定checkpoint/source/特徴のhash、出所、採用理由、未実施範囲 |

1. Kaggleで**新しいprivate Dataset**を作り、名前を`rsna-f002-head-fold0-v1`にして上記ZIP一つをアップロードします。展開後に直下の`best.pt`・`config.json`・`checkpoint-provenance.json`を確認します。既存encoder Datasetへ追加して版を変える必要はありません。
2. 新しいprivate Notebookを作り、上記の設定済み07をImportします。Inputsに**Competition、既存の`rsraki/rsna-frozen-feature-inputs-v1`、新しい`rsraki/rsna-f002-head-fold0-v1`の三つ**を追加します。4,207件の学習特徴cacheを追加する必要はありません。
3. **T4 GPU・Internet OFF**にし、新しい保存実行で全セルを実行します。T4×2でもこの単独候補が使うのはGPU0だけです。学習は行わず、test MRI→特徴→f002予測を処理します。表示されたInputのmount名が違う場合のみ`INPUT`/`HEAD_INPUT`を実際の名前へ合わせます。codeは既存code配下だけを探索し、Competition全体の再帰探索はしません。
4. 完走したらOutputの**`F002_INFERENCE_SUMMARY.json`**を確認します。`status=passed`、`cache_complete=true`、`studies=cached_studies`、`receipt.valid=true`が必要です。時間は`extract_seconds`・`head_and_validation_seconds`・`total_seconds`に記録します。まずこの小さいJSONをレビューへ渡し、NPZ一式をダウンロードする必要はありません。失敗した場合は末尾のtracebackを渡します。
5. 保存実行の成功・実時間・receiptを確認後、Kaggleの**Submit to Competition**で`submission.csv`を手動提出します。自己Publicと保存Notebook/Input版を記録し、その後にp002との補完性と追加時間を検討します。可視testの完走だけで隠しtestの時間内完走を保証しません。p003の例外修正は別工程です。

梱包したheadは元checkpointとbyte一致し、所見別混合・fold平均・p002混合はありません。Inputパス設定・固定hash検査・phase/時間/失敗summaryを追加しただけで、学習/推論実装は変更していません。人工特徴で既存f002 headのGPU forwardとReportなし提出契約、異なるcheckpointの拒否、抽出例外の再送出を確認しました。実Kaggle MRI推論/隠し時間/Publicはこれから測定します。[準備と検証記録](../experiments/f002-fold0-single-handoff-20261006.json)

## fold0の比較結果と次のfold1確認

以下はfold0レビュー時点の履歴です。fold0/1とも完了済みなので、下記の学習コマンドは再実行せず、現在は上記の10へ進みます。

両方ともRTX4090で12 epochを完走し、学習3,386/検証821、seed・特徴・教師・分割・環境・実装が一致しました。全epochのCSVとbest/last checkpointを照合し、両方ともBCE最良のepoch11を正しく保存しています。[117項目の監査](../experiments/f001-f002-fold0-review-20261006.json)

| 指標 | f001：平均集約 | f002：Attention |
| --- | ---: | ---: |
| 採用epoch（BCE選択） | 11 | 11 |
| weak BCE：低いほど良い | 0.332475 | 0.319130 |
| weak 12所見平均AUC | 0.770865 | 0.813290 |
| 補助11平均AUC：Synovitis除外 | 0.800035 | 0.811771 |
| head学習時間 | 2分44.7秒 | 2分59.9秒 |

9/12所見で改善しましたが、12平均差の74.6%はSynovitisの0.45→0.83によるもので、検証側の陰性は1件しかありません。MCLは0.7550→0.6572と低下し、陽性11件です。補助11平均差+0.011736のpaired group bootstrap 95%区間は[-0.002762,+0.024256]で0を跨ぎます。Attentionは有望な暫定候補として扱い、fold0の12平均差だけで採用を確定しません。[不確実性集計](../experiments/f001-f002-fold0-bootstrap-20261006.json)

短時間で学習できたのは、392px DINOv2の特徴計算をKaggleで済ませ、小さいheadだけを学習しているためです。Kaggleでの画像→特徴作成時間や提出推論時間を含む値ではありません。12 epoch目は両方BCEが11より悪化しているので、まず分割を変えて比較し、epoch延長や教師変更を同時に加えません。weak AUC0.8133は公開スコア0.943と直接比較できません。

次はこのPCのPowerShellで、**同じ二つのconfig・12 epoch・seed・教師・特徴を使い、foldだけを1へ変えて比較**します。fold1は学習3,316/検証891で、Synovitisの陰性25/陽性97があります。分割を変えて改善の方向とMCLの低下が続くか確認します。完全未観測のholdoutや患者独立評価とは呼びません。

```powershell
Set-Location 'C:\Users\Casper4\Python\ueki\shibasaki\pj-RSNA-Knee-Abnormality-Detection'
$featureCache = 'data/features/f001-weak-sv355640726'
$featureManifest = 'data/manifests/v1-vmohitrao-research'
$featureAudit = "$featureManifest/fold-audit-v2.json"

.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest $featureManifest --fold-audit $featureAudit --cache $featureCache `
  --config configs/experiments/f001-dinov2-frozen.json --fold 1 `
  --output artifacts/runs/f001-dinov2-frozen-fold1

.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest $featureManifest --fold-audit $featureAudit --cache $featureCache `
  --config configs/experiments/f002-dinov2-attention.json --fold 1 `
  --output artifacts/runs/f002-dinov2-attention-fold1
```

f001の正常終了後にf002を実行し、各fold1フォルダのrun.jsonを結果レビューへ渡します。fold0のrunは保持し、06の再実行や特徴の再取得は不要です。fold1比較の確認後に、採用モデルの07推論候補を準備します。p002のPublic0.937を保持し、p003の隠し例外は独自head学習とは別に診断します。

## 全件の取得・検証完了と今実行する操作

以下は全件取得直後のfold0学習手順の履歴です。fold0/1は完了済みで、現在は上記の10を行います。

`rsraki/notebookbd6efc1cf7` Version 2（出力scriptVersionId `355640726`）は、PILOT=Falseの全件特徴作成を完了しました。4,207検査・12,621系列・386,196中心、特徴作成3時間29分8秒、NPZ計281.6MBです。保存版sourceは既存の梱包06からPILOTだけを変更しており、元CSV・encoder出所・入力/実装hash・固定weak/foldと一致しました。[全件監査](../experiments/f001-full-feature-review-20261006.json)

Outputの全NPZ・studies.csv・export.jsonをこのPCの`data/features/f001-weak-sv355640726/`へ取得済みです。全ファイルのサイズ・SHA-256、各NPZのshape/dtype/finite値・mask・位置・plane・flag・fingerprintを検証しました。**06の再実行・再ダウンロードは不要です。ここまでが特徴作成で、モデルのhead学習はまだです。** 05は隠し再実行の例外が続いたため同じ版の再提出を保留し、採点済みp002 0.937を保持します。

次の操作はこのPCのPowerShellで行います。まず下記のf001を実行し、正常終了を確認してからf002を実行します。両方とも12 epoch、fold0の学習3,386/検証821、同じseed・教師・特徴・BCE選択です。変えるのは集約方式だけです。MRIの再デコードやencoderの再学習は行いません。

```powershell
Set-Location 'C:\Users\Casper4\Python\ueki\shibasaki\pj-RSNA-Knee-Abnormality-Detection'
$featureCache = 'data/features/f001-weak-sv355640726'
$featureManifest = 'data/manifests/v1-vmohitrao-research'
$featureAudit = "$featureManifest/fold-audit-v2.json"

.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest $featureManifest --fold-audit $featureAudit --cache $featureCache `
  --config configs/experiments/f001-dinov2-frozen.json --fold 0 `
  --output artifacts/runs/f001-dinov2-frozen-fold0

.\.venv\Scripts\python.exe scripts/frozen_features.py train `
  --manifest $featureManifest --fold-audit $featureAudit --cache $featureCache `
  --config configs/experiments/f002-dinov2-attention.json --fold 0 `
  --output artifacts/runs/f002-dinov2-attention-fold0
```

1本目がエラーの場合は2本目を始めず、末尾のエラーと作成されたrun.jsonを確認します。どちらも既存runへの上書きを拒否します。途中停止したrunを消して使い回さず、再実行は新しいoutput名にします。今回の二つの既定outputは未作成と確認しました。

完了時は各runの`run.json`が`status=complete`になり、`best.pt`と`weak_valid_predictions.csv`が保存されます。まず二つのrun.jsonをレビューします。選択epoch、weak BCE、12所見別AUCと陽性/陰性/欠損数を比較し、単に最後のepochや最大AUCのepochを採用しません。weak AUCはPublic 0.943と直接比較できません。

その後はfold1で同じf001/f002比較を新しいrunへ保存し、結果を確認してから07の単独推論候補へ進みます。07を今すぐ実行する必要はありません。統合は完走・採点済みp002を先に検討し、p003は例外が解消し採点できた後に検討します。学習時間・精度向上・Publicはこれから測定します。

## 32検査pilotの確認結果と次の操作

以下は全件実行前の履歴です。現在の操作は上記のローカル学習です。

ユーザー提供の`export.json`は`complete=true`・`limited=true`・`requested=32`でした。32検査それぞれでAxial/Sagittal/Coronalの3系列、計96系列・2,953中心を処理し、全系列が物理位置順、PixelSpacingは正の有限値でした。固定seed・fold0の学習側32件、元CSV、config/実装、ローカルInputの出所記録との一致を確認しました。ユーザーはNotebook末尾の画像に明らかな崩れはないと回答しています。[確認記録](../experiments/f001-pilot-review-20261006.json)

特徴作成時間はTesla T4で83.018秒、NPZ合計2.15MB。4,207検査への単純換算は約3時間2分・約283MBです。初期化・重み読込み・初期hash照合・最後の画像表示を含まない計測で、検査の大きさやI/Oによって変わる目安です。NPZ本体のshape/finite値・hashをこの添付だけから再検査したわけではなく、精度評価もありません。

次はpilot保存版を保持して、同じ06とInputで以下に変更し、**新しい保存実行・新しいカーネル**で全セルを実行します。

```python
PILOT = False
FOLD = 0
SHARD_COUNT = 1
SHARD_INDEX = 0
```

本質的な変更は`PILOT=True`から`False`だけです。`FOLD=0`はpilotの選択と分割検査に使う値で、`PILOT=False`では全weak 4,207検査を対象にします。既にDINOv2をimportした対話カーネルで続けて全セルを再実行すると、出所検査が停止させるため、新規保存実行を使います。Inputの再作成やモデルの追加学習は不要です。

完了後は`frozen-weak-v1-shard0/export.json`の`complete=true`・`limited=false`・`requested=4207`を確認し、この新しいJSONをレビューへ渡します。学習にはJSONだけでなく、同フォルダの全NPZ・`studies.csv`・`export.json`が必要です。Output全体を手動で`data/features/f001-weak/`へ配置し、下記「3」のverifyに成功してから4090学習へ進めます。32件pilotは本学習へ流用しません。

## 05の提出待ちと並行して進める場合

05は学習済み公開モデルの提出、こちらは精度改善用モデルの準備です。全weak特徴の取得・検証とf001/f002両fold学習は完了しており、現在は上記の10で固定f002の提出経路を確認します。05は例外終了しており同じ版の再提出は保留します。以下の初回Input準備を繰り返す必要はありません。

初回準備時はencoder Inputが未作成でしたが、その後ユーザーが`artifacts/kaggle/f001-inputs-v1/`へ一括Inputを作り、Kaggleでpilotと全件処理を完了しました。今回のexportはこのローカル梱包記録と一致します。以下の初回手順は、新たに環境を作る場合の説明として残します。現在のユーザーはInput準備や全件処理を繰り返しません。

この準備で0.943超が確定するわけではありません。同一特徴・教師・foldを使って平均集約とAttention集約だけを比較し、独自モデルの弱点と追加効果を測るための経路です。

## 06を開始する具体的な操作

今回は**32検査のpilotを完了して結果を確認するところまで**進めます。以下が一括Inputを使う手順です。後段の個別Input説明を重ねて実行する必要はありません。ローカル梱包にはGPUを使いません。

### A. 公式の2ファイルをDownloadsへ保存する

- [固定版DINOv2ソースZIP](https://github.com/facebookresearch/dinov2/archive/7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip)：`dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip`として保存。
- [公式ViT-S/14の重み](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)：`dinov2_vits14_pretrain.pth`として保存。

ZIPは展開せず、両方を`C:\Users\Casper4\Downloads`へ置きます。ブラウザで`(1)`などが付いた場合は、下記コマンドを実際の名前へ合わせてください。ソースは[公式commit](https://github.com/facebookresearch/dinov2/commit/7764ea0f912e53c92e82eb78a2a1631e92725fc8)、重みは[公式READMEのViT-S/14・registerなし](https://github.com/facebookresearch/dinov2#pretrained-models)を指定しています。

### B. PowerShellで一括Inputを作る

以下はユーザーがローカルで実行するコマンドです。取得済み2ファイル、準備済みコード、既存分割データを読み、ハッシュと出所JSONを作ってZIPへまとめます。ネット通信・モデル読込み・MRI処理・学習・アップロードは行いません。

```powershell
Set-Location 'C:\Users\Casper4\Python\ueki\shibasaki\pj-RSNA-Knee-Abnormality-Detection'
.\.venv\Scripts\python.exe scripts/package_frozen_feature_inputs.py `
  --source-zip 'C:\Users\Casper4\Downloads\dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip' `
  --weights 'C:\Users\Casper4\Downloads\dinov2_vits14_pretrain.pth' `
  --output artifacts/kaggle/f001-inputs-v1
```

成功すると`status: packaged_not_executed`と保存先が表示されます。出力先が存在する場合は上書きせず停止するので、再作成時は`f001-inputs-v2`など新しい名前へ変更します。

| 出力先`artifacts/kaggle/f001-inputs-v1/`の中身 | 用途 |
| --- | --- |
| `rsna-frozen-feature-inputs-v1.zip` | 次のprivate Datasetへ手動アップロード |
| `06_prepare_frozen_features.ipynb` | 今回KaggleへImportする設定済みNotebook |
| `package.json` | 入力・重み・ソース・出力のhashと未実行状態を記録 |

ローカルhashは今回渡したバイト列を固定します。配布元の署名や既知の正解checksumとの照合、実重みの互換性確認ではありません。必ずAの公式リンクから取得した2ファイルを渡します。元ソースのライセンス等はZIP内へ保持します。

### C. Kaggleにprivate Datasetを一つ作る

Dataset名を`rsna-frozen-feature-inputs-v1`にし、BのZIPをアップロードします。**既存の検査・fold情報を含むためprivateにします。** Datasetのファイル一覧で、ZIPが展開されて直下に`code/`・`encoder/`・`manifest/`があることを確認します。ZIP一個のまま添付された場合は、展開したこの3フォルダを同じ構造でアップロードします。

### D. Bで生成した06を別NotebookへImportする

Bの出力先にある`06_prepare_frozen_features.ipynb`を新しいprivate NotebookへImportします。今回のZIPに合わせたパスを持つコピーで、元の`notebooks/public/06_prepare_frozen_features.ipynb`はひな形のまま保持します。

Inputへ追加するのは**公式CompetitionとCのprivate Datasetの二つ**です。T4 GPU・Internet OFFに設定します。準備コピーの先頭code cellは、ユーザー`rsraki`のDatasetパスと旧形式の短いパスに対応します。Dataset名やユーザー名が異なる場合は、表示される実際のInputパスを`INPUT`へ指定します。`INPUT`は`code/`・`encoder/`・`manifest/`が直下にあるディレクトリです。

最初は`PILOT=True`・`FOLD=0`・`SHARD_COUNT=1`・`SHARD_INDEX=0`を維持し、新しいカーネルで保存して全セルを実行します。提出用05とは別のNotebookです。

### E. 32検査の結果を確認する

特徴作成セルが`complete=True`・`requested=32`を表示し、最後の画質確認セルまで完走することを確認します。画像の向き・縦横比・切り出し・系列選択も見ます。`frozen-pilot-v1/export.json`が主な確認ファイルです。

まずこの`export.json`を結果レビューへ渡します。エラーの場合は末尾のtracebackを渡します。32件の時間・coverageと画質を確認してから、全件特徴作成と4090学習へ進みます。ここでは提出CSVは作りません。

## 今回の実装

元DICOMをKaggleで読み、物理座標順に並べ、PixelSpacingによる縦横比を保って392pxへletterboxします。既存の決定的なfluid-sensitive優先選択で最大3系列、各系列は最大48中心、隣接3スライスをRGBへ並べます。48枚以下なら全中心、超える場合は全範囲から等間隔で中心を選びます。全系列のp1/p99正規化、MONOCHROME1反転、RescaleSlope/Intercept、ImageNet mean/stdを学習用特徴抽出と提出推論で共有します。

汎用DINOv2 ViT-S/14を凍結し、各中心から384次元のCLS特徴だけをFP16で保存します。最大3×48×384×2 bytesで1検査110,592 bytes、4,207検査の特徴値のみなら約0.465GBです。位置・mask・出所等は別で、これは圧縮後実測容量ではありません。392はpatch14の倍数です。公式は汎用の凍結特徴利用と、より大きな14倍数画像を説明しています。[DINOv2公式](https://github.com/facebookresearch/dinov2)、[モデルカード](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)。

ローカルGPUではplane、fluid/fat suppression、正規化slice位置を加えた軽量headだけを学習します。[f001](../configs/experiments/f001-dinov2-frozen.json)は系列内・系列間の平均、[f002](../configs/experiments/f002-dinov2-attention.json)は集約だけを所見別Attentionへ変更します。画像、encoder、教師、fold、seed、12 epoch、optimizer、**weak BCEによるcheckpoint選択**を固定します。所見別AUC・supportと12所見macro AUCも全epoch保存し、陰性が極少の所見を含むAUCだけで自由な探索をしません。

goldは学習・選択・評価とも使用しません。既存`audit_folds`の成功JSONと現manifest三ファイルのhash一致を必須にし、gold/group混入を拒否します。汎用の事前学習画像と競技画像の重複を完全に監査できたという主張や、患者独立性の主張はしません。

## 1. 手動で汎用encoderのInputを準備する

用意するものは公式DINOv2 sourceの固定commit、公式の`dinov2_vits14_pretrain.pth`、[出所ひな形](../configs/frozen-encoder-provenance.example.json)を埋めた`encoder-provenance.json`です。今回これらをダウンロードしていません。公開競技用にfine-tuneされた重み、goldで選んだ重みはこの経路に使いません。

ローカルに手動で用意した後、hashを表示します。認証情報・他プロジェクトの資産は読みません。

```powershell
.\.venv\Scripts\python.exe scripts/frozen_features.py inspect-encoder --repo data/pretrained/dinov2-source --weights data/pretrained/dinov2_vits14_pretrain.pth
```

出力の二つのSHA-256、取得元の正確なcommit、確認日、利用条件、既知の競技データ露出に関する監査をprovenanceへ記入します。LVD-142Mの全画像との重複は未確認と正確に残してください。`TODO`、不完全なhash、競技fine-tune/gold選択の申告は実行時に拒否します。コードではネット取得を呼ばず、hash済のlocal sourceから`pretrained=False`でモデルを作り、local state_dictをstrictに読み込みます。既importのDINOv2を使い回さないため**新しいNotebook/kernel**を使います。

## 2. Kaggleで32検査の品質・時間を確認する

[06_prepare_frozen_features.ipynb](../notebooks/public/06_prepare_frozen_features.ipynb)を新しいprivate NotebookへImportします。T4 GPU・Internet OFFに設定し、以下を手動でInputへ追加します。

- 公式Competition。
- 汎用encoder source、重み、記入済provenance。
- `weak.csv`、`gold.csv`、`manifest.json`、`fold-audit-v2.json`を含む既存`v1-vmohitrao-research` manifest。privateのまま扱います。
- この実装のコードbundle。次のコマンドで新しいディレクトリへ作成できます。

```powershell
.\.venv\Scripts\python.exe scripts/build_frozen_feature_handoff.py --output artifacts/kaggle/frozen-features-next
```

zip名は`rsna-frozen-features-code.zip`です。既存directoryへは上書きしません。生成物に実データ・重み・認証情報は含めず、自動アップロードしません。setupセルの`CODE_INPUT`にはこのコードだけが入ったInputディレクトリを指定します。Competition全体を探索する処理はありません。

Notebookの`ENCODER_REPO`、`ENCODER_WEIGHTS`、`PROVENANCE`、`MANIFEST`を実際のInputパスへ変更します。最初は`PILOT=True`を維持します。fold 0の**学習partitionだけ**からhash順で32検査を選び、goldと検証partitionを品質調整に使いません。Outputの`export.json`と図で系列選択、元の細部、物理slice順、縦横比、端の構造、coverageを確認します。PixelSpacing欠損・不整合、向き混在、空系列は停止し、架空のspacingや定数予測で補いません。

全件の時間は32件の線形換算だけで保証しません。大きい検査・series数・GPU差を考慮した余裕を取り、全件が一回の制限に近い場合は次のshardを使います。古い192px画像を392pxに引き伸ばす経路ではありません。

## 3. 全weak特徴を作成・手動転送・検証する

QC後、**別の保存実行**で`PILOT=False`にします。一回で収まるなら`SHARD_COUNT=1`、時間が長ければ例えば`SHARD_COUNT=4`にし、`SHARD_INDEX=0,1,2,3`を別実行します。元のweak UIDをソートしてstrideで分けるため、fold割当を変えません。同じconfig・source・汎用重みを全shardで固定してください。pilotは学習に受理されません。

Outputを手動で`data/features/f001-shard0`などへ取得します。4分割した場合の結合はローカルのファイル検証・コピーだけで、GPU推論を再実行しません。

```powershell
.\.venv\Scripts\python.exe scripts/frozen_features.py merge --cache data/features/f001-shard0 --cache data/features/f001-shard1 --cache data/features/f001-shard2 --cache data/features/f001-shard3 --studies-csv data/manifests/v1-vmohitrao-research/weak.csv --config configs/experiments/f001-dinov2-frozen.json --output data/features/f001-weak
.\.venv\Scripts\python.exe scripts/frozen_features.py verify --cache data/features/f001-weak --config configs/experiments/f001-dinov2-frozen.json
```

全件を1回で作った場合は取得先を`data/features/f001-weak`とし、verifyだけ行います。完了flag、UID集合、各NPZのサイズ・SHA256・shape・finite値、元CSV、前処理実装とencoder資産のhashを検査します。shardの重複、欠落、別Input混入は拒否します。転送後の検査が成功してから学習へ進みます。

## 4. RTX 4090でheadのみを学習する

以下は学習前に準備した汎用コマンドの履歴です。今回のf001/f002 fold0/1は完了済みで、既存runへ再実行しません。

```powershell
.\.venv\Scripts\python.exe scripts/frozen_features.py train --manifest data/manifests/v1-vmohitrao-research --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json --cache data/features/f001-weak-sv355640726 --config configs/experiments/f001-dinov2-frozen.json --fold 0 --output artifacts/runs/f001-dinov2-frozen-fold0
.\.venv\Scripts\python.exe scripts/frozen_features.py train --manifest data/manifests/v1-vmohitrao-research --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json --cache data/features/f001-weak-sv355640726 --config configs/experiments/f002-dinov2-attention.json --fold 0 --output artifacts/runs/f002-dinov2-attention-fold0
```

CUDAがなければ、入出力を読む前に停止します。欠損ラベルはlossからmask除外します。`run.json`にconfig、seed、fold、理由、原入力/manifest/fold監査/export/source hash、環境、選択規則、全epoch指標を保存します。`best.pt`、`last.pt`、各epochのweak予測、選択された`weak_valid_predictions.csv`を新しいrun内へ保存します。checkpointは推論用で、学習途中からのresumeには対応しません。途中停止時のrunは完成実験として扱わず、新しいrunを作成してください。

f001/f002で同じ特徴を再利用します。教師改善もUID/group/foldが同一なら同じcacheを使えますが、変更教師から作った新manifestと、それに一致する**新しい**fold監査が必要です。元のfeature抽出に使った教師CSV hashは履歴として残し、ラベル値とfeature identityを分離しています。gold、固定group、欠損扱いは緩めません。

採用候補はまず別foldでも同じpaired比較を行います。自己モデルのweak評価と、公開モデルが同じ検査へ学習露出していた可能性のある混合評価を分けます。採点済みp002を先の統合相手とし、p003は採点成功後に扱います。5%/10%等の少数の事前固定した統合は、両方の同一UID予測と実測時間が揃った次段階で行います。この実装は自動混合しません。

## 5. Kaggleで単独の提出候補を作る

[07_predict_frozen_features.ipynb](../notebooks/public/07_predict_frozen_features.ipynb)を新しいprivate NotebookへImportします。手順2と同一のcode bundle・汎用encoder・provenance、学習後の`best.pt`を手動Inputに追加し、`CHECKPOINT`パスを指定します。学習manifestやReportは不要です。

同じDICOM→特徴変換をtestへ適用してからheadで予測し、`submission.csv`と`submission.receipt.json`を生成します。学習時と異なるencoder、前処理実装、head実装、12所見順、UID集合は拒否します。sourceを変更した場合に、同じ形状だから互換と黙認しません。保存実行の完走・時間・receiptを確認後に手動提出します。

## 確認済みと残り

人工DICOMの物理順・非正方PixelSpacingとletterbox、人工encoderの特徴化、両poolingのGPU forward/backward、人工特徴4検査の1 epoch学習、checkpoint保存とReportなし提出契約を確認しました。全件shardの結合と重複/欠落拒否、古いfold監査・payload破損・head実装hash不一致の拒否も確認しています。

32検査pilotに続き、全4,207検査の保存Version 2 source・実行ログ・出力容量と、全NPZ本体の検証を完了しました。Notebook版・出力版ID・Input参照・内容hashは確認済みですが、過去のDataset版番号は未確定です。f001/f002のfold0/1実head学習・weak指標・保存予測/checkpointと、07のVersion2での公開test3件推論/提出CSVを確認済みで、独自Publicと隠しtest完走は未測定です。一般画像のCLSのみでは細い局所病変を落とす可能性があり、今回の集約比較から392px化やencoder変更の効果を分離して一般化しません。空間特徴やfine-tuningの追加は単独推論の結果と別の固定比較として判断します。
