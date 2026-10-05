# Public 0.937からの次の比較

更新日：2026年10月5日 JST。

**次の提出候補はp003です。** haideptryのPublic 0.943・V2に対応する公開ソースを直接取得し、追加CoAtNet 4系統の完走と統合を検査するNotebookを作りました。自己Publicは未測定です。0.945まで+0.008、0.955まで+0.018が必要であり、この準備だけで到達したとは判断しません。

## 今回確認したこと

- p002は公式APIで **COMPLETE / Public 0.937**。提出ref `56840796`、保存版1、scriptVersionId `355324155`。保存コード16セルは前回のhandoffと一致しました。[監査記録](../experiments/p002-scored-20261005.json)
- ユーザー報告の約8時間と、可視3検査のguard計測164.699秒を分けて記録しました。隠しtestの正味実行時間はAPIで取得できていません。
- p002は画像cacheと一部の予測集約を共有していますが、DINO encoderの前半計算は各memberで繰り返します。p003は前半6 blockの重みhashと各memberの人工forward一致を実行時に検査して共有します。実MRIでp002と同一予測となることを今回証明したわけではありません。
- [haideptry V2の公開ページ](https://www.kaggle.com/code/haideptry/rsna-knee-speedy-raptors-coatnet-d4-0943/comments)にはPublic／Best 0.943・V2が表示されています。表示4分27秒は本番全testの時間として使いません。作者の歴史的Input版や採点CSVとの同一性は未確定です。

添付のGitHub保存コピーをそのまま採用せず、KaggleのV2を取得しました。p003は追加モデル以外にも入力範囲、A5の配分、Raptorの読み方、所見別係数が違う**候補一式の比較**です。改善しても一つの要因の効果とは解釈しません。[差分・利用条件・限界](research/p003-candidate-review-20261005.json)

## 用意したNotebook

|ファイル|用途|出力／扱い|
|---|---|---|
|[成功版p002](../notebooks/public/01_submit_p002_scored.ipynb)|0.937の復帰先|今回再提出する必要はありません|
|[p003提出候補](../notebooks/public/02_submit_p003.ipynb)|V2の全構成で次のPublic比較|`P003_READY.json`と`submission.csv`|
|[A5精度診断](../notebooks/public/03_a5_precision_probe.ipynb)|p002のBF16／FP16／FP32比較|ラベル不使用、提出CSVを作りません|
|[p003・64検査の時間診断](../notebooks/public/04_profile_p003_64.ipynb)|本番に近い逐次CoAt分岐の動作と時間内訳|train画像による診断。AUC／OOFではなく、提出用ではありません|

まず64検査の時間診断、次にp003提出候補を勧めます。A5精度診断は別セッションで実行でき、p003の係数やBF16を変更せずに進められます。全Notebookは新しいprivate NotebookへImportし、**GPU T4 x2・Internet OFF**で実行します。ローカルへDICOMや新しい重みを取得する必要はありません。

## p003でユーザーが行う操作

1. 上表の64検査診断Notebookを新しいprivate NotebookにImportします。下記14 Inputsを指定版で追加します。既存p002用のTonylicaや半月板専門Inputをそのまま追加しないでください。余分なInputは同名ファイルの誤選択を防ぐため拒否します。
2. `Save Version → Save & Run All`。診断の`P003_PROFILE.json`、各段階の秒数、失敗ログを確認します。選んだUIDは`P003_PROFILE_SELECTION.json`、予測は`profile_predictions.csv`へ保存します。64件なので、元コードの「48件以下ならCoAtを並列」という分岐を抜けます。偏りのない本番時間を保証するサンプルではありません。
3. p003提出候補を別の新しいprivate NotebookにImportし、同じ14 Inputs・T4 x2・Internet OFFで `Save & Run All`。最後の **P003 READY FOR MANUAL SUBMISSION** と `P003_READY.json`、`submission.csv` を確認します。
4. 保存実行が成功した版の `submission.csv` を手動Submitします。失敗時の途中CSVや、p003の各モデル単体CSVは提出しません。採点後のPublic、提出ID、保存版、経過時間を残します。

Kaggleの公式上限は9時間ですが、原版には開始後**8時間の内部予算**もあります。モデル追加のためにこれを延長したり、失敗した系統を削ったりせず、診断で時間の内訳を確認します。[公式実行条件](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

Inputの版選択が難しい場合でも、別版を黙って使用しないでください。期待hashが異なる場合は推論前に停止します。固定公開ファイル148件のhashと、大会CSV4件の実行時hashを記録します。約6.475GBは公開資産の一覧サイズであり、大会データや実行中のcache容量を含みません。詳細は[Input契約](research/p003-input-contract-20261005.json)です。

|Input|指定版|
|---|---|
|[dreaddevelopment/raptor-knee-maxspan](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan/versions/1)|1|
|[dreaddevelopment/raptor-knee-native384](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384/versions/1)|1|
|[dreaddevelopment/raptor-knee-native384dense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense/versions/1)|1|
|[mattiaangeli/knee-mri-fold-weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights/versions/2)|2|
|[mattiaangeli/opencv-python-headless-4120088-x86](https://www.kaggle.com/datasets/mattiaangeli/opencv-python-headless-4120088-x86/versions/1)|1|
|[marwanmath/resnet-50-radimagenet-marwan](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan/versions/1)|1|
|[mattiaangeli/rsna-knee-coat-resgated-ep10-top3](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3/versions/3)|3|
|[mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2/versions/2)|2|
|[mattiaangeli/rsna-knee-coatnet-global96-top3](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3/versions/1)|1|
|[antoinegg1/rsna-knee-e9-radimagenet-heads-v15](https://www.kaggle.com/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15/versions/3)|3|
|[pilkwang/rsna-knee-weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights/versions/1)|1|
|[sofiaanjenje/rsna-knee-e13-train](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e13-train)|1|
|[metaresearch/dinov2/PyTorch/small/1](https://www.kaggle.com/models/metaresearch/dinov2/frameworks/PyTorch/variations/small/versions/1)|1|
|[rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)|competition|

## Add Inputの検索で見つからない場合

名前だけで見つからないDatasetは、Add Inputの検索欄へ`https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan`のようにページURLを貼り付ける方法を使います。まず`/versions/1`等を除いた基本URLで対象を探し、追加時に上表の指定版を確認してください。URLを使った追加は[TU Delftの授業用手順](https://github.com/tudelft/AE4353-Y24/blob/main/kaggle_guide.md#step-3-load-the-dataset-in-kaggle)でも説明されています。今回ユーザーの画面で動作確認したものではありません。

検索対象の種類も確認します。1〜11はDatasets、12はNotebookのOutput、13はModels、14はCompetitionsです。特に`rsna-knee-e13-train`とDINOv2はDatasetだけに絞った検索では対象外です。[Kaggle公式のInput説明](https://www.kaggle.com/docs/notebooks)。今回の配布Notebookの`metadata.kaggle.dataSources`は空であり、Importだけで14 Inputsを自動追加する構成にはしていません。

## p003に加えた検査

元23コードセルの数値処理は保持します。事前の入力hash確認、timm 1.0.22の添付wheelによるオフライン固定、実行順、段階別時間、以下の完了条件を追加しています。

- DINO全20 memberと共通前半の人工forward比較、A5全5 fold、固定Rad校正器、Raptorの3入力準備×全UIDと4 viewを確認します。metadata上の系列不在と、読み込み・推論失敗を分けます。
- Residual（epoch 4/6/8）、Global96（16/23/18）、D4（親＋深さadapter）、Repair-v1（12/7/11）の全4系統を要求します。子処理の失敗、定数埋め、前処理が別方式に戻る`dense-fallback`、確率ファイル不足を拒否します。
- 生確率を全検査のUIDに揃え、「4系統の確率平均→全体順位化→Raptor 0.60／追加CoAt 0.40→所見別の外側統合」を再計算して照合します。rankをGPU・chunkごとに分割しません。CSV保存／再読による微小丸めを区別して、原版と同じ値で最後の順位を照合します。
- 失敗すると、この実行の`submission.csv`を提出不可の拡張子へ退避し、`P003_FAILED.json`に理由を残します。過去の実験は上書きしません。

保存するのは`P003_PREFLIGHT.json`、`P003_TIMINGS.json`、`P003_READY.json`、原版の`diagnostics/`と各CoAt生確率・receipt・子処理ログです。Kaggleが隠し再実行のOutputを公開しない場合、保存実行で見えた情報から隠しtestの内訳を推定しないでください。

## A5診断の読み方

成功版p002と同じ5重み・前処理・GPU0・MICRO=8で、固定64検査の画像を一度だけ準備し、BF16／FP16／FP32を各3回測ります。CUDA同期したforward時間、VRAM、raw確率とlogit、非有限値、順位逆転を保存します。warmup・重み読込・画像準備は別の時間です。

速いという理由だけでFP16へ変更しません。異常値がなくても順位が変わり得ます。A5単体の短縮時間が全体の短縮にどれだけ寄与するかを確認し、その後に**A5の精度だけを変えた別候補**を一つ作ります。ここで保存する出力はA5単体であり、最終p002アンサンブルへの影響はまだ測定していません。

## 0.945、続いて0.955へ進む判断

1. **p003一式の比較**：自身のPublicが0.937を超え、完全構成で時間内に完走した場合に主力へ昇格します。作者0.943を得ても0.945にはまだ0.002あります。改善しなければ成功版p002を維持します。
2. **保存予測で差分を調べる**：追加CoAtの確率平均／順位平均、系統ごとの除外を調べます。ただしtrain上の相関や変化だけで汎化改善と判断せず、goldやPublicを使った大量の所見別係数探索は行いません。比較する変更を先に固定し、一つずつ採点します。
3. **0.955へ向けた学習**：最初の学習候補は、強いencoderを固定し、系列・スライス位置・所見別の小さな残差集約を学習するものです。元の生logitに上限付き補正を加え、補正を0初期化し、epoch 0が元モデルと一致する契約を先に作ります。p002の最終rankを病変確率としてBCEへ入れません。

3は今回の提出Notebookに入っていません。基盤の特徴export・元foldの学習除外UID・checkpoint選択履歴が必要で、監査不明のまま学習すると評価の独立性を失います。次の実装は、(a) Kaggleで元前処理の特徴cacheを作成、(b) hash／UID／系列・位置を固定、(c) 既存group・同一reportを保つweak分割、(d) RTX 4090で補正なし／ありを同じseed・foldで比較、(e) 別fold確認、の順です。公開重みがその検査を既に学習していれば「競技用診断」と記録し、独立OOFにはしません。

追加CoAt 4系統には、作者がgold58でcheckpoint等を選択した履歴があります。今回は凍結済み公開モデルとして利用し、自分のgold学習・checkpoint選択・係数fitは行いません。患者独立性も未確認です。0.955が銀メダル境界であることは今回確認できていません。

## 再生成と今回の検証範囲

成功版と原版は`notebooks/public/`へ固定し、Git除外の調査フォルダがなくても生成できるようにしました。重み・MRIをGitに含めません。

```powershell
.\.venv\Scripts\python.exe scripts/build_public_candidate.py --output-dir artifacts/kaggle/p002-rebuild-new
.\.venv\Scripts\python.exe scripts/build_p003_candidate.py --output-dir artifacts/kaggle/p003-rebuild-new
.\.venv\Scripts\python.exe scripts/build_a5_precision_probe.py --output artifacts/kaggle/a5-probe-new.ipynb
.\.venv\Scripts\python.exe scripts/build_p003_profile.py --output-dir artifacts/kaggle/p003-profile-new
```

上記はNotebookの作成だけで、実データ処理を始めません。実行済みなのは人工データの検査と静的検査です。p003／A5診断／64検査診断の実MRI完走、隠しtest時間、新候補Public、学習による0.955到達は未確認です。検査結果は[validation](validation.md)、凍結元の帰属は[原版情報](../notebooks/public/vendor/haideptry-speedy-v2.provenance.json)に残します。
