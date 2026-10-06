# 392px凍結DINOv2から独自headを学習する

更新日：2026年10月6日 JST。**コードと人工データでの経路確認を完了した未学習の候補**です。自己Public 0.943や、それを超える性能はまだありません。p003の公開競技checkpointを再学習する経路ではなく、汎用DINOv2と既存のweakラベル・固定group foldから自分のheadを作ります。

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

以下は**今後ユーザーが実行する実学習コマンド**です。今回は実行していません。

```powershell
.\.venv\Scripts\python.exe scripts/frozen_features.py train --manifest data/manifests/v1-vmohitrao-research --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json --cache data/features/f001-weak --config configs/experiments/f001-dinov2-frozen.json --fold 0 --output artifacts/runs/f001-dinov2-frozen-fold0
.\.venv\Scripts\python.exe scripts/frozen_features.py train --manifest data/manifests/v1-vmohitrao-research --fold-audit data/manifests/v1-vmohitrao-research/fold-audit-v2.json --cache data/features/f001-weak --config configs/experiments/f002-dinov2-attention.json --fold 0 --output artifacts/runs/f002-dinov2-attention-fold0
```

CUDAがなければ、入出力を読む前に停止します。欠損ラベルはlossからmask除外します。`run.json`にconfig、seed、fold、理由、原入力/manifest/fold監査/export/source hash、環境、選択規則、全epoch指標を保存します。`best.pt`、`last.pt`、各epochのweak予測、選択された`weak_valid_predictions.csv`を新しいrun内へ保存します。checkpointは推論用で、学習途中からのresumeには対応しません。途中停止時のrunは完成実験として扱わず、新しいrunを作成してください。

f001/f002で同じ特徴を再利用します。教師改善もUID/group/foldが同一なら同じcacheを使えますが、変更教師から作った新manifestと、それに一致する**新しい**fold監査が必要です。元のfeature抽出に使った教師CSV hashは履歴として残し、ラベル値とfeature identityを分離しています。gold、固定group、欠損扱いは緩めません。

採用候補はまず別foldでも同じpaired比較を行います。自己モデルのweak評価と、公開モデルが同じ検査へ学習露出していた可能性のあるp003混合評価を分けます。p003との5%/10%等の少数の事前固定した統合は、両方の同一UID予測と実測時間が揃った次段階で行います。この実装は自動混合しません。

## 5. Kaggleで単独の提出候補を作る

[07_predict_frozen_features.ipynb](../notebooks/public/07_predict_frozen_features.ipynb)を新しいprivate NotebookへImportします。手順2と同一のcode bundle・汎用encoder・provenance、学習後の`best.pt`を手動Inputに追加し、`CHECKPOINT`パスを指定します。学習manifestやReportは不要です。

同じDICOM→特徴変換をtestへ適用してからheadで予測し、`submission.csv`と`submission.receipt.json`を生成します。学習時と異なるencoder、前処理実装、head実装、12所見順、UID集合は拒否します。sourceを変更した場合に、同じ形状だから互換と黙認しません。保存実行の完走・時間・receiptを確認後に手動提出します。

## 確認済みと残り

人工DICOMの物理順・非正方PixelSpacingとletterbox、人工encoderの特徴化、両poolingのGPU forward/backward、人工特徴4検査の1 epoch学習、checkpoint保存とReportなし提出契約を確認しました。全件shardの結合と重複/欠落拒否、古いfold監査・payload破損・head実装hash不一致の拒否も確認しています。

公式DINOv2の実source/実checkpoint読み込み、圧縮された実MRIのdecode、Input版、実cacheサイズ、Kaggle T4速度、実学習、weak指標とPublicは未確認です。一般画像のCLSのみでは細い局所病変を落とす可能性があり、392px化やAttentionが精度向上を保証するものではありません。まずこの一系統の実測を取り、空間特徴やfine-tuningを追加する判断はその後にします。
