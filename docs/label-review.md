# 学習側のレポート根拠監査

更新日2026年10月7日。現行weakラベルを変更する前に、学習側150検査の原文と判定根拠を確認するための手順です。[作成ツール](../scripts/build_label_review_queue.py)はCSVだけを処理します。**実150検査の抽出と重点30セルの一次レビューを完了し、再ラベル付け・実学習は行っていません。**

f002のMCL/PF OAの両foldでの低下、Synovitisの少数陰性、EffusionのN/Bの違いを調べる動機はありますが、モデルの弱点を教師誤りと断定しません。goldとfold0/1の連結groupを**読む前に除外**し、キュー150検査・1,800セルでgold_selected=holdout_selected=0を確認しました。患者独立性は未確認です。これは教師規則の検討資料であり、画像由来正解ではありません。

## 実キューと30セル一次レビューの結果

保存先は[新規private artifact](../artifacts/label-review/20261007-training-evidence-v1/manifest.json)です。元の`studies.csv`、未記入`review.csv`、hash manifestを保持し、[重点30セルの一次レビュー](../artifacts/label-review/20261007-training-evidence-v1/preliminary-reviewed-30.csv)を別ファイルへ保存しました。根拠は原文に存在する部分文字列で、`proposed_label`は全件空欄です。原Report/UIDはGitに含めず、[公開する集計JSON](../experiments/training-report-evidence-review-20261007.json)には含めていません。

| 確認 | 件数 |
| --- | ---: |
| MCL／PF OA／Effusion／Synovitis | 9／7／7／7 |
| 陽性根拠の記載あり／明示的否定 | 5／2 |
| 閾値未満／言及なし | 2／4 |
| 程度・範囲・言語等の再確認が必要 | 17 |

**17は未確定セル数で、誤ラベル17件ではありません。** 意図的に希少state等を拾う標本なので、母集団の誤り率も推定しません。提供`label_provenance.csv`には所見ごとの根拠文や閾値判断がなく、元promptがどう判定したかまでは確認できませんでした。

[大会Hostの所見定義](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)を10月7日に確認しました。MCLは急性の高度部分/完全断裂が対象で、低度捻挫・古い変化は陰性です。PF OAは高度軟骨欠損（厚さ50%超）に加え、中～大範囲（概ね1cm以上）が必要です。Effusionは中～大量、Synovitisは滑膜自体の炎症/肥厚を対象とします。したがって「捻挫」「OA」「少量の液体」という語だけから陽性へ変更しません。

次はこの30セルを原文・言語・否定の作用範囲・現在/既往・程度/範囲について再確認し、明記されない閾値を推測で補わず判断規則を固定します。未記載は欠損のまま、疑いと未記載、明示的否定と閾値未満を区別します。EffusionをSynovitisへ流用しません。その後、別版教師候補を作り、同じ画像特徴・モデル・seed・固定foldで**教師だけの比較**へ進みます。現在のf003と準備f004は元教師のままです。

既存監査ではSynovitisの陰性は全weakで49件、fold別に1／25／7／10／6件です。fold 0は陽性100・陰性1・欠損87.7%で、評価も不安定です。Effusionの陰性はN 1,230件＋B 1,444件で、同じ0でも「明示的な否定」と「所見の閾値未満」を区別する必要があります。[集計根拠](../experiments/weak-state-audit-20261004.json)。この不足を欠損→0やSynovitis＝Effusionへの置換で解決したとは扱いません。

## キュー作成の実施手順

以下は今回実施したキュー作成の記録です。既存の固定分割・供給group・出所・入力hashを先に再監査し、fold0とfold1を両方保護して、それ以外から抽出しました。どのfoldを保護するかはレポートを読む前に決め、その記録を将来のラベル規則・promptの開発にも引き継ぎます。**同じoutputへ再実行しません。** fold1のReportを監査に使った後で、fold1を独立した追試と呼ばないでください。現fold0/1もモデルの選択には既使用です。

```powershell
.\.venv\Scripts\python.exe scripts/build_label_review_queue.py `
  --train-csv data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train.csv `
  --weak-labels data/labels/vmohitrao-v3/weak_labels.csv `
  --provenance data/labels/vmohitrao-v3/provenance.json `
  --groups data/labels/vmohitrao-v3/groups.csv `
  --manifest-dir data/manifests/v1-vmohitrao-research `
  --config configs/baseline.json `
  --label-states data/labels/sources/vmohitrao-v3/label_states.csv `
  --holdout-fold 0 --holdout-fold 1 `
  --size 150 --seed 20261006 `
  --reason "Training-side evidence audit before changing missing and negative label rules" `
  --output-dir artifacts/label-review/20261007-training-evidence-v1
```

`--label-states`は任意ですが、指定時は出所JSONに記録された`label_states.csv`のhashとP/N/B/U/M→値の対応を検証します。現在の保存先と異なる場合は、監査済み元ファイルの実パスを指定してください。省略しても陽性／陰性／欠損／softを区別し、softを閾値で二値化しません。

出力は、このリポジトリの`artifacts/`または`data/`配下で、Git除外を確認できる新規ディレクトリに限定します。同名runは上書きしません。実行は取得・LLM通信・MRI処理・GPU学習・提出を開始しません。

## 抽出と記入

希少な「所見×現在のラベル種別」を先に巡回し、seed付きSHA-256順で最大150検査を選びます。stateを指定した場合はNとB、UとMも別の層にします。同一レポート・供給groupの推移的な連結を再構成し、1連結成分につき1検査に限定します。goldや保護したfoldに連結される検査は入りません。候補が不足する場合は水増しせず、実際の件数と不足をmanifestへ記録します。

| 出力 | 内容 |
| --- | --- |
| `studies.csv` | 抽出順、UID、元fold/group、重点所見、原Report、Report hash、言語の記入欄 |
| `review.csv` | 150検査なら1,800所見セル。現行値とstateを保存し、証拠文・否定・疑い・現在・既往・言及なし・レビュー結果を記入 |
| `manifest.json` | seed、保護fold、理由、出所、入力・コード・出力hash、環境、層別件数、抽出できない所見／種別 |
| `fold-audit.json` | 既存の分割・入力・gold予約監査結果 |
| `codebook.json` | 記入値と解釈、独立性・保存範囲 |

まず`primary_review=1`の150セルを確認し、必要に応じて同じ検査の他所見へ広げます。根拠文は原文から引用し、否定・疑い・現在・既往・言及なしは`yes / no / unclear`で記入します。空欄は未確認です。言語は人が記録し、今回の抽出で言語分布を網羅できたとはしません。`proposed_label`は検討案だけで、学習ラベルへ適用する機能はありません。

この抽出は教師規則の問題を探すための意図的な標本です。母集団の陽性率、ラベル正解率、モデル精度の推定には使えません。Reportの根拠があることも画像由来の正解と同義ではありません。全12項目が欠損で既存`prepare`から除かれた検査は、fold所属がないため今回も対象外です。患者独立性は未確認のままです。

## ラベル変更へ進む条件

原文の根拠から、所見ごとにどの規則を変えるかを固定し、元CSVを保持した別版の教師候補にします。画像・モデル・seed・fold・選択規則を固定し、教師だけを変えた比較を別runで行います。根拠の強さによる重み付けや新しいencoderは、その後の別要因にします。goldは規則、prompt、checkpoint、混合係数の選択に使いません。

検査は標準ライブラリだけで実行できます。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_label_review_queue.py -v
```
