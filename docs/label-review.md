# 学習側のレポート根拠監査

2026年10月6日。現行weakラベルを変更する前に、学習側150検査の原文と判定根拠を確認するための手順です。[作成ツール](../scripts/build_label_review_queue.py)はCSVだけを処理します。今回の実装検査は人工CSVだけであり、実レポートの抽出・再ラベル付け・学習はまだ行っていません。

既存監査ではSynovitisの陰性は全weakで49件、fold別に1／25／7／10／6件です。fold 0は陽性100・陰性1・欠損87.7%で、評価も不安定です。Effusionの陰性はN 1,230件＋B 1,444件で、同じ0でも「明示的な否定」と「所見の閾値未満」を区別する必要があります。[集計根拠](../experiments/weak-state-audit-20261004.json)。この不足を欠損→0やSynovitis＝Effusionへの置換で解決したとは扱いません。

## キュー作成

既存の固定分割・供給group・出所・入力hashを先に再監査します。例では今後比較するfold 0とfold 1を両方保護し、それ以外から抽出します。どのfoldを保護するかはレポートを読む前に決め、その記録を将来のラベル規則・promptの開発にも引き継ぎます。fold 1のレポートを監査に使った後で、fold 1を独立した追試と呼ばないでください。

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
  --output-dir artifacts/label-review/20261006-training-evidence-v1
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
