# 最終提出に向けた検証監査（2026-10-22）

この文書は読み取り監査と保存済みCSVの探索診断を扱う。学習、MRIデコード、モデル取得、checkpointのデシリアライズ、外部書込み、提出は行わない。PROJECT/docsの最終統合はrootが行う。

## 解析前に固定した比較と解釈

R6と独自nativeの補完性を、既に資格確認済みのR6 probe（original weak fold0/1の809/855検査）と対応する同foldの独自予測だけで調べる。R6は公開trainer宣言の除外fold、独自は元のweak除外foldを用いる。Raptor384/224の同一検査・同一除外foldの予測はないため、P033全体のOOFは作れない。

最初の数値を読む前に親へ送った固定表は、native比率0/10/20/30/35%、native側をF006 CLS+REGの50%平均、F013 CLS+REGの50%平均、F013 CLS単独、REG単独の4通りとするもの。各枝をbank内の全検査で平均tie順位にし、FP32の積・和で合成する。観測ラベルだけで固定10所見の平均AUCを比較する。MCL/Synovitisを含めた救済や、所見別係数の選択はしない。全件の表を開示し、表の最大値を検証済み最適値として採用しない。20%から増量・減量する方向が両bankで一貫するかだけを補助的に解釈する。

追加のsnapshot診断は、この追記時点で未実施。F013のE10/E11/E12の保存CSVが両foldに揃い、元run historyのSHA・UID・mask・source等と一致する場合に限り、確率の固定3epoch単純平均を作る。既存REG（fold0 E8、fold1 E10）との固定50%平均について現E12との固定10所見AUC差を調べる。これ以外のepoch集合・係数は試さない。各snapshotの数値を見て集合を変えず、途中epoch・BCE bestへの選択変更を行わない。これは新しい探索的な推論ensembleの比較で、元E12の事前検証やF009 engineering HOLDを変更しない。正であっても実tensorの有限性・厳密復元・実Kaggle GPU/MRI・CSV資格は別途必要になる。

## 現在地の訂正

2026-10-21のPROJECT/docsはF014/F015を学習中と記録しているが、2026-10-22の実run読取りでは両方failedだった。

- Tiny：`artifacts/runs/f014-convnext-tiny-fold0-20261021-v1/run.json`、E3まで完了、elapsed 7,612.603秒、`MemoryError: Unable to allocate 63.3 MiB for an array with shape (66382848,) and data type uint8`。E12・fold1は存在しない。
- Small：`artifacts/runs/f015-convnext-small-fold0-20261021-v1/run.json`、完了epoch 0、elapsed 1,164.318秒、`MemoryError`。E12・fold1は存在しない。
- 当時のWindows resource exhaustion eventはsystem commit約49.88GB / limit49.92GBを記録している。物理RAMの空きだけを見た停止条件ではcommit枯渇を検出できていなかった。VRAM OOMが原因だと断定しない。`artifacts/research/20261021-F015-MemoryError-root-diagnosis-v1/windows-resource-exhaustion-events-redacted.json`。
- 2026-10-22 13:51 JST頃のnvidia-smiは4090使用18,923MiB・99%。既存処理は停止・変更していない。CIMのprocess列挙はアクセス拒否のため実行者を断定しない。

新CNNは現有候補にできない。新学習2fold・故障対策・servingを前提に最終5枠を割り当てる根拠はない。

## 主要な確定済み根拠

| 系列 | 実施内容と現在の判断 |
| --- | --- |
| F001→F002 | 凍結DINO特徴のmean poolingからattentionへ。BCE選択済みweak10はfold0 0.804538→0.827229、fold1 0.792198→0.822494。両foldで改善。単独初期servingにはhidden例外の履歴があり、可視成功と提出成功を区別した。 |
| F003/F004 | 24epoch化はfold1の選択が同じE12のまま。dropout0.4も一貫した利益がなく、延長・dropoutを優先しなかった。 |
| F005 | native392・encoder末尾2blockの学習で、凍結対照より大きく改善。BCE選択weak10はfold0 0.818085→0.880569、fold1 0.803667→0.880062。現在の独自native系の基礎。 |
| F006 | late cosine12epochを両fold完了。後の固定E12の10所見比較と復元・native64等を通過し、P039で使うCLS対照。BCE選択E8/E9と固定E12を混同しない。 |
| F007 | 元BCE CLSを保持する局所残差head12epochを両fold完了。元F005比のweak改善と可視20出力資格はあるが、F006固定E12が両foldで上回り、固定50% CLS混合もAUCを下げたためF006を優先。正式Submitは0。旧F005残差を現F013へそのまま足すことは同じ比較ではない。 |
| F008 | F007を24epochへ延長。fold1のweak10は0.881632→0.879104で低下し、延長を優先しない。 |
| F009→F013 | gamma拡張を一因子で追加。F009の旧zero-skip HOLDを保持したまま、新F013 scopeで固定E12の厳密復元・有限性・GPU等を資格確認し、P048として正式比較した。 |
| F010 | DINO末尾4blockへ拡大。固定E12 weak差が負で、既存工学HOLDも保持。 |
| F011/F012 | JEV教師置換・範囲拡張・train-only補正のCPU screenは旧教師を超えず、画像モデル学習を追加しない。 |
| F014/F015 | ConvNeXt Tiny/SmallはMemoryErrorでE12未到達。途中epochは候補にしない。 |

F001〜F008の上記数値は各 `artifacts/runs/f00*/run.json` の元BCE選択結果・`per_target` の10所見平均からの再集計で、固定E12の主比較とは区別する。F007を優先しなかった直接根拠は[PROJECTの当時の判断](../PROJECT.md)の「弱ラベルの直接比較ではF006 E12がF007 E12を両foldで上回り」と [three-submitted-two-native64-pass-F007-visible-error-20261014.json](../experiments/three-submitted-two-native64-pass-F007-visible-error-20261014.json)。Public増分への換算はしていない。

- F013は同じ固定E12のCLSについてfold0 +0.001692782、fold1 +0.000902383。fold1のgroup bootstrap95%区間は[-0.000297706,+0.002114496]で0を含む。元F009のskip1によるHOLDは保持し、別scopeで厳密復元・有限性・GPUを確認した経路である。
- 現P048のnative家族50%は既存REGを固定し、fold0 +0.000701087 / fold1 +0.000370281。反復した既使用weakであり、実際の2fold平均やP033全体のOOFではない。`experiments/P048-native50-paired-weak-positive-sourceV2-20261020.json`。
- P048 V4は人工CPU/GPUと実可視3/3成功、元親・native平均・最終CSVの独立算術照合71項目等を通過して提出済み。正式ref57464683。今回rootのlive照会ではCOMPLETE / Public0.958、全44件terminal、残5件。今回の公式観測正本はroot担当。
- R6のTTAなし・TTA1/4・補正なし・crop320は、共通成功1,664件で全て両bank正条件を満たさず棄却。P047を実候補化しない。`experiments/R6-four-prespecified-variants-nonpositive-no-P047-20261021.json`。
- P045 DINOv3追加10%は正式0.956なので、条件付き20%のP046も棄却。`experiments/P045-formal-COMPLETE0956-DINO-rejected-20261021.json`。
- F010末尾4blockは固定E12の探索差-0.004133893。教師置換F011/F012も旧教師より低く、追加fold・採用へ進めない。
- OrthoDiffusionのglobal256/20%、fluid拡張/20%、regional1024/20%は所定weak条件で棄却。別C=0.01のinner4モデルは学習・予測が封印済みだが、規定のF006 fold2対照が未学習で外側評価未実施。既存fold0/1のF006はfold2を学習に含むため、それで代用できない。

## R6とnativeの順位合成proxy（探索結果）

以下は新たな実MRI推論を使わず、元run/CSV/checkpointのbyte hashと元R6のtransport/source/replay資格を照合して計算した。checkpointはbyte hashだけで、tensorロードしていない。Python標準ライブラリだけを使用し、Torch未importをassertした。平均AUCは固定10所見、R6/native両側で当該検査を除外するモデルだけを使用した。公開重みの上流学習集団・Gold/補助学習履歴の独立証明はなく、患者独立性も未確認。

| native枝 | bank | 0% | 10% | 20% | 30% | 35% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F006 CLS+REG 50% | 0 | 0.952241962 | 0.952995383 | 0.952040086 | 0.949569194 | 0.947776163 |
| F006 CLS+REG 50% | 1 | 0.961766418 | 0.962292222 | 0.960744876 | 0.957598632 | 0.955333321 |
| F013 CLS+REG 50% | 0 | 0.952241962 | 0.953072671 | 0.952130706 | 0.949681859 | 0.948005307 |
| F013 CLS+REG 50% | 1 | 0.961766418 | 0.962311873 | 0.960686418 | 0.957593118 | 0.955436552 |
| F013 CLS単独 | 0 | 0.952241962 | 0.952963920 | 0.952010441 | 0.949343482 | 0.947519399 |
| F013 CLS単独 | 1 | 0.961766418 | 0.962479106 | 0.960716486 | 0.957334704 | 0.954951902 |
| REG単独 | 0 | 0.952241962 | 0.952879817 | 0.951407720 | 0.948447973 | 0.946292795 |
| REG単独 | 1 | 0.961766418 | 0.962097712 | 0.960364629 | 0.956921584 | 0.954489481 |

F013家族50%の20→10%はbank0 +0.000941965 / bank1 +0.001625456で両方同方向。20→30%は-0.002448847 / -0.003093300、20→35%は-0.004125399 / -0.005249866。したがって次の一因子候補としてP048のnative量20→10%は合理的な探索候補、30/35%を優先する根拠はない。10%はR6単独にも両bankで+0.000830710 / +0.000545455だった。

この結果はP033全体への同じ増分を予測するものではない。全親はRaptorを含むアンサンブル、今回はR6の単一除外foldで、実提出のnative側も2fold平均である。nativeの20%がPublicで不適切だと断定せず、10%が0.959になるとの確率・保証も与えない。全体OOFがない条件で、無差別増量より減量を優先する補助根拠とする。gamma50とgammaCLSの優劣はbank間で一貫せず、REG除去を優先しない。

入力集団のUID hash（改行join）はbank0 `561517a8a4ae6432cf244efbef62837992dbe8f7173b2a1a27f4b482fae79394`、bank1 `9d88426b8d8efc96d7960f95a9d75885be2fe90fa8e3b19bcea37251cbe29160`。R6元出力は `artifacts/research/20261020-R6-pair-probe-root-execution-v1/poll-20261020T122939651701Z/files/R6-paired-weak-probe/`、native入力/各SHAは `artifacts/research/20261020-P048-native50-paired-weak-prospective-v1/policy.json` に固定されている。

### FP64の全比較と再現資料

rootの追加許可に基づき、再現スクリプトと全出力を `artifacts/research/20261022-native-r6-proxy-v1/` に保存した。`reproduce.py --output <新規artifacts/research配下dir>` で同じ結果を新規ディレクトリへ再現できる。初回出力は `reproduction-v1/proxy.json`（52,835B・SHA `ef11e781ae00afabb4ea008f3f5c3673604fde599d8f5c1d6ae0243f6a5bdb7a`）、`snapshot.json`（5,838B・SHA `19e7315e60a88fd196258504a049e7b8c5eb7e5929c7bbce9ee17e8b5bd5261e`）、`inputs.json`（26,998B・SHA `398017d7938279e7ce4790019619e5af1b767eafa32785ae6c6a89e412b6dc99`）。script自体のSHAは `ed95c0bd0b75cd7a9493edfeb58ed4a642182b4cbac36cb1f0c6b8ad43b691f7`。

保存済みnative FP32確率平均は同じまま、順位と外側積和だけをFP64で丸めず計算した全表は以下。gamma50の20→10%差はbank0 +0.000966317 / bank1 +0.001615765で同方向。FP32の丸めだけに依存した減量の推奨ではない。

| native枝 | bank | 0% | 10% | 20% | 30% | 35% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F006 CLS+REG 50% | 0 | 0.952241962 | 0.952999787 | 0.952034203 | 0.949565496 | 0.947774714 |
| F006 CLS+REG 50% | 1 | 0.961766418 | 0.962295100 | 0.960744347 | 0.957597349 | 0.955334167 |
| F013 CLS+REG 50% | 0 | 0.952241962 | 0.953079700 | 0.952113383 | 0.949681092 | 0.948007268 |
| F013 CLS+REG 50% | 1 | 0.961766418 | 0.962313677 | 0.960697912 | 0.957568196 | 0.955437433 |
| F013 CLS単独 | 0 | 0.952241962 | 0.952957545 | 0.952014379 | 0.949335756 | 0.947514780 |
| F013 CLS単独 | 1 | 0.961766418 | 0.962486541 | 0.960724627 | 0.957341919 | 0.954956006 |
| REG単独 | 0 | 0.952241962 | 0.952881247 | 0.951412264 | 0.948448300 | 0.946296545 |
| REG単独 | 1 | 0.961766418 | 0.962105004 | 0.960367482 | 0.956926056 | 0.954489435 |

実P048の外側は `src/rsna_knee/fixed20_blend.py` と `success20_blend.py` のFP32順位・係数・積2回・加算1回である。FP32 proxyの外側丸めはこの仕様に揃えている。ただし実親P033は再順位化せず数値を用い、今回のproxyはR6順位を用いた点が異なる。

再現資料へのリンク：[再現スクリプト](../artifacts/research/20261022-native-r6-proxy-v1/reproduce.py)、[FP32/FP64全比較](../artifacts/research/20261022-native-r6-proxy-v1/reproduction-v1/proxy.json)、[snapshot全比較](../artifacts/research/20261022-native-r6-proxy-v1/reproduction-v1/snapshot.json)、[全入力SHA](../artifacts/research/20261022-native-r6-proxy-v1/reproduction-v1/inputs.json)。rootは別の `.venv` Pythonで[別ディレクトリへの再実行](../artifacts/research/20261022-native-r6-proxy-v1/root-reproduction-v1/proxy.json)を行い、20→10%のFP32/FP64差とsnapshot負を完全一致で確認した。これは別実装ではなく同一scriptの別実行による再現確認である。

## 固定3epoch snapshot平均（探索結果・棄却）

E10/E11/E12の保存CSVとcheckpointは両fold全6本存在した。各CSV SHA、元run/history/epoch checkpoint indexと全CSV UID、保存checkpointのbyte SHA/size、元writerの全所見AUCの完全一致を確認した。元macro平均だけ加算順で最大3.33e-16差があり、所見ごとの厳密一致とmacro1e-12以内を要求した。torch・tensorロード・GPU・新forwardは使用していない。

| fold | 現E12+既存REG 50% | 固定E10/E11/E12確率平均+同REG 50% | 差 |
| --- | ---: | ---: | ---: |
| 0（821件） | 0.889624244 | 0.888668953 | -0.000955291 |
| 1（891件） | 0.889027342 | 0.888680750 | -0.000346592 |

両foldで悪化し、snapshot案は最終候補にしない。epoch集合・平均係数を替える追加sweepは行わない。現E12の資格と旧engineering HOLDは変更しない。

## 分割・教師・採否の解釈

元weakは4,207件、供給group・同一レポートを連結した5fold（821/891/888/778/829件）。Gold58件と同groupの4件、全欠損138件を除外している。欠損はloss/AUCのmaskで除き、0へ埋めていない。Gold由来数値の自分の学習・checkpoint/係数選択は今回も0。初期研究にはGoldの監査評価を既に見た履歴があるため、Goldを完全未観測holdoutと呼ばない。公開重みには上流Gold選択/学習が確認済みまたは不明なものがあり、自分のGold非利用と区別する。

MCL/Synovitisを固定10の外にしたことは公式12所見評価を変更する意味ではない。fold0 Synovitisは陽性100/陰性1などの不安定性があり、weak12の最大化はPublic12の最大化を保証しない。所見ごとの後付け重み最適化には特に証拠が乏しい。P040の複数係数同時変更が0.957だったことから、その逆方向が改善すると数学的に推論できない。

AMPのskipはその更新を実行しなかった事象であり、checkpointが破損した証拠ではない。旧zero-skip基準のHOLDは履歴として保持する一方、現在の採用可能性は別に固定重みの有限性・CPU/GPU厳密復元・数値一致で評価できる。F013/P048はこの区別を満たしている。新たに古いHOLDを解除する作業を最終候補作成の目的にしない。
