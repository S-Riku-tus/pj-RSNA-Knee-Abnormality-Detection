# 最終5提出の判断 — 2026-10-22

2026年10月22日13:58 JST時点の統合分析。**第1候補はP048の独自native比率を20%から10%へ減らす一因子比較**とする。新しい公開上位版のコピー、新学習、独自枝の増量より根拠が具体的である。公式0.959は未達で、今回の診断は到達保証やPublic増分の推定ではない。

## 現在地を公式結果・実ログで更新

本人のKaggle SDK Get/List/limitsを13:51:48 JSTに読み取った。全44提出は終端、34件採点・10件例外。本人の公式Public最高は0.958で、該当8件、未返却0件。本日の残枠は5件。P048 ref57464683もCOMPLETE／0.958だった。[本人の今回の公式観測](../artifacts/research/20261022-final-readonly-state-v1/observation-20261022T045148409831Z/summary.json)。過去の「P048採点待ち」「残2/4枠」は当時の履歴である。

最初にREADME、PROJECT、validation、台帳405記録を確認し、正式提出を全件照合した。経緯と全44件のref・版・スコアは[全履歴監査](audit-handoff-history-20261022.md)にある。初期はKaggleキャッシュ→ローカルRTX 4090学習→Kaggle提出を整備し、欠損mask、12列順、group/report分離を共通化した。自作ResNetのPublic0.749と弱検証の乖離を受け、公開強モデルを主力とし、独自DINOv2のCLS/REGを補完枝として育てた。

| 到達点・比較 | 本人の公式Public | 今回の意味 |
| --- | ---: | --- |
| 公開Raptor P001 → 公開DINO P002 | 0.924 → 0.937 | 採点成功を確保し、公開重みの再現へ |
| Apex P005 → S5 P012 → S6 P020 | 0.950 → 0.951 → 0.954 | 主力画像モデル・組合せを改善 |
| V22R/R7 → R6＋Raptor224 P029M | 0.956 → 0.957 | 相補的な画像モデルと入力を追加 |
| P033の所見別配合 | **0.958** | 現在も保持する基準 |
| 224反転P036、Reader P041、Gaussian融合P043、Reader＋反転P044 | 各0.958 | 表示上の上積みなし。真の同点・非劣性の証明ではない |
| 所見別配合の別ベクトルP040／中間順位化P042 | 各0.957 | この変更は不採用 |
| DINOv3 5fold追加10% P045 | 0.956 | 条件付き20% P046も棄却 |
| native20% P039／gamma CLSへ置換したP048 | 各0.958 | 独自枝の表示上の寄与は未確定 |

P033はRaptor384、Raptor224、R6の4fold TTAを使う。Raptor内は384:224=65:35。その後のRaptor比率はLateral OA=.20、PF OA=.60、Synovitis=.80、Baker's=.35、その他=.50。P033の正式ref57290868／scriptVersion358955110、Notebook・Input版・source SHAは履歴監査に固定している。以後の実験はこの成功版を保存し、新規runへ作る。

**F014 TinyとF015 Smallはホストメモリ確保のMemoryErrorで停止済みだった。** TinyはE3完了、Smallは完了epoch0、両方E12・fold1なし。Windows resource exhaustion記録はcommit枯渇を示す。人工GPU成功やE1復元成功を最終精度の資格にはしない。GPUの既存処理には触れず、新CNNの再学習を最終提出の前提にしない。[実ログと検証監査](audit-handoff-validation-20261022.md)。

## Webで何が分かったか

13:50–13:57 JSTにKaggle公式匿名APIとWeb検索を更新した。スコア降順100件、DATE_RUN/DATE_CREATED各100件、0.959/0.960/0.96/pseudo label等の検索、版別コード・Input、当日Discussionを点検した。公開スコア上位100件の作者最高は0.958で、その最高6件のsource/Inputは10月20日21時から不変。今回の範囲では、採点済みで再現できる新公開0.959以上は見つからなかった。[公開調査と全一次証拠](audit-handoff-public-20261022.md)。非公開手法や全公開物に存在しないという主張ではない。

- [heliosliの「0.9586」](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586?scriptVersionId=359224783)は公式作者表示0.958。タイトルを未丸めの公式値と認定しない。latest V7とlast executed V6を混同しない。
- [evgendvorkin V23](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7?scriptVersionId=360116244)は既知のgoodpjw系科学コード。新しい独立モデルではない。
- [S6をR6へ追加した作者比較](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957?scriptVersionId=359343463)は元0.958に対して0.957、Goldから選んだ別係数も0.957/0.955と記載する。単純な枝の増加を支持しない。
- [single-model議論](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)にはConvNeXt 5fold=0.959等の作者自己申告があるが、その値と結びつく公開checkpoint・版・再現レシピは得られていない。今から同じ値を再現できる配布資産と区別する。

作者値、本人再現値、期待値、weak、Goldを分けた。公開重みの上流Gold接触には確認済み・不明なものが残る。今回自分のGold画像ラベルを学習・checkpoint・係数選択に使っていない。

## 最も有望な方向を保存済み予測で検証

R6の既存probe809/855検査に、対応する除外foldのF006/F013 CLSとREGの予測CSVをUIDで結合した。係数0/10/20/30/35%、家族50%・CLS単独・REG単独などの小表を解析前に固定し、全結果を保存した。新MRIデコード、forward、学習、tensorロードは行っていない。

主に見るF013 CLS＋既存REGの50%平均は以下だった。数値は**再利用した弱ラベルの固定10所見平均AUC**で、公式Publicではない。

| native比率 | bank0：809検査 | bank1：855検査 |
| --- | ---: | ---: |
| 0% | 0.952242 | 0.961766 |
| **10%** | **0.953073** | **0.962312** |
| 現行20% | 0.952131 | 0.960686 |
| 30% | 0.949682 | 0.957593 |
| 35% | 0.948005 | 0.955437 |

20→10%は+0.000941965／+0.001625456で両bank正、FP64検算も+0.000966317／+0.001615765で同方向。30/35%への増量は両bankで悪化した。家族をF006・CLS単独・REG単独に替えた表でも減量方向は一貫した。rootも別出力ディレクトリで再実行し、数値を再現した。[再現script](../artifacts/research/20261022-native-r6-proxy-v1/reproduce.py)、[root再現結果](../artifacts/research/20261022-native-r6-proxy-v1/root-reproduction-v1/proxy.json)。

ただしRaptor384/224の同じ除外fold予測はなく、**P033全体のOOFは作れていない**。今回の代理親は単一除外foldのR6を順位化したもの。実P048の親はP033の合成値で、再順位化せず使う。nativeも実提出では2fold平均になる。集団、分布、上流監査、公式12所見との違いがあり、この差をPublicの予想増分や10%の最適性へ読み替えない。既使用weakの探索として「増やすより減らす」を支持する根拠に限定する。

別の未試行案として、gamma CLSの最後3epoch E10/E11/E12の確率を単純平均する比較も、集合を固定して既存CSVだけで調べた。REG固定50%との家族平均は現E12よりfold0 −0.000955291／fold1 −0.000346592で両方低下。snapshot案は棄却し、別のepoch集合を追加探索しない。[結果](../artifacts/research/20261022-native-r6-proxy-v1/root-reproduction-v1/snapshot.json)。

## 残5枠の配分

以下は提出済みの実験ではなく、今回固定する探索候補。P049〜P053は計画上の識別名で、Save・モデル実行・正式Submitはまだ0である。**P049を最優先、P050とP052を独立に準備し、P051はP049との比較、P053は最後の予約枠**とする。順位や同じ0.958の表示から細かな差を推測して候補を入れ替えない。

| 優先枠 | 親と、一つだけ変える要因 | 具体値 | 根拠の強さ・条件 |
| --- | --- | --- | --- |
| 1：P049案 | P048のnative比率のみ | 20%→**10%**。同じF013 CLS2＋REG2、家族50%、fold平均、P0338セル | 今回の両bank診断が減量を支持。最も優先 |
| 2：P050案 | P033の所見別配合ベクトルのみ | LOA=.25、PF=.55、Synovitis=.75、Baker=.40。他=.50 | P040で悪化した方向の反対側を一度だけ比較。反対なら改善するという証明はなく、探索1枠まで |
| 3：P051案 | P049へ既存P041 Readerを追加する要因のみ | 同じReader3本、k16、既存所見別係数.003–.035、end方式 | nativeとReaderの組合せは未試行。P049完成CSVへReaderを最後に適用し、順序を固定。両単体が0.958だったことから相乗効果を仮定しない |
| 4：P052案 | P033のRaptor内A224だけ | .35→**.20** | 同じ重みで384寄りを探索。過去44件に同じ親のA224比較はない。方向の優位性は未証明 |
| 5：P053案・予約 | P033のRaptor内A224だけ | .35→**.50** | 反対側の対照。新しい強い公開証拠がなければこれを予約案とし、0.959達成後は追加不要 |

P049は成功した対象セルだけ、`r = f32(average_tie_rank(f32(native_success)) / N_success)`、`out = f32(f32(f32(.9) * f32(parent_s)) + f32(f32(.1) * r))` とする。順位は1始まりの平均tie順位で、係数・入力も乗算前にFP32化する。親と最終結果は再順位化せず、nativeの順位母集団、FP32の積和、失敗セルの元文字列保持を既存契約のままにする。MCL/Synovitisは元親の文字列を保持する。家族やREGの削除・checkpoint変更は混ぜない。

P050のベクトル全体は一つの事前固定した配合方針の比較で、4所見の個別効果を識別するものではない。現行P033が局所最良で、P040と反対方向の両方が悪い可能性もある。単一所見の実AUCだけでmacroを.001上げるなら、その所見に約.012の改善が必要になる。根拠のない一所見ずつの探索で枠を消費しない。

P051はP049のPublicが0.957以下と先に判明した場合、標準候補から外す。残り枠を埋めるためだけに弱い組合せを提出しない。P049が未採点でも独立候補の準備は進める。20:00 JSTを判断の目安とし、結果待ちと実行待ちを区別する。

P052/P053は.20/.50を事前に決めた両側比較であり、可視3検査の順位を見て値を選んだものではない。公開上位もA224=.35のままで、この2案にはP049のような弱検証の方向根拠はない。新たな公開資産を優先するなら、未採点のタイトルではなく、採点版・取得可能な重み/Input・成功親との差分が確認できた場合だけ予約枠と置換する。

## 実行の順序と締切の扱い

[当日公式Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)で日次最大5件・最終選択最大2件、[Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)で**10月23日08:59 JST**、Code RequirementsでInternet OFF・9時間以内・submission.csvを再確認した。

当日の[Scoring backlog報告](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751502)には5–6時間以上のPENDING、[Kernel Stuck報告](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751576)には昨日7分のcodeが40分RUNNINGという参加者の記述がある。これは自分の所要時間やSLAではないが、全候補を一件ずつ採点待ちしてから準備する日程は危険である。

1. まずP049の比率だけを変えたsourceと算術を固定する。P033/P048の重み・decoder・GPU経路はhash一致を確認して再利用し、既存の人工GPU資格を理由なく一から繰り返さない。変更したCSV計算は標準ライブラリで検算する。
2. P050/P052を並行して準備する。Kaggleの利用可能枠を確認し、各Save版とInputを照合、今回の可視出力・失敗の有無・CSV計算を確認して手動Submitする。可視3件成功とhidden採点成功を区別する。ソースに自動アップロード・自動提出を追加しない。
3. **夕方から20:00 JST頃までを主要候補の手動提出目標**とし、採点待ち中も独立候補の準備を続ける。これは余裕を確保する作業上の目安で、完了保証やKaggleの別締切ではない。待ち行列により残5枠を必ず使い切れるとは約束しない。
4. 本人の正式COMPLETE／Public≥0.959を確認できたらref・版・Input・sourceを記録し、目標のための追加探索は終了する。未採点・weakの改善・作者の値を達成として扱わない。
5. 最終選択は成功済みP033を予備として保持し、正式に改善した候補を優先する。表示同点だけなら、単純なP033と相補的な成功候補を、検証範囲・運用実績も踏まえて最大2件選ぶ。最終Privateの改善を保証する判断ではなく、選択操作は別途現在の状態を確認して行う。

今回は調査・保存済みCSV分析・計画の固定までを実施した。新しい学習、実MRI推論、モデル取得、外部アップロード、Save、正式提出、最終選択操作は行っていない。既存runは上書きせず、GPUの既存処理も停止・変更していない。
