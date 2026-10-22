# 最終日の公開資産調査（2026-10-22）

調査時刻は2026年10月22日13:50–13:57 JST。README.md、PROJECT.md、docs/validation.md、experiments/ledger.csvと10月19–21日の既存公開調査を読んだ後、Web検索とKaggle公式の匿名読取APIで更新した。自分の提出結果・残枠の当日正本はrootの所有者API監査であり、本資料の作者値と混同しない。

**調べた範囲で、採点済みの公開0.959以上と、そのまま再現すべき新しい上位レシピは見つからなかった。作者最高0.958の6件は10月20日21時の保存記録からNotebook source/Inputとも変化していない。** 同じforkの追加提出より、既存の自己0.958を親にした根拠のある変更を優先する。

## 当日の一次証拠

raw応答・取得日時・要求payload・SHAは [20261022-public-final-audit-v1](../artifacts/research/20261022-public-final-audit-v1/) に保存した。認証情報は読まず、重み・ラベルCSV・MRIは取得せず、公開コードを実行していない。学習、Save、Submit、アップロードは0。

| 調査 | 範囲と結果 | 保存物 |
| --- | --- | --- |
| 公式公開Notebook | SCORE_DESCENDING先頭100件、最大作者0.958、0.958超0件 | score_descending.json / receipt |
| 別の公式一覧 | DATE_RUN、DATE_CREATEDを各100件。新しいsource/依存を点検 | date_run.json、date_created.json |
| 正しい全文検索 | `kernelFilterCriteria.search`で0.959、0.960、0.96、pseudo label、s6fall。最初の2検索は各1件/14件 | query-*.json / receipt |
| 0.958資産の更新比較 | 6件とも10月20日21時からsource/Input差分0 | top958-version-comparison.json |
| 版別表示 | evgendvorkin、goodpjw2008、heliosli、nartaa、basith007の表示JSONを確認 | view-*.json / receipt |
| 科学コードの静的確認 | source360116244、359343463、360278037のNotebookと358439050のPythonを保存。実行なし | source-* / receipt |
| 新しいDiscussion | 公式SDKと同じ公開POST APIで新着一覧と本文/返信を取得 | sdk-topics-new.json、topic-*.json、comments-*.json |
| 当日Rules等 | ListCompetitionPagesで全ページ取得、ルール・期限・提出契約を再確認 | all-competition-pages.json、page-*.txt / receipt |

調査の限界として、全公開Notebookを走査した証明ではなく、非公開・限定公開の手法は含まない。DATE_RUNは実行日時の一覧で、source作成時刻順とは一致しない。全文検索の先頭100件だけのものは検索全件とは呼ばない。

初回の`date_run-0.959`等は検索語をlistRequest内へ置いていたため絞込が効かなかった。これらを検索の証拠には使わず、正しい位置へ直した`query-*.json`を使用した。v1 GET topicsは文字列sort指定が反映されず同じ順を返したため、新着の根拠は正しいSDKエンドポイントの`sdk-topics-new.json`である。

## 公開候補の差分と判断

| 資産 | 当日確認した版・Input参照 | 作者値／自己値 | 本当の差分と判断 |
| --- | --- | --- | --- |
| [goodpjw2008のCoAtNet + R6 + Reader](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | current実行V27/source359009299/ref57313125、COMPLETE。Input版ID20297367、20325538、20440357、20601735、20763353。best表示欄はV26/source358890016 | 当日作者公式0.958。これは自己値ではない | Raptor384/224、R6、Readerの既知の家族。新しい独立重みではない。既存の自分の比較との対応を維持し、fork単位で新規候補に数えない |
| [evgendvorkin RSNA_Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7?scriptVersionId=360116244) | V23/source360116244/ref57429282、COMPLETE、同じ5 Input版ID | 当日作者公式0.958。自己再現は今回0件 | 元goodpjwのレシピを説明する12科学セルのコピー。10月20日からsource/Input不変。再提出する情報増分なし |
| [heliosli title 0.9586](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586?scriptVersionId=359224783) | 一覧latestはV7/source359224783、best欄もV7。Get current実行はV6/source359117069/ref57325533。Input版ID20297367、20325538、20440357、20611930、Notebook359009299 | どちらの公式表示も作者0.958。タイトル0.9586を未丸め公式値と認定しない | latest source、last executed run、best欄が別。版を混ぜない。P038等の既存比較を無視して再度同じ全レシピを出す根拠なし |
| [nartaa S6fall](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo?scriptVersionId=358439050) | V2/source358439050/ref57194443。bestはV1/source357577090。Input版ID20297367、20440357、20733226 | 作者公式0.950。純粋単体全検査の自己再現値ではない | ConvNeXt-smallの全データ学習とCoAtNet fallbackを含む既知pipeline。10月19日にexact training textとJEV由来Gold選択曝露を調査済み。今日の新発見として扱わない |
| [goodpjwのS6追加比較](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957?scriptVersionId=359343463) | V8/source359343463、静的本文を本日再取得 | 作者記述はS6追加V4が0.957、元レシピ0.958。Goldで選んだ係数V6/V7も0.957/0.955。今回自己再現なし | S6をR6の横へ追加するだけの案を支持しない。係数の極端化や58例選択を改善の確証にしない |
| [basith007 Model Ensembling](https://www.kaggle.com/code/basith007/model-ensembling?scriptVersionId=360278037) | V9/source360278037、COMPLETE、submission57421750にscoreなし。Dataset20128434/20153863とModel1028372/1028761/1028763/1035889がprivate | 作者Public値null。自己値なし | 独自DINOv2/CoAtNetの30/70確率平均だが必要な依存6個が非公開。新しい再現可能な0.959候補ではない。nullを0扱いしない |

0.959検索の1件は既調査のS6_60/N40で、作者値nullの期待値記載。0.960検索も以前のA5や旧pipelineと同じだった。タイトル、期待スコア、非公開チームの自己申告から、実行可能な公開上位資産の存在を推定しない。

## 公開情報が今回の判断へ与える意味

公開bestにもA224=0.35と既知の所見別融合が残る。P033に対してA224だけ変えた効果を直接裏づける新資料はない。他作者のw30/w50は異なる親レシピなので、係数だけの比較として転用できない。224の鏡映はすでにP036/P044で表示同点であり、同じ比較を繰り返さない。

P040の所見ベクトル変更（Lateral OA .20→.15、PF OA .60→.65、Synovitis .80→.85、Baker's .35→.30）が自己0.957だった事実は、その方向へさらに動かす根拠を弱める。しかし反対方向へ動かせば改善するとは証明しない。逆方向の比較は探索候補であり、既存の独立診断で両集団が正になった変更より上位に置く根拠はない。

Readerは自己P041=0.958で表示上の主効果を確認できていない。native10が別途支持された場合に、その親へReaderだけを加える比較は一要因として定義できるが、組合せの改善は未測定である。母体の変更とReader追加を同時に比較して、Readerの効果とは呼ばない。

[単体モデルDiscussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)には、作者自己申告のConvNeXt-T 5fold=0.959、別作者のConvNeXt-T=.953/DINOv2=.951/平均=.955などがある。しかしcheckpoint・同じteacher・Input版を備えた公開0.959レシピはそこに提示されていない。CNNを異なる帰納バイアスとして調べる仮説は支持するが、実行済みのF014/F015の自己結果を置き換えない。

[当日LLMラベルDiscussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)には、10月22日11:23 JSTにk256.devが画像とreportの併用ラベルと初回学習Public0.950超を報告した。Gold58での診断改善も作者報告であり、利用可能な全非Goldラベルやその版、0.959のcheckpointを提供するものではない。新ラベル生成・再学習・監査を最終日に追加する根拠は弱い。

外部の[Tejas公開実装](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection)は作者READMEでPublic0.941、親0.940から2学習armで+0.001を報告する。これは親0.958へも同じ+0.001が移る証拠ではない。[3DReasonKnee](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee)と[SGMRI-VQA](https://lamawmouk.github.io/SGMRI-VQA/)は空間注釈・VQAの一次資料として関連するが、現提出契約へ直結する12列の公開0.959モデルではない。今からデータ接続・再前処理・学習・serving監査を追加する時間コストに見合う直接根拠を確認できなかった。

## 当日の時間制約と提出順序

[公式Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)は**日次最大5件、最終審査用の選択最大2件**を本日再確認した。[Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#timeline)の最終提出は10月22日23:59 UTC、すなわち**10月23日08:59 JST**。Code RequirementsはCPU/GPU各9時間以内、Internet OFF、submission.csvである。本人の残5枠は別途live照会する。

新しい一次情報として、以下の参加者報告がある。Kaggleの確定SLAや、自分の提出が同じ状態になる証明とは扱わない。

| 報告 | 投稿者・時刻 | 内容と判断への影響 |
| --- | --- | --- |
| [Scoring backlog?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751502) | Masum Huq、2026-10-22 06:26:23 JST | 複数提出が5–6時間以上PENDING、対応NotebookはCOMPLETEという本人の報告。可視完走と正式採点を区別し、結果待ちで次候補の準備を止めない |
| [Kernel Stuck](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751576) | Carrot Production、2026-10-22 13:06:49 JST | 昨日7分だった同じ2T4 codeが40分RUNNING、ログ空という報告。通常所要時間だけで締切直前の完了を約束しない |

したがって、まず候補の優先順・一要因・採否を固定し、利用可能な既存診断と現在の実行資格を点検する。資格済み候補は早い時間帯に手動提出できるよう準備し、複数時間の採点待ちを見込む。遅い時間まで0.959探索を広げて成功bestの保全・最終2選択を失うことは避ける。少なくとも既存の公式成功0.958を保持し、候補の可視成功だけで置き換えない。

本調査からは、既存の失敗案・不変fork・タイトルだけの高得点候補を再試行するより、root側で両bankの改善を確認したnative20→10のような軽量で根拠のある比較を最優先にする判断が妥当である。改善がPublicの表示境界を超える保証はない。Reader追加、逆ベクトル、A224の順序は、それぞれの追加診断と有限な提出予算を踏まえてrootの統合判断へ渡す。
