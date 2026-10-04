# 準備内容の検証範囲

更新日 2026年10月4日 JST。全件キャッシュ転送、weakラベルの監査・固定分割、A/B/C対照に続き、fold 1の初期化対照2本とfold 0の20 epoch対照2本を完了した。固定50:50 rank ensembleの追加診断とローカル提出準備も進めた。以下は新しい確認から順に記し、過去の「未実施」は各作業時点の履歴として残す。

## Kaggle Version 3の表示用test実行（10月4日、完了）

- ユーザーの提示ログを構造化して候補asset-manifestと照合した。表示用test 3検査の前処理はwarnings=0・debug_limit=None。通常BN／固定BNのCUDA推論は各3検査×12所見、最終rank ensembleもvalid=true。両checkpoint・全source・rank helperファイル/関数・pixel fingerprintのhashが一致し、RANK_MODULEはInput直下のrank50.pyを参照した。[確認記録](../experiments/r001-kaggle-v3-review-20261004.json)。
- 公式SDKのread-only確認でもNotebook `rsraki/rsna-knee-r001-rank50` はVersion 3・COMPLETE・GPU有効・Internet OFF。Output一覧に最終submission.csv（852 bytes）、単体予測2 CSV（各885 bytes）、submission.meta.json（871 bytes）を確認。Output本体は取得せず、CSVのvalidは実行ログの検査結果に基づく。Notebook内計測47.9549秒、提示ログは起動/変換を含み68.2秒まで。Debugger・mistune・nbconvertの警告は完走を妨げていない。
- [公式データ説明](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)のtest.csvは3検査の例で、Submit時に本番testへ差し替わる。現時点の提出履歴は旧公開モデルのPublic 0.924の1件だけで、r001の本番採点・全体時間・Publicは未確認。Version 3のOutputからsubmission.csvをユーザーがSubmitし、その結果を別途記録する。今回のレビューではMRI/重み/ラベル取得・decode・forward・学習・外部書込・代理提出は行っていない。

## Kaggleのrank helper参照先の修正（10月4日、再依頼）

- ユーザーの再実行ログは最初のcode cellのrank helper確認で停止し、直前の2 checkpointの存在/SHA検査は通過した。公式SDKの通常認証・read-only要求で実際の保存済みNotebook `rsraki/rsna-knee-r001-rank50` Version 2とInputを確認した。GPU有効・Internet OFF・競技Inputは設定済み。保存NotebookのInput参照は得られるが、厳密な添付版番号はAPI metadataに含まれない。[原因・根拠](../experiments/r001-kaggle-pathfix-20261004.json)。
- 現在のコードDataset Version 1から小さいcode/manifestだけ75,459 bytesを取得し、全15ファイルのSHAが候補資産と一致。ZIPの展開13ファイルと別添rank50.pyは正しい。保存コードのCODE_ROOTは`.../rsna-knee-code`だが、元の条件分岐はhelperもその中にあると仮定する。実際はInput直下のrank50.pyなので、ZIPを更新しても参照先は直らない。Debugger・nbconvertの警告は停止原因ではない。
- [修正版Notebook](../artifacts/kaggle/r001-bn-rank50-pathfix-20261004/01_submit_r001_pathfix.ipynb) はRANK_MODULEを`CODE_ZIP.parent / "rank50.py"`へ変更し、欠損とhash不一致を別メッセージで示す。実保存版のCODE_ROOT・後続code cell・全期待hash/provenance・前処理・重み・rank係数は保持。元のローカル資産を上書きせず、修正済み未実行Notebookを新フォルダに保存した。
- `artifacts/qa/20261004-r001-pathfix-v1/checks.json`で取得した実コードInputを人工mountへ配置し、元Notebookの同じ参照先エラーを再現。修正Notebookの初め2 cellを実行し、全source/hash確認・frozenコード配置・共通helper importが成功した。4 code cellのcompileと空output、実2 checkpointのepoch/config/所見順/共通fingerprintをCPUで照合。共通rank関数の対向順位が厳密に0.5になることも確認。
- CUDA初期化・forward・実MRI decode・追加学習・gold/ラベル読取・新しい重み取得・代理アップロード/保存実行/提出は行っていない。検査環境はローカルPython 3.12.13で、KaggleログのPython 3.13による全MRI処理と実採点は修正後の実行で確認する。元の97 unittest/GPU契約は以前の結果で、今回は設定・配置変更の検査を追加した。

## 続行依頼後の保存・更新計測と長期実験（10月4日、完了）

- 推論互換schema 1のbest_bce・best_auc・last・5/10/15/20 milestone保存を実装し、best.ptと選択予測CSVを設定した規則へのaliasにした。checkpoint indexにepoch・BCE/AUC・予測/metrics/重みhashを対応付ける。旧configのBCE規則、diagnosticのBCE強制、AUC未定義の非採用、gold非選択を維持する。lastはoptimizer/scaler/RNGを含まず、resume非対応を明示。
- optimizer post-step hookでAMP skipを除いた実更新を測定。機会・実更新・skipをepoch別と累積で記録する。500 microbatch毎、検証前、epoch保存後、完了時にprogress.jsonを更新する。追加forwardや学習lossの変更はない。
- checkpoint 7件・config 3件・extended集計8件・weak bootstrap 8件を含む全97 unittest成功（14.462秒）。CPU/CUDAで追加保存前後のRNG・sample順・BN・optimizer・scalerが完全一致し、AMP overflow skip、Reportなし推論、人工cache CUDA学習全経路を確認。Ruff check/formatは38対象ファイル、git diff --checkも成功。CSV比較のAUCは独立の人工5,400ケースでもpairwise定義と一致した。
- `artifacts/qa/20261004-extended-pipeline-v2/checks.json`で人工ノイズCUDAのAMP/backward/save/mask/提出3×12契約と旧e002 checkpoint互換を確認。QA 3.25秒、最大torch割当246,000,128 bytes。v1はlegacy専用補助チェックへImageNet正規化Cを渡す引数誤りで停止し、正しい旧e002を指定したv2で成功。productionコードはこの修正で変更していない。
- `artifacts/kaggle/20261004-extended-source-v1/rsna-knee-code.zip`を凍結（SHA-256 `9c85e2f120ba2eb9f0b3fc44b99ad328783f78026e46a52e8073b377d9bd496f`）。zip全ファイルが学習開始前の現sourceと一致し、imaging.pyのhashと既存cache fingerprintは不変。実学習中のsrc変更を行わない。
- [事前計画](../experiments/e006-e009-plan-20261004.json)に選択規則とconfig/input/source hashを保存して、fold 1のB/C各5 epoch、fold 0のC20 epochとBN統計固定20 epochを新規runで完了した。[完成集計](../experiments/e006-e009-extended-summary-20261004.json) は全4本valid=true。config・開始/終了/現source・入力hash、各epoch予測/mask/metrics、checkpoint/indexと選択、実更新数を照合し、CPUの実checkpoint読み込みでも内部epoch/fold/config/所見順/fingerprintを確認した。全体時間合計10,025.703秒、既存run上書きなし、gold cache/評価は無効、自作Public未測定。
- 実更新/機会/AMP skipはe006が4,142/4,145/3、e007が4,143/4,145/2、e008が16,932/16,940/8、e009が16,930/16,940/10。e008の最初5 epochのBCE・AUC・予測CSV hashは旧e005と全て完全一致した。BN固定モデルの20 BN module・60 running bufferは初期状態と完全一致し、affine 40/40・encoder 60/60のparameter tensorは学習で変化した。
- fold 1の事前学習差は12平均+0.093513、95%区間[+0.060727,+0.127135]。通常BNの5→8 epoch差は+0.009802、区間[−0.004061,+0.024398]。通常BN8→固定BN9は+0.019204、区間[−0.006118,+0.041423]で、平均優位は未確定。固定BNの5→9 epoch差は+0.024807、区間[+0.008788,+0.042459]。両20 epochとも終点は選択epochを下回った。[詳細と所見別の入れ替わり](controlled-experiments.md#続行後の実行結果e006e009実行完了)。
- weak CSV・供給groupだけのpaired bootstrapはPCG64、seed 20261004、3,000反復。fold 0の12平均はSynovitis陰性1groupの不在により1,124反復未定義で、区間は1,876反復に条件付き。補助11平均は全3,000反復で定義される。同foldのepoch/config選択・患者独立性・画像ラベルの正確さは保証しない。入力hashの解析前後一致を確認し、新規共有JSONには検査/group/Report識別子を保存しない。
- e008/e009の凍結コード・実重みから人工192px/24窓/3検査×12所見のCPU提出契約が成功。初期ImageNetファイル、Report、URL取得、CUDA初期化をCPU検査で禁止しても推論可能。実GPUでも同じ人工cacheから契約を確認し、CPUとの差は最大2.8313e-5/1.9849e-5。`artifacts/qa/20261004-extended-candidates-cuda-v2/checks.json` がvalid=true。最初のGPU検査は両forward成功後、CPU比較CSVのファイル名を誤って停止した。既存GPU出力の比較先だけを修正し、追加forwardやproduction変更なしでv2記録を作成、v1失敗記録は保持した。
- 所見別の入れ替わりを見た後の固定50:50 rank診断は12平均0.830249307、補助11平均0.815726517。gold・画像・モデルを読まず、係数/所見weight探索はしていない。全821検査を所見別に順位化し、欠損maskは評価時だけ適用。exact tieの二倍平均順位を整数のまま加算してから一度だけfloat化し、数学的な同順位を保った。[診断記録](../experiments/e008-e009-rank50-review-20261004.json) の共通helperと確定CSV全9,852値が完全一致、独立人工34拒否ケースと行順/左右交換/順位不変性も成功。固定BNとの差は補助11平均+0.029057、区間[+0.015129,+0.043443]だが、後付け候補選択の偏りを区間へ織り込んだものではない。順位値のBCEを確率校正の改善とは扱わない。
- [r001提出資産](../artifacts/kaggle/r001-bn-rank50-fold0/) は実選択2重み・凍結zip・診断とbyte同一の共通helper・未実行Notebookを含む。`artifacts/qa/20261004-rank50-contract/checks.json` で標準ライブラリ6契約チェックが成功し、保存済みCPU/GPU人工出力を各3×12で合わせたCSV契約もvalid=true。既存人工UIDだけを新QA内でnumericの架空UIDへ同じ対応で置換し、scoreと元CSVを保持した。モデルforwardやtorch importは行っていない。Notebookは2 checkpoint/source/helper hash、共通pixel fingerprint、cache1個・推論2回・全test一括rank・提出契約を確認する構成で、全code cellのcompileと未実行状態を検査した。
- 固定BN単体の [e009 v2資産](../artifacts/kaggle/e009-pretrained20-auc-bnfreeze-fold0-v2/) は元bundleに残った説明文のepoch誤記だけを修正した。`artifacts/qa/20261004-e009-bundle-v2-metadata/checks.json` でcode cell完全一致、重み/zip/index/config同一、旧6資産hash保持を確認した。追加モデルforwardは行わず、元Notebookと履歴を保持する。
- 公開方式の追加監査は [再現契約](research/public-reproduction-contract-20261004.json)。取得は公開code・metadata・小さい説明文書だけで、重み・MRI・ラベル表・予測配列取得、公開コード実行、外部書き込みはしていない。
- SDK版指定を`version_label="v32"`へ修正し、DINOsaur V32ソースを取得・hash固定した。[V32追補](research/public-reproduction-v32-followup-20261004.json)。表示0.937は作者報告で、歴史的Inputの厳密版と採点CSVは未確認。物理cropは [入力契約](research/physical-crop-contract-20261004.json) を調査しただけで、元MRIの追加取得・実decode・crop cache再生成はしていない。新しいKaggleアップロード・実test実行・提出も未実施。
- [完了時の検証記録](../experiments/e006-e009-verification-20261004.json) にテスト・各QA・主担当のrank9,852値再照合・Notebook/asset hash・JSON/文書リンク/台帳の確認をまとめた。[判断記録](../experiments/e006-e009-decision-20261004.json) は実施済みと次の提案を区別する。新しい台帳5行のgold/Public列は空欄で、r001へ学習時間を割り当てていない。

## 追加の添付分析と優先順位の再監査（10月4日）

- 新しい添付分析をHEAD `05b0710`、runtime/model/imaging/config、保存済み集計/epoch評価・予測CSVと照合した。原文を [user-analysis-review-20261004.txt](research/user-analysis-review-20261004.txt)、識別子を含まない数値/入力hash/調査根拠を [strategy-audit-20261004.json](research/strategy-audit-20261004.json) に新規保存した。
- C epoch 2→5は12所見AUC+0.007713、Synovitisは0.98→0.96、補助11平均+0.010232。7所見改善/5所見悪化で、Medial MeniscusはAUC−0.048177。同所見の悪化が全セルBCE増加の約77.8%を占める。添付の「Synovitis以外も改善」は支持するが、全所見の改善や長期化の成功を証明しない。
- goldを読まず、weak fold 0の812供給group/821検査を単位にpaired bootstrap（3,000回、seed20261004）。補助11所見C2−B1差+0.120629、percentile 95%区間[+0.078707,+0.162513]。C5−C2差+0.010232、区間[−0.005521,+0.025498]。12平均はSynovitis陰性1groupの不在で1,124回未定義となり、有効標本だけの区間を全体の確証に使わない。同じfoldでのepoch/config選択バイアス・患者独立性・教師誤差はこの区間の対象外。
- 現実装はBCE改善時のbest.ptだけ保存し、optimizer/scaler/RNG/Generatorの再開状態は保存しないことを確認。Cのepoch 5モデルも未保存。蓄積末尾はgroup_sizeで正しく補正され、fold 0の847回は更新機会でありAMPの実更新数の実測ではない。batch変更でmasked lossの検査重みも変わり得ること、config seedとmanifest seedの分離が未対応であることを確認した。
- 公式Overviewの検索索引でmacro12 AUC・Internet OFF/9時間・最終10月22日23:59 UTC（10月23日08:59 JST）を再確認。Host733826の索引本文で画像由来の正解とReportの不一致を確認。DINOsaur V4は表示Best 0.937 V32/最新V35、V32の厳密Inputは未取得。CoAtNet Training V8の未配布softラベルとgold選択は既存監査済みsourceを再読して確認し、gold選択を移植しない判断とした。
- 次の提案はAUC/BCE/lastモデル保存→fold 1のB/C各5 epoch→Cの20 epoch→BN固定だけの20 epoch比較。公開方式監査と自作Kaggle動作確認を並行させる。詳細は [対照実験の追加分析](controlled-experiments.md#追加分析に基づく優先順位10月4日)。
- 今回は調査とCSV/JSON再集計、文書更新のみ。src/config/script/test/Notebook/実験台帳を変更せず、学習・MRIデコード・GPU推論・データ/重み取得・外部アップロード・提出は実施していない。新しい提案を実装済み/学習済みと扱わない。
- 主担当も同じweak CSVと固定予測からbootstrapを独立に再計算し、上記区間を1e-12以内で再現した。調査JSONの厳密parse、保存source/添付hash、変更資料のローカルリンク40件とgit diff --checkが成功。GPUコード変更がないためunittest/人工CUDA検証は再実行せず、以前の71件成功と今回の調査検査を区別する。

## 対照実験の実装と診断（10月4日、再分析後）

- 共通tensor正規化、明示ローカルImageNet初期化、BN running statistics固定、epoch別weak CSV・所見別AUC/BCE/観測数/予測分布、固定train subsetのeval、gold監査無効化を実装。旧config/checkpointはrandom＋legacyを維持し、推論が初期重みを要求する経路はない。
- 追加評価でglobal Python/NumPy/torch CPU/CUDA RNG、DataLoader/sampler Generator、mixed train/eval modeを復元。人工CPU/CUDAでログ有無のsample順、model state、BN buffers、AdamW state/step、乱数状態が完全一致。詳細loss集計を付けても元のcanonicalセル平均BCEが完全一致した。
- 全71 unittestが成功。以前の31件にcache互換性13、model/input12、epoch診断3、対照集計7、実行記録5を追加。Ruff check/format（49 Pythonファイル）と差分の空白確認も成功。テストでは人工DICOMとノイズを使う。
- `scripts/check_synthetic_gpu.py`を追加。新規private QAに24架空weak＋予約2goldを用意し、gold cacheを作らず全cache readでgold UIDがないことをassert。両正規化でAMP/backward/optimizer/save、欠損mask、無効windowの予測不変、Reportなし3×12提出契約、メタデータ不整合の拒否を確認。実e002 checkpointも独立人工192px入力で契約がvalid=true。最大torch割当246,000,128 bytes、QA関数2.86秒で、実コンペ時間・性能とは区別する。
- 保存exportの完全性とpixel互換性を別検査するよう変更。実4,407検査・4,423ファイルのhash/CSV/UID/保存src/configが一致し、現repoのモデル・ログ変更を差分として明示しながら既存画像を再利用できた。保存exportを変更せず、imaging.pyのhashは従来と同じ。
- 公式URLからResNet18 ImageNet1K V1だけを明示取得。46,830,571 bytes、SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。公式prefixを照合しfull hashをローカル計算。コードlicenseと重みの条件を区別して記録した。実公式重みのfc込みstrict load、head/CPU RNG保持、人工CPU forwardを確認。元DICOMや新ラベルを取得していない。
- q001は原manifestを変えずfold 0学習側から16検査を選択し、dropout=0・100 epochで診断。初期同一subset eval BCE 0.694520、最良0.124106（epoch 32）、最終train-mode BCE 0.000188／eval BCE 0.179478、146.78秒。goldはpreflight/評価とも無効。定義可能AUCは7所見だけでmacro12は未定義。記録は [q001](../experiments/q001-train-fit.json)。これは少数trainへの適合診断で、独立CV・画像ラベルの正確さ・Publicの証明ではない。
- 公式SDKの通常認証とread-only要求でRules全文、Hostの外部LLM案内と全コメントを取得。認証値の表示・手動読み出し・コピー、外部投稿は行っていない。HostはreportからのLLMラベル抽出を条件付き許可し、NC制限だけでデータを禁止せず、賞金受領だけで商用とは扱わないと説明。公開READMEの想定用途とCC本文も照合し、現在の研究・学習目的と帰属等を維持して通常ライセンスで採用を進める判断に更新した。作者への追加許可を一律必須とはしない。確認UTC・URL・短い引用・根拠hashは [利用条件監査](research/kaggle-source-eligibility-20261004.json)。入賞時のWinner公開条文の不整合はその段階で確認する。
- [weak state監査](../experiments/weak-state-audit-20261004.json) でP/N/B/U/Mをweak全体・train・各foldに集計。Synovitisの全weak陰性は49件、fold別は1/25/7/10/6件で、全体の不足とfold 0の偏りを区別した。Effusion陰性2,674件はN 1,230＋B 1,444。N/Bは現行で同じ0でも意味が異なる。gold値・UID・group・reportは共有集計へ保存していない。既存分割とラベルは変更していない。
- e003/e004/e005を同じ192px・fold 0・seed・5 epochで新規実行。A→Bは正規化、B→Cはencoder初期化だけを変更し、MIL head初期hashも全て一致。Aの全5 BCEとbest予測CSVは旧e002と完全一致した。BCE選択の結果はA epoch 1: 0.430481/AUC 0.557023、B epoch 1: 0.411992/0.644652、C epoch 2: 0.371395/0.766895。cache事前検査込み・import除外の所要時間は1,104.29/1,043.59/1,039.36秒。gold cacheのpreflightと評価を無効にし、Publicは未測定。
- CSV/JSONのみの集計器でsource/config/manifest/input/subset/hash、全epochのUID集合・所見順・観測数・AUC/分布・historyとのBCE整合、first minimum-BCE選択とbest予測hashを監査しvalid=true。実checkpointのepoch/config/foldは別のCPU読み込みでも照合した。[比較集計](../experiments/e003-e005-controlled-summary-20261004.json)。CはBより12/12所見AUCが改善し、補助11所見平均も0.626893→0.747522。Cを暫定比較基準としたが、C epoch 5のAUC最大値0.774608を後付け採用していない。[判断記録](../experiments/e005-decision-20261004.json)。追加fold 1のB/C学習は未実行。
- 凍結code zipを新しいPythonプロセスで読み込み、実A/B/C checkpointと既存の独立人工192px・24窓cacheでReportなし3検査×12所見の提出契約が全てvalid=true。初期化関数とtorch.hub取得関数を呼べないようpatchし、Cの初期化pathを不存在に変えたQAコピーでもCSV hash一致。code zip SHA-256 `5f4f7a016225beec5e30d1696030f4ee879e7db1ceb600575396c436c6cacd78`、C checkpoint `677f5e793ebc9cfbb8eb76ab267582d7916b23034056af0bc3ac7c68bd448084`。[提出契約集計](../experiments/controlled-submit-contract-20261004.json)。実testのDICOM decode・Kaggle実行時間・新規採点を検証した結果ではない。
- `artifacts/kaggle/e005-pretrained-imagenet-fold0/` に上記code zipとbest.pt、hash/帰属記録、未実行の専用Notebookを新規保存。Input名は提案名で実mountは手動追加後に確認する。元MRI・レポート・ラベルCSV・認証値はbundleに含めず、アップロード・提出はしていない。過去e002/run/cacheを上書きしていない。
- C終了後、wrapperのKeyboardInterrupt/SystemExit記録とsource変更時のstatusを修正し、mockで中断/通常失敗/不一致/成功/再実行拒否/CUDA guardを確認。学習処理は変更せず、旧runの記録も書き換えない。今後は終了hashもcompleted保存前に確認し、対照集計器は記録済みの場合だけ追加照合する。
- 最終確認で共有JSON 21ファイル、Markdown 12ファイルのローカルリンク104件、PowerShellブロック22件を検査。台帳7行のID重複がなく、A/B/Cのgold/Public列は空欄。現sourceと実run、C bundleのコード・重みhash、専用Notebookの未実行状態と全コードcellのcompileが一致した。Gitで追跡されるdata/artifacts配下は従来の各READMEのみ。

## 添付分析の照合と次工程の調査（10月4日）

- 添付原文を [user-analysis-20261004.txt](research/user-analysis-20261004.txt) へ保存。SHA-256は `ee844f85bf1281db9d8e3a1a1f0261249ea887414c472dfd97a60e927fcb9f97`。判断は [次の実験計画](research/next-experiments-20261004.md)、出典・既存記録hash・容量計算は [調査記録](research/reanalysis-20261004.json)。
- 今回は実データ・重み取得、実MRIデコード、GPU推論、追加学習、Kaggle実行・アップロード・提出を行っていない。src・config・Notebook・script・test・実験台帳に変更はない。新しい学習結果を記録していない。
- runtime/model/imaging/contractsと現行設定を照合。所見別Attentionは既存、ResNet18はrandom、画像正規化は[-1,1]。batch 1でもencoderは同じ検査の24 windowを処理する。batch変更はBNだけでなく欠損mask付きlossの検査寄与も変える。学習と検証のloss集計方法も異なり、train-valid差を純粋な過学習量と解釈しない。
- installed PyTorch 2.10のDataLoader/RandomSamplerソースで、shuffle=Falseでもiterator生成がGeneratorを消費することを確認。train/valid共有Generatorに追加評価を入れると次epochの順序が変わり得る。専用評価GeneratorとRNG・module状態の復元、人工データでの学習不変性の確認は次工程の設計であり、まだ実装していない。
- export検査器が現在のsrc全体のhash一致を要求し、画像fingerprintがimaging.pyとpreprocess設定から決まることを確認。画像を変えないログ・モデル変更でのcache再利用には検査の分離が必要。既存hash保護を緩めていない。
- Notebook generatorの未圧縮容量式を再計算。4,407検査×3系列×8 window×3 channelで192/224/256/288/336pxはそれぞれ11,697,094,656／15,921,045,504／20,794,834,944／26,318,462,976／35,822,352,384 bytes。現行18,000,000,000 bytes未満の事前ガードは256px以上を止める。この値はrepoの保守的予算で、現在のKaggle公式枠や最大RAM使用量ではない。LIMITはprefix制限で全件shard処理ではない。
- Torchvision 0.25とPyTorch 2.10の公式資料、作者のsoft/hard比較、EfficientNet入力比較、既存CoAtNet model card、別Notebookのスコア表示、2024腰椎優勝者の記述を照合した。作者報告を自分の再現値と区別し、別条件の改善幅を現在の期待値へ転用しない。別NotebookのPublic 0.942は表示確認のみで、内部処理・Input・重みの来歴は未監査。
- 公式Overviewの索引本文で12所見平均AUC、Internet OFF・9時間、期限を再確認。最終期限は10月23日08:59 JST。CC公式資料とHostの別データへの回答から、提供元の条件と大会の条件を分けて確認する計画とした。Rules・外部LLM案内本文は取得できず、採用ラベルの許可・禁止は未確定。外部への問い合わせは送っていない。
- goldは既に監査値を見ているため完全未観測の最終holdoutとは呼ばない。学習・checkpoint選択・prompt・ensemble係数の調整には使わない。公開重みを後付けfoldで評価しても独立OOFとは主張しない。
- 調査JSONのparse、添付原文・保存コピー・既存2記録のSHA-256、全解像度の容量式、Markdown 11ファイルのローカルリンク、差分の空白を確認。実装・台帳に差分がないことも確認した。コード変更がないため31 unittestやGPU契約を再実行しておらず、10月3日の成功結果と今回の文書検査を区別する。Gitで追跡されるdata/artifacts配下は従来のREADMEだけ。

## e002の提出判断のための追加診断（10月3日）

- 今回の質問には保存済みCSVの再集計で対応し、GPU推論・実MRIデコード・追加学習・モデル取得・アップロード・提出は行っていない。学習や推論コードも変更していない。
- fold 0の学習3,386件で観測された所見別陽性率を定数予測として計算。検証やgoldから定数をfitしていない。同じ検証821件・観測5,727セルに欠損maskを適用したBCEは0.415326862。e002の保存予測からのBCEは0.430481425で、記録されたtorch集計値との差は1.2e-9未満。定数予測のBCEが低いことから、BCE単独を画像学習の有効性の証拠とはしない。
- 観測されたbinary weakラベルのmacro AUCは自作モデル0.557022616、定数予測0.5。BCEとAUCは異なる性質を測るため、「画像信号を全く学習していない」とは断定しない。特にSynovitisは陽性100・陰性1であり、weak macro AUCも不安定な指標を含む。gold 58件のAUCを隠しtestの予測値へ読み替えない。
- e001/e002のweak予測CSV同士、gold予測CSV同士のSHA-256がそれぞれ一致。5 epoch実験から提出するbest.ptはepoch 1が選ばれており、新しい予測改善はない。
- 識別子やレポートを含まない集計・元ファイルhash・計算方法・判断は `experiments/e002-submission-review-20261003.json`。学習側だけの陽性率計算、UID集合一致、BCEの再現、定数予測AUC=0.5をassertで確認した。
- 公式Overviewの検索索引で評価が12所見平均ROC-AUC、Internet OFF・9時間制限を再確認。Rules本文と採用ラベルのページ本文は今回のWeb取得でも開けず、大会利用の許可・禁止を確定していない。汎用事前学習候補について [Torchvision ResNet18の公式資料](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) を確認。実装・重み取得・効果検証は次工程。
- 診断JSONの構造と比較結果、ローカルリンク49件、差分の空白を確認。学習・推論・Notebook・設定・既存チェックの実装には変更がなく、全31 unittestは前回の成功結果を維持している。今回のCSV診断には上記assertを使用し、GPUテストは再実行していない。

## ラベル監査・固定分割・ローカル研究（10月3日）

- 認証済み公式APIから自分の提出履歴を読み、既存の1提出がCOMPLETE・Public 0.924と確認。提出日時は10月2日05:57:22 UTC、提出URLのscriptVersionId=354569007。該当Notebookの最新版metadataはVersion 1で、Inputはraptor-knee-widedense、Internet OFF。APIが返すInput参照は版なしなので、保存Input版・実行時間・global IDと表示版の直接照合は未確認。新規提出はしていない。結果はp001として台帳へ保存。
- 公開7候補のDataset版、metadata、ライセンス、生成方法とgold利用を公式CLI/SDKで監査。作者の学習コードはDreaddevelopment Version 8、Pilkwang Version 15を取得して読み、実行していない。前者のgold AUCによるcheckpoint選択、後者の学習分岐での公式ラベル混合を確認した。公開重みの厳密な履歴まで証明したものではない。
- vmohitrao Dataset Version 3を非商用研究候補として取得。元train.csv hash、weak_labels・weights・states・provenanceの4つの公開hash、4,349行のUID集合、全セルのstate/value/maskが一致。goldのUIDは含まれない。作者READMEが58検査を抽出・prompt開発から除外したと宣言しており、その根拠のhashを保存した。開発ログを独立監査してはいない。CC BY-NC 4.0の大会提出への利用可否は未確認。
- prepareでweak 4,207検査・gold 58検査を作成。全欠損138件、goldと同一レポート・groupで連結される4件を除外。供給report_groupを保った5-foldは821・891・888・778・829件。fold 0は学習3,386／検証821件。元入力hash、元ラベルとの一致、gold予約、group連結、fold別・所見別分布を追加監査した。
- fold 0のSynovitisは観測101件のうち陽性100／陰性1、欠損率87.7%。CSV整合性の成功はラベルの臨床的正確さやクラス分布の妥当性を保証しない。患者独立性も未確認。
- 新規の標準ライブラリunittest 12件でUNK→欠損、0とsoft labelの保持、出所hash・state/mask不一致、gold宣言不足、改ざんgroup、除外行を橋渡しにするgold漏洩、未観測クラス等を検査。既存を含む31件が成功。srcは変更しておらず、exportのsource hashと前処理fingerprintを維持。
- ユーザーはKaggleで元画像とcacheを目視比較し、見た範囲では問題なしと回答。件数とUIDは未報告であり、全画像の品質確認や医学的判定として扱わない。
- 人工ノイズ22検査とtest 3検査を独立したQAディレクトリに生成。192px・24 windowの設定でCUDA/AMP/backward/optimizer、1 epoch完走、checkpoint保存・再読込、Reportなしの提出契約を確認。全3行×12所見の契約がvalid=true。最大torch割当467,144,192 bytes。人工データの精度・時間・VRAMを実コンペ性能として記録しない。結果は `artifacts/qa/20261003-synthetic-cuda/checks.json`。
- 実キャッシュでe001-smoke-fold0が完了。weak検証BCE 0.4304814244、選択後のgold macro AUC 0.4871779862。checkpointは44,830,657 bytesで保存。historyの205.33秒はepoch内の経過時間で、cache事前検査と最後のgold評価を除く。コマンド全体の所要時間は未計測。
- e002-baseline-fold0は同じラベル・fold・seedでepoch数だけ5へ変更し、新規runで完了。weak検証BCEはepoch順に0.430481、0.430906、0.447424、0.502899、0.448503。最小のepoch 1を選択し、その後にgoldを評価した。gold macro AUCはe001と同じ0.487178。train関数は事前検査・最後のgold評価を含め1,082.83秒、Python importは計測外。epoch部分は1,034.21秒。実行環境・入力hash・config・全history・checkpoint hashはrunと共有可能な実験JSONへ保存した。Public LBは未測定。
- `artifacts/kaggle/e002-baseline-fold0/` を新規作成。code zipは許可されたsrc 9ファイルとbaseline.jsonだけで、SHA-256は9fd958c4585d3bced5552cb1c949a103ba6209fc4ddaa3e18b6d07219d49545f。別ファイルで置いたbest.ptのSHA-256は903cce571904a8e74ff54d3242e3cb6293db747b293475be378578767eb3f29b。展開zipのsource hashと学習記録が一致した。
- 新しいPythonプロセスで展開zipのsrcを読み込み、実e002 checkpointと人工test cacheで予測。Reportなしの3行×12所見について、UID集合・順序・列・有限値・0〜1範囲の提出契約がvalid=true。結果は `artifacts/qa/20261003-e002-submission/checks.json`。実MRI/testのdecodeとKaggle実行は未検証。bundleにRESEARCH-ONLY.jsonを追加し、アップロード・提出はしていない。
- 最終資料のローカルリンク48件、PowerShell手順ブロック15個、実験JSONと台帳3行の構造、保存run configと元configの一致を確認。run configのhashと元ファイルのhashは記録上区別した。Ruff・全31 unittest・差分の空白検査が成功。data/・artifacts/の追跡対象は各READMEだけで、ラベル・画像・分割・予測・重みはGit除外を確認した。

## 実学習へ移る前の確認（10月3日、以下は当時の状態）

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
