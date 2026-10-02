# GPU端末で準備するデータ

このディレクトリの生成物はGit対象外です。ローカルPCには競技データを取得していません。

GPU端末での推奨配置:

```text
data/raw/           train.csv, train_series.csv, train_series/, test.csv, ...
data/labels/        weak_labels.csv, provenance.json
data/manifests/v1/  weak.csv, gold.csv, manifest.json
data/cache/train-v1/  cache.json, coverage.json, <study>.npz
data/cache/test-v1/   提出時にその回のtest.csvから構築
```

元データは別ドライブにも置けます。CLIの `--data-root` で絶対パスを渡してください。元画像は約570GBと報告されているため、GPU端末で実際の一覧・サイズを確認してから容量を確保します。ZIPと展開後データが共存すると容量が増えます。

この実装は最大3シリーズ、8窓、各窓3枚、192px、uint8を保存します。4,407検査と仮定した非圧縮画像配列の概算は約10.9GiBです。これは設計上の計算であり、実測容量や最適な画像設定ではありません。DICOM全情報の可逆圧縮ではありません。
