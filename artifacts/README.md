# GPU端末で生成する成果物

重み・予測・ログ・コードzipはGit対象外です。

```text
artifacts/runs/e001-fold0/  best.pt, config.json, run.json, history.json, gold_metrics.json
artifacts/kaggle/          rsna-knee-code.zip, bundle-manifest.json
artifacts/submissions/     submission.csv, submission.meta.json
```

実験ごとに新しいrunを作ります。GPU端末のrunディレクトリをバックアップし、実測スコアと環境情報の要約を experiments/ に記録してください。コードzipにはsrcとconfigsだけを含めます。モデル重みの受け渡しとKaggle入力への追加は別途行います。
