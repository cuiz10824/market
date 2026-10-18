# 研究数据库工具

数据下载、原始快照重建与审计从 `code/tools` 移到这里，不属于每日推理。仓库根目录执行：

```sh
python data/tools/market_data.py audit
python data/tools/market_data.py build
python data/tools/market_data.py fetch --provider tencent
python data/tools/review_database.py --as-of 2026-10-18
```

`build` 是主动重建研究库操作，不会被网页或每日服务调用。已训练模型的历史重放需要原训练数据库，日常新行情应使用 `html/predict.py --refresh`，写入单独的增量库。数据解析测试：`python -m unittest discover -s data/tools -p 'test_*.py' -v`。
