# 本轮实际训练：CatBoost、改进LightGBM、Kronos-mini

结果见 [完整对比](../../data/reports/training/enhanced_20261018_5ccccf4b_95bacd54_2e2174c4/README.md)。本轮实际训练4种树模型候选，8指数×5期限；保存60个共享原生模型包，重载95,928条候选测试预测通过。按预设验证门槛0/40任务获得替换资格，保留失败结果，网页继续按旧模型输出。候选的原始非零预测也完整保存供检查。

## 环境

```sh
cd /workspace/market
.venv/bin/python -m pip install -r code/requirements-enhanced.txt
# Kronos可选依赖：先装官方CPU torch，避免无用CUDA依赖
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r code/requirements-kronos.txt
```

CatBoost1.2.10、LightGBM4.7.0、pandas2.2.3、NumPy2.3.5和CPU torch2.14.1实际安装并验证。日常界面仅需要html/requirements.txt，不需要安装PyTorch或下载Kronos。

## 数据、训练和验证

```sh
.venv/bin/python data/tools/build_ohlcv.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python code/training/train_enhanced.py
.venv/bin/python code/training/verify_enhanced.py
.venv/bin/python code/training/report_enhanced.py
```

相同运行指纹拒绝覆盖。数据为原始快照中已保存的OHLCV，80个特征由html/daily/rich_features.py共享，训练与日常完全同定义。不允许全序列拟合归一化、补造缺失交易日或把未来行业涨跌用于当前输入。

新候选在3个扩展训练验证窗选型；每个验证起点之前，所有指数的训练标签必须已结束。测试使用2024起点的固定模型，A500另用2025起点固定模型。单个指数的随机行切分或混排后随机切分都不采用。本轮测试价格区间已被观察，是探索性复测。

## 日常加载

`code/models/latest_enhanced.json`指向本轮结果。日常程序读取验证选择，只为获批任务加载新模型。获批后量价日线与基准日历、历史重叠成交量单位核对通过才更新；数据或新特征未就绪时明确提示并沿用旧模型。原权重、旧库和旧报告保留。

## Kronos实际阻塞

官方源码按`67b630e67f6a18c9e9be918d9b4337c960db1e9a`固定并保留MIT许可，位于code/references/kronos。官方类与CPU torch导入通过；预训练权重下载的Hugging Face请求被当前代理CONNECT 403拒绝，**尚未运行模型预测，不能报其性能**。

恢复官方域名联网后运行：

```sh
.venv/bin/python code/training/benchmark_kronos.py
.venv/bin/python code/training/report_enhanced.py
```

脚本从官方模型仓库获取修订号和safetensors，记录SHA256、核对官方LFS摘要；仅缓存到忽略目录`.model-cache`。没有改用镜像绕过网络拒绝。只执行预训练模型的零样本对照，不自动微调或将未知预训练截止的数据结果用于日常选择。测试起点、参数、缺失成交额通道及未来收益取值规则写在协议和状态文件中。

联网要求：保留腾讯与东方财富，补充huggingface.co及其官方文件下载域名。配置草稿保存不代表当前运行环境已经放行，应用后须实际重试。
