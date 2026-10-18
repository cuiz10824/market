# 小型共享神经网络：实际实验

本轮结果见 [完整实验报告](../../data/reports/training/neural_20261018_14abddff_9a829429/README.md)。测试前验证选择的组合，半年MAE从12.306降至11.613个百分点，但一月/三月变差，短期基本无改善；没有自动接入每日网页。

训练与测试入口：`train_neural.py`；固定协议：`neural_protocol.json`。使用DLinear、NLinear、小TCN、小PatchTST四种项目适配结构；不依赖外部预训练权重，不涉及Kronos的下载阻塞。

## 输入与目标

8指数共享一套网络参数及指数ID偏置，每个网络直接输出未来1/5/21/63/126交易日累计价格收益。输入比较252/756/1260交易日；每个起点先确保有1261个连续观测，因此各长度使用同样的合格起点。

过去日线按5交易日聚合成OHLCV组，得到51/152/252个分组，6通道由`code/research/sequence_data.py`定义。按预测日向前对齐，第一组可短于5日；组内开高低收和成交量保留，但不保留每根日线的组内排列。PatchTST再将8组（约40交易日）切成patch。此版本不同于142汇总特征的树模型输入，也不是逐根1260日的原样Transformer。

目标不转成分类。五个输出按固定`0.01*sqrt(h)`缩放后使用MAE训练，报告还原为累计收益百分点。损失对未知标签逐头掩码：所有被用于训练的标签结束日期必须严格早于切分点，最新训练也能利用近期短期标签，不填造尚未发生的半年收益。

## 实验顺序

1. 第一个种子分别比较四种结构×三种输入长度×三个历史验证窗，共36次训练；按等验证窗、等指数、等期限的缩放MAE为每种结构选长度。
2. 第二个种子复现选中的四种长度×三个验证窗，共12次训练；两个种子的预测等权集成。
3. 预先冻结逐任务研究选择与训练轮数。训练轮数为该结构两个种子、三个验证窗所选轮数的中位数取整。
4. 每种结构用两个种子在2024/2025切分点前重训，保存16个测试模型；另保存8个latest模型。总计72份原生模型包。
5. 全部测试输出与旧模型的真实收益和日期逐条对齐；latest只是测试后的重训权重，不另算独立成绩。

训练每5个观测交易日采样，早停验证同样每5日计算；最终验证指标和测试都逐日计算。每次最多12轮、最少2轮、耐心3轮，CPU固定2线程，使用固定的小容量结构。所有日志和各候选窗口的失败结果保留。只测试验证选中的长度，不根据测试重挑长度或种子。

本轮两个随机种子为20261018、20261019。两个种子不是统计稳定性的证明。既有测试区间此前已观察，属于探索性复测；跨指数相关和长期标签重叠也限制结论，不宣称已建立显著优势。

## 环境和复现

当前云环境已安装所需CPU PyTorch与safetensors。新环境先装官方CPU wheel，再安装研究依赖：

```sh
cd /workspace/market
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r code/requirements-neural.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python code/training/train_neural.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python code/training/verify_neural.py
.venv/bin/python code/training/report_neural.py
```

相同运行指纹拒绝覆盖。模型权重只保存为`safetensors`，加载时核对SHA256；不读取不可信pickle，不下载第三方模型权重。共享网络定义在`code/research/neural_models.py`，训练/预测/保存逻辑在`neural_engine.py`。

本轮实际执行的训练入口原始字节已按SHA归档到`data/training_sources/`，独立重放使用现有的历史源码校验机制。后续只修复了运行管理的指针保护规则：新的运行允许更新自己的latest_neural研究指针，不把该指针误判为被改写的历史权重；已训练网络、数据处理和损失实现未改动。

结果指针为`code/models/latest_neural.json`，报告在`data/reports/training/neural_*/`。指针标记`research_only=true, auto_deploy=false`：本轮是训练测试实验，当前每日网页与Windows依赖不变。验证改善≥1%的任务会标为研究候选，仍不等于已证明未来优势或已上线。

首轮在首个模型导出前遇到NumPy字符串日期归约兼容问题并终止；修复后的运行拥有独立源码指纹。失败尝试记录和源码单独保留，未计入性能。
