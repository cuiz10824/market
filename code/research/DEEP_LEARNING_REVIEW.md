# 深度学习方法与Kronos权重来源调查（2026-10-18）

本轮只调查公开仓库、文件清单、摘要声明和官方文档，没有下载镜像权重，没有运行新的模型或更改当前网页预测。来源证据见 [sources.json](../../data/reports/model_research/2026-10-18/sources.json)。搜索结果不是GitHub全库穷尽检索；本次主要查询Kronos-mini、Kronos weights、Kronos safetensors及官方问题区。

## 找到的权重转存仓库

[Alexptully/kronos-weights](https://github.com/Alexptully/kronos-weights)，本次观察提交为 `e743182a4a8961e814c63ca84ec73af39b3573e5`。

其README声明是NeoQuasar官方Hugging Face权重的未修改镜像，列出：

| 模型目录 | 配套tokenizer | 仓库声明模型大小 |
|---|---|---:|
| Kronos-mini | Kronos-Tokenizer-2k | 约16MB，不含tokenizer |
| Kronos-small | Kronos-Tokenizer-base | 约99MB，不含tokenizer |

已读取mini与2k目录的公开清单，均列有`config.json`及`model.safetensors`；没有读取这些模型内容。其[SHA256SUMS](https://github.com/Alexptully/kronos-weights/blob/e743182a4a8961e814c63ca84ec73af39b3573e5/SHA256SUMS)声明4份权重的摘要与Hugging Face LFS对象ID一致，摘要记录于sources.json。这是镜像作者的声明，**尚未与官方固定修订的摘要独立核对，也没有计算实际权重字节的摘要**。README声称MIT许可，不等于已经完成原始权重许可核查。

当前存在明确的Hugging Face代理CONNECT403拒绝。环境网络规则要求通过支持的配置流程更改权限，不能把同一被拒绝权重改由镜像下载来绕过。此前官方域名配置草稿已经保存，但没有证据表明当前运行环境已应用；本次没有重复未改变条件的官方下载，也没有clone包含权重的镜像。恢复官方访问之后优先下载固定官方修订，或在合法放行后核对上述镜像与官方摘要，再做推理。

官方[Kronos README](https://github.com/shiyu-coder/Kronos)当前列mini为4.1M参数、2048上下文，small为24.7M参数、512上下文。因此“模型更大”不意味着能直接读取五年1260日日线：small的512限制不适合直接照搬1260日输入；mini的上下文长度在文档上可覆盖，但仍需确认实际checkpoint配置。已有项目诊断脚本使用256日输入，尚未运行，不能将其描述为完成了五年序列对照。

官方问题区[#265](https://github.com/shiyu-coder/Kronos/issues/265)、[#307](https://github.com/shiyu-coder/Kronos/issues/307)存在关于公开checkpoint训练截止日期、归一化修复与旧权重关系的询问。本次没有核验到足以消除歧义的官方权重版本证明；这些社区讨论不是已证实权重泄漏的结论。结果仍只能作为预训练模型诊断，不能直接当新独立未来成绩。

## 针对本项目的优先级

当前只有8个相关指数、36,857条日线，五年输入预热后有效训练起点更少，标签也重叠。更合适的初始方案是小模型跨指数共享参数，输入加入指数身份，直接输出1/5/21/63/126交易日收益；不要给8个指数各训练一个大Transformer。下表是设计建议，尚未在本项目训练验证。

| 方法 | 参考代码 | 项目用途 | 是否依赖外部预训练权重 |
|---|---|---|---|
| DLinear / NLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | 小容量神经基线，检验线性时序模型；NLinear的归一化设计可作为分布变化对照 | 不需要，使用项目数据训练 |
| 小型TCN | [Qlib TCN](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcn.py) | 因果扩张卷积覆盖长历史，同时保留短期变化；可设计日/周双分支 | 不需要 |
| 小型PatchTST | [官方实现](https://github.com/yuqinie98/PatchTST) | 将五年日线切成约20/21日的patch，降低注意力序列长度，保留原始时序形态 | 不需要；可在项目训练期内自监督预训练，之后用收益标签微调 |
| N-HiTS | [NeuralForecast官方实现](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py) | 多尺度采样与插值，比较短中长期变化，不假设牛熊严格固定周期 | 不需要 |
| 小型GRU/ALSTM | [Qlib GRU](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gru.py) | 简单循环神经网络对照；五年逐日展开成本较高，可用日/周多尺度输入 | 不需要 |
| Chronos-2-small / Chronos-Bolt-tiny | [Chronos官方库](https://github.com/amazon-science/chronos-forecasting) | 通用预训练模型对照，官方表列28M/9M参数；不是A股行业专用模型 | 需要下载权重，官方文档指向Hugging Face；不能当作当前网络问题已经解决 |

建议先做DLinear/NLinear和小TCN两个低容量基线，再做小PatchTST、N-HiTS；官方权重可用后补充Kronos-mini与Chronos对照。PatchTST更贴近“直接读五年序列”的要求，优先于继续往树模型追加更多长期汇总特征。N-HiTS是适合比较的结构，不代表其在股市中已优于现有模型。

Qlib公开榜主要针对个股横截面IC、选股组合和超额收益，不等于8个行业指数的绝对收益MAE。不能拿其公开年化收益率替代本项目的误差测试；参考网络结构时必须更换数据组织、标签和评价指标。

## 推荐的测试协议

1. 对照输入252/756/1260交易日；每组使用匹配训练起点和同样预测日期，避免长窗口预热损失样本造成混淆。
2. 直接预测五个累计收益，或预测日收益路径后按同一定义累计；不以价格点位拟合的漂亮曲线代替收益误差。
3. 每个窗口只用预测日前的观测；归一化仅使用过去窗口或训练集，验证/测试不参与拟合。自监督预训练也遵守日期隔离。
4. 共享网络跨8指数训练；先限制参数量及训练预算，固定少量随机种子，验证早停。参数选择只用验证期，不在已有测试期反复搜索。
5. 报告MAE、带符号误差均值及标准差、RMSE、各行业和各年表现，同时保留零预测、当前模型、一年对照。长期标签重叠需要块式不确定性估计，不能假设所有行独立。
6. 原有2024—2026测试价格已经观察过，继续使用时标为探索性比较；预训练模型额外检查训练截止日期。

这轮没有开始上述训练，没有任何新性能数字。已有五年树模型实验未改善，不能据此断言时序网络也必然失败；同样，网络更复杂也不保证改善。
