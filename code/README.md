# 训练、测试与模型

此目录保留训练和测试相关代码、算法参考及已训练参数。每日服务、行情增量更新、实际运行的回归推理和H5页面均放在 [html](../html/README.md)。研究数据库管理工具放在 [data/tools](../data/tools/README.md)。

|位置|用途|
|---|---|
|[training](training/README.md)|训练、滚动验证、历史测试、报告和模型重放|
|[research](research/README.md)|训练样本、交易日特征和标签、时间切分及模型适配|
|[references](references/README.md)|已下载的参考算法、固定版本与许可|
|[models](models/README.md)|已训练模型及索引；每日推理读取这里的原回归参数|
|`tests/`|训练、特征、选型和模型保存/重载测试|
|`requirements-research.txt`|训练研究依赖；日常依赖另见html/requirements.txt|

`training/smoke_research.py` 检查研究模型接口及时间因果性。历史完整模型加载器移到html/inference，训练入口已更新导入路径。迁移前的训练源码原始字节保存在data/training_sources，旧报告与权重未重写，旧实验仍可按原指纹和预测记录复验。
