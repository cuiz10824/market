# 本项目的研究基础代码

目标：对单个行业或宽基指数，预测未来5、21、63、126个交易日的方向和收益。当前包含可运行的数据准备及模型适配，不输出正式预测，也没有接入网页。

后续完整训练已移到 [training](../training/README.md)，参数在 [models](../models/README.md)。这里保留共享特征、早期研究模块及原始功能检查说明；旧功能检查不是完整训练的成绩。

## 文件

- `dataset.py`：从SQLite读取14条主要指数；默认拒绝28条旧申万社区样本。严格按观测交易日历对齐，构造历史动量、波动、均线偏离、回撤等10个价格特征，以及未来涨跌标签。
- `models.py`：LightGBM二分类基础适配，以及statsmodels状态切换拟合、固定参数前向过滤。库在函数内加载，数据准备无需安装这些包。
- [开源参考代码](../references/README.md)：原始示例、许可证、提交号与SHA-256。
- [数据库复查](../../data/reports/readiness_2026-10-18.md)：当前覆盖及缺失项目。

## 数据准备：不需要机器学习依赖

在仓库根目录执行：

```sh
python code/research/dataset.py --code 000300 --horizon 21
python code/research/dataset.py --code 399989 --horizon 126 --output medical_research.csv
python code/tools/review_database.py --as-of 2026-10-18
```

`--output`是可选的研究导出路径，避免把实验大表重复提交到Git。导出的`target_return`用小数表示收益，`0.05`是5%；`target_up=1`表示严格上涨，持平或下跌为0。`date`是特征截止的收盘日，`label_end_date`是未来第N个交易日。

每条样本要求此前252个交易日和未来整个标签区间连续可得；不补零、不填充缺口。末尾尚无结果的记录不生成训练标签。时间切分保留扩展训练窗，并剔除所有标签结束日大于或等于测试起点的训练样本，防止半年标签泄漏。

`review_database.py`离线复核数据库，引用同日期已保存的联网核对记录，不自动把旧记录当成新一次联网检查。没有该日期核对记录时，报告会将最新行情核对标为未确认。

不同代码、不同期限分别构造样本和切分。当前缺少正式指数发布日期审计，即使能生成标签，也只能用于探索；尤其不要把中证A500发布前的回溯记录当作当年真实可投资的历史。特征在收盘后才可用，标签不是假设当天收盘成交后的可实现策略收益。

## 运行模型适配检查

研究依赖与网页分开，Python **3.12** 已验证。推荐独立虚拟环境，按 [requirements-research.txt](../requirements-research.txt) 安装。网页和数据库审计仍可使用系统Python。

Windows PowerShell，在仓库根目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements-research.txt
.\.venv\Scripts\python.exe code/tools/smoke_research.py
```

Linux：

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r code/requirements-research.txt
.venv/bin/python code/tools/smoke_research.py
```

本次已在Linux云环境执行模型检查：LightGBM使用609条沪深300训练样本，验证126条后续样本能输出有效概率；状态模型使用1000条真实历史收益拟合并成功收敛，随后检查1100条过滤结果。改变末尾未来收益，没有改变此前过滤概率；更换原训练前缀会被拒绝。记录见 [运行报告](../../data/reports/code_smoke_2026-10-18.json)。Windows命令尚未在真实Windows机器执行。

这两个检查是分别执行的功能检查，未将状态概率与分类模型串联，也未计算或声称预测准确率。概率未经校准，不发布模型、不覆盖网页现有结果，也不保存训练后的模型文件。

## 后续正式实验的顺序

1. 建立固定指数定义、发布日和数据版本；必要时限定回溯数据的研究用途。
2. 比较无条件上涨概率、趋势基准、LightGBM，以及加入训练窗状态信息的模型。
3. 按时间滚动，训练、内层验证、最终测试分离；相邻标签重叠不能当成独立样本。
4. 评估Brier分数、概率校准、方向表现、收益误差与分位数区间覆盖，按牛熊时段分别检查稳定性。
5. 验证优于基准后，再生成网页读取的预测JSON。基金实际表现还需接入净值、分红、费用等数据。
