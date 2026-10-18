# A股行业观察

本地H5网页：上栏显示实际涨跌，下栏显示已训练回归模型的未来累计涨跌幅。白底黑字，上涨红色、下跌绿色。当前仅8个方向：中证A500、沪深300、能源、医药、消费、制造、金融、科技，各一个代表指数。

## Windows 使用

1. 下载仓库ZIP并解压，保持完整目录结构。
2. 安装官方 **Python 3.12**，双击 `html/启动网页.bat`。首次自动创建 `.venv-daily` 并安装每日预测依赖，随后打开 `http://127.0.0.1:8765`。
3. 保持命令窗口开启。程序启动时及每30分钟检查已收盘行情，页面每60秒读取结果，也可点击“更新行情与预测”。关闭窗口即停止。
4. 只看结果可直接双击 `html/index.html`，显示随仓库保存的真实数据快照。HTML内置全部样式、脚本和数据，只下载这一个文件也可查看。需要更新但不启动服务时运行 `html/每日更新.bat`。

当前选取的8个指数日线均已联网取得并与本地历史核对；当前数据截至 **2026-10-16**。公开来源可能延迟或不可用，失败时保留已有数据并显示日期与错误提示。

## 目录

```text
market/
├── data/                  # 冻结的研究数据库、分类历史和报告
│   ├── daily/             # 每日新增收盘价和交易日，独立SQLite（不提交）
│   ├── tools/             # 研究库下载、重建与审计工具
│   └── training_sources/  # 历史训练源码的原始字节，用于迁移后复验
├── html/                  # 每日预测程序、网页和Windows入口
│   ├── daily/             # 行情更新、特征和纯回归推理
│   ├── server.py          # 本机服务和30分钟更新调度
│   ├── predict.py         # 一次性预测及快照生成
│   ├── snapshot.js        # 可直接打开HTML查看的实际预测快照
│   └── inference/         # 历史模型加载器，供旧实验重放
└── code/                  # 训练、测试与模型；没有每日服务
    ├── training/
    ├── research/          # 训练数据、标签及研究模型适配
    ├── references/        # 训练参考算法及许可
    ├── tests/
    └── models/            # 已保存权重及模型索引
```

详细使用见 [每日程序说明](html/README.md)、[训练说明](code/training/README.md) 和 [数据库说明](data/README.md)。

## 当前8个指数

范围统一配置在 [data/active_indices.json](data/active_indices.json)，供训练、每日采集、模型加载和网页共用。医药采用中证医药，制造采用800工业，科技采用中证信息。它们是类别趋势代理，不等同列表中具体基金的跟踪指数或主动持仓。原14指数训练模型与研究库保留，日常仅使用所选8个。选取说明见 [配置说明](data/active_indices.md)。

## 预测口径

最新 [预测误差均值与方差性能表](data/reports/performance/eight_indices_2026-10-18/README.md) 覆盖当前8指数×5期限，含完整CSV及逐年统计。它使用已保存的历史测试预测，供检查现有模型效果，不是CatBoost、Kronos等尚未训练方案的成绩。

实际变化使用1/5/21/63/126个交易日前的收盘价。“当天”是最近已收盘交易日，不是盘中报价。预测只加载已经训练的1/5/21/63/126交易日回归模型，输入为最新收盘及过去252个交易日价格特征。输出是未来整个期限累计涨跌幅；页面不使用分类，不再使用旧的均值外推预测或80%区间。

单日列预测下一交易日收盘相对此次收盘的涨跌幅。当前8个指数各含1/5/21/63/126交易日，共40个预测；从原独立模型中按配置加载，对应参数不变。模型类型逐格标注。0%表示模型未验证出有效幅度信号，不表示未来一定不涨不跌。较小MAE也可能只是市场波动较小，并不代表模型准确掌握方向。

价格收益不包含分红、基金跟踪误差与费用。现有模型未证明稳定预测优势。原训练数据库、模型和历史报告保持原样，新行情单独写入每日增量库；源码迁移前保存了按原指纹命名的原始字节，历史实验仍可重放。

## 开发与验证

使用研究虚拟环境或按 `html/requirements.txt` 安装日常依赖：

```sh
.venv/bin/python html/predict.py --refresh
.venv/bin/python html/server.py --no-browser
.venv/bin/python -m unittest discover -s html/tests -v
node --test html/tests/model.test.cjs
.venv/bin/python -m unittest discover -s data/tools -p 'test_*.py' -v
cd code
../.venv/bin/python -m unittest discover -s tests -v
```

历史模型重放：

```sh
.venv/bin/python code/training/verify.py --run 20261018_5333c8a6_f18f8f5f_288545ff
.venv/bin/python code/training/verify_three_class.py
.venv/bin/python code/training/verify_one_day.py
```

训练测试结果见 [回归与原方向报告](data/reports/training/20261018_5333c8a6_f18f8f5f_288545ff/performance_report.md) 和 [三分类实验报告](data/reports/training/threeclass_20261018_7258daaf_6a5f7840/README.md)。三分类保留为历史研究，日常显示只使用回归。
