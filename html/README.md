# 每日预测和网页

Windows 安装 Python 3.12，双击 `启动网页.bat`。首次会创建仓库根目录的 `.venv-daily` 并安装 `requirements.txt`，随后打开 `http://127.0.0.1:8765`。保持命令窗口开启。日常推理安装 NumPy、LightGBM、pandas、CatBoost、CPU PyTorch和safetensors。启动批处理会先安装官方CPU版PyTorch，首次安装需要联网；不需要显卡。

当前每日输出由 `daily/model_selection.json` 控制：40组分别在已训练的CatBoost、LightGBM、DLinear、NLinear、TCN、PatchTST候选中按滚动验证MAE取最小者，不再要求先胜过零预测；固定零与历史均值基准不参与选择。模型真实结果即使很小也保留精度，不人为添加涨跌。选择细节见 [每日选择报告](../data/reports/deployment/2026-10-19/README.md)。这不意味着每个模型都已优于零基准。

`index.html` 是自包含文件，内置样式、脚本、行情与预测，即使只下载这一个文件也能显示，无需依赖旁边的CSS/JS。直接双击 `index.html` 无需 Python，可查看随仓库保存的真实数据与模型预测快照；此方式不能自行联网或执行 Python。`每日更新.bat` 用于只更新并生成快照，之后可直接打开 HTML。若希望每天定时运行，可在 Windows 任务计划程序中调用该批处理；本仓库未擅自注册系统任务。

## 目录与流程

当前加载7个树模型任务与33个神经网络任务；神经网络按训练方式平均两个种子的输出，实际权重会复用。输入使用与训练相同的一年/三年/五年历史，不进行每日训练。此前 `latest_five_year.json` 和 `latest_neural.json` 保持原实验记录，本次用户授权的每日选择使用独立配置。

- `daily/feeds.py`：腾讯日线，失败尝试东方财富；代码、名称、日期和开高低收校验，15:10前剔除当日日线。
- `daily/service.py`：原数据库 + `data/daily/market.sqlite3` 增量；以沪深300新日线扩展观测交易日历，校验其他指数连续性。仅为8个指数采集新行情；范围由data/active_indices.json统一配置。`best_runtime.py`统一调用每日所选模型，`ohlcv_store.py`保留验证后的量价增量。
- `daily/features.py / rich_features.py / cycle_features.py / neural_features.py`：与训练核对一致的价格、量价、长期和序列特征。
- `daily/regression.py / enhanced_predictor.py / neural_predictor.py`：只做保存权重推理，验证SHA256；神经权重使用safetensors，不加载pickle。
- `server.py`：只绑定本机地址，启动时及每30分钟检查行情；`GET /api/dashboard` 读取结果，`POST /api/refresh` 请求更新。
- `predict.py`：执行一次，可加 `--refresh` 更新日线，生成 `snapshot.js`。
- `index.template.html` 是页面模板，`daily/page.py` 在每次预测导出时生成自包含的 `index.html`，同时预先渲染表格，脚本不可用时仍显示数据。修改排版时编辑模板，避免生成时覆盖。
- `index.html / app.js / style.css / model.js`：上栏实际涨跌，下栏模型预测；每60秒读取服务结果，可手动刷新、导出CSV。白底黑字，上涨红、下跌绿。
- `inference/`：原完整模型加载器，为历史实验重放而迁移，日常服务不调用分类。

当前8个指数都展示实际1/5/21/63/126交易日涨跌。预测以各自可用的收盘日期为起点，输出未来累计价格收益；单日列为下一实际交易日。实际模型及输入年数逐格显示，单位为百分数，接口的0.1表示0.1%。

如果新收盘价已经更新但量价输入未同步，继续用所选模型的最后有效历史计算，逐格标明较早的预测基准日期；CSV也包含forecast_as_of。没有合格输入则显示“—”与错误，不补零、不悄悄改用其他模型。原始研究数据库和权重不会改写。

休市时维持最近收盘数据。网络、来源或校验失败时保留对应指数已有数据，显示日期和更新失败提示。原数据库、分类CSV、训练模型及历史报告不会被每日程序改写。指数价格收益不等于某只基金的含分红收益；模型未证明稳定预测优势。

## Linux / 开发

仓库根目录，已有研究虚拟环境可以直接运行；新环境先装CPU PyTorch再安装本目录requirements：

```sh
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r html/requirements.txt
.venv/bin/python code/training/select_daily_models.py
.venv/bin/python html/predict.py --refresh
.venv/bin/python html/server.py --no-browser
# 不做自动联网刷新，便于离线测试；仍可手动更新
.venv/bin/python html/server.py --no-browser --offline
```

验证：

```sh
.venv/bin/python -m unittest discover -s html/tests -v
node --test html/tests/model.test.cjs
```

测试使用临时增量库模拟新交易日、量价缺失、成交量单位变化和进程重启，核对40项输出与冻结候选结果一致，并保留原模型独立重放检查。模拟数据不会写入提交的快照或研究数据库。

表头逐列显示日期：实际涨跌为起始收盘至最新收盘，预测为模型基准收盘至第N个交易日。日期随数据刷新，不依赖电脑日历强行移动；各指数或模型基准不同则在单元格分别列明。直接打开HTML保留快照日期，启动网页.bat后自动检查并刷新。CSV包含comparison_start、comparison_end、target_date_estimated。

交易日历配置为`daily/trading_calendar.json`，当前收录交易所2026年休市安排（来源见文件）。不把周末调休上班算成交易；未收录年份的日期按工作日暂估并明确标注。新增年度应根据交易所公告补充closed_dates和confirmed_years，再生成快照。
