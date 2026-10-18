# 每日预测和网页

Windows 安装 Python 3.12，双击 `启动网页.bat`。首次会创建仓库根目录的 `.venv-daily` 并安装 `requirements.txt`，随后打开 `http://127.0.0.1:8765`。保持命令窗口开启。日常推理只需要 NumPy、LightGBM 及其依赖，不安装整套研究环境。

`index.html` 是自包含文件，内置样式、脚本、行情与预测，即使只下载这一个文件也能显示，无需依赖旁边的CSS/JS。直接双击 `index.html` 无需 Python，可查看随仓库保存的真实数据与模型预测快照；此方式不能自行联网或执行 Python。`每日更新.bat` 用于只更新并生成快照，之后可直接打开 HTML。若希望每天定时运行，可在 Windows 任务计划程序中调用该批处理；本仓库未擅自注册系统任务。

## 目录与流程

- `daily/feeds.py`：腾讯日线，失败尝试东方财富；代码、名称、日期和开高低收校验，15:10前剔除当日日线。
- `daily/service.py`：原数据库 + `data/daily/market.sqlite3` 增量；以沪深300新日线扩展观测交易日历，校验其他指数连续性。启动时加载70份回归参数（14份单日及原56份长周期）。
- `daily/features.py`：过去252个交易日的特征，与训练时逐值核对一致。
- `daily/regression.py`：只计算保存模型的收益回归分支；不加载分类权重，不训练、不调整参数。验证模型和原生权重SHA256。
- `server.py`：只绑定本机地址，启动时及每30分钟检查行情；`GET /api/dashboard` 读取结果，`POST /api/refresh` 请求更新。
- `predict.py`：执行一次，可加 `--refresh` 更新日线，生成 `snapshot.js`。
- `index.template.html` 是页面模板，`daily/page.py` 在每次预测导出时生成自包含的 `index.html`，同时预先渲染表格，脚本不可用时仍显示数据。修改排版时编辑模板，避免生成时覆盖。
- `index.html / app.js / style.css / model.js`：上栏实际涨跌，下栏模型预测；每60秒读取服务结果，可手动刷新、导出CSV。白底黑字，上涨红、下跌绿。
- `inference/`：原完整模型加载器，为历史实验重放而迁移，日常服务不调用分类。

所有14个指数都展示实际1/5/21/63/126交易日涨跌。预测使用已训练的1/5/21/63/126交易日模型，以最新收盘为起点预测未来累计价格收益。单日列为下一实际交易日的收盘到收盘收益，单日模型使用原有价格特征及新增短期收益/波动，独立指针为code/models/latest_one_day.json。各周期模型可能保留零变化基准；0%不等于确定不会变化。实际模型名称逐格展示，单位为百分数，接口的0.1表示0.1%。

休市时维持最近收盘数据。网络、来源或校验失败时保留对应指数已有数据，显示日期和更新失败提示。原数据库、分类CSV、训练模型及历史报告不会被每日程序改写。指数价格收益不等于某只基金的含分红收益；模型未证明稳定预测优势。

## Linux / 开发

仓库根目录，使用已有研究虚拟环境或按本目录requirements创建每日环境：

```sh
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

测试使用临时增量库模拟新交易日、部分来源故障、基准日历故障和进程重启，同时核对原56份回归输出不变以及14份单日预测与其训练保存结果一致。模拟数据不会写入提交的快照或研究数据库。
