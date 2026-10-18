# 代码目录

| 位置 | 用途 |
| --- | --- |
| `server.py` | 网页本地行情服务 |
| `tools/market_data.py` | 数据下载、重建及完整性审计 |
| `tools/review_database.py` | 数据覆盖与建模准备度复查 |
| `tools/smoke_research.py` | 用真实历史检查模型接口及过滤的时间因果性 |
| [research](research/README.md) | 项目数据准备、标签、时间切分和模型适配 |
| [references](references/README.md) | 已下载的开源参考代码、许可及固定版本 |
| `tests/` | 数据处理及现有网页计算测试 |
| `requirements-research.txt` | 可选研究依赖，不影响原网页使用 |

数据库继续保存在 `data/`，前端继续保存在 `html/`。
