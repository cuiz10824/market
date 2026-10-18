# 历史模型重放加载接口

实际每日执行入口已接入 [每日程序](../README.md)，只调用 `daily/regression.py` 的收益分支。这里保留原完整加载器，供历史二分类/三分类训练与测试重放使用，页面不展示分类结果。

三分类加载器为 `three_class_predictor.py`。沿用下方示例，把指针改为 `code/models/latest_three_class.json`，并使用 `from inference.three_class_predictor import ThreeClassPredictor` 和 `ThreeClassPredictor(root / entry["latest"])`。返回 `class_label`（大跌/波动/大涨）、`p_big_down`、`p_neutral`、`p_big_up` 和保持原模型结果的 `expected_price_return`。分类阈值为各期限累计收益的 ±5%，不是每日涨幅阈值；概率未经校准。

新模型包通过路径和 SHA256 引用原回归模型，因此部署时需要同时保留原模型目录。网页仅接入原数值回归，三分类结果不展示。

`predictor.py` 和 `three_class_predictor.py` 保持原始字节，迁移到 html 后仍可用于历史实验的模型重放。

加载模型不会拟合、不读未来标签、不发网络请求。示例（仓库根目录，研究虚拟环境）：

```python
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "html"))
from inference.predictor import Predictor
sys.path.insert(0, str(root / "code"))
from research.dataset import build_feature_rows, load_history

pointer = json.loads((root / "code/models/latest.json").read_text(encoding="utf-8"))
index = json.loads((root / pointer["index"]).read_text(encoding="utf-8"))
entry = next(m for m in index["models"] if m["code"] == "000300" and m["horizon"] == 21)
calendar, prices, _ = load_history("000300")
latest_features = build_feature_rows(calendar, prices, code="000300")[-1]
predictor = Predictor(root / entry["latest"])
result = predictor.predict([latest_features])[0]
print(latest_features["date"], result)
```

输入是同一套10个历史特征；`p_up`为未经另行校准的上涨概率，`expected_price_return`为小数收益估计，`0.05`表示5%。期限按交易日计数。数据为日收盘价，当前只能解释为收盘后预测，不能冒充盘中实时模型。

界面应保留数据日期、研究状态和模型类型。若`predictor.bundle['return']['kind']`为`zero`，应显示“零变化基准/暂无有效幅度信号”，不把0当确定不涨不跌；若分类为`constant`，说明这是历史上涨比例基准。返回值明确带有`research_only=True`。

当前参考交易日历截止2026-10-16。未来每日追加数据时必须同步有效交易日历，不能只追加价格；当前构造器遇到日历外观测会拒绝，避免悄悄按行数错算期限。

latest是测试后重训模型；历史测试报告对应另存的holdout模型。模型结果未证明稳定优于基准，不能将“成功加载和运行”解释成预测能力已验证。
