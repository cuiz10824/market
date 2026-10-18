# 每日推理部分：模型加载接口

本轮先完成训练和测试。这里只提供保存参数的加载器 `predictor.py`，验证未来日常程序能调用已训练模型。行情更新、定时执行、网页预测接口尚未实现，网页继续使用原有逻辑。

加载模型不会拟合、不读未来标签、不发网络请求。示例（仓库根目录，研究虚拟环境）：

```python
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "code"))
from inference.predictor import Predictor
from research.dataset import build_feature_rows, load_history

pointer = json.loads((root / "code/models/latest.json").read_text())
index = json.loads((root / pointer["index"]).read_text())
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
