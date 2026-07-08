# 这是一个交易回测工具
快速回测你的历史交易，生成包含Win rate、Profit factor、Drawdown analysis、Sortino/Calmar ratio的交易报告。

## 快速使用
1.数据导入:
将交易所导出的交易账单放置到项目根目录

2.运行:
```bash
uv sync             # 安装依赖
python main.py      # 运行程序
```

3.查看回测报告(文件名带生成日期，重名时自动追加`_02`等后缀):
```text
trading-backtester/
└─reports/
  ├─equity_curve_20260708.png            # 权益曲线
  ├─pnl_r_distribution_20260708.png      # 盈亏分布图
  ├─report_20260708.md                   # 交易报告
  └─trades_20260708.csv                  # 交易记录
```

(可选)在`config.py`中设置单笔风险:
```python
# Single Trade Risk (the proportion of total capital allocated to a single trade)
SINGLE_TRADE_RISK = 0.03        # 默认值为0.03(3%),可更改
```
