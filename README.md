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

3.查看回测报告:
```text
trading-backtester/
└─reports/
  ├─equity_curve.png            # 权益曲线
  ├─pnl_r_distribution.png      # 盈亏分布图
  ├─report.md                   # 交易报告
  └─trades.csv                  # 交易记录
```