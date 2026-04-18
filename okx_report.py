from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Iterable

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

getcontext().prec = 28

OPEN_TYPES = {"开多": "long", "开空": "short"}
CLOSE_TYPES = {"平多": "long", "平空": "short"}
FUNDING_TYPES = {"资金费收入", "资金费支出"}
MANUAL_MARGIN_TYPES = {"手动追加保证金"}
ZERO = Decimal("0")

from config import SINGLE_TRADE_RISK

R_MULTIPLE_EQUITY_FRACTION = Decimal(SINGLE_TRADE_RISK)

@dataclass(slots=True)
class LedgerRow:
    row_id: int
    order_id: str
    timestamp: datetime
    bill_type: str
    symbol: str
    trade_type: str
    quantity: Decimal
    price: Decimal
    pnl: Decimal
    fee: Decimal
    position_delta: Decimal
    position_balance: Decimal
    account_delta: Decimal
    account_balance: Decimal
    currency: str

    @property
    def equity_delta(self) -> Decimal:
        return self.account_delta + self.position_delta


@dataclass(slots=True)
class TradeCycle:
    symbol: str
    direction: str | None = None
    carryover: bool = False
    excluded_reason: str | None = None
    open_equity: Decimal | None = None
    open_rows: list[LedgerRow] = field(default_factory=list)
    close_rows: list[LedgerRow] = field(default_factory=list)
    funding_rows: list[LedgerRow] = field(default_factory=list)

    @property
    def opened_at(self) -> datetime | None:
        return self.open_rows[0].timestamp if self.open_rows else None

    @property
    def closed_at(self) -> datetime | None:
        return self.close_rows[-1].timestamp if self.close_rows else None

    @property
    def gross_pnl(self) -> Decimal:
        return sum((row.pnl for row in self.open_rows + self.close_rows), ZERO)

    @property
    def fees(self) -> Decimal:
        return sum((row.fee for row in self.open_rows + self.close_rows), ZERO)

    @property
    def funding(self) -> Decimal:
        return sum((row.pnl for row in self.funding_rows), ZERO)

    @property
    def net_pnl(self) -> Decimal:
        return self.gross_pnl + self.fees + self.funding

    @property
    def risk_unit(self) -> Decimal | None:
        if self.open_equity is None or self.open_equity <= ZERO:
            return None
        return self.open_equity * R_MULTIPLE_EQUITY_FRACTION

    @property
    def r_multiple(self) -> Decimal | None:
        risk_unit = self.risk_unit
        if risk_unit is None or risk_unit == ZERO:
            return None
        return self.net_pnl / risk_unit

    def iter_rows(self) -> Iterable[LedgerRow]:
        return [*self.open_rows, *self.close_rows, *self.funding_rows]


@dataclass(slots=True)
class SummaryMetrics:
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    loss_rate: float
    expectancy: Decimal
    payoff_ratio: float
    profit_factor: float
    annualized_return: float
    max_drawdown: float
    max_consecutive_losses: int
    sortino_ratio: float
    calmar_ratio: float
    start_equity: Decimal
    end_equity: Decimal


@dataclass(slots=True)
class AnalysisResult:
    csv_path: Path
    metadata_line: str
    interval_start: datetime
    interval_end: datetime
    included_cycles: list[TradeCycle]
    carryover_cycles: list[TradeCycle]
    unfinished_cycles: list[TradeCycle]
    daily_equity: list[tuple[date, Decimal]]
    summary: SummaryMetrics


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("\ufeff", "").replace("﻿", "").strip()


def decimal_from_text(value: str | None) -> Decimal:
    text = normalize_text(value)
    if not text:
        return ZERO
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Unable to parse decimal value: {value!r}") from exc


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(normalize_text(value), "%Y-%m-%d %H:%M:%S")


def discover_csv_path(project_root: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    candidates = sorted(project_root.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No CSV file found in the project root.")
    candidate_list = ", ".join(path.name for path in candidates)
    raise ValueError(f"Multiple CSV files found. Please pass one explicitly: {candidate_list}")


def load_ledger(csv_path: Path) -> tuple[str, list[LedgerRow]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_line = handle.readline().strip()
        reader = csv.DictReader(handle)
        rows: list[LedgerRow] = []
        for raw_row in reader:
            normalized = {normalize_text(key): normalize_text(value) for key, value in raw_row.items()}
            rows.append(
                LedgerRow(
                    row_id=int(normalized["id"]),
                    order_id=normalized["关联订单id"],
                    timestamp=parse_datetime(normalized["时间"]),
                    bill_type=normalized["账单类型"],
                    symbol=normalized["交易品种"],
                    trade_type=normalized["交易类型"],
                    quantity=decimal_from_text(normalized["数量"]),
                    price=decimal_from_text(normalized["成交价"]),
                    pnl=decimal_from_text(normalized["收益"]),
                    fee=decimal_from_text(normalized["手续费"]),
                    position_delta=decimal_from_text(normalized["仓位余额变动"]),
                    position_balance=decimal_from_text(normalized["仓位余额"]),
                    account_delta=decimal_from_text(normalized["交易账户余额变动"]),
                    account_balance=decimal_from_text(normalized["交易账户余额"]),
                    currency=normalized["交易账户余额单位"],
                )
            )
    rows.sort(key=lambda row: (row.timestamp, row.row_id))
    return metadata_line, rows


def classify_row(row: LedgerRow) -> str:
    if row.trade_type in OPEN_TYPES:
        return "open"
    if row.trade_type in CLOSE_TYPES:
        return "close"
    if row.trade_type in FUNDING_TYPES:
        return "funding"
    if row.trade_type in MANUAL_MARGIN_TYPES:
        return "manual_margin"
    raise ValueError(f"Unsupported trade type: {row.trade_type}")


def reconstruct_trade_cycles(rows: list[LedgerRow]) -> tuple[list[TradeCycle], list[TradeCycle], list[TradeCycle]]:
    by_symbol: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in rows:
        by_symbol[row.symbol].append(row)

    included_cycles: list[TradeCycle] = []
    carryover_cycles: list[TradeCycle] = []
    unfinished_cycles: list[TradeCycle] = []

    for symbol, symbol_rows in by_symbol.items():
        current: TradeCycle | None = None

        for row in symbol_rows:
            row_kind = classify_row(row)
            if row_kind == "manual_margin":
                continue

            if row_kind == "funding":
                if current is None:
                    current = TradeCycle(symbol=symbol, carryover=True, excluded_reason="carryover")
                current.funding_rows.append(row)
                continue

            direction = OPEN_TYPES.get(row.trade_type) or CLOSE_TYPES.get(row.trade_type)
            if current is None:
                current = TradeCycle(symbol=symbol, direction=direction)
            elif current.direction is None:
                current.direction = direction
            elif current.direction != direction:
                raise ValueError(
                    f"Encountered a {row.trade_type} row for {symbol} while an opposite-direction cycle is open."
                )

            if row_kind == "open":
                if current.carryover:
                    raise ValueError(f"Carryover cycle for {symbol} encountered a visible opening trade.")
                current.open_rows.append(row)
                continue

            if not current.open_rows:
                current.carryover = True
                current.excluded_reason = "carryover"
            current.close_rows.append(row)

            if row.position_balance == ZERO:
                if current.carryover:
                    carryover_cycles.append(current)
                else:
                    included_cycles.append(current)
                current = None

        if current is not None:
            current.excluded_reason = "unfinished"
            unfinished_cycles.append(current)

    included_cycles.sort(key=lambda cycle: (cycle.closed_at or cycle.opened_at or datetime.min, cycle.symbol))
    carryover_cycles.sort(key=lambda cycle: (cycle.closed_at or cycle.opened_at or datetime.min, cycle.symbol))
    unfinished_cycles.sort(key=lambda cycle: (cycle.opened_at or cycle.closed_at or datetime.min, cycle.symbol))
    return included_cycles, carryover_cycles, unfinished_cycles


def build_equity_curve(cycles: list[TradeCycle]) -> tuple[Decimal, list[tuple[datetime, Decimal]], Decimal]:
    if not cycles:
        raise ValueError("No complete trade cycles were found in the CSV window.")

    rows = sorted(
        (row for cycle in cycles for row in cycle.iter_rows()),
        key=lambda row: (row.timestamp, row.row_id),
    )
    first_row = rows[0]
    current_equity = first_row.account_balance + first_row.position_balance - first_row.equity_delta
    start_equity = current_equity

    first_open_row_to_cycle = {cycle.open_rows[0].row_id: cycle for cycle in cycles if cycle.open_rows}
    event_points: list[tuple[datetime, Decimal]] = []

    for row in rows:
        cycle = first_open_row_to_cycle.get(row.row_id)
        if cycle is not None and cycle.open_equity is None:
            cycle.open_equity = current_equity
        current_equity += row.equity_delta
        event_points.append((row.timestamp, current_equity))

    return start_equity, event_points, current_equity


def build_daily_equity(
    event_points: list[tuple[datetime, Decimal]],
    start_equity: Decimal,
) -> list[tuple[date, Decimal]]:
    if not event_points:
        return []

    index = pd.DatetimeIndex([timestamp for timestamp, _ in event_points])
    values = [float(equity) for _, equity in event_points]
    series = pd.Series(values, index=index)
    daily = series.groupby(index.normalize()).last()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, method="ffill")
    if daily.empty:
        return []

    daily_index = pd.DatetimeIndex(daily.index).to_pydatetime()
    daily_values = daily.to_list()
    daily_series = [
        (timestamp.date(), Decimal(str(value)))
        for timestamp, value in zip(daily_index, daily_values)
    ]
    if daily_series and daily_series[0][1] == ZERO:
        daily_series[0] = (daily_series[0][0], start_equity)
    return daily_series


def compute_trade_metrics(cycles: list[TradeCycle]) -> tuple[int, int, int, float, float, Decimal, float, float, int]:
    ordered_cycles = sorted(cycles, key=lambda cycle: (cycle.closed_at or datetime.min, cycle.symbol))
    net_values = [cycle.net_pnl for cycle in ordered_cycles]

    wins = [value for value in net_values if value > ZERO]
    losses = [value for value in net_values if value < ZERO]
    breakeven = [value for value in net_values if value == ZERO]

    decisive_total = len(wins) + len(losses)
    win_rate = len(wins) / decisive_total if decisive_total else 0.0
    loss_rate = len(losses) / decisive_total if decisive_total else 0.0
    expectancy = sum(net_values, ZERO) / Decimal(len(net_values)) if net_values else ZERO

    average_win = (sum(wins, ZERO) / Decimal(len(wins))) if wins else ZERO
    average_loss = (sum((-value for value in losses), ZERO) / Decimal(len(losses))) if losses else ZERO
    if average_loss == ZERO:
        payoff_ratio = math.inf if average_win > ZERO else 0.0
    else:
        payoff_ratio = float(average_win / average_loss)

    gross_profit = sum(wins, ZERO)
    gross_loss = sum((-value for value in losses), ZERO)
    if gross_loss == ZERO:
        profit_factor = math.inf if gross_profit > ZERO else 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)

    current_streak = 0
    max_streak = 0
    for value in net_values:
        if value < ZERO:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return (
        len(wins),
        len(losses),
        len(breakeven),
        win_rate,
        loss_rate,
        expectancy,
        payoff_ratio,
        profit_factor,
        max_streak,
    )


def compute_risk_metrics(
    start_equity: Decimal,
    end_equity: Decimal,
    daily_equity: list[tuple[date, Decimal]],
) -> tuple[float, float, float, float]:
    if not daily_equity:
        return 0.0, 0.0, 0.0, 0.0

    start_date = daily_equity[0][0]
    end_date = daily_equity[-1][0]
    period_days = max((end_date - start_date).days, 1)

    if start_equity <= ZERO or end_equity <= ZERO:
        annualized_return = math.nan
    else:
        annualized_return = (float(end_equity / start_equity) ** (365 / period_days)) - 1

    peak_equity = start_equity
    max_drawdown = 0.0
    for _, equity in daily_equity:
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > ZERO:
            drawdown = float((peak_equity - equity) / peak_equity)
            max_drawdown = max(max_drawdown, drawdown)

    returns: list[float] = []
    previous_equity = start_equity
    for _, equity in daily_equity:
        if previous_equity == ZERO:
            returns.append(0.0)
        else:
            returns.append(float((equity - previous_equity) / previous_equity))
        previous_equity = equity

    if not returns:
        sortino_ratio = 0.0
    else:
        returns_array = np.asarray(returns, dtype=float)
        mean_daily_return = float(np.mean(returns_array))
        downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(returns_array, 0.0)))))
        if downside_deviation == 0:
            sortino_ratio = math.inf if mean_daily_return > 0 else 0.0
        else:
            sortino_ratio = (mean_daily_return / downside_deviation) * math.sqrt(365)

    if max_drawdown == 0:
        calmar_ratio = math.inf if annualized_return > 0 else 0.0
    else:
        calmar_ratio = annualized_return / max_drawdown

    return annualized_return, max_drawdown, sortino_ratio, calmar_ratio


def analyze_csv(csv_path: Path) -> AnalysisResult:
    metadata_line, rows = load_ledger(csv_path)
    included_cycles, carryover_cycles, unfinished_cycles = reconstruct_trade_cycles(rows)
    start_equity, event_points, end_equity = build_equity_curve(included_cycles)
    daily_equity = build_daily_equity(event_points, start_equity)

    wins, losses, breakeven, win_rate, loss_rate, expectancy, payoff_ratio, profit_factor, max_consecutive_losses = (
        compute_trade_metrics(included_cycles)
    )
    annualized_return, max_drawdown, sortino_ratio, calmar_ratio = compute_risk_metrics(
        start_equity,
        end_equity,
        daily_equity,
    )

    summary = SummaryMetrics(
        total_trades=len(included_cycles),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=win_rate,
        loss_rate=loss_rate,
        expectancy=expectancy,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        max_consecutive_losses=max_consecutive_losses,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        start_equity=start_equity,
        end_equity=end_equity,
    )

    interval_rows = sorted(
        (row for cycle in included_cycles for row in cycle.iter_rows()),
        key=lambda row: (row.timestamp, row.row_id),
    )
    return AnalysisResult(
        csv_path=csv_path,
        metadata_line=metadata_line,
        interval_start=interval_rows[0].timestamp,
        interval_end=interval_rows[-1].timestamp,
        included_cycles=included_cycles,
        carryover_cycles=carryover_cycles,
        unfinished_cycles=unfinished_cycles,
        daily_equity=daily_equity,
        summary=summary,
    )


def decimal_to_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def trade_dataframe(cycles: list[TradeCycle]) -> pd.DataFrame:
    records = []
    for cycle in sorted(cycles, key=lambda item: (item.closed_at or datetime.min, item.symbol)):
        records.append(
            {
                "symbol": cycle.symbol,
                "direction": cycle.direction or "",
                "opened_at": cycle.opened_at.isoformat(sep=" ") if cycle.opened_at else "",
                "closed_at": cycle.closed_at.isoformat(sep=" ") if cycle.closed_at else "",
                "gross_pnl": decimal_to_text(cycle.gross_pnl),
                "fees": decimal_to_text(cycle.fees),
                "funding": decimal_to_text(cycle.funding),
                "net_pnl": decimal_to_text(cycle.net_pnl),
                "open_equity": decimal_to_text(cycle.open_equity),
                "r_multiple": decimal_to_text(cycle.r_multiple),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "symbol",
            "direction",
            "opened_at",
            "closed_at",
            "gross_pnl",
            "fees",
            "funding",
            "net_pnl",
            "open_equity",
            "r_multiple",
        ],
    )


def write_trade_csv(cycles: list[TradeCycle], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trade_dataframe(cycles).to_csv(output_path, index=False)


def plot_equity_curve(daily_equity: list[tuple[date, Decimal]], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    data = pd.DataFrame(
        {
            "date": [day for day, _ in daily_equity],
            "equity": [float(value) for _, value in daily_equity],
        }
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x="date", y="equity", ax=axis, linewidth=2.2, color="#1f77b4")
    axis.set_title("Equity Curve")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity (USDT)")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_r_distribution_table(r_values: list[float]) -> pd.DataFrame:
    if not r_values:
        return pd.DataFrame({"bucket_start": [0], "label": ["0-1"], "count": [0]})

    lower_bound = math.floor(min(r_values))
    upper_bound = math.floor(max(r_values)) + 1
    bucket_edges = np.arange(lower_bound, upper_bound + 1, 1, dtype=float)
    bucket_counts, _ = np.histogram(np.asarray(r_values, dtype=float), bins=bucket_edges)
    bucket_starts = list(range(lower_bound, upper_bound))
    labels = [f"{start}-{start + 1}" for start in bucket_starts]

    return pd.DataFrame(
        {
            "bucket_start": bucket_starts,
            "label": labels,
            "count": bucket_counts.astype(int),
        }
    )


def plot_r_distribution(cycles: list[TradeCycle], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    r_values = [float(cycle.r_multiple) for cycle in cycles if cycle.r_multiple is not None]
    distribution = build_r_distribution_table(r_values)
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if bucket_start < 0 else "#2ca02c" for bucket_start in distribution["bucket_start"]]
    axis.bar(distribution["label"], distribution["count"], color=colors, edgecolor="white")
    axis.set_title("Trade PnL Distribution (R)")
    axis.set_xlabel("R Bucket")
    axis.set_ylabel("Trade Count")
    if len(distribution) > 6:
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def interval_calendar_days(start: datetime, end: datetime) -> int:
    return (end.date() - start.date()).days + 1


def next_available_output_path(output_dir: Path, stem: str, suffix: str) -> Path:
    candidate = output_dir / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter:02d}{suffix}"
        counter += 1
    return candidate


def build_output_paths(output_dir: Path, generated_at: datetime) -> dict[str, Path]:
    date_stamp = generated_at.strftime("%Y%m%d")
    return {
        "trades_csv": next_available_output_path(output_dir, f"trades_{date_stamp}", ".csv"),
        "equity_curve": next_available_output_path(output_dir, f"equity_curve_{date_stamp}", ".png"),
        "pnl_r_distribution": next_available_output_path(output_dir, f"pnl_r_distribution_{date_stamp}", ".png"),
        "report_markdown": next_available_output_path(output_dir, f"report_{date_stamp}", ".md"),
    }


def write_outputs(result: AnalysisResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(output_dir, datetime.now())
    write_trade_csv(result.included_cycles, paths["trades_csv"])
    plot_equity_curve(result.daily_equity, paths["equity_curve"])
    plot_r_distribution(result.included_cycles, paths["pnl_r_distribution"])
    write_markdown_report(result, paths)
    return paths


def write_markdown_report(result: AnalysisResult, output_paths: dict[str, Path]) -> None:
    summary = result.summary
    interval_days = interval_calendar_days(result.interval_start, result.interval_end)
    lines = [
        f"Interval: {result.interval_start} -> {result.interval_end} ({interval_days} days)",
        f"Included trades: {summary.total_trades}",
        f"Excluded carryover trades: {len(result.carryover_cycles)}",
        f"Excluded unfinished trades: {len(result.unfinished_cycles)}",
        f"Wins / Losses / Breakeven: {summary.wins} / {summary.losses} / {summary.breakeven}",
        f"Win rate: {format_percent(summary.win_rate)}",
        f"Loss rate: {format_percent(summary.loss_rate)}",
        f"Expectancy: {format_money(summary.expectancy)} USDT",
        f"Payoff ratio: {format_ratio(summary.payoff_ratio)}",
        f"Profit factor: {format_ratio(summary.profit_factor)}",
        f"Annualized return: {format_percent(summary.annualized_return)}",
        f"Max drawdown: {format_percent(summary.max_drawdown)}",
        f"Max consecutive losses: {summary.max_consecutive_losses}",
        f"Sortino ratio: {format_ratio(summary.sortino_ratio)}",
        f"Calmar ratio: {format_ratio(summary.calmar_ratio)}",
        f"Start equity: {format_money(summary.start_equity)} USDT",
        f"End equity: {format_money(summary.end_equity)} USDT"
    ]
    output_paths["report_markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.2%}"


def format_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def format_money(value: Decimal) -> str:
    return f"{value:.4f}"


def print_summary(result: AnalysisResult, output_paths: dict[str, Path]) -> None:
    summary = result.summary
    interval_days = interval_calendar_days(result.interval_start, result.interval_end)
    print(f"CSV: {result.csv_path}")
    print(f"Interval: {result.interval_start} -> {result.interval_end} ({interval_days} days)")
    print(f"Included trades: {summary.total_trades}")
    print(f"Excluded carryover trades: {len(result.carryover_cycles)}")
    print(f"Excluded unfinished trades: {len(result.unfinished_cycles)}")
    print(f"Wins / Losses / Breakeven: {summary.wins} / {summary.losses} / {summary.breakeven}")
    print(f"Win rate: {format_percent(summary.win_rate)}")
    print(f"Loss rate: {format_percent(summary.loss_rate)}")
    print(f"Expectancy: {format_money(summary.expectancy)} USDT")
    print(f"Payoff ratio: {format_ratio(summary.payoff_ratio)}")
    print(f"Profit factor: {format_ratio(summary.profit_factor)}")
    print(f"Annualized return: {format_percent(summary.annualized_return)}")
    print(f"Max drawdown: {format_percent(summary.max_drawdown)}")
    print(f"Max consecutive losses: {summary.max_consecutive_losses}")
    print(f"Sortino ratio: {format_ratio(summary.sortino_ratio)}")
    print(f"Calmar ratio: {format_ratio(summary.calmar_ratio)}")
    print(f"Start equity: {format_money(summary.start_equity)} USDT")
    print(f"End equity: {format_money(summary.end_equity)} USDT")
    print(f"Trades CSV: {output_paths['trades_csv']}")
    print(f"Report Markdown: {output_paths['report_markdown']}")
    print(f"Equity curve: {output_paths['equity_curve']}")
    print(f"R distribution: {output_paths['pnl_r_distribution']}")
