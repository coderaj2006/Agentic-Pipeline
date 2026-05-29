"""
cash_forecasting.py — Startup Cash Runway Forecasting Model
============================================================

Standalone statistical script that:
  1. Constructs a realistic 6-month historical cash balance dataset for a
     startup with fixed overhead and variable operating costs.
  2. Fits a LinearRegression model on the cumulative burn trajectory to
     project the cash balance forward month by month.
  3. Mathematically solves for the exact fractional month (and calendar date)
     at which the projected balance crosses zero — the "zero-cash date".
  4. Renders a publication-quality Matplotlib chart showing historical balance,
     forecast line, zero-cash threshold, and the intersection marker.
  5. Prints a clean terminal financial summary.

This script is completely decoupled from the agent pipeline files
(step1_dynamic_rag.py, step2_google_adk.py) and has no shared imports,
state, or side effects with them.

INSTALLATION
------------
    pip install numpy pandas matplotlib scikit-learn

USAGE
-----
    python cash_forecasting.py

OUTPUT
------
  - cash_runway_forecast.png  (saved to the working directory, 300 DPI)
  - Terminal financial summary printed to stdout
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import math
from datetime import date, timedelta
from typing import Tuple

# ---------------------------------------------------------------------------
# Third-party (pip install numpy pandas matplotlib scikit-learn)
# ---------------------------------------------------------------------------
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ===========================================================================
# SECTION 1 — HISTORICAL DATASET CONSTRUCTION
# ===========================================================================

def build_historical_data(
    start_date: str = "2024-08-01",
    initial_cash: float = 500_000.0,
) -> pd.DataFrame:
    """
    Construct a realistic 6-month historical cash balance dataset.

    The burn model combines:
      - Fixed monthly overhead  : salaries, rent, SaaS subscriptions
      - Variable operating costs: cloud compute, marketing, contractor spend
      - Occasional one-off items: equipment purchase, legal fees

    Each month's ending cash = previous month's ending cash + net_cash_flow,
    where net_cash_flow is negative (outflow exceeds inflow at this stage).

    Parameters
    ----------
    start_date : str
        ISO date string for the first month's start (YYYY-MM-DD).
    initial_cash : float
        Cash balance at the beginning of the first recorded month.

    Returns
    -------
    pd.DataFrame
        Columns: date (month-start), revenue, fixed_costs, variable_costs,
                 one_off_costs, net_cash_flow, cash_balance.
    """

    # --- Month-start dates (6 months) -------------------------------------
    base = pd.Timestamp(start_date)
    months = [base + pd.DateOffset(months=i) for i in range(6)]

    # --- Revenue (early-stage startup: small and growing slowly) ----------
    # Month 0 = $18k MRR, growing ~$2k/month
    revenue = [18_000 + i * 2_000 for i in range(6)]

    # --- Fixed monthly overhead -------------------------------------------
    # Salaries (3 engineers + 1 ops): $42k
    # Office / co-working:             $3k
    # SaaS tools & subscriptions:      $2k
    # Total fixed:                     $47k / month
    fixed_costs = [47_000] * 6

    # --- Variable operating costs (fluctuate with activity) ---------------
    # Cloud compute, marketing spend, contractor hours
    variable_costs = [
        12_000,   # Month 0: baseline
        14_500,   # Month 1: marketing push
        11_000,   # Month 2: cost optimisation
        15_200,   # Month 3: new contractor sprint
        13_800,   # Month 4: moderate
        16_400,   # Month 5: scaling cloud infra
    ]

    # --- One-off / exceptional items --------------------------------------
    one_off_costs = [
        8_000,    # Month 0: legal fees (incorporation docs)
        0,
        22_000,   # Month 2: equipment purchase (dev workstations)
        0,
        5_500,    # Month 4: conference + travel
        0,
    ]

    # --- Derive net cash flow and running balance -------------------------
    records = []
    balance = initial_cash

    for i in range(6):
        net = revenue[i] - fixed_costs[i] - variable_costs[i] - one_off_costs[i]
        balance += net
        records.append(
            {
                "date":           months[i],
                "revenue":        revenue[i],
                "fixed_costs":    fixed_costs[i],
                "variable_costs": variable_costs[i],
                "one_off_costs":  one_off_costs[i],
                "net_cash_flow":  net,
                "cash_balance":   balance,
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# ===========================================================================
# SECTION 2 — LINEAR REGRESSION FORECASTING MODEL
# ===========================================================================

def fit_burn_model(
    df: pd.DataFrame,
) -> Tuple[LinearRegression, float, float]:
    """
    Fit a LinearRegression model on the historical cash balance time series.

    We use integer month indices (0, 1, 2, ...) as the single feature X and
    the cash_balance series as the target y. This captures the average linear
    burn trajectory across the historical window.

    Why linear regression over a simple average burn rate?
      - It accounts for the trend direction (accelerating or decelerating burn)
        rather than assuming a flat constant burn each month.
      - The slope coefficient directly represents the average monthly change
        in cash balance (negative = net outflow).
      - The intercept anchors the line to the historical data, not just the
        starting balance, making the projection more robust to early one-offs.

    Parameters
    ----------
    df : pd.DataFrame
        Historical dataframe with a 'cash_balance' column.

    Returns
    -------
    model : LinearRegression
        Fitted sklearn model.
    slope : float
        Monthly cash change ($/month). Negative means burning cash.
    intercept : float
        Projected cash balance at month index 0.
    """

    # X: integer month indices [0, 1, 2, 3, 4, 5] reshaped for sklearn
    X = np.arange(len(df)).reshape(-1, 1)

    # y: cash balance at each month-end
    y = df["cash_balance"].values

    model = LinearRegression()
    model.fit(X, y)

    slope     = float(model.coef_[0])      # $/month (expected negative)
    intercept = float(model.intercept_)    # projected balance at index 0

    return model, slope, intercept


def calculate_zero_cash_date(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
) -> Tuple[float, date, float]:
    """
    Solve analytically for the exact fractional month index where the
    linear projection crosses zero, then convert to a calendar date.

    The regression line is:  balance(t) = intercept + slope * t
    Setting balance(t) = 0:  t_zero = -intercept / slope

    t_zero is a fractional month index relative to the start of the
    historical series. We convert the fractional part to days within
    the target calendar month to produce an exact date.

    Parameters
    ----------
    df : pd.DataFrame
        Historical dataframe (index = month-start dates).
    slope : float
        Monthly cash change from the fitted model.
    intercept : float
        Intercept from the fitted model.

    Returns
    -------
    t_zero : float
        Fractional month index at which balance = 0.
    zero_date : date
        Exact calendar date of the zero-cash crossing.
    remaining_months : float
        Months of runway remaining from the last historical data point.
    """

    if slope >= 0:
        raise ValueError(
            "The fitted slope is non-negative — the model predicts the cash "
            "balance is not declining. Check the historical data."
        )

    # Exact fractional month index where balance = 0
    t_zero: float = -intercept / slope

    # Convert to calendar date -------------------------------------------
    # The historical series starts at df.index[0].
    # Each unit of t = 1 month (30.4375 days average).
    series_start: pd.Timestamp = df.index[0]

    whole_months  = int(math.floor(t_zero))
    frac_month    = t_zero - whole_months          # 0.0 – 1.0

    # Advance whole_months from series_start
    target_month_start: pd.Timestamp = series_start + pd.DateOffset(
        months=whole_months
    )

    # Days in the target calendar month
    next_month_start = target_month_start + pd.DateOffset(months=1)
    days_in_month    = (next_month_start - target_month_start).days

    # Fractional day offset within the target month
    day_offset = int(round(frac_month * days_in_month))
    zero_date  = (target_month_start + timedelta(days=day_offset)).date()

    # Remaining runway from the last historical observation
    last_hist_index   = len(df) - 1
    remaining_months  = t_zero - last_hist_index

    return t_zero, zero_date, remaining_months


def generate_forecast_series(
    df: pd.DataFrame,
    model: LinearRegression,
    t_zero: float,
    n_forecast_months: int = 18,
) -> pd.DataFrame:
    """
    Generate the projected cash balance series from the last historical
    month through the zero-cash date (plus a small tail for visual clarity).

    Parameters
    ----------
    df : pd.DataFrame
        Historical dataframe.
    model : LinearRegression
        Fitted model.
    t_zero : float
        Fractional month index of the zero-cash crossing.
    n_forecast_months : int
        How many months beyond the last historical point to project.
        Automatically extended to cover t_zero if needed.

    Returns
    -------
    pd.DataFrame
        Columns: date, cash_balance (projected), month_index.
    """

    series_start = df.index[0]
    last_hist_idx = len(df) - 1

    # Ensure we project at least 2 months past the zero-cash point
    end_idx = max(last_hist_idx + n_forecast_months, int(math.ceil(t_zero)) + 2)

    # Month indices for the forecast window (starts at last historical point)
    forecast_indices = np.arange(last_hist_idx, end_idx + 1)

    # Predicted balances — allow negative values so the line crosses zero
    predicted_balances = model.predict(forecast_indices.reshape(-1, 1))

    # Build date series
    forecast_dates = [
        series_start + pd.DateOffset(months=int(i))
        for i in forecast_indices
    ]

    forecast_df = pd.DataFrame(
        {
            "date":         forecast_dates,
            "cash_balance": predicted_balances,
            "month_index":  forecast_indices,
        }
    )
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df.set_index("date")

    return forecast_df


# ===========================================================================
# SECTION 3 — MATPLOTLIB VISUALISATION
# ===========================================================================

def plot_cash_runway(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    zero_date: date,
    slope: float,
    intercept: float,
    t_zero: float,
    avg_burn: float,
    remaining_months: float,
    initial_cash: float,
    output_path: str = "cash_runway_forecast.png",
) -> None:
    """
    Render and save a professional cash runway forecast chart.

    Chart elements
    --------------
    - Blue solid line  : Historical cash balance (actual data)
    - Blue scatter dots: Historical data point markers
    - Orange dashed line: Linear regression forecast
    - Red dashed horizontal line: Zero-cash threshold ($0)
    - Red star marker  : Exact zero-cash intersection point
    - Red vertical dashed line: Drops from intersection to x-axis
    - Shaded region    : Forecast uncertainty band (±1 std of residuals)
    - Annotations      : Zero-cash date label, starting balance label

    Parameters
    ----------
    hist_df : pd.DataFrame
        Historical cash balance data.
    forecast_df : pd.DataFrame
        Projected cash balance data.
    zero_date : date
        Exact calendar date of zero-cash crossing.
    slope : float
        Model slope ($/month).
    intercept : float
        Model intercept.
    t_zero : float
        Fractional month index of zero crossing.
    avg_burn : float
        Average monthly net cash outflow (positive number).
    remaining_months : float
        Months of runway from last historical point.
    initial_cash : float
        Starting cash balance for annotation.
    output_path : str
        File path for the saved PNG.
    """

    # --- Compute residual std for uncertainty band ------------------------
    X_hist = np.arange(len(hist_df)).reshape(-1, 1)
    y_hist = hist_df["cash_balance"].values
    y_pred_hist = intercept + slope * X_hist.flatten()
    residuals = y_hist - y_pred_hist
    residual_std = float(np.std(residuals))

    # --- Exact zero-cash point coordinates --------------------------------
    zero_cash_ts = pd.Timestamp(zero_date)

    # --- Figure setup ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    # --- Plot historical cash balance -------------------------------------
    ax.plot(
        hist_df.index,
        hist_df["cash_balance"],
        color="#1A73E8",
        linewidth=2.5,
        label="Historical Cash Balance",
        zorder=4,
    )
    ax.scatter(
        hist_df.index,
        hist_df["cash_balance"],
        color="#1A73E8",
        s=60,
        zorder=5,
        label="_nolegend_",
    )

    # --- Plot forecast line -----------------------------------------------
    # Only show forecast from the last historical point onward
    ax.plot(
        forecast_df.index,
        forecast_df["cash_balance"],
        color="#F4A300",
        linewidth=2.5,
        linestyle="--",
        label="Projected Cash Balance (Linear Trend)",
        zorder=4,
    )

    # --- Uncertainty band (±1 std of historical residuals) ----------------
    ax.fill_between(
        forecast_df.index,
        forecast_df["cash_balance"] - residual_std,
        forecast_df["cash_balance"] + residual_std,
        color="#F4A300",
        alpha=0.12,
        label=f"Forecast Uncertainty Band (±${residual_std:,.0f})",
        zorder=2,
    )

    # --- Zero-cash threshold line -----------------------------------------
    ax.axhline(
        y=0,
        color="#D93025",
        linewidth=1.8,
        linestyle="-",
        label="Zero Cash Threshold ($0)",
        zorder=3,
    )

    # --- Vertical dashed line at zero-cash date ---------------------------
    ax.axvline(
        x=zero_cash_ts,
        color="#D93025",
        linewidth=1.4,
        linestyle=":",
        alpha=0.7,
        zorder=3,
        label="_nolegend_",
    )

    # --- Zero-cash intersection marker ------------------------------------
    ax.scatter(
        [zero_cash_ts],
        [0],
        color="#D93025",
        s=180,
        marker="*",
        zorder=6,
        label=f"Zero-Cash Date: {zero_date.strftime('%b %d, %Y')}",
    )

    # --- Annotation: zero-cash date label ---------------------------------
    ax.annotate(
        f"  Cash Out\n  {zero_date.strftime('%b %d, %Y')}",
        xy=(zero_cash_ts, 0),
        xytext=(zero_cash_ts, max(hist_df["cash_balance"]) * 0.12),
        fontsize=10,
        color="#D93025",
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#D93025",
            lw=1.5,
        ),
        zorder=7,
    )

    # --- Annotation: starting balance label -------------------------------
    ax.annotate(
        f"  Starting Balance\n  ${initial_cash:,.0f}",
        xy=(hist_df.index[0], hist_df["cash_balance"].iloc[0]),
        xytext=(
            hist_df.index[0] + pd.DateOffset(months=1),
            hist_df["cash_balance"].iloc[0] * 1.04,
        ),
        fontsize=9,
        color="#1A73E8",
        arrowprops=dict(arrowstyle="->", color="#1A73E8", lw=1.2),
        zorder=7,
    )

    # --- Axes formatting --------------------------------------------------
    # Y-axis: dollar formatting with comma separators
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    # X-axis: month + year labels, rotated for readability
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # --- Grid -------------------------------------------------------------
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color="#CCCCCC")
    ax.grid(axis="x", linestyle=":",  linewidth=0.4, alpha=0.4, color="#CCCCCC")
    ax.set_axisbelow(True)

    # --- Titles and labels ------------------------------------------------
    ax.set_title(
        "Startup Cash Runway Forecast",
        fontsize=18,
        fontweight="bold",
        pad=18,
        color="#202124",
    )
    ax.set_xlabel("Month", fontsize=12, labelpad=10, color="#5F6368")
    ax.set_ylabel("Cash Balance (USD)", fontsize=12, labelpad=10, color="#5F6368")

    # --- Summary stats box (top-right inset) ------------------------------
    stats_text = (
        f"Avg Monthly Burn:  ${avg_burn:>10,.0f}\n"
        f"Remaining Runway:  {remaining_months:>8.1f} months\n"
        f"Zero-Cash Date:    {zero_date.strftime('%b %d, %Y')}"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#FFF8E1", alpha=0.85, edgecolor="#F4A300")
    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        fontfamily="monospace",
        color="#202124",
    )

    # --- Legend -----------------------------------------------------------
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.78),
        fontsize=9.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    legend.get_frame().set_linewidth(0.8)

    # --- Spine styling ----------------------------------------------------
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.8)

    # --- Save and close ---------------------------------------------------
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[CHART] Saved to: {output_path}")


# ===========================================================================
# SECTION 4 — TERMINAL FINANCIAL SUMMARY
# ===========================================================================

def print_financial_summary(
    hist_df: pd.DataFrame,
    initial_cash: float,
    avg_burn: float,
    remaining_months: float,
    zero_date: date,
    slope: float,
    intercept: float,
    r_squared: float,
) -> None:
    """
    Print a clean, formatted financial summary to stdout.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Historical dataframe.
    initial_cash : float
        Starting cash balance.
    avg_burn : float
        Average monthly net cash outflow (positive = burning cash).
    remaining_months : float
        Months of runway from the last historical data point.
    zero_date : date
        Exact predicted zero-cash date.
    slope : float
        Model slope ($/month).
    intercept : float
        Model intercept.
    r_squared : float
        R² score of the fitted model on historical data.
    """

    last_balance = hist_df["cash_balance"].iloc[-1]
    last_date    = hist_df.index[-1].strftime("%B %Y")
    total_burned = initial_cash - last_balance
    months_elapsed = len(hist_df)

    divider = "=" * 60

    print(f"\n{divider}")
    print("  CASH RUNWAY FORECAST — FINANCIAL SUMMARY")
    print(divider)
    print(f"  {'Starting Cash Balance':<30} ${initial_cash:>14,.2f}")
    print(f"  {'Current Cash Balance':<30} ${last_balance:>14,.2f}  ({last_date})")
    print(f"  {'Total Cash Burned (historical)':<30} ${total_burned:>14,.2f}")
    print(f"  {'Historical Period':<30} {'6 months':>15}")
    print(divider)
    print(f"  {'Average Monthly Burn Rate':<30} ${avg_burn:>14,.2f} / month")
    print(f"  {'Model Slope ($/month)':<30} ${slope:>14,.2f}")
    print(f"  {'Model Intercept':<30} ${intercept:>14,.2f}")
    print(f"  {'Model R² Score':<30} {r_squared:>15.4f}")
    print(divider)
    print(f"  {'Remaining Runway':<30} {remaining_months:>13.2f}  months")
    print(f"  {'Predicted Zero-Cash Date':<30} {zero_date.strftime('%B %d, %Y'):>15}")
    print(divider)

    # Risk assessment
    if remaining_months < 3:
        risk = "CRITICAL — Raise capital or cut costs immediately."
    elif remaining_months < 6:
        risk = "HIGH     — Begin fundraising process now."
    elif remaining_months < 12:
        risk = "MODERATE — Plan next funding round within 3 months."
    else:
        risk = "LOW      — Comfortable runway; monitor burn rate."

    print(f"  {'Runway Risk Assessment':<30} {risk}")
    print(divider)
    print()


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":

    print("\n[INIT] Building historical financial dataset...")

    # --- 1) Construct historical data -------------------------------------
    INITIAL_CASH = 500_000.0
    hist_df = build_historical_data(
        start_date="2024-08-01",
        initial_cash=INITIAL_CASH,
    )

    print(f"[INIT] Historical dataset ready — {len(hist_df)} months of data.")
    print("\n  Month-by-Month Summary:")
    print(f"  {'Date':<12} {'Revenue':>10} {'Total Costs':>12} "
          f"{'Net Flow':>12} {'Balance':>12}")
    print("  " + "-" * 60)
    for dt, row in hist_df.iterrows():
        total_costs = (
            row["fixed_costs"] + row["variable_costs"] + row["one_off_costs"]
        )
        print(
            f"  {dt.strftime('%b %Y'):<12}"
            f"  ${row['revenue']:>9,.0f}"
            f"  ${total_costs:>10,.0f}"
            f"  ${row['net_cash_flow']:>10,.0f}"
            f"  ${row['cash_balance']:>10,.0f}"
        )

    # --- 2) Fit the forecasting model -------------------------------------
    print("\n[MODEL] Fitting LinearRegression on cash balance time series...")
    model, slope, intercept = fit_burn_model(hist_df)

    # R² score on historical data
    X_hist = np.arange(len(hist_df)).reshape(-1, 1)
    r_squared = float(model.score(X_hist, hist_df["cash_balance"].values))
    print(f"[MODEL] Slope: ${slope:,.2f}/month | "
          f"Intercept: ${intercept:,.2f} | R²: {r_squared:.4f}")

    # --- 3) Calculate zero-cash date --------------------------------------
    print("[MODEL] Solving for zero-cash intersection...")
    t_zero, zero_date, remaining_months = calculate_zero_cash_date(
        hist_df, slope, intercept
    )
    print(f"[MODEL] Zero-cash at month index t={t_zero:.4f} "
          f"→ {zero_date.strftime('%B %d, %Y')}")

    # --- 4) Generate forecast series --------------------------------------
    forecast_df = generate_forecast_series(hist_df, model, t_zero)

    # Average monthly burn rate (positive = outflow)
    avg_burn = float(-hist_df["net_cash_flow"].mean())

    # --- 5) Plot and save chart -------------------------------------------
    print("[CHART] Rendering cash runway forecast chart...")
    plot_cash_runway(
        hist_df=hist_df,
        forecast_df=forecast_df,
        zero_date=zero_date,
        slope=slope,
        intercept=intercept,
        t_zero=t_zero,
        avg_burn=avg_burn,
        remaining_months=remaining_months,
        initial_cash=INITIAL_CASH,
        output_path="cash_runway_forecast.png",
    )

    # --- 6) Print financial summary ---------------------------------------
    print_financial_summary(
        hist_df=hist_df,
        initial_cash=INITIAL_CASH,
        avg_burn=avg_burn,
        remaining_months=remaining_months,
        zero_date=zero_date,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
    )
