PLOTLY_CONVENTIONS = """
Plotting conventions:
- Use Plotly with the 'plotly_dark' template.
- Prefer log scale for price series; normal scale for returns.
- For time series, show shaded drawdown regions.
- Save interactive charts as standalone HTML files under ${DATA_DIR}.
- Do not call any network APIs in plots (no map tiles, no web requests).
"""

BACKTEST_HYGIENE = """
Backtesting hygiene (MANDATORY):
- No look-ahead: when computing signals, only use information available at time t, trade at t+1 open or next bar.
- Apply realistic frictions: include commissions and slippage; assume a trade delay of one bar.
- Enforce position limits, leverage caps, and (if shorting) borrow constraints.
- Corporate actions handled via adjusted prices.
- Report turnover, gross/net exposures, max drawdown.
"""

STATS_RIGOR = """
Statistical rigor:
- Report Sharpe and Probabilistic Sharpe Ratio (PSR).
- Bootstrap the equity curve for 2,000 resamples to report 95% CI of Sharpe.
- Compute hit rate, skew, kurtosis, and HHI of weights if a portfolio.
- If you try >1 hypothesis/parameter set, state the count and warn about multiple testing (White's Reality Check or SPA suggested).
"""

OUTPUT_CONTRACT = """
Output contract (MANDATORY):
- Save all tabular outputs to CSV/Parquet inside ${DATA_DIR}.
- Save all plots as Plotly HTML (and PNG/SVG if helpful) inside ${DATA_DIR}.
- Write a 'result.json' in the working directory with keys: tables, figures, metrics, explanation.
- Do not access any network resources. Read only the local artifacts provided to you.
- Each tables[] and figures[] entry MUST be a JSON object (not a string). For tables include "rows" (int) and "columns" (list of strings).
"""
