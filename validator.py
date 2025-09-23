from datetime import datetime
from typing import Tuple, Dict, Any, List


DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
]


def validate_date_format(value: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except Exception:
            continue
    return False


def _coerce_date(value: Any, default: str) -> Tuple[str, List[str]]:
    issues: List[str] = []
    if isinstance(value, str) and validate_date_format(value):
        # Normalize to YYYY-MM-DD when only a date is provided; keep as-is otherwise
        try:
            if len(value) == 10:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d"), issues
            else:
                for fmt in DATE_FORMATS[1:]:
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.strftime("%Y-%m-%d %H:%M:%S"), issues
                    except Exception:
                        pass
        except Exception:
            pass
    issues.append(f"Invalid date '{value}', falling back to {default}")
    return default, issues


def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate and normalize tool parameters. Returns (ok, fixed_params, issues)."""
    params = dict(params or {})
    issues: List[str] = []
    ok = True

    if tool_name == "extract_daily_stock_data":
        # Required
        ticker = params.get("ticker") or "AAPL"
        if not isinstance(ticker, str) or not ticker.strip():
            issues.append("ticker missing; defaulting to AAPL")
            ticker = "AAPL"
        params["ticker"] = ticker.upper()

        start_default = "2023-01-01"
        end_default = "2023-12-31"
        start_date, start_issues = _coerce_date(params.get("start_date"), start_default)
        end_date, end_issues = _coerce_date(params.get("end_date"), end_default)
        issues.extend(start_issues + end_issues)
        params["start_date"] = start_date
        params["end_date"] = end_date
        try:
            if datetime.strptime(start_date, DATE_FORMATS[0]) > datetime.strptime(end_date, DATE_FORMATS[0]):
                params["start_date"], params["end_date"] = params["end_date"], params["start_date"]
                issues.append("start_date > end_date; swapped")
        except Exception:
            pass

        interval = params.get("interval") or "1d"
        if interval not in {"1d", "1wk", "1mo"}:
            issues.append(f"Invalid interval '{interval}'; defaulting to 1d")
            interval = "1d"
        params["interval"] = interval

    elif tool_name == "extract_economic_data_from_fred":
        series_id = params.get("series_id") or "UNRATE"
        if not isinstance(series_id, str) or not series_id.strip():
            issues.append("series_id missing; defaulting to UNRATE")
            series_id = "UNRATE"
        params["series_id"] = series_id.upper()

        start_default = "2023-01-01"
        end_default = "2023-12-31"
        start_date, start_issues = _coerce_date(params.get("start_date"), start_default)
        end_date, end_issues = _coerce_date(params.get("end_date"), end_default)
        issues.extend(start_issues + end_issues)
        params["start_date"] = start_date
        params["end_date"] = end_date

    elif tool_name == "extract_fundamentals_from_fmp":
        # Prefer tickers list; fallback to single ticker
        tickers = params.get("tickers")
        ticker = params.get("ticker")
        if isinstance(tickers, list) and tickers:
            params["tickers"] = [str(t).upper() for t in tickers if isinstance(t, str) and t.strip()]
        elif isinstance(ticker, str) and ticker.strip():
            params["tickers"] = [ticker.upper()]
        else:
            issues.append("tickers/ticker missing; defaulting to ['AAPL']")
            params["tickers"] = ["AAPL"]
        start_default = "2023-01-01"; end_default = "2023-12-31"
        start_date, s_iss = _coerce_date(params.get("start_date"), start_default)
        end_date, e_iss = _coerce_date(params.get("end_date"), end_default)
        issues.extend(s_iss + e_iss)
        params["start_date"] = start_date; params["end_date"] = end_date

    elif tool_name == "extract_analyst_estimates_from_fmp":
        ticker = params.get("ticker") or "AAPL"
        if not isinstance(ticker, str) or not ticker.strip():
            issues.append("ticker missing; defaulting to AAPL")
            ticker = "AAPL"
        params["ticker"] = ticker.upper()
        start_default = "2023-01-01"; end_default = "2023-12-31"
        start_date, s_iss = _coerce_date(params.get("start_date"), start_default)
        end_date, e_iss = _coerce_date(params.get("end_date"), end_default)
        issues.extend(s_iss + e_iss)
        params["start_date"] = start_date; params["end_date"] = end_date
        period = (params.get("period") or "quarter").lower()
        if period not in {"quarter", "annual"}:
            issues.append(f"Invalid period '{period}'; defaulting to quarter")
            period = "quarter"
        params["period"] = period

    elif tool_name == "bulk_extract_daily_closing_prices_from_polygon":
        tickers = params.get("tickers") or ["AAPL", "MSFT"]
        if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
            issues.append("tickers invalid; defaulting to ['AAPL','MSFT']")
            tickers = ["AAPL", "MSFT"]
        params["tickers"] = [t.upper() for t in tickers]
        start_default = "2023-01-01"; end_default = "2023-12-31"
        start_date, s_iss = _coerce_date(params.get("start_date"), start_default)
        end_date, e_iss = _coerce_date(params.get("end_date"), end_default)
        issues.extend(s_iss + e_iss)
        params["start_date"] = start_date; params["end_date"] = end_date

    if issues:
        ok = False  # indicate we changed or corrected something (caller can decide policy)
    return ok, params, issues
