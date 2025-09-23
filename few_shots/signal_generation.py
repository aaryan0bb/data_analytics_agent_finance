[{
    "question": "given a list of tickers, calculate the time series momentum factor, defined as return over the last 12 months (252 days) minus returns over the last 1 month (22 days)",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - ticker data would be extracted beforehand:\n\
# from tools_clean import bulk_extract_daily_closing_prices_from_polygon\n\
# tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
# merged_data = bulk_extract_daily_closing_prices_from_polygon(tickers, '2020-01-01', '2023-12-31')\n\
# state['dataframes']['merged_closing_prices'] = merged_data\n\
# OR individual ticker data:\n\
# for ticker in tickers:\n\
#     state['dataframes'][f'{ticker}_daily'] = extract_daily_stock_data(ticker, '2020-01-01', '2023-12-31', '1d')\n\
\n\
def calculate_time_series_momentum(state, tickers):\n\
    momentum_factors = {}\n\
    \n\
    for ticker in tickers:\n\
        # Try to access data from state - check multiple possible keys\n\
        stock_data = None\n\
        if 'merged_closing_prices' in state.get('dataframes', {}):\n\
            merged_df = state['dataframes']['merged_closing_prices']\n\
            if ticker in merged_df.columns:\n\
                stock_data = pd.DataFrame({ticker: merged_df[ticker]})\n\
                stock_data.columns = ['Close']\n\
        elif f'{ticker}_daily' in state.get('dataframes', {}):\n\
            stock_data = state['dataframes'][f'{ticker}_daily']\n\
        \n\
        if stock_data is None:\n\
            raise ValueError(f'Stock data for {ticker} not found in state. Ensure data extraction was performed first.')\n\
        \n\
        # Calculate momentum factor\n\
        daily_returns = stock_data['Close'].pct_change()\n\
        one_month_returns = daily_returns.rolling(window=22).apply(lambda x: np.prod(1 + x) - 1, raw=True)\n\
        twelve_month_returns = daily_returns.rolling(window=252).apply(lambda x: np.prod(1 + x) - 1, raw=True)\n\
        momentum_factors[ticker] = twelve_month_returns - one_month_returns\n\
    \n\
    momentum_df = pd.DataFrame(momentum_factors)\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})['time_series_momentum'] = momentum_df\n\
    state.setdefault('signals', {})['momentum_signals'] = momentum_df\n\
    \n\
    return momentum_df\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract ticker data:\n\
#    tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
#    state['dataframes']['merged_closing_prices'] = bulk_extract_daily_closing_prices_from_polygon(tickers, '2020-01-01', '2023-12-31')\n\
# 2. Then calculate momentum factor:\n\
tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
momentum_factor = calculate_time_series_momentum(state, tickers)\n\
print(momentum_factor)\n\
",
    "code_description": "This Python code calculates the time series momentum factor for a list of stocks using LangGraph state. It accesses stock data from the state dict (populated by previous tool calls), computes cumulative returns over 1-month and 12-month periods, and derives the momentum factor. Results are stored in state under both 'factors' and 'signals' for use by subsequent portfolio construction or trading agents in the LangGraph workflow."
},
{
    "question": "Generate momentum factor for the list of stocks, where the momentum factor is calculated as SMA 20 - SMA200",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - ticker data would be extracted beforehand:\n\
# from tools_clean import bulk_extract_daily_closing_prices_from_polygon\n\
# tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
# merged_data = bulk_extract_daily_closing_prices_from_polygon(tickers, '2020-01-01', '2023-12-31')\n\
# state['dataframes']['merged_closing_prices'] = merged_data\n\
\n\
def calculate_momentum_factor(state, tickers):\n\
    momentum_factors = {}\n\
    \n\
    for ticker in tickers:\n\
        # Try to access data from state - check multiple possible keys\n\
        stock_data = None\n\
        if 'merged_closing_prices' in state.get('dataframes', {}):\n\
            merged_df = state['dataframes']['merged_closing_prices']\n\
            if ticker in merged_df.columns:\n\
                stock_data = pd.DataFrame({ticker: merged_df[ticker]})\n\
                stock_data.columns = ['Close']\n\
        elif f'{ticker}_daily' in state.get('dataframes', {}):\n\
            stock_data = state['dataframes'][f'{ticker}_daily']\n\
        \n\
        if stock_data is None:\n\
            raise ValueError(f'Stock data for {ticker} not found in state. Ensure data extraction was performed first.')\n\
        \n\
        # Calculate SMA-based momentum factor\n\
        sma_20 = stock_data['Close'].rolling(window=20).mean()\n\
        sma_200 = stock_data['Close'].rolling(window=200).mean()\n\
        momentum_factors[ticker] = sma_20 - sma_200\n\
    \n\
    momentum_df = pd.DataFrame(momentum_factors)\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})['sma_momentum'] = momentum_df\n\
    state.setdefault('signals', {})['sma_momentum_signals'] = momentum_df\n\
    \n\
    return momentum_df\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract ticker data:\n\
#    tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
#    state['dataframes']['merged_closing_prices'] = bulk_extract_daily_closing_prices_from_polygon(tickers, '2020-01-01', '2023-12-31')\n\
# 2. Then calculate SMA momentum factor:\n\
tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
momentum_factor = calculate_momentum_factor(state, tickers)\n\
print(momentum_factor)\n\
",
    "code_description": "This Python code computes the SMA-based momentum factor for a list of stocks using LangGraph state. It accesses stock data from the state dict, calculates the difference between 20-day and 200-day simple moving averages for each ticker, and stores results in state under both 'factors' and 'signals'. This momentum signal can be used by subsequent portfolio optimization or trading strategy agents in the LangGraph workflow."
},
{
    "question": "Compute and rank the time series factor (call_oi-put_oi)/(call_oi + put_oi)/2 for the entire history of a list of tickers, and generate dynamic rank-weighted portfolio weights",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - options data would be extracted beforehand:\n\
# This would typically come from specialized options data providers or previous analysis:\n\
# for ticker in tickers:\n\
#     options_data = extract_options_data(ticker, '2020-01-01', '2023-12-31')  # hypothetical function\n\
#     state['dataframes'][f'{ticker}_options'] = options_data\n\
\n\
def load_data_and_calculate_factor(state, tickers):\n\
    all_factors = pd.DataFrame()\n\
    \n\
    for ticker in tickers:\n\
        # Access options data from LangGraph state\n\
        options_key = f'{ticker}_options'\n\
        if options_key not in state.get('dataframes', {}): \n\
            raise ValueError(f'Options data for {ticker} not found in state. Ensure options data extraction was performed first.')\n\
        \n\
        data = state['dataframes'][options_key]\n\
        data['factor'] = (data['call_oi'] - data['put_oi']) / ((data['call_oi'] + data['put_oi']) / 2)\n\
        all_factors[ticker] = data['factor']\n\
\n\
    # Transpose the DataFrame for easier manipulation\n\
    factor_data = all_factors.transpose()\n\
\n\
    # Rank the factors cross-sectionally for each date\n\
    ranked_factors = factor_data.rank(axis=0, ascending=False)\n\
\n\
    # Calculate rank-weighted weights\n\
    weights = ranked_factors.div(ranked_factors.sum(axis=0), axis=1)\n\
    weights_df = weights.transpose()\n\
\n\
    # Store results back in state\n\
    state.setdefault('factors', {})['options_oi_factor'] = all_factors\n\
    state.setdefault('portfolio_weights', {})['oi_rank_weighted'] = weights_df\n\
    state.setdefault('signals', {})['oi_ranking_signals'] = ranked_factors.transpose()\n\
\n\
    return weights_df\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract options data for each ticker:\n\
#    for ticker in ['AAPL', 'MSFT', 'GOOGL']:\n\
#        state['dataframes'][f'{ticker}_options'] = extract_options_data(ticker, '2020-01-01', '2023-12-31')\n\
# 2. Then calculate factor and portfolio weights:\n\
tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
weights_df = load_data_and_calculate_factor(state, tickers)\n\
print(weights_df.head())\n\
",
    "code_description": "This Python code computes an options open interest factor for a list of stocks using LangGraph state, ranks stocks cross-sectionally at each date, and calculates dynamic rank-weighted portfolio weights. The function accesses options data from state, computes the factor, performs cross-sectional ranking, and stores results including portfolio weights and ranking signals for use by subsequent trading or risk management agents."
},
{
    "question": "Create the factor (call_oi-put_oi)/(call_oi + put_oi)/2, calculate z-scores, rank them, and generate rank weighted portfolio weights for a list of tickers",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - options data would be extracted beforehand:\n\
# for ticker in tickers:\n\
#     options_data = extract_options_data(ticker, '2020-01-01', '2023-12-31')  # hypothetical function\n\
#     state['dataframes'][f'{ticker}_options'] = options_data\n\
\n\
def calculate_factor_and_weights(state, tickers, zscore_window=20):\n\
    all_z_scores = pd.DataFrame()\n\
    \n\
    for ticker in tickers:\n\
        # Access options data from LangGraph state\n\
        options_key = f'{ticker}_options'\n\
        if options_key not in state.get('dataframes', {}): \n\
            raise ValueError(f'Options data for {ticker} not found in state. Ensure options data extraction was performed first.')\n\
        \n\
        data = state['dataframes'][options_key].copy()\n\
        data['factor'] = (data['call_oi'] - data['put_oi']) / ((data['call_oi'] + data['put_oi']) / 2)\n\
        \n\
        # Compute z-scores of the factor over a rolling window\n\
        data['z_score'] = (data['factor'] - data['factor'].rolling(window=zscore_window).mean()) / data['factor'].rolling(window=zscore_window).std()\n\
        all_z_scores[ticker] = data['z_score']\n\
\n\
    # Transpose for easier manipulation\n\
    z_scores_transposed = all_z_scores.transpose()\n\
    \n\
    # Rank the z-scores cross-sectionally for each date\n\
    ranked_z_scores = z_scores_transposed.rank(axis=0, ascending=False)\n\
    \n\
    # Calculate rank-weighted weights\n\
    weights = ranked_z_scores.div(ranked_z_scores.sum(axis=0), axis=1)\n\
    weights_df = weights.transpose()\n\
\n\
    # Store results back in state\n\
    state.setdefault('factors', {})['options_oi_zscore_factor'] = all_z_scores\n\
    state.setdefault('portfolio_weights', {})['oi_zscore_rank_weighted'] = weights_df\n\
    state.setdefault('signals', {})['oi_zscore_ranking_signals'] = ranked_z_scores.transpose()\n\
    state.setdefault('analysis_results', {})['oi_zscore_analysis'] = {\n\
        'raw_factors': all_z_scores,\n\
        'ranked_scores': ranked_z_scores.transpose(),\n\
        'final_weights': weights_df\n\
    }\n\
\n\
    return weights_df\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract options data for each ticker:\n\
#    for ticker in ['AAPL', 'MSFT', 'GOOGL']:\n\
#        state['dataframes'][f'{ticker}_options'] = extract_options_data(ticker, '2020-01-01', '2023-12-31')\n\
# 2. Then calculate z-score factor and portfolio weights:\n\
tickers = ['AAPL', 'MSFT', 'GOOGL']\n\
weights_df = calculate_factor_and_weights(state, tickers, zscore_window=20)\n\
print(weights_df.head())\n\
",
    "code_description": "This Python code calculates an options open interest factor with z-score normalization for a list of stocks using LangGraph state. It computes rolling z-scores of the factor, performs cross-sectional ranking, and generates dynamic rank-weighted portfolio weights. The function stores comprehensive results including raw factors, ranked scores, and final weights in state for use by portfolio optimization and risk management agents in the LangGraph workflow."
}]