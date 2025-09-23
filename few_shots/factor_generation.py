[{
    "question": "Calculate moving averages for metric A and metric B, with a window of 20 and calculate their difference",
    "executable_code": "\
import pandas as pd\n\
\n\
# Tool call example - metrics would typically come from previous analysis stored in state:\n\
# state['dataframes']['metrics_data'] = some_previous_analysis_result\n\
# OR from external data sources like FRED:\n\
# from tools_clean import extract_macro_data_from_fred\n\
# metric_a_data = extract_macro_data_from_fred('UNRATE', '2020-01-01', '2023-12-31')  # unemployment rate\n\
# metric_b_data = extract_macro_data_from_fred('FEDFUNDS', '2020-01-01', '2023-12-31')  # fed funds rate\n\
# state['dataframes']['metric_a'] = metric_a_data\n\
# state['dataframes']['metric_b'] = metric_b_data\n\
\n\
def calculate_moving_averages_and_difference(state, metric_a_key, metric_b_key, window=20):\n\
    # Access metrics data from LangGraph state\n\
    if metric_a_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Metric A data ({metric_a_key}) not found in state. Ensure data extraction was performed first.')\n\
    if metric_b_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Metric B data ({metric_b_key}) not found in state. Ensure data extraction was performed first.')\n\
    \n\
    metric_a_df = state['dataframes'][metric_a_key]\n\
    metric_b_df = state['dataframes'][metric_b_key]\n\
    \n\
    # Merge the metrics on their index (typically date)\n\
    df = pd.concat([metric_a_df.iloc[:, 0], metric_b_df.iloc[:, 0]], axis=1)\n\
    df.columns = ['Metric_A', 'Metric_B']\n\
    \n\
    # Calculate moving average for Metric A and Metric B using the provided window size\n\
    df['MA_A'] = df['Metric_A'].rolling(window=window).mean()\n\
    df['MA_B'] = df['Metric_B'].rolling(window=window).mean()\n\
    \n\
    # Calculate the difference between the two moving averages\n\
    df['MA_Difference'] = df['MA_A'] - df['MA_B']\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})['ma_difference'] = df['MA_Difference']\n\
    state.setdefault('dataframes', {})['ma_analysis'] = df\n\
    \n\
    return df\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract macro data using tool calls:\n\
#    state['dataframes']['unemployment'] = extract_macro_data_from_fred('UNRATE', '2020-01-01', '2023-12-31')\n\
#    state['dataframes']['fed_funds'] = extract_macro_data_from_fred('FEDFUNDS', '2020-01-01', '2023-12-31')\n\
# 2. Then calculate moving averages and difference:\n\
result_df = calculate_moving_averages_and_difference(state, 'unemployment', 'fed_funds', window=20)\n\
print(result_df[['MA_A', 'MA_B', 'MA_Difference']])\n\
",
    "code_description": "This Python code defines a function 'calculate_moving_averages_and_difference' that works with LangGraph state to calculate moving averages for two metrics (e.g., economic indicators from FRED) and compute their difference. The function accesses metric data from the state dict, merges them on their date index, calculates rolling means, and stores the results back in state for use by subsequent agents. This pattern enables seamless factor generation from various data sources in a LangGraph workflow."
},
{
    "question": "calculate rolling beta for factor A with forward returns of SPY with a window of 20 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - factor and SPY data would be extracted beforehand:\n\
# from tools_clean import extract_daily_stock_data\n\
# spy_data = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# state['dataframes']['SPY_daily'] = spy_data\n\
# factor_a would come from previous analysis:\n\
# state['factors']['factor_a'] = some_calculated_factor_series\n\
\n\
def calculate_rolling_beta(state, factor_key, spy_key='SPY_daily', window=20):\n\
    # Access factor data from LangGraph state\n\
    if factor_key not in state.get('factors', {}) and factor_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Factor data ({factor_key}) not found in state. Ensure factor calculation was performed first.')\n\
    \n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    # Get factor data (could be in factors or dataframes)\n\
    if factor_key in state.get('factors', {}):\n\
        factor_data = state['factors'][factor_key]\n\
    else:\n\
        factor_data = state['dataframes'][factor_key]\n\
        # If it's a DataFrame, take the first column\n\
        if isinstance(factor_data, pd.DataFrame):\n\
            factor_data = factor_data.iloc[:, 0]\n\
    \n\
    # Get SPY data and calculate returns\n\
    spy_data = state['dataframes'][spy_key]\n\
    spy_returns = spy_data['Close'].pct_change()\n\
    \n\
    # Combine the datasets on their index\n\
    data = pd.concat([factor_data, spy_returns], axis=1)\n\
    data.columns = ['Factor_A', 'SPY_Returns']\n\
    data = data.dropna()\n\
    \n\
    # Calculate rolling covariance between Factor A and SPY returns\n\
    rolling_cov = data['Factor_A'].rolling(window=window).cov(data['SPY_Returns'])\n\
    \n\
    # Calculate rolling variance of SPY returns\n\
    rolling_var = data['SPY_Returns'].rolling(window=window).var()\n\
    \n\
    # Calculate the rolling beta\n\
    rolling_beta = rolling_cov / rolling_var\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})[f'{factor_key}_rolling_beta'] = rolling_beta\n\
    \n\
    return rolling_beta\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract SPY data:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# 2. Assume we have a factor from previous analysis:\n\
#    state['factors']['momentum_factor'] = some_momentum_calculation\n\
# 3. Then calculate rolling beta:\n\
beta_series = calculate_rolling_beta(state, 'momentum_factor', window=20)\n\
print(beta_series)\n\
",
    "code_description": "This Python code defines a function 'calculate_rolling_beta' that calculates the rolling beta of a factor against SPY returns using LangGraph state. The function accesses factor data and SPY data from the state dict, calculates SPY returns, computes rolling covariance and variance, and derives the beta. Results are stored back in state for use by subsequent agents. This is crucial for understanding how a factor's returns relate to market fluctuations in a LangGraph workflow."
},
{
    "question": "Calculate rolling standard deviation of SPY returns with a rolling window of 20 days",
    "executable_code": "\
import pandas as pd\n\
\n\
# Tool call example - SPY data would be extracted beforehand:\n\
# from tools_clean import extract_daily_stock_data\n\
# spy_data = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# state['dataframes']['SPY_daily'] = spy_data\n\
\n\
def calculate_rolling_std(state, spy_key='SPY_daily', window=20):\n\
    # Access SPY data from LangGraph state\n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    spy_data = state['dataframes'][spy_key]\n\
    # Calculate daily returns based on the 'Close' price\n\
    spy_returns = spy_data['Close'].pct_change()\n\
    \n\
    # Calculate rolling standard deviation of SPY returns\n\
    rolling_std = spy_returns.rolling(window=window).std()\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})['spy_rolling_volatility'] = rolling_std\n\
    \n\
    return rolling_std\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract SPY data:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# 2. Then calculate rolling volatility:\n\
rolling_std_series = calculate_rolling_std(state, window=20)\n\
print(rolling_std_series)\n\
",
    "code_description": "This Python code defines a function 'calculate_rolling_std' that calculates the rolling standard deviation of SPY returns using LangGraph state. The function accesses SPY data from the state dict, computes daily returns, calculates rolling volatility over the specified window, and stores results back in state. This volatility metric is often used to assess market risk and can be used by subsequent agents in the LangGraph workflow."
},
{
    "question": "perform regression of factor A on SPY returns, compute the residuals and calculate the rolling z-score of these residuals with a window of 20 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
from sklearn.linear_model import LinearRegression\n\
\n\
# Tool call example - factor and SPY data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# state['factors']['factor_a'] = some_calculated_factor_series\n\
\n\
def perform_regression_and_calculate_zscore(state, factor_key, spy_key='SPY_daily', window=20):\n\
    # Access data from LangGraph state\n\
    if factor_key not in state.get('factors', {}) and factor_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Factor data ({factor_key}) not found in state. Ensure factor calculation was performed first.')\n\
    \n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    # Get factor data\n\
    if factor_key in state.get('factors', {}):\n\
        factor_data = state['factors'][factor_key]\n\
    else:\n\
        factor_df = state['dataframes'][factor_key]\n\
        factor_data = factor_df.iloc[:, 0] if isinstance(factor_df, pd.DataFrame) else factor_df\n\
    \n\
    # Get SPY data and calculate returns\n\
    spy_data = state['dataframes'][spy_key]\n\
    spy_returns = spy_data['Close'].pct_change()\n\
    \n\
    # Merge datasets on Date index\n\
    data = pd.concat([factor_data, spy_returns], axis=1)\n\
    data.columns = ['Factor_A', 'SPY_Returns']\n\
    data = data.dropna()\n\
    \n\
    # Regression: Factor A on SPY Returns\n\
    model = LinearRegression()\n\
    model.fit(data[['SPY_Returns']], data['Factor_A'])\n\
    predictions = model.predict(data[['SPY_Returns']])\n\
    data['Residuals'] = data['Factor_A'] - predictions\n\
    \n\
    # Calculate the rolling z-score of residuals\n\
    mean_resid = data['Residuals'].rolling(window=window).mean()\n\
    std_resid = data['Residuals'].rolling(window=window).std()\n\
    data['Residuals_Z_Score'] = (data['Residuals'] - mean_resid) / std_resid\n\
    \n\
    # Store results back in state\n\
    state.setdefault('factors', {})[f'{factor_key}_residuals_zscore'] = data['Residuals_Z_Score']\n\
    state.setdefault('analysis_results', {})[f'{factor_key}_regression'] = {\n\
        'residuals': data['Residuals'],\n\
        'z_score': data['Residuals_Z_Score'],\n\
        'beta': model.coef_[0],\n\
        'intercept': model.intercept_\n\
    }\n\
    \n\
    return data['Residuals_Z_Score']\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract SPY data and have factor ready:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
#    state['factors']['momentum_factor'] = some_momentum_calculation\n\
# 2. Then perform regression and z-score calculation:\n\
residuals_z_score = perform_regression_and_calculate_zscore(state, 'momentum_factor', window=20)\n\
print(residuals_z_score)\n\
",
    "code_description": "This Python code performs a linear regression of Factor A on SPY returns using LangGraph state data. It calculates residuals and computes a rolling z-score to normalize residuals over time. The function stores regression results and z-scores back in state, enabling subsequent agents to access both the statistical analysis and the normalized residuals for further processing or decision-making in the LangGraph workflow."
},
{
    "question": "For SPY calculate the difference between closing price and put_wall and call_wall and closing price and divide both the factors with the difference between call_wall and put_wall; then calculate rolling z-score of this metric with a window of 30 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
\n\
# Tool call example - SPY options gamma data would be available in state:\n\
# This data might come from a specialized options data provider or previous analysis:\n\
# state['dataframes']['SPY_options_gamma'] = options_gamma_data\n\
\n\
def calculate_metrics_and_zscores(state, options_data_key='SPY_options_gamma', window=30):\n\
    # Access options gamma data from LangGraph state\n\
    if options_data_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Options gamma data ({options_data_key}) not found in state. Ensure options data extraction was performed first.')\n\
\n\
    data = state['dataframes'][options_data_key].copy()\n\
\n\
    # Calculate metrics\n\
    denominator = data['Call_Wall'] - data['Put_Wall']\n\
    data['Metric1'] = (data['Close'] - data['Put_Wall']) / denominator\n\
    data['Metric2'] = (data['Call_Wall'] - data['Close']) / denominator\n\
\n\
    # Calculate the rolling z-score for both metrics\n\
    data['Metric1_Z_Score'] = (data['Metric1'] - data['Metric1'].rolling(window=window).mean()) / data['Metric1'].rolling(window=window).std()\n\
    data['Metric2_Z_Score'] = (data['Metric2'] - data['Metric2'].rolling(window=window).mean()) / data['Metric2'].rolling(window=window).std()\n\
\n\
    # Store results back in state\n\
    state.setdefault('factors', {})['spy_gamma_wall_metric1'] = data['Metric1_Z_Score']\n\
    state['factors']['spy_gamma_wall_metric2'] = data['Metric2_Z_Score']\n\
    state.setdefault('dataframes', {})['spy_gamma_wall_analysis'] = data\n\
\n\
    return data[['Metric1_Z_Score', 'Metric2_Z_Score']]\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. Assume options gamma data is available in state:\n\
#    state['dataframes']['SPY_options_gamma'] = options_gamma_data  # from external source\n\
# 2. Then calculate gamma wall metrics:\n\
metrics_z_scores = calculate_metrics_and_zscores(state, window=30)\n\
print(metrics_z_scores)\n\
",
    "code_description": "This Python code computes two normalized metrics for SPY options gamma walls using LangGraph state: the position of closing price relative to put wall and call wall, normalized by the wall spread. It calculates rolling z-scores for these metrics to identify significant deviations from normal market positioning. Results are stored in state for use by subsequent agents in options trading strategies."
},
{
    "question": "perform rolling regression (OLS) of factor A on SPY returns with window of 250 days, compute the residuals and calculate the rolling z-score of these residuals with a window of 20 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
import statsmodels.api as sm\n\
from statsmodels.regression.rolling import RollingOLS\n\
\n\
# Tool call example - factor and SPY data would be available in state:\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2023-01-01', '2023-12-31', '1d')\n\
# state['factors']['factor_a'] = some_calculated_factor_series\n\
\n\
def perform_rolling_regression_and_zscore(state, factor_key, spy_key='SPY_daily', rolling_window=250, zscore_window=20):\n\
    # Access data from LangGraph state\n\
    if factor_key not in state.get('factors', {}) and factor_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Factor data ({factor_key}) not found in state.')\n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state.')\n\
\n\
    # Get factor and SPY data\n\
    if factor_key in state.get('factors', {}):\n\
        factor_data = state['factors'][factor_key]\n\
    else:\n\
        factor_df = state['dataframes'][factor_key]\n\
        factor_data = factor_df.iloc[:, 0] if isinstance(factor_df, pd.DataFrame) else factor_df\n\
\n\
    spy_data = state['dataframes'][spy_key]\n\
    spy_returns = spy_data['Close'].pct_change()\n\
\n\
    # Merge data on Date index\n\
    data = pd.concat([factor_data, spy_returns], axis=1)\n\
    data.columns = ['Factor_A', 'SPY_Returns']\n\
    data = data.dropna()\n\
\n\
    # Perform rolling OLS regression\n\
    model = RollingOLS(endog=data['Factor_A'], exog=sm.add_constant(data['SPY_Returns']), window=rolling_window)\n\
    fitted_model = model.fit()\n\
    data['Residuals'] = fitted_model.resids\n\
\n\
    # Calculate rolling z-score of residuals\n\
    data['Residuals_Z_Score'] = (data['Residuals'] - data['Residuals'].rolling(window=zscore_window).mean()) / data['Residuals'].rolling(window=zscore_window).std()\n\
\n\
    # Store results back in state\n\
    state.setdefault('factors', {})[f'{factor_key}_rolling_regression_zscore'] = data['Residuals_Z_Score']\n\
    state.setdefault('analysis_results', {})[f'{factor_key}_rolling_regression'] = {\n\
        'residuals': data['Residuals'],\n\
        'z_score': data['Residuals_Z_Score'],\n\
        'rolling_betas': fitted_model.params.iloc[:, 1],\n\
        'rolling_alphas': fitted_model.params.iloc[:, 0]\n\
    }\n\
\n\
    return data['Residuals_Z_Score']\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. Extract data and calculate factor:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
#    state['factors']['momentum_factor'] = some_momentum_calculation\n\
# 2. Perform rolling regression analysis:\n\
residuals_z_score = perform_rolling_regression_and_zscore(state, 'momentum_factor', rolling_window=250, zscore_window=20)\n\
print(residuals_z_score)\n\
",
    "code_description": "This Python code performs a rolling regression of Factor A on SPY returns using LangGraph state, with a 250-day rolling window for regression and 20-day window for z-score normalization. It uses statsmodels RollingOLS for efficient computation, calculates residuals and their z-scores, and stores comprehensive regression results in state. This enables dynamic assessment of factor-market relationships over time in a LangGraph workflow."}
,
{
    "question": "perform rolling regression (OLS) of factor A on 1 day forward SPY returns with window of 250 days, compute the residuals and calculate the rolling percentile rank of these residuals with a window of 20 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
from statsmodels.regression.rolling import RollingOLS\n\
import statsmodels.api as sm\n\
\n\
# Tool call example - factor and SPY data would be available in state:\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
# state['factors']['factor_a'] = some_calculated_factor_series\n\
\n\
def perform_rolling_regression_and_percentile_rank(state, factor_key, spy_key='SPY_daily', rolling_window=250, rank_window=20):\n\
    # Access data from LangGraph state\n\
    if factor_key not in state.get('factors', {}) and factor_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Factor data ({factor_key}) not found in state.')\n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state.')\n\
\n\
    # Get factor and SPY data\n\
    if factor_key in state.get('factors', {}):\n\
        factor_data = state['factors'][factor_key]\n\
    else:\n\
        factor_df = state['dataframes'][factor_key]\n\
        factor_data = factor_df.iloc[:, 0] if isinstance(factor_df, pd.DataFrame) else factor_df\n\
\n\
    spy_data = state['dataframes'][spy_key]\n\
    # Calculate 1-day forward returns for SPY\n\
    forward_returns = spy_data['Close'].pct_change().shift(-1)\n\
\n\
    # Merge data on Date index\n\
    data = pd.concat([factor_data, forward_returns], axis=1)\n\
    data.columns = ['Factor_A', 'Forward_SPY_Returns']\n\
    data = data.dropna()\n\
\n\
    # Perform rolling OLS\n\
    model = RollingOLS(endog=data['Forward_SPY_Returns'], exog=sm.add_constant(data['Factor_A']), window=rolling_window)\n\
    fitted_model = model.fit()\n\
    data['Residuals'] = fitted_model.resids\n\
\n\
    # Calculate rolling percentile rank of residuals\n\
    data['Residuals_Percentile_Rank'] = data['Residuals'].rolling(window=rank_window).apply(\n\
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)\n\
\n\
    # Store results back in state\n\
    state.setdefault('factors', {})[f'{factor_key}_forward_regression_percentile'] = data['Residuals_Percentile_Rank']\n\
    state.setdefault('analysis_results', {})[f'{factor_key}_forward_regression'] = {\n\
        'residuals': data['Residuals'],\n\
        'percentile_ranks': data['Residuals_Percentile_Rank'],\n\
        'predictive_power': fitted_model.rsquared.mean() if hasattr(fitted_model.rsquared, 'mean') else None\n\
    }\n\
\n\
    return data[['Residuals_Percentile_Rank']]\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. Extract data and calculate factor:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
#    state['factors']['momentum_factor'] = some_momentum_calculation\n\
# 2. Perform forward-looking regression analysis:\n\
percentile_ranks = perform_rolling_regression_and_percentile_rank(state, 'momentum_factor', rolling_window=250, rank_window=20)\n\
print(percentile_ranks)\n\
",
    "code_description": "This Python code conducts a rolling regression of Factor A on 1-day forward SPY returns using LangGraph state, enabling predictive analysis of factor performance. It computes residuals and calculates rolling percentile ranks to identify relative performance within rolling windows. Results are stored in state for subsequent agents to use in trading signal generation or risk assessment workflows."
},
{
    "question": "perform rolling regression (OLS) of factor A on 1 day forward SPY returns with window of 250 days, compute the residuals and calculate the rolling quartiles of these residuals with a window of 20 days",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
from statsmodels.regression.rolling import RollingOLS\n\
import statsmodels.api as sm\n\
\n\
# Tool call example - factor and SPY data would be available in state:\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
# state['factors']['factor_a'] = some_calculated_factor_series\n\
\n\
def perform_rolling_regression_and_quartiles(state, factor_key, spy_key='SPY_daily', rolling_window=250, quartile_window=20):\n\
    # Access data from LangGraph state\n\
    if factor_key not in state.get('factors', {}) and factor_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'Factor data ({factor_key}) not found in state.')\n\
    if spy_key not in state.get('dataframes', {}):\n\
        raise ValueError(f'SPY data ({spy_key}) not found in state.')\n\
\n\
    # Get factor and SPY data\n\
    if factor_key in state.get('factors', {}):\n\
        factor_data = state['factors'][factor_key]\n\
    else:\n\
        factor_df = state['dataframes'][factor_key]\n\
        factor_data = factor_df.iloc[:, 0] if isinstance(factor_df, pd.DataFrame) else factor_df\n\
\n\
    spy_data = state['dataframes'][spy_key]\n\
    # Calculate 1-day forward returns for SPY\n\
    forward_returns = spy_data['Close'].pct_change().shift(-1)\n\
\n\
    # Merge data on Date index\n\
    data = pd.concat([factor_data, forward_returns], axis=1)\n\
    data.columns = ['Factor_A', 'Forward_SPY_Returns']\n\
    data = data.dropna()\n\
\n\
    # Perform rolling OLS\n\
    model = RollingOLS(endog=data['Forward_SPY_Returns'], exog=sm.add_constant(data['Factor_A']), window=rolling_window)\n\
    fitted_model = model.fit()\n\
    data['Residuals'] = fitted_model.resids\n\
\n\
    # Calculate rolling quartiles for residuals (fixed typo: quantate -> quantile)\n\
    data['Quartile_1'] = data['Residuals'].rolling(window=quartile_window).quantile(0.25)\n\
    data['Quartile_2'] = data['Residuals'].rolling(window=quartile_window).quantile(0.50)  # Median\n\
    data['Quartile_3'] = data['Residuals'].rolling(window=quartile_window).quantile(0.75)\n\
\n\
    # Store results back in state\n\
    quartiles_data = data[['Quartile_1', 'Quartile_2', 'Quartile_3']]\n\
    state.setdefault('factors', {})[f'{factor_key}_forward_regression_quartiles'] = quartiles_data\n\
    state.setdefault('analysis_results', {})[f'{factor_key}_forward_quartiles_analysis'] = {\n\
        'residuals': data['Residuals'],\n\
        'quartiles': quartiles_data,\n\
        'distribution_stats': {\n\
            'q1_mean': quartiles_data['Quartile_1'].mean(),\n\
            'median_mean': quartiles_data['Quartile_2'].mean(),\n\
            'q3_mean': quartiles_data['Quartile_3'].mean()\n\
        }\n\
    }\n\
\n\
    return quartiles_data\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. Extract data and calculate factor:\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
#    state['factors']['momentum_factor'] = some_momentum_calculation\n\
# 2. Perform quartiles analysis:\n\
quartiles = perform_rolling_regression_and_quartiles(state, 'momentum_factor', rolling_window=250, quartile_window=20)\n\
print(quartiles)\n\
",
    "code_description": "This Python code performs a rolling regression of Factor A on 1-day forward SPY returns using LangGraph state, computes residuals, and calculates rolling quartiles to understand residual distribution over time. The function fixes the original typo (quantate->quantile) and stores comprehensive quartile analysis in state, enabling subsequent agents to assess factor performance distribution and identify outliers in predictive accuracy."
}]