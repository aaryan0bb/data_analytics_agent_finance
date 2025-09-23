[{
    "question": "Calculate spreads between AAA and BBB credit rates and predict 1D forward SPY returns; also analyze the results using the statsmodel library",
    "executable_code": "\
import pandas as pd\n\
import statsmodels.api as sm\n\
import pickle\n\
\n\
# Tool call example - credit data and SPY data would be extracted beforehand:\n\
# from tools_clean import extract_macro_data_from_fred, extract_daily_stock_data\n\
# aaa_yields = extract_macro_data_from_fred('BAMLC0A1CAAAEY', '2020-01-01', '2023-12-31')  # AAA credit yields\n\
# bbb_yields = extract_macro_data_from_fred('BAMLC0A1CBBBEY', '2020-01-01', '2023-12-31')  # BBB credit yields\n\
# spy_data = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
# state['dataframes']['aaa_yields'] = aaa_yields\n\
# state['dataframes']['bbb_yields'] = bbb_yields\n\
# state['dataframes']['SPY_daily'] = spy_data\n\
\n\
# Load Data from LangGraph state\n\
def load_data_from_state(state):\n\
    # Access credit yields and SPY data from state\n\
    if 'aaa_yields' not in state.get('dataframes', {}):\n\
        raise ValueError('AAA yield data not found in state. Ensure extract_macro_data_from_fred was called first.')\n\
    if 'bbb_yields' not in state.get('dataframes', {}):\n\
        raise ValueError('BBB yield data not found in state. Ensure extract_macro_data_from_fred was called first.')\n\
    if 'SPY_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('SPY data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    aaa_data = state['dataframes']['aaa_yields']\n\
    bbb_data = state['dataframes']['bbb_yields']\n\
    spy_data = state['dataframes']['SPY_daily']\n\
    \n\
    # Calculate credit spread\n\
    credit_spread = pd.DataFrame({\n\
        'Credit_Spread': bbb_data.iloc[:, 0] - aaa_data.iloc[:, 0]\n\
    })\n\
    \n\
    return credit_spread, spy_data\n\
\n\
# Prepare Data\n\
def prepare_data(credit_spread, spy_data):\n\
    spy_data['Forward_Return'] = spy_data['Close'].pct_change().shift(-1)\n\
    combined_data = credit_spread.join(spy_data['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Fit Model using statsmodels\n\
def fit_model(data):\n\
    X = data[['Credit_Spread']]\n\
    X = sm.add_constant(X)  # Adds a constant term to the predictor\n\
    y = data['Forward_Return']\n\
    model = sm.OLS(y, X).fit()\n\
    return model\n\
\n\
# Model Analysis and Storage\n\
def model_analysis_and_store(state, model):\n\
    print(model.summary())\n\
    \n\
    # Store model results in state\n\
    state.setdefault('models', {})['credit_spread_spy_regression'] = model\n\
    state.setdefault('analysis_results', {})['credit_spread_analysis'] = {\n\
        'model_summary': model.summary().as_text(),\n\
        'rsquared': model.rsquared,\n\
        'pvalues': model.pvalues.to_dict(),\n\
        'coefficients': model.params.to_dict()\n\
    }\n\
\n\
# Execute the workflow\n\
def run_credit_spread_analysis(state):\n\
    credit_spread, spy_data = load_data_from_state(state)\n\
    combined_data = prepare_data(credit_spread, spy_data)\n\
    model = fit_model(combined_data)\n\
    model_analysis_and_store(state, model)\n\
    return model\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First extract data using tool calls:\n\
#    state['dataframes']['aaa_yields'] = extract_macro_data_from_fred('BAMLC0A1CAAAEY', '2020-01-01', '2023-12-31')\n\
#    state['dataframes']['bbb_yields'] = extract_macro_data_from_fred('BAMLC0A1CBBBEY', '2020-01-01', '2023-12-31')\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
# 2. Run analysis:\n\
model = run_credit_spread_analysis(state)\n\
",
    "code_description": "This Python script calculates credit spreads between AAA and BBB yields using LangGraph state, computes 1-day forward SPY returns, and performs regression analysis using statsmodels. The function accesses macro data from FRED and SPY data from state, calculates credit spreads, and stores comprehensive regression results including model summary, R-squared, p-values, and coefficients back in state for use by subsequent risk assessment or trading strategy agents in the LangGraph workflow."
},
{
    "question": "Create a dataframe of factors A, B, C, D, E and merge it with QQQ 1 day forward returns as the dependent variable, then fit and save the linear regression model and perform model analysis.",
    "executable_code": "\
import pandas as pd\n\
from sklearn.linear_model import LinearRegression\n\
import statsmodels.api as sm\n\
import pickle\n\
\n\
# Load data\n\
def load_data(factor_path, qqq_path):\n\
    factors = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    qqq_prices = pd.read_csv(qqq_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, qqq_prices\n\
\n\
# Prepare data\n\
def prepare_data(factors, qqq_prices):\n\
    qqq_prices['Forward_Return'] = qqq_prices['Close'].pct_change().shift(-1)\n\
    combined_data = factors.join(qqq_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Fit linear regression model\n\
def fit_model(data):\n\
    X = data[['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']]\n\
    y = data['Forward_Return']\n\
    X = sm.add_constant(X)  # Adding a constant for the intercept\n\
    model = sm.OLS(y, X).fit()\n\
    return model\n\
\n\
# Save the model\n\
def save_model(model, filename):\n\
    with open(filename, 'wb') as file:\n\
        pickle.dump(model, file)\n\
\n\
# Model analysis\n\
def analyze_model(model):\n\
    print(model.summary())\n\
\n\
# Tool call example - factors and QQQ data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['QQQ_daily'] = extract_daily_stock_data('QQQ', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Main execution using LangGraph state\n\
def run_factor_regression_analysis(state):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'QQQ_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('QQQ data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    qqq_prices = state['dataframes']['QQQ_daily']\n\
    data = prepare_data(factors, qqq_prices)\n\
    model = fit_model(data)\n\
    \n\
    # Store model in state instead of file\n\
    state.setdefault('models', {})['factor_qqq_linear_regression'] = model\n\
    state.setdefault('analysis_results', {})['factor_regression_analysis'] = {\n\
        'model_summary': model.summary().as_text(),\n\
        'rsquared': model.rsquared,\n\
        'pvalues': model.pvalues.to_dict(),\n\
        'coefficients': model.params.to_dict()\n\
    }\n\
    \n\
    analyze_model(model)\n\
    return model\n\
\n\
# Example usage:\n\
model = run_factor_regression_analysis(state)\n\
",
    "code_description": "This script performs several steps: it loads factor data and QQQ price data, calculates 1-day forward returns for QQQ, merges the factors with these returns, fits a linear regression model using statsmodels to provide detailed regression output, saves the model, and finally, prints a comprehensive summary of the regression analysis. This approach helps to understand the impact of each factor on QQQ's returns and assess the model's overall fit and predictive power."
},
{
    "question": "Create a dataframe of factor A, factor B, factor C, factor D, factor E and merge it with QQQ 1 day forward returns as the dependent variable, then fit and save the lasso regression model and identify important variables.",
    "executable_code": "\
import pandas as pd\n\
from sklearn.linear_model import Lasso\n\
from sklearn.preprocessing import StandardScaler\n\
import pickle\n\
\n\
# Load data\n\
def load_data(factor_path, qqq_path):\n\
    factors = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    qqq_prices = pd.read_csv(qqq_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, qqq_prices\n\
\n\
# Prepare data\n\
def prepare_data(factors, qqq_prices):\n\
    qqq_prices['Forward_Return'] = qqq_prices['Close'].pct_change().shift(-1)\n\
    combined_data = factors.join(qqq_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Fit Lasso regression model\n\
def fit_model(data):\n\
    X = data[['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']]\n\
    y = data['Forward_Return']\n\
    scaler = StandardScaler()\n\
    X_scaled = scaler.fit_transform(X)  # Standardizing the data is crucial for Lasso\n\
    lasso = Lasso(alpha=0.01)  # Alpha is a hyperparameter that controls regularization\n\
    lasso.fit(X_scaled, y)\n\
    return lasso, scaler\n\
\n\
# Save the model\n\
def save_model(model, scaler, filename):\n\
    with open(filename, 'wb') as file:\n\
        pickle.dump({'model': model, 'scaler': scaler}, file)\n\
\n\
# Identify important variables\n\
def identify_important_variables(model, feature_names):\n\
    print('Coefficients:')\n\
    for i, coef in enumerate(model.coef_):\n\
        print(f'{feature_names[i]}: {coef}')\n\
\n\
# Tool call example - factors and QQQ data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['QQQ_daily'] = extract_daily_stock_data('QQQ', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Main execution using LangGraph state\n\
def run_lasso_regression_analysis(state, alpha=0.01):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'QQQ_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('QQQ data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    qqq_prices = state['dataframes']['QQQ_daily']\n\
    data = prepare_data(factors, qqq_prices)\n\
    lasso_model, scaler = fit_model(data)\n\
    \n\
    # Store model and results in state\n\
    state.setdefault('models', {})['factor_qqq_lasso_regression'] = {\n\
        'model': lasso_model,\n\
        'scaler': scaler\n\
    }\n\
    \n\
    # Store analysis results\n\
    feature_names = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']\n\
    coefficients = {name: coef for name, coef in zip(feature_names, lasso_model.coef_)}\n\
    important_features = {name: coef for name, coef in coefficients.items() if abs(coef) > 0.001}\n\
    \n\
    state.setdefault('analysis_results', {})['lasso_regression_analysis'] = {\n\
        'coefficients': coefficients,\n\
        'important_features': important_features,\n\
        'alpha': alpha,\n\
        'n_selected_features': len(important_features)\n\
    }\n\
    \n\
    identify_important_variables(lasso_model, feature_names)\n\
    return lasso_model, scaler\n\
\n\
# Example usage:\n\
lasso_model, scaler = run_lasso_regression_analysis(state, alpha=0.01)\n\
",
    "code_description": "This Python script performs Lasso regression analysis using LangGraph state to predict QQQ forward returns from financial factors. It accesses factor and QQQ data from state, applies regularization to identify the most important features, and stores comprehensive results including coefficients and selected features back in state. The Lasso model helps with feature selection by shrinking less important coefficients to zero, enabling subsequent agents to focus on the most predictive factors in the workflow."
},
{
    "question": "Create a dataframe of factor A, factor B, factor C, factor D, factor E and merge it with SPY 1 day forward returns as the dependent variable, split the train and test set, then fit the model and perform inference on the test set using an elastic net model.",
    "executable_code": "\
import pandas as pd\n\
from sklearn.linear_model import ElasticNet\n\
from sklearn.model_selection import train_test_split\n\
from sklearn.metrics import mean_squared_error\n\
from sklearn.preprocessing import StandardScaler\n\
\n\
# Load data function\n\
def load_data(factors_path, spy_path):\n\
    factors = pd.read_csv(factors_path, parse_dates=['Date'], index_col='Date')\n\
    spy_prices = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, spy_prices\n\
\n\
# Prepare data function\n\
def prepare_data(factors, spy_prices):\n\
    spy_prices['Forward_Return'] = spy_prices['Close'].pct_change().shift(-1)\n\
    combined_data = factors.join(spy_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Split data function\n\
def split_data(data):\n\
    X = data[['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']]\n\
    y = data['Forward_Return']\n\
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\
    return X_train, X_test, y_train, y_test\n\
\n\
# Fit Elastic Net model function\n\
def fit_model(X_train, y_train):\n\
    scaler = StandardScaler()\n\
    X_train_scaled = scaler.fit_transform(X_train)\n\
    model = ElasticNet(alpha=0.1, l1_ratio=0.7)\n\
    model.fit(X_train_scaled, y_train)\n\
    return model, scaler\n\
\n\
# Predict and analyze function\n\
def predict_and_analyze(model, scaler, X_test, y_test):\n\
    X_test_scaled = scaler.transform(X_test)\n\
    predictions = model.predict(X_test_scaled)\n\
    mse = mean_squared_error(y_test, predictions)\n\
    print('Mean Squared Error on Test Set:', mse)\n\
\n\
# Tool call example - factors and SPY data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Main execution using LangGraph state\n\
def run_elastic_net_analysis(state, alpha=0.1, l1_ratio=0.7, test_size=0.2):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'SPY_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('SPY data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    spy_prices = state['dataframes']['SPY_daily']\n\
    data = prepare_data(factors, spy_prices)\n\
    X_train, X_test, y_train, y_test = split_data(data)\n\
    model, scaler = fit_model(X_train, y_train)\n\
    \n\
    # Make predictions and calculate metrics\n\
    X_test_scaled = scaler.transform(X_test)\n\
    predictions = model.predict(X_test_scaled)\n\
    mse = mean_squared_error(y_test, predictions)\n\
    \n\
    # Store model and results in state\n\
    state.setdefault('models', {})['factor_spy_elastic_net'] = {\n\
        'model': model,\n\
        'scaler': scaler\n\
    }\n\
    \n\
    # Store comprehensive analysis results\n\
    feature_names = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']\n\
    state.setdefault('analysis_results', {})['elastic_net_analysis'] = {\n\
        'test_mse': mse,\n\
        'coefficients': {name: coef for name, coef in zip(feature_names, model.coef_)},\n\
        'alpha': alpha,\n\
        'l1_ratio': l1_ratio,\n\
        'train_size': len(X_train),\n\
        'test_size': len(X_test),\n\
        'predictions': predictions.tolist(),\n\
        'actual_values': y_test.tolist()\n\
    }\n\
    \n\
    predict_and_analyze(model, scaler, X_test, y_test)\n\
    return model, scaler, mse\n\
\n\
# Example usage:\n\
model, scaler, mse = run_elastic_net_analysis(state, alpha=0.1, l1_ratio=0.7)\n\
",
    "code_description": "This Python script performs Elastic Net regression analysis using LangGraph state to predict SPY forward returns with train/test split validation. It accesses factor and SPY data from state, applies regularization combining Lasso and Ridge penalties, and stores comprehensive results including test performance metrics and model coefficients back in state. The analysis enables subsequent agents to evaluate predictive capabilities and use the trained model for forecasting in the LangGraph workflow."
},
{
    "question": "Create a dataframe of factor A, factor B, factor C, factor D, factor E and merge it with SPY 1 day forward returns as the dependent variable, split the train and test set, then fit the regression model and perform inference on test set using randomforestregression model?",
    "executable_code": "\
import pandas as pd\n\
from sklearn.ensemble import RandomForestRegressor\n\
from sklearn.model_selection import train_test_split\n\
from sklearn.metrics import mean_squared_error\n\
\n\
# Load data\n\
def load_data(factor_path, spy_path):\n\
    factors = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_prices = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, spy_prices\n\
\n\
# Prepare data\n\
def prepare_data(factors, spy_prices):\n\
    spy_prices['Forward_Return'] = spy_prices['Close'].pct_change().shift(-1)\n\
    combined_data = factors.join(spy_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Split data\n\
def split_data(data):\n\
    X = data[['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']]\n\
    y = data['Forward_Return']\n\
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\
    return X_train, X_test, y_train, y_test\n\
\n\
# Fit RandomForest model\n\
def fit_model(X_train, y_train):\n\
    model = RandomForestRegressor(n_estimators=100, random_state=42)\n\
    model.fit(X_train, y_train)\n\
    return model\n\
\n\
# Make predictions and analyze\n\
def predict_and_analyze(model, X_test, y_test):\n\
    predictions = model.predict(X_test)\n\
    mse = mean_squared_error(y_test, predictions)\n\
    print('Mean Squared Error on Test Set:', mse)\n\
\n\
# Tool call example - factors and SPY data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Main execution using LangGraph state\n\
def run_random_forest_regression_analysis(state, n_estimators=100):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'SPY_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('SPY data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    spy_prices = state['dataframes']['SPY_daily']\n\
    data = prepare_data(factors, spy_prices)\n\
    X_train, X_test, y_train, y_test = split_data(data)\n\
    model = fit_model(X_train, y_train)\n\
    \n\
    # Make predictions and calculate metrics\n\
    predictions = model.predict(X_test)\n\
    mse = mean_squared_error(y_test, predictions)\n\
    \n\
    # Store model and results in state\n\
    state.setdefault('models', {})['factor_spy_random_forest_regression'] = model\n\
    \n\
    # Store comprehensive analysis results\n\
    feature_names = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']\n\
    state.setdefault('analysis_results', {})['random_forest_regression_analysis'] = {\n\
        'test_mse': mse,\n\
        'feature_importance': {name: importance for name, importance in zip(feature_names, model.feature_importances_)},\n\
        'n_estimators': n_estimators,\n\
        'train_size': len(X_train),\n\
        'test_size': len(X_test),\n\
        'predictions': predictions.tolist(),\n\
        'actual_values': y_test.tolist()\n\
    }\n\
    \n\
    predict_and_analyze(model, X_test, y_test)\n\
    return model, mse\n\
\n\
# Example usage:\n\
model, mse = run_random_forest_regression_analysis(state, n_estimators=100)\n\
",
    "code_description": "This Python script performs Random Forest regression analysis using LangGraph state to predict SPY forward returns with comprehensive feature importance analysis. It accesses factor and SPY data from state, handles non-linear relationships through ensemble learning, and stores detailed results including feature importance rankings and performance metrics back in state. The Random Forest model provides robustness against overfitting and enables subsequent agents to understand which factors contribute most to market predictions in the LangGraph workflow."
},
{
    "question": "Create a dataframe of factor A, factor B, factor C, factor D, factor E and merge it with SPY 1 day forward returns, create the dependent categorical variables on SPY where if returns >0.25% variable = 1, elif returns <-0.25% variable = 0, split the train and test set, then fit the classification model and perform inference on test set using a randomforest model?",
    "executable_code": "\
import pandas as pd\n\
from sklearn.ensemble import RandomForestClassifier\n\
from sklearn.model_selection import train_test_split\n\
from sklearn.metrics import accuracy_score, classification_report\n\
\n\
# Load data function\n\
def load_data(factor_path, spy_path):\n\
    factors = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_prices = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, spy_prices\n\
\n\
# Prepare data function\n\
def prepare_data(factors, spy_prices):\n\
    spy_prices['Forward_Return'] = spy_prices['Close'].pct_change().shift(-1)\n\
    spy_prices['Target'] = spy_prices['Forward_Return'].apply(lambda x: 1 if x > 0.0025 else (-1 if x < -0.0025 else 0))\n\
    combined_data = factors.join(spy_prices['Target']).dropna()\n\
    return combined_data\n\
\n\
# Split data function\n\
def split_data(data):\n\
    X = data[['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']]\n\
    y = data['Target']\n\
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\
    return X_train, X_test, y_train, y_test\n\
\n\
# Fit RandomForest model function\n\
def fit_model(X_train, y_train):\n\
    model = RandomForestClassifier(n_estimators=100, random_state=42)\n\
    model.fit(X_train, y_train)\n\
    return model\n\
\n\
# Predict and analyze function\n\
def predict_and_analyze(model, X_test, y_test):\n\
    predictions = model.predict(X_test)\n\
    accuracy = accuracy_score(y_test, predictions)\n\
    report = classification_report(y_test, predictions)\n\
    print('Accuracy:', accuracy)\n\
    print('Classification Report:\\n', report)\n\
\n\
# Tool call example - factors and SPY data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Main execution using LangGraph state\n\
def run_random_forest_classification_analysis(state, up_threshold=0.0025, down_threshold=-0.0025, n_estimators=100):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'SPY_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('SPY data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    spy_prices = state['dataframes']['SPY_daily']\n\
    data = prepare_data(factors, spy_prices)\n\
    X_train, X_test, y_train, y_test = split_data(data)\n\
    model = fit_model(X_train, y_train)\n\
    \n\
    # Make predictions and calculate metrics\n\
    predictions = model.predict(X_test)\n\
    accuracy = accuracy_score(y_test, predictions)\n\
    classification_rep = classification_report(y_test, predictions, output_dict=True)\n\
    \n\
    # Store model and results in state\n\
    state.setdefault('models', {})['factor_spy_random_forest_classification'] = model\n\
    \n\
    # Store comprehensive analysis results\n\
    feature_names = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']\n\
    state.setdefault('analysis_results', {})['random_forest_classification_analysis'] = {\n\
        'test_accuracy': accuracy,\n\
        'classification_report': classification_rep,\n\
        'feature_importance': {name: importance for name, importance in zip(feature_names, model.feature_importances_)},\n\
        'up_threshold': up_threshold,\n\
        'down_threshold': down_threshold,\n\
        'n_estimators': n_estimators,\n\
        'train_size': len(X_train),\n\
        'test_size': len(X_test),\n\
        'predictions': predictions.tolist(),\n\
        'actual_labels': y_test.tolist()\n\
    }\n\
    \n\
    predict_and_analyze(model, X_test, y_test)\n\
    return model, accuracy\n\
\n\
# Example usage:\n\
model, accuracy = run_random_forest_classification_analysis(state, up_threshold=0.0025, down_threshold=-0.0025)\n\
",
    "code_description": "This Python script performs Random Forest classification analysis using LangGraph state to predict SPY return direction (up/down/neutral) based on financial factors. It accesses factor and SPY data from state, transforms continuous returns into categorical labels using configurable thresholds, and stores comprehensive classification results including accuracy metrics, feature importance, and confusion matrix data back in state. This enables subsequent trading strategy agents to make directional market predictions in the LangGraph workflow."
}]