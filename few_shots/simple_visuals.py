[
    {
    "question": "Create a scatter plot between factor A and SPY 1D forward returns using native Plotly with the 'plotly_dark' template and save the plot in a temporary location.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
import tempfile\n\
\n\
# Tool call example - factor and SPY data would be available in state:\n\
# from tools_clean import extract_daily_stock_data\n\
# state['dataframes']['factors'] = some_calculated_factors_dataframe  # from previous analysis\n\
# state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
\n\
# Load data from LangGraph state\n\
def load_data_from_state(state):\n\
    if 'factors' not in state.get('dataframes', {}):\n\
        raise ValueError('Factors data not found in state. Ensure factor calculation was performed first.')\n\
    if 'SPY_daily' not in state.get('dataframes', {}):\n\
        raise ValueError('SPY data not found in state. Ensure extract_daily_stock_data was called first.')\n\
    \n\
    factors = state['dataframes']['factors']\n\
    spy_prices = state['dataframes']['SPY_daily']\n\
    return factors, spy_prices\n\
\n\
# Prepare data function\n\
def prepare_data(factors, spy_prices):\n\
    spy_prices['Forward_Return'] = spy_prices['Close'].pct_change().shift(-1)\n\
    combined_data = factors[['Factor A']].join(spy_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Plotting function with state storage\n\
def plot_data(state, data, factor_name='Factor A'):\n\
    fig = go.Figure(\n\
        data=[go.Scatter(x=data[factor_name], y=data['Forward_Return'], mode='markers')], \n\
        layout=go.Layout(template='plotly_dark')\n\
    )\n\
    fig.update_layout(\n\
        title=f'Scatter Plot of {factor_name} vs. SPY 1D Forward Returns',\n\
        xaxis_title=factor_name,\n\
        yaxis_title='SPY 1D Forward Returns'\n\
    )\n\
    \n\
    # Save to temporary file\n\
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')\n\
    fig.write_html(temp_file.name)\n\
    print(f'Saved plot to {temp_file.name}')\n\
    \n\
    # Store plot reference and data in state\n\
    state.setdefault('visualizations', {})[f'{factor_name}_spy_scatter'] = {\n\
        'figure': fig,\n\
        'file_path': temp_file.name,\n\
        'data_summary': {\n\
            'correlation': data[factor_name].corr(data['Forward_Return']),\n\
            'data_points': len(data),\n\
            'factor_range': [data[factor_name].min(), data[factor_name].max()]\n\
        }\n\
    }\n\
    \n\
    return fig\n\
\n\
# Main execution function\n\
def create_factor_spy_scatter(state, factor_name='Factor A'):\n\
    factors, spy_prices = load_data_from_state(state)\n\
    data = prepare_data(factors, spy_prices)\n\
    plot = plot_data(state, data, factor_name)\n\
    plot.show()\n\
    return plot\n\
\n\
# Example usage in LangGraph agent:\n\
# 1. First ensure factor and SPY data are in state:\n\
#    state['dataframes']['factors'] = some_calculated_factors_dataframe\n\
#    state['dataframes']['SPY_daily'] = extract_daily_stock_data('SPY', '2020-01-01', '2023-12-31', '1d')\n\
# 2. Create visualization:\n\
plot = create_factor_spy_scatter(state, 'Factor A')\n\
",
    "code_description": "This Python script creates a scatter plot to analyze the relationship between Factor A and SPY's 1-day forward returns using LangGraph state, with Plotly's dark mode template. The function accesses factor and SPY data from state, creates an interactive visualization, stores the plot and correlation statistics back in state, and saves to a temporary HTML file. This enables subsequent agents in the LangGraph workflow to access visualization results and build upon the analysis."
},

{
    "question": "Create a scatter plot between Factor A and Factor B and SPY 1D forward returns with Factor A and Factor B on the primary Y-axis and SPY 1D forward returns on the secondary axis using Plotly dark mode theme.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
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
    combined_data = factors[['Factor A', 'Factor B']].join(spy_prices[['Forward_Return']]).dropna()\n\
    return combined_data\n\
\n\
# Plotting function\n\
def plot_data(data):\n\
    # Create figure with secondary y-axis\n\
    fig = go.Figure()\n\
\n\
    # Add Factor A trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Factor A'], name='Factor A', mode='markers', marker=dict(color='blue')),\n\
                  secondary_y=False)\n\
\n\
    # Add Factor B trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Factor B'], name='Factor B', mode='markers', marker=dict(color='red')),\n\
                  secondary_y=False)\n\
\n\
    # Add SPY Forward Returns trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Forward_Return'], name='SPY 1D Forward Returns', mode='lines', line=dict(color='green')),\n\
                  secondary_y=True)\n\
\n\
    # Layout updates\n\
    fig.update_layout(\n\
        title_text='Dual Y-Axis Plot: Factors A & B and SPY 1D Forward Returns',\n\
        plot_bgcolor='black',\n\
        paper_bgcolor='black',\n\
        font=dict(color='white'),\n\
        template='plotly_dark'\n\
    )\n\
\n\
    # X-axis and Y-axis titles\n\
    fig.update_xaxes(title_text='Date')\n\
    fig.update_yaxes(title_text='<b>Primary</b>: Factors A & B', secondary_y=False)\n\
    fig.update_yaxes(title_text='<b>Secondary</b>: SPY 1D Forward Returns', secondary_y=True)\n\
\n\
    fig.show()\n\
\n\
# Main execution\n\
factor_path = '/path/to/factor.csv'\n\
spy_path = '/path/to/SPY_daily_ohlcv.csv'\n\
factors, spy_prices = load_data(factor_path, spy_path)\n\
data = prepare_data(factors, spy_prices)\n\
plot_data(data)\n\
",
    "code_description": "This Python script creates an interactive scatter plot using Plotly with a dark mode theme, which plots Factor A and Factor B on the primary Y-axis and SPY's 1-day forward returns on a secondary Y-axis. The plot is designed to analyze how these factors relate to stock performance simultaneously on a single chart. The use of Plotly's 'plotly_dark' template ensures that the visualization is both aesthetically pleasing and easy to interpret in low-light conditions, enhancing the readability of different data points with distinct colors for clarity."
},

{
    "question": "Create a scatter plot between Factor A and Factor B and SPY 1D forward returns with Factor A and Factor B on the primary Y-axis and SPY 1D forward returns on the secondary axis using Plotly dark mode theme.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
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
    combined_data = factors[['Factor A', 'Factor B']].join(spy_prices[['Forward_Return']]).dropna()\n\
    return combined_data\n\
\n\
# Plotting function\n\
def plot_data(data):\n\
    # Create figure with secondary y-axis\n\
    fig = go.Figure()\n\
\n\
    # Add Factor A trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Factor A'], name='Factor A', mode='markers', marker=dict(color='blue')),\n\
                  secondary_y=False)\n\
\n\
    # Add Factor B trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Factor B'], name='Factor B', mode='markers', marker=dict(color='red')),\n\
                  secondary_y=False)\n\
\n\
    # Add SPY Forward Returns trace\n\
    fig.add_trace(go.Scatter(x=data.index, y=data['Forward_Return'], name='SPY 1D Forward Returns', mode='lines', line=dict(color='green')),\n\
                  secondary_y=True)\n\
\n\
    # Layout updates\n\
    fig.update_layout(\n\
        title_text='Dual Y-Axis Plot: Factors A & B and SPY 1D Forward Returns',\n\
        plot_bgcolor='black',\n\
        paper_bgcolor='black',\n\
        font=dict(color='white'),\n\
        template='plotly_dark'\n\
    )\n\
\n\
    # X-axis and Y-axis titles\n\
    fig.update_xaxes(title_text='Date')\n\
    fig.update_yaxes(title_text='<b>Primary</b>: Factors A & B', secondary_y=False)\n\
    fig.update_yaxes(title_text='<b>Secondary</b>: SPY 1D Forward Returns', secondary_y=True)\n\
\n\
    fig.show()\n\
\n\
# Main execution\n\
factor_path = '/path/to/factor.csv'\n\
spy_path = '/path/to/SPY_daily_ohlcv.csv'\n\
factors, spy_prices = load_data(factor_path, spy_path)\n\
data = prepare_data(factors, spy_prices)\n\
plot_data(data)\n\
",
    "code_description": "This Python script creates an interactive scatter plot using Plotly with a dark mode theme, which plots Factor A and Factor B on the primary Y-axis and SPY's 1-day forward returns on a secondary Y-axis. The plot is designed to analyze how these factors relate to stock performance simultaneously on a single chart. The use of Plotly's 'plotly_dark' template ensures that the visualization is both aesthetically pleasing and easy to interpret in low-light conditions, enhancing the readability of different data points with distinct colors for clarity."
},

{
    "question": "Perform decile analysis for factor A against SPY 1D forward returns; calculate percentile ranks with a rolling window of 60 days for factor A, bin them into 5 equi-spaced percentiles; merge with the SPY returns, then group by (mean) and plot the deciles vs returns as a bar chart using Plotly with dark mode theme.",
    "executable_code": "\
import pandas as pd\n\
import numpy as np\n\
import plotly.graph_objects as go\n\
\n\
# Load data function\n\
def load_data(factor_path, spy_path):\n\
    factors = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_prices = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    return factors, spy_prices\n\
\n\
# Prepare data function\n\
def prepare_data(factors, spy_prices, num_bins):\n\
    spy_prices['Forward_Return'] = spy_prices['Close'].pct_change().shift(-1)\n\
    factors['Factor_A_Rank'] = factors['Factor A'].rolling(window=60).apply(lambda x: pd.qcut(x.rank(method='first'), num_bins, labels=range(1, num_bins + 1)))\n\
    combined_data = factors[['Factor_A_Rank']].join(spy_prices['Forward_Return']).dropna()\n\
    return combined_data\n\
\n\
# Analyze and plot function\n\
def analyze_and_plot(data, num_bins):\n\
    grouped_data = data.groupby('Factor_A_Rank')['Forward_Return'].mean()\n\
    fig = go.Figure(data=[go.Bar(x=list(range(1, num_bins + 1)), y=grouped_data, marker_color='lightskyblue')])\n\
    fig.update_layout(\n\
        title='Decile Analysis of Factor A vs. SPY 1D Forward Returns',\n\
        xaxis_title='Decile of Factor A',\n\
        yaxis_title='Average SPY 1D Forward Returns',\n\
        template='plotly_dark'\n\
    )\n\
    fig.show()\n\
\n\
# Main execution\n\
factor_path = '/path/to/factor.csv'\n\
spy_path = '/path/to/SPY_daily_ohlcv.csv'\n\
num_bins = 5\n\
factors, spy_prices = load_data(factor_path, spy_path)\n\
data = prepare_data(factors, spy_prices, num_bins)\n\
analyze_and_plot(data, num_scores)\n\
",
    "code_description": "This Python script uses Plotly to perform a decile (quintile) analysis of Factor A against SPY's 1-day forward returns. It calculates rolling percentile ranks for Factor A over a 60-day window and bins these ranks into 5 equi-spaced percentiles. The script then merges these binned ranks with SPY's forward returns, computes the average return for each bin, and visualizes the results in a bar chart using Plotly's dark mode theme. This visualization helps to understand how different levels of Factor A influence SPY returns, providing a clear and interactive means to assess the factor's predictive power on stock market movements."
},

{
    "question": "Create histogram plots for Factor A and SPY returns on the same plot but at different scales using Plotly in dark mode, assuming factor_a and spy_close_prices are in different CSV files.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
from plotly.subplots import make_subplots\n\
import numpy as np\n\
\n\
# Load data function for Factor A from its CSV\n\
def load_factor_a_data(factor_path):\n\
    factor_a_data = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    return factor_a_data\n\
\n\
# Load data function for SPY prices from its CSV\n\
def load_spy_data(spy_path):\n\
    spy_data = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data['Forward_Return'] = spy_data['Close'].pct_change().shift(-1)\n\
    return spy_data\n\
\n\
# Creating a figure with secondary y-axis for different scales\n\
def create_histograms(factor_a_data, spy_data):\n\
    fig = make_subplots(specs=[[{'secondary_y': True}]])\n\
\n\
    # Adding histogram for Factor A\n\
    fig.add_trace(go.Histogram(x=factor_a_data['Factor A'], name='Factor A', marker_color='blue'), secondary_y=False)\n\
\n\
    # Adding histogram for SPY 1D Forward Returns\n\
    fig.add_trace(go.Histogram(x=spy_data['Forward_Return'], name='SPY Returns', marker_color='red'), secondary_y=True)\n\
\n\
    # Update layout to include titles and use the dark theme\n\
    fig.update_layout(\n\
        title='Histogram of Factor A and SPY 1D Forward Returns',\n\
        xaxis_title='Value',\n\
        template='plotly_dark'\n\
    )\n\
\n\
    # Update y-axes titles\n\
    fig.update_yaxes(title_text='Count for Factor A', secondary_y=False)\n\
    fig.update_yaxes(title_text='Count for SPY Returns', secondary_y=True)\n\
\n\
    fig.show()\n\
\n\
# Paths to CSV files\n\
factor_path = '/path/to/factor_a.csv'\n\
spy_path = '/path/to/spy_prices.csv'\n\
\n\
# Load data\n\
factor_a_data = load_factor_a_data(factor_path)\n\
spy_data = load_spy_data(spy_path)\n\
\n\
# Create and display histograms\n\
create_histograms(factor_a_data, spy_data)\n\
",
    "code_description": "This Python script demonstrates how to visualize the distribution of Factor A and SPY's 1-day forward returns on a shared plot with histograms plotted on separate y-axes to accommodate their differing scales, loading the data from separate CSV files. The plot is generated using Plotly, with histograms for Factor A on the primary y-axis and SPY returns on the secondary y-axis. This setup highlights the distributions effectively, using Plotly's 'plotly_dark' theme for enhanced visibility and a modern aesthetic. It allows for easy comparison of the data distributions while handling their scale differences appropriately."
},

{
    "question": "Create joint KDE plots for Factor A and 1D forward SPY returns using seaborn in dark mode, saving the plot as a PNG file. Assume data is stored in two different files, one for Factor A and another for SPY closing prices.",
    "executable_code": "\
import pandas as pd\n\
import seaborn as sns\n\
import matplotlib.pyplot as plt\n\
\n\
# Load data from CSV files\n\
def load_data(factor_path, spy_path):\n\
    factor_data = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data['Forward_Return'] = spy_data['Close'].pct_change().shift(-1)\n\
    return factor_data, spy_data\n\
\n\
# Create and save joint KDE plot using seaborn\n\
def create_and_save_kde_plot(factor_data, spy_data, png_file_path):\n\
    combined_data = pd.DataFrame({'Factor A': factor_data['Factor A'], 'SPY Forward Returns': spy_data['Forward_Return']})\n\
    sns.set(style='darkgrid')\n\
    plt.figure(figsize=(10, 6))\n\
    joint_kde = sns.jointplot(x='Factor A', y='SPY Forward Returns', data=combined_data, kind='kde', fill=True)\n\
    joint_kde.fig.suptitle('Joint KDE of Factor A and SPY 1D Forward Returns', fontsize=16)\n\
    joint_kde.fig.subplots_adjust(top=0.95)  # Adjust title position\n\
    plt.savefig(png_file_path)\n\
    return joint_kde\n\
\n\
# File paths to the CSV files\n\
factor_path = '/path/to/factor_a.csv'\n\
spy_path = '/path/to/spy_prices.csv'\n\
png_file_path = '/path/to/joint_kde_plot.png'\n\
\n\
# Load data\n\
factor_data, spy_data = load_data(factor_path, spy_path)\n\
\n\
# Create and save the KDE plot\n\
kde_plot = create_and_save_kde_plot(factor_data, spy_data, png_file_path)\n\
kde_plot.fig.show()\n\
",
    "code_description": "This Python script loads Factor A and SPY forward returns data from two separate CSV files, generates a joint KDE plot using the seaborn library in dark mode, and saves the plot as a PNG file. The script returns the seaborn joint KDE plot object for further use or inspection in a Python environment. This approach provides a detailed visualization of the relationship between Factor A and SPY returns."
},

{
    "question": "Create a histogram of rolling z-scores with a window of 20 days for Factor A and 1-day forward SPY returns on the same plot but at different scales.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
from plotly.subplots import make_subplots\n\
import numpy as np\n\
\n\
# Load data from CSV files\n\
def load_data(factor_path, spy_path):\n\
    factor_data = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data['Forward_Return'] = spy_data['Close'].pct_change().shift(-1)\n\
    return factor_data, spy_data\n\
\n\
# Calculate rolling z-scores\n\
def rolling_z_score(series, window=20):\n\
    rolling_mean = series.rolling(window=window).mean()\n\
    rolling_std = series.rolling(window=window).std()\n\
    z_scores = (series - rolling_mean) / rolling_std\n\
    return z_scores\n\
\n\
# Create and save histograms for rolling z-scores\n\
def create_histogram(factor_data, spy_data):\n\
    factor_z_scores = rolling_z_score(factor_data['Factor A'], window=20)\n\
    spy_z_scores = rolling_z_score(spy_data['Forward_Return'], window=20)\n\
\n\
    fig = make_subplots(specs=[[{'secondary_y': True}]])\n\
\n\
    # Add histogram for Factor A z-scores\n\
    fig.add_trace(go.Histogram(x=factor_z_scores, name='Factor A Z-Scores', marker_color='blue'), secondary_y=False)\n\
\n\
    # Add histogram for SPY forward return z-scores\n\
    fig.add_trace(go.Histogram(x=spy_z_scores, name='SPY Forward Return Z-Scores', marker_color='red'), secondary_y=True)\n\
\n\
    # Update layout to include titles and use dark mode\n\
    fig.update_layout(\n\
        title='Histograms of Rolling Z-Scores for Factor A and SPY 1D Forward Returns',\n\
        xaxis_title='Z-Score',\n\
        template='plotly_dark'\n\
    )\n\
\n\
    # Update y-axes titles\n\
    fig.update_yaxes(title_text='Count for Factor A', secondary_y=False)\n\
    fig.update_yaxes(title_text='Count for SPY Returns', secondary_y=True)\n\
\n\
    fig.show()\n\
    return fig\n\
\n\
# File paths to the CSV files\n\
factor_path = '/path/to/factor_a.csv'\n\
spy_path = '/path/to/spy_prices.csv'\n\
\n\
# Load data\n\
factor_data, spy_data = load_data(factor_path, spy_path)\n\
\n\
# Create histograms of rolling z-scores\n\
fig = create_histogram(factor_data, spy_data)\n\
",
    "code_description": "This Python script loads Factor A and SPY forward returns data from two separate CSV files, calculates rolling z-scores with a 20-day window for both datasets, and generates histograms of these z-scores on the same plot using Plotly in dark mode. The histograms are plotted on separate y-axes to accommodate their differing scales, and the plot is displayed interactively. This visualization helps to compare the distributions and variability of Factor A and SPY returns over time."
},

{
    "question": "Create joint KDE plots for rolling z-scores of Factor A (with a window of 20 days) and 1D forward SPY returns.",
    "executable_code": "\
import pandas as pd\n\
import seaborn as sns\n\
import matplotlib.pyplot as plt\n\
\n\
# Load data from CSV files\n\
def load_data(factor_path, spy_path):\n\
    factor_data = pd.read_csv(factor_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')\n\
    spy_data['Forward_Return'] = spy_data['Close'].pct_change().shift(-1)\n\
    return factor_data, spy_data\n\
\n\
# Calculate rolling z-scores for Factor A\n\
def rolling_z_score(series, window=20):\n\
    rolling_mean = series.rolling(window=window).mean()\n\
    rolling_std = series.rolling(window=window).std()\n\
    z_scores = (series - rolling_mean) / rolling_std\n\
    return z_scores\n\
\n\
# Create and save joint KDE plot using seaborn\n\
def create_and_save_kde_plot(factor_data, spy_data, png_file_path):\n\
    factor_z_scores = rolling_z_score(factor_data['Factor A'], window=20)\n\
    combined_data = pd.DataFrame({\n\
        'Factor A Z-Scores': factor_z_scores,\n\
        'SPY Forward Returns': spy_data['Forward_Return']\n\
    }).dropna()\n\
    sns.set(style='darkgrid')\n\
    plt.figure(figsize=(10, 6))\n\
    joint_kde = sns.jointplot(\n\
        x='Factor A Z-Scores', \n\
        y='SPY Forward Returns', \n\
        data=combined_data, \n\
        kind='kde', \n\
        fill=True\n\
    )\n\
    joint_kde.fig.suptitle('Joint KDE of Rolling Z-Scores for Factor A and SPY 1D Forward Returns', fontsize=16)\n\
    joint_kde.fig.subplots_adjust(top=0.95)  # Adjust title position\n\
    plt.savefig(png_file_path)\n\
    return joint_kde\n\
\n\
# File paths to the CSV files\n\
factor_path = '/path/to/factor_a.csv'\n\
spy_path = '/path/to/spy_prices.csv'\n\
png_file_path = '/path/to/joint_kde_plot.png'\n\
\n\
# Load data\n\
factor_data, spy_data = load_data(factor_path, spy_path)\n\
\n\
# Create and save the KDE plot\n\
kde_plot = create_and_save_kde_plot(factor_data, spy_data, png_file_path)\n\
kde_plot.fig.show()\n\
",
    "code_description": "This Python script loads Factor A and SPY forward returns data from two separate CSV files, calculates rolling z-scores with a 20-day window for Factor A, and generates joint KDE plots using seaborn. The plot visualizes the relationship between the rolling z-scores of Factor A and the 1-day forward SPY returns. The plot is saved as a PNG file and displayed for further inspection."
},

{
    "question": "Plot box plots for Factor A, B, C, D, E to analyze their distribution using Plotly in dark mode, and save the figure as an HTML file. Assume all factors are from the same CSV file.",
    "executable_code": "\
import pandas as pd\n\
import plotly.graph_objects as go\n\
import plotly.io as pio\n\
\n\
# Load data from CSV file\n\
def load_data(csv_path):\n\
    data = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')\n\
    return data\n\
\n\
# Create and save box plots using Plotly\n\
def create_and_save_box_plots(data, html_file_path):\n\
    fig = go.Figure()\n\
    factors = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E']\n\
\n\
    for factor in factors:\n\
        fig.add_trace(go.Box(y=data[factor], name=factor))\n\
\n\
    # Update layout to include titles and use dark mode\n\
    fig.update_layout(\n\
        title='Box Plots of Factors A, B, C, D, E',\n\
        yaxis_title='Value',\n\
        xaxis_title='Factors',\n\
        template='plotly_dark'\n\
    )\n\
\n\
    # Save the plot as an HTML file\n\
    pio.write_html(fig, file=html_file_path, auto_open=False)\n\
    return fig\n\
\n\
# File path to the CSV file\n\
csv_path = '/path/to/factors.csv'\n\
html_file_path = '/path/to/box_plots.html'\n\
\n\
# Load data\n\
data = load_data(csv_path)\n\
\n\
# Create and save the box plots\n\
fig = create_and_save_box_plots(data, html_file_path)\n\
fig.show()\n\
",
    "code_description": "This Python script loads data for Factors A, B, C, D, and E from a CSV file, creates box plots to analyze their distributions using Plotly in dark mode, and saves the figure as an HTML file. Each factor's distribution is visualized in a separate box plot, allowing for a comparative analysis of their distributions."
}
]