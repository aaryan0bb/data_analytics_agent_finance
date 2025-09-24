"""
Clean, agent-ready financial data extraction tools with Pydantic validation.
Designed for use with LangGraph + PydanticAI agents.
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional, List, Union, Dict, Any
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
# from pydantic_ai import Agent, RunContext  # Commented out for testing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS FOR INPUT/OUTPUT VALIDATION
# =============================================================================

class IntervalType(str, Enum):
    """Supported intervals for stock data"""
    DAILY = "1d"
    HOURLY = "60m"
    THIRTY_MIN = "30m"
    FIFTEEN_MIN = "15m"
    FIVE_MIN = "5m"
    ONE_MIN = "1m"

class PeriodType(str, Enum):
    """Supported periods for estimates and fundamentals"""
    QUARTER = "quarter"
    ANNUAL = "annual"

# INPUT MODELS
class StockDataRequest(BaseModel):
    """Request model for stock data extraction"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    interval: IntervalType = Field(default=IntervalType.DAILY, description="Data interval")

    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string")
        return v.upper().strip()

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class MacroDataRequest(BaseModel):
    """Request model for FRED macro data"""
    series_id: str = Field(..., description="FRED series ID (e.g., GDP, CPIAUCSL)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

    @validator('series_id')
    def validate_series_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Series ID must be a non-empty string")
        return v.upper().strip()

class AnalystEstimatesRequest(BaseModel):
    """Request model for analyst estimates data"""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    period: PeriodType = Field(default=PeriodType.QUARTER, description="Data period")

class FundamentalsRequest(BaseModel):
    """Request model for fundamentals data"""
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    tickers: Optional[List[str]] = Field(None, description="List of stock ticker symbols")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class InsiderTradesRequest(BaseModel):
    """Request model for insider trades data"""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class ESGRatingsRequest(BaseModel):
    """Request model for ESG ratings data"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")

    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string")
        return v.upper().strip()

# OUTPUT MODELS
class DataExtractionResponse(BaseModel):
    """Generic response model for data extraction"""
    success: bool = Field(..., description="Whether the extraction was successful")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted data as list of records")
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")
    records_count: int = Field(0, description="Number of records returned")
    ticker: Optional[str] = Field(None, description="Ticker symbol (for stock data)")
    series_id: Optional[str] = Field(None, description="Series ID (for macro data)")

# =============================================================================
# CORE DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_daily_stock_data(ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Extract daily/intraday stock data from Polygon API
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1d', '60m', '30m', '15m', '5m', '1m')
    
    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    try:
        from polygon import RESTClient
        from polygon.rest.models import Agg
        import datetime

        # Validate API key
        POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
        if not POLYGON_API_KEY:
            raise ValueError("POLYGON_API_KEY environment variable not set")
        
        client = RESTClient(POLYGON_API_KEY)

        # Convert interval to Polygon API format
        interval_mapping = {
            '1d': ('day', 1),
            '60m': ('hour', 1),
            '30m': ('minute', 30),
            '15m': ('minute', 15),
            '5m': ('minute', 5),
            '1m': ('minute', 1)
        }
        
        if interval not in interval_mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        
        timespan, multiplier = interval_mapping[interval]

        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        logger.info(f"Fetching {interval} data for {ticker} from {start_date} to {end_date}")

        # Fetch data from Polygon API
        aggs = []
        data = client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_dt.strftime("%Y-%m-%d"),
            to=end_dt.strftime("%Y-%m-%d"),
            limit=50000
        )
        aggs.extend(data)

        if not aggs:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Convert aggs to DataFrame
        records = []
        for agg in aggs:
            if isinstance(agg, Agg):
                records.append({
                    "datetime": datetime.datetime.fromtimestamp(agg.timestamp / 1000),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                })

        df = pd.DataFrame(records)
        
        if df.empty:
            return df

        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Retrieved {len(df)} records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def extract_intraday_stock_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Extract intraday stock data - delegates to extract_daily_stock_data
    """
    return extract_daily_stock_data(ticker, start_date, end_date, interval)

def extract_macro_data_from_fred(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract macroeconomic data from FRED API
    
    Args:
        series_id: FRED series identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with date and series data
    """
    try:
        from fredapi import Fred

        # Validate API key
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable not set")
        
        fred = Fred(api_key=api_key)
        
        # Convert string dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        logger.info(f"Fetching FRED data for series {series_id} from {start_date} to {end_date}")
        
        # Fetch data from FRED API
        data = fred.get_series(series_id, start_date=start_dt, end_date=end_dt)
        
        if data.empty:
            logger.warning(f"No data returned for series {series_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[series_id])
        df.index.name = 'date'
        df.reset_index(inplace=True)
        
        logger.info(f"Retrieved {len(df)} records for {series_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving FRED data for {series_id}: {e}")
        return pd.DataFrame()

def extract_analyst_estimates_data_from_fmp(ticker: str, start_date: str, end_date: str, period: str = "quarter") -> pd.DataFrame:
    """
    Extract analyst estimates data from Financial Modeling Prep API
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Data period ('quarter' or 'annual')
    
    Returns:
        DataFrame with analyst estimates data
    """
    try:
        from urllib.request import urlopen
        import certifi
        import json

        # Validate API key
        api_key = os.getenv('FMP_API_KEY')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable not set")

        def get_jsonparsed_data(url):
            with urlopen(url, cafile=certifi.where()) as response:
                data = response.read().decode("utf-8")
            return json.loads(data)

        logger.info(f"Fetching analyst estimates for {ticker}, period: {period}")

        # Fetch estimates data
        base_url = "https://financialmodelingprep.com/api/v3/analyst-estimates"
        url = (f"{base_url}/{ticker}"
               f"?period={period}"
               f"&limit=500"
               f"&apikey={api_key}")
        
        json_data = get_jsonparsed_data(url)
        
        if not json_data or not isinstance(json_data, list):
            logger.warning(f"No estimates data found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        
        # Filter by date range if date column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]
            
            df.sort_values('date', inplace=True, ascending=False)
        
        logger.info(f"Retrieved {len(df)} analyst estimate records for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching analyst estimates for {ticker}: {e}")
        return pd.DataFrame()

def extract_fundamentals_data_from_fmp(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract fundamental data from Financial Modeling Prep API
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)  
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with key fundamental metrics
    """
    try:
        from urllib.request import urlopen
        import certifi
        import json

        # Validate API key
        FMP_API_KEY = os.getenv('FMP_API_KEY')
        if not FMP_API_KEY:
            raise ValueError("FMP_API_KEY environment variable not set")

        def get_jsonparsed_data(url):
            with urlopen(url, cafile=certifi.where()) as response:
                data = response.read().decode("utf-8")
            return json.loads(data)

        logger.info(f"Fetching fundamental data for {ticker}")

        # Fetch key metrics data
        FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/"
        url = (f"{FMP_BASE_URL}key-metrics/{ticker}"
               f"?period=quarter&limit=60&apikey={FMP_API_KEY}")

        # url = (f"https://financialmodelingprep.com/api/v3/key-metrics/AAPL?period=quarter&limit=60&apikey=4659ab78d056051dc50f892d465471f1")
        
        data = get_jsonparsed_data(url)
        
        if not data:
            logger.warning(f"No fundamental data returned for {ticker}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Convert date column and filter by date range
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]
            
            df.sort_values('date', inplace=True)
        
        # Ensure symbol column
        df['symbol'] = ticker
        
        logger.info(f"Retrieved {len(df)} fundamental records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching fundamentals data for {ticker}: {e}")
        return pd.DataFrame()

def extract_latest_insider_trades_from_fmp(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract latest insider trades data from Financial Modeling Prep API

    Args:
        start_date: Start date (YYYY-MM-DD) - used for post-processing filtering
        end_date: End date (YYYY-MM-DD) - used for post-processing filtering

    Returns:
        DataFrame with insider trades data filtered by date range
    """
    try:
        from urllib.request import urlopen
        import certifi
        import json

        # Validate API key
        api_key = os.getenv('FMP_API_KEY')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable not set")

        def get_jsonparsed_data(url):
            with urlopen(url, cafile=certifi.where()) as response:
                data = response.read().decode("utf-8")
            return json.loads(data)

        logger.info("Fetching latest insider trades data")
        logger.info(
            "Insider trades call parameters: start_date=%s, end_date=%s",
            start_date,
            end_date,
        )

        # Fetch insider trades data
        base_url = "https://financialmodelingprep.com/api/v3/"
        url = f"{base_url}insider-trading/latest?limit=100&apikey={api_key}"

        json_data = get_jsonparsed_data(url)

        if not json_data or not isinstance(json_data, list):
            logger.warning("No insider trades data found")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(json_data)

        # Filter by date range if date column exists (similar to fundamentals filtering)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]

            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]

            df.sort_values('date', inplace=True, ascending=False)
        elif 'filingDate' in df.columns:
            # Alternative date column sometimes used in insider trades
            df['date'] = pd.to_datetime(df['filingDate'])

            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]

            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]

            df.sort_values('date', inplace=True, ascending=False)

        logger.info(f"Retrieved {len(df)} insider trades records")
        return df

    except Exception as e:
        logger.error(f"Error fetching insider trades data: {e}")
        return pd.DataFrame()

def extract_esg_ratings_from_fmp(ticker: str) -> pd.DataFrame:
    """
    Extract ESG ratings data from Financial Modeling Prep API

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with ESG ratings data
    """
    try:
        from urllib.request import urlopen
        import certifi
        import json

        # Validate API key
        api_key = os.getenv('FMP_API_KEY')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable not set")

        def get_jsonparsed_data(url):
            with urlopen(url, cafile=certifi.where()) as response:
                data = response.read().decode("utf-8")
            return json.loads(data)

        logger.info(f"Fetching ESG ratings for {ticker}")

        # Fetch ESG ratings data
        base_url = "https://financialmodelingprep.com/api/v3/"
        url = f"{base_url}esg-ratings?symbol={ticker}&apikey={api_key}"

        json_data = get_jsonparsed_data(url)

        if not json_data:
            logger.warning(f"No ESG ratings data found for {ticker}")
            return pd.DataFrame()

        # Convert to DataFrame - handle both single object and list responses
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        else:
            df = pd.DataFrame([json_data])

        # Ensure symbol column
        df['symbol'] = ticker

        logger.info(f"Retrieved {len(df)} ESG ratings records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error fetching ESG ratings for {ticker}: {e}")
        return pd.DataFrame()

# =============================================================================
# AGENT-READY TOOL WRAPPERS
# =============================================================================

def get_stock_data(request: StockDataRequest) -> DataExtractionResponse:
    """
    Agent-ready wrapper for stock data extraction
    """
    try:
        df = extract_daily_stock_data(
            ticker=request.ticker,
            start_date=request.start_date, 
            end_date=request.end_date,
            interval=request.interval.value
        )
        
        if df.empty:
            return DataExtractionResponse(
                success=False,
                error_message=f"No stock data found for {request.ticker}",
                ticker=request.ticker,
                records_count=0
            )
        
        # Convert DataFrame to records
        df_reset = df.reset_index()
        data = df_reset.to_dict('records')
        
        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data),
            ticker=request.ticker
        )
        
    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            ticker=request.ticker,
            records_count=0
        )

def get_macro_data(request: MacroDataRequest) -> DataExtractionResponse:
    """
    Agent-ready wrapper for FRED macro data extraction
    """
    try:
        df = extract_macro_data_from_fred(
            series_id=request.series_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if df.empty:
            return DataExtractionResponse(
                success=False,
                error_message=f"No macro data found for {request.series_id}",
                series_id=request.series_id,
                records_count=0
            )
        
        data = df.to_dict('records')
        
        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data),
            series_id=request.series_id
        )
        
    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            series_id=request.series_id,
            records_count=0
        )

def get_analyst_estimates(request: AnalystEstimatesRequest) -> DataExtractionResponse:
    """
    Agent-ready wrapper for analyst estimates data extraction
    """
    try:
        df = extract_analyst_estimates_data_from_fmp(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            period=request.period.value
        )
        
        if df.empty:
            return DataExtractionResponse(
                success=False,
                error_message=f"No analyst estimates found for {request.ticker}",
                ticker=request.ticker,
                records_count=0
            )
        
        data = df.to_dict('records')
        
        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data),
            ticker=request.ticker
        )
        
    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            ticker=request.ticker,
            records_count=0
        )

def get_fundamentals_data(request: FundamentalsRequest) -> DataExtractionResponse:
    """Agent-ready wrapper for fundamentals data extraction.
    Supports single ticker (request.ticker) or multiple (request may include 'tickers').
    Returns a long/stacked DataFrame as records.
    """
    try:
        # Accept both single and multi ticker inputs
        tickers = []
        if hasattr(request, 'tickers') and request.tickers:
            tickers = list(request.tickers)
        if getattr(request, 'ticker', None):
            tickers = tickers or [request.ticker]
        if not tickers:
            return DataExtractionResponse(success=False, error_message="No tickers provided", records_count=0)

        frames = []
        for t in tickers:
            df = extract_fundamentals_data_from_fmp(
                ticker=t,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            if not df.empty:
                # Ensure symbol column is uppercase
                if 'symbol' in df.columns:
                    df['symbol'] = df['symbol'].astype(str).str.upper()
                else:
                    df['symbol'] = str(t).upper()
                frames.append(df)

        if not frames:
            return DataExtractionResponse(
                success=False,
                error_message=f"No fundamentals data found for {tickers}",
                records_count=0
            )

        df_all = pd.concat(frames, axis=0, ignore_index=True)
        # Sort by symbol then date if present
        if 'date' in df_all.columns:
            df_all = df_all.sort_values(['symbol', 'date'])
        else:
            df_all = df_all.sort_values(['symbol'])

        data = df_all.to_dict('records')
        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data),
        )
    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            records_count=0
        )

def get_insider_trades(request: InsiderTradesRequest) -> DataExtractionResponse:
    """
    Agent-ready wrapper for insider trades data extraction
    """
    try:
        df = extract_latest_insider_trades_from_fmp(
            start_date=request.start_date,
            end_date=request.end_date
        )

        if df.empty:
            return DataExtractionResponse(
                success=False,
                error_message="No insider trades data found",
                records_count=0
            )

        data = df.to_dict('records')

        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data)
        )

    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            records_count=0
        )

def get_esg_ratings(request: ESGRatingsRequest) -> DataExtractionResponse:
    """
    Agent-ready wrapper for ESG ratings data extraction
    """
    try:
        df = extract_esg_ratings_from_fmp(
            ticker=request.ticker
        )

        if df.empty:
            return DataExtractionResponse(
                success=False,
                error_message=f"No ESG ratings data found for {request.ticker}",
                ticker=request.ticker,
                records_count=0
            )

        data = df.to_dict('records')

        return DataExtractionResponse(
            success=True,
            data=data,
            records_count=len(data),
            ticker=request.ticker
        )

    except Exception as e:
        return DataExtractionResponse(
            success=False,
            error_message=str(e),
            ticker=request.ticker,
            records_count=0
        )

def bulk_extract_daily_closing_prices_from_polygon(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract daily closing prices for multiple tickers from Polygon API, merge and clean data
    
    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with date as index and ticker symbols as columns containing closing prices.
        Rows with any NaN values are removed after forward filling.
    """
    try:
        logger.info(f"Bulk extracting daily closing prices for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Initialize list to store individual ticker dataframes
        ticker_dataframes = []
        successful_tickers = []
        
        # Extract data for each ticker
        for ticker in tickers:
            try:
                logger.info(f"Extracting data for {ticker}")
                df = extract_daily_stock_data(ticker, start_date, end_date, interval='1d')
                
                if not df.empty and 'close' in df.columns:
                    # Keep only the close price and rename column to ticker symbol
                    close_df = df[['close']].copy()
                    close_df.columns = [ticker]
                    ticker_dataframes.append(close_df)
                    successful_tickers.append(ticker)
                    logger.info(f"Successfully extracted {len(close_df)} records for {ticker}")
                else:
                    logger.warning(f"No data or missing 'close' column for {ticker}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract data for {ticker}: {e}")
                continue
        
        if not ticker_dataframes:
            logger.error("No data extracted for any tickers")
            return pd.DataFrame()
        
        logger.info(f"Successfully extracted data for {len(successful_tickers)} out of {len(tickers)} tickers")
        
        # Perform outer join to merge all ticker data
        logger.info("Merging data using outer join")
        merged_df = ticker_dataframes[0]
        
        for df in ticker_dataframes[1:]:
            merged_df = merged_df.join(df, how='outer')
        
        # Sort by date index
        merged_df.sort_index(inplace=True)
        
        logger.info(f"Merged dataframe shape before cleaning: {merged_df.shape}")
        logger.info(f"Missing values per column:\n{merged_df.isnull().sum()}")
        
        # Forward fill missing values
        logger.info("Forward filling missing values")
        merged_df = merged_df.ffill()
        
        # Remove rows where any column still has NaN values
        logger.info("Removing rows with any remaining NaN values")
        initial_rows = len(merged_df)
        merged_df = merged_df.dropna()
        final_rows = len(merged_df)
        
        logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        logger.info(f"Final dataframe shape: {merged_df.shape}")
        
        if merged_df.empty:
            logger.warning("No data remaining after cleaning")
            return pd.DataFrame()
        
        # Add summary statistics
        logger.info("Final dataset summary:")
        logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        logger.info(f"Tickers included: {list(merged_df.columns)}")
        logger.info(f"Total trading days: {len(merged_df)}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error in bulk extraction: {e}")
        return pd.DataFrame()

def date_alignment_for_series(series_dict: dict) -> pd.DataFrame:
    """
    Align time series data with different frequencies using merge_asof to avoid forward bias.
    
    Args:
        series_dict: Dictionary where keys are series names and values are DataFrames.
                    Each DataFrame should have a date column and data columns.
                    Expected format: {
                        'daily_stocks': DataFrame with date index or 'date' column,
                        'monthly_cpi': DataFrame with date column and CPI data,
                        'quarterly_gdp': DataFrame with date column and GDP data,
                        ...
                    }
    
    Returns:
        DataFrame with aligned data where higher frequency series (daily) is the base,
        and lower frequency series (monthly/quarterly) are merged using backward search
        to avoid forward bias. Rows with any NaN values are removed after forward fill.
    """
    try:
        if not series_dict:
            logger.error("Empty series dictionary provided")
            return pd.DataFrame()
            
        logger.info(f"Aligning {len(series_dict)} time series with different frequencies")
        
        # Convert all DataFrames to have 'date' as a column (not index)
        processed_series = {}
        base_df = None
        base_name = None
        max_records = 0
        
        for name, df in series_dict.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame: {name}")
                continue
                
            # Copy DataFrame to avoid modifying original
            df_copy = df.copy()
            
            # Ensure date column exists
            if 'date' not in df_copy.columns:
                if hasattr(df_copy.index, 'name') and df_copy.index.name in ['date', 'datetime']:
                    df_copy = df_copy.reset_index()
                    # Rename the index column to 'date' if it's not already named 'date'
                    if df_copy.columns[0] != 'date':
                        df_copy = df_copy.rename(columns={df_copy.columns[0]: 'date'})
                elif df_copy.index.dtype.kind in ['M', 'O']:  # datetime-like index
                    df_copy = df_copy.reset_index()
                    if df_copy.columns[0] != 'date':
                        df_copy = df_copy.rename(columns={df_copy.columns[0]: 'date'})
                elif pd.api.types.is_datetime64_any_dtype(df_copy.index):  # datetime index without name
                    df_copy = df_copy.reset_index()
                    df_copy = df_copy.rename(columns={df_copy.columns[0]: 'date'})
                else:
                    raise ValueError(f"DataFrame {name} must have a 'date' column or datetime index")
            
            # Ensure date column is datetime
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # Sort by date
            df_copy = df_copy.sort_values('date')
            
            logger.info(f"Series {name}: {len(df_copy)} records, date range {df_copy['date'].min()} to {df_copy['date'].max()}")
            
            # Find the series with most records (likely highest frequency) as base
            if len(df_copy) > max_records:
                max_records = len(df_copy)
                base_df = df_copy
                base_name = name
            
            processed_series[name] = df_copy
        
        if base_df is None:
            logger.error("No valid DataFrames found")
            return pd.DataFrame()
        
        logger.info(f"Using '{base_name}' as base series with {len(base_df)} records")
        
        # Start with base DataFrame
        aligned_df = base_df.copy()
        
        # Merge other series using merge_asof (backward search to avoid forward bias)
        for name, df in processed_series.items():
            if name == base_name:
                continue
                
            logger.info(f"Merging {name} using merge_asof with backward search")
            
            # Perform merge_asof - looks backward to find closest date without forward bias
            aligned_df = pd.merge_asof(
                aligned_df.sort_values('date'),
                df.sort_values('date'),
                on='date',
                direction='backward',  # Only look backward to avoid forward bias
                suffixes=('', f'_{name}')
            )
            
            logger.info(f"After merging {name}: shape {aligned_df.shape}")
        
        # Set date as index
        aligned_df.set_index('date', inplace=True)
        aligned_df.sort_index(inplace=True)
        
        logger.info(f"Merged dataframe shape before cleaning: {aligned_df.shape}")
        logger.info(f"Missing values per column:\n{aligned_df.isnull().sum()}")
        
        # Forward fill missing values (common for economic indicators)
        logger.info("Forward filling missing values")
        aligned_df = aligned_df.ffill()
        
        # Remove rows where any column still has NaN values
        logger.info("Removing rows with any remaining NaN values")
        initial_rows = len(aligned_df)
        aligned_df = aligned_df.dropna()
        final_rows = len(aligned_df)
        
        logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        logger.info(f"Final aligned dataframe shape: {aligned_df.shape}")
        
        if aligned_df.empty:
            logger.warning("No data remaining after alignment and cleaning")
            return pd.DataFrame()
        
        # Summary statistics
        logger.info("Final aligned dataset summary:")
        logger.info(f"Date range: {aligned_df.index.min()} to {aligned_df.index.max()}")
        logger.info(f"Columns: {list(aligned_df.columns)}")
        logger.info(f"Total aligned records: {len(aligned_df)}")
        
        return aligned_df
        
    except Exception as e:
        logger.error(f"Error in date alignment: {e}")
        return pd.DataFrame()

# =============================================================================
# EXAMPLE AGENT INTEGRATION
# =============================================================================

# Example of how to register these as PydanticAI agent tools:
"""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

agent = Agent(
    OpenAIResponsesModel("gpt-4o-mini"),
    instructions="You are a financial data analyst. Use the available tools to extract financial data.",
)

@agent.tool
def stock_data_tool(_: RunContext[None], ticker: str, start_date: str, end_date: str, interval: str = "1d") -> DataExtractionResponse:
    \"\"\"Extract stock price data for analysis\"\"\"
    request = StockDataRequest(ticker=ticker, start_date=start_date, end_date=end_date, interval=interval)
    return get_stock_data(request)

@agent.tool
def macro_data_tool(_: RunContext[None], series_id: str, start_date: str, end_date: str) -> DataExtractionResponse:
    \"\"\"Extract macroeconomic data from FRED\"\"\"
    request = MacroDataRequest(series_id=series_id, start_date=start_date, end_date=end_date)
    return get_macro_data(request)

@agent.tool  
def analyst_estimates_tool(_: RunContext[None], ticker: str, start_date: str, end_date: str, period: str = "quarter") -> DataExtractionResponse:
    \"\"\"Extract analyst estimates data\"\"\"
    request = AnalystEstimatesRequest(ticker=ticker, start_date=start_date, end_date=end_date, period=period)
    return get_analyst_estimates(request)

@agent.tool
def fundamentals_tool(_: RunContext[None], ticker: str, start_date: str, end_date: str) -> DataExtractionResponse:
    \"\"\"Extract fundamental financial metrics\"\"\"
    request = FundamentalsRequest(ticker=ticker, start_date=start_date, end_date=end_date)
    return get_fundamentals_data(request)
"""