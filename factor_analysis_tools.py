"""Factor and alpha analysis tools for quantitative research.

This module provides comprehensive factor analysis capabilities including:
- Alpha predictiveness testing using alphalens-reloaded
- Risk factor attribution using statistical factor models
- Robust statistical testing with HAC/Newey-West standard errors
- Bootstrap and jackknife resampling for stability assessment
- Information coefficient (IC) and information ratio (IR) analysis
- Quantile spread analysis and factor decay testing

The tools integrate with the existing quant research framework and support
both forward-looking alpha analysis and backward-looking risk attribution.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Configure logging
logger = logging.getLogger(__name__)

# Factor analysis dependencies
try:
    import alphalens as al
    HAS_ALPHALENS = True
    logger.info("alphalens-reloaded imported successfully")
except ImportError:
    logger.warning("alphalens-reloaded not available - factor analysis will be limited")
    HAS_ALPHALENS = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.weightstats import DescrStatsW
    from statsmodels.sandbox.stats.runs import runstest_1samp
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import het_white
    from arch.bootstrap import IIDBootstrap, CircularBlockBootstrap
    HAS_STATSMODELS = True
    logger.info("statsmodels and arch imported successfully")
except ImportError:
    logger.warning("statsmodels/arch not available - advanced statistical tests will be limited")
    HAS_STATSMODELS = False

try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
    logger.info("pandas-datareader imported successfully")
except ImportError:
    logger.warning("pandas-datareader not available - benchmark factor loading will be limited")
    HAS_DATAREADER = False

try:
    from scipy import stats
    import scipy.optimize as opt
    HAS_SCIPY = True
    logger.info("scipy imported successfully")
except ImportError:
    logger.warning("scipy not available - statistical tests will be limited")
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    logger.info("matplotlib and seaborn imported successfully")
except ImportError:
    logger.warning("matplotlib/seaborn not available - visualization will be disabled")
    HAS_PLOTTING = False

# Suppress warnings from libraries
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Data Models and Enums
# =============================================================================

class FactorAnalysisType(str, Enum):
    """Type of factor analysis to perform."""
    ALPHA_PREDICTIVENESS = "alpha_predictiveness"
    RISK_ATTRIBUTION = "risk_attribution"
    COMBINED_ANALYSIS = "combined_analysis"


class StatisticalTest(str, Enum):
    """Statistical tests for factor analysis."""
    T_TEST = "t_test"
    NEWEY_WEST = "newey_west"
    HAC_ROBUST = "hac_robust"
    BOOTSTRAP = "bootstrap"
    JACKKNIFE = "jackknife"


class FactorModel(str, Enum):
    """Factor models for risk analysis."""
    CAPM = "capm"
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    CUSTOM = "custom"


class FactorAnalysisRequest(BaseModel):
    """Request model for factor and alpha analysis."""

    factor_data: List[Dict[str, Any]] = Field(..., description="Factor exposures or alpha signals")
    returns_data: List[Dict[str, Any]] = Field(..., description="Asset returns data")
    price_data: Optional[List[Dict[str, Any]]] = Field(None, description="Price data for quantile analysis")

    # Analysis configuration
    analysis_type: FactorAnalysisType = Field(FactorAnalysisType.COMBINED_ANALYSIS, description="Type of analysis")

    # Statistical testing options
    statistical_tests: List[StatisticalTest] = Field([StatisticalTest.T_TEST, StatisticalTest.NEWEY_WEST], description="Statistical tests to perform")
    confidence_level: float = Field(0.95, description="Confidence level for tests")

    # Bootstrap/resampling options
    bootstrap_samples: int = Field(1000, description="Number of bootstrap samples")
    block_size: Optional[int] = Field(None, description="Block size for block bootstrap (auto if None)")

    # Factor model options
    factor_model: FactorModel = Field(FactorModel.FAMA_FRENCH_3, description="Factor model for risk attribution")
    custom_factors: Optional[List[Dict[str, Any]]] = Field(None, description="Custom factor data")

    # Alphalens options (for alpha predictiveness)
    quantiles: int = Field(5, description="Number of quantiles for analysis")
    periods: List[int] = Field([1, 5, 10], description="Forward-looking periods for alpha analysis")

    # Output options
    save_charts: bool = Field(True, description="Generate visualization artifacts")
    output_dir: str = Field("artifacts", description="Directory for output files")

    # Risk-free rate
    risk_free_rate: float = Field(0.0, description="Risk-free rate for calculations")

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        return v

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: int) -> int:
        if v < 2 or v > 10:
            raise ValueError("Quantiles must be between 2 and 10")
        return v

    @field_validator("bootstrap_samples")
    @classmethod
    def validate_bootstrap_samples(cls, v: int) -> int:
        if v < 100:
            raise ValueError("Bootstrap samples must be at least 100")
        return min(v, 10000)  # Cap for performance

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class FactorAnalysisResponse(BaseModel):
    """Response model for factor and alpha analysis."""

    success: bool
    analysis_type: str

    # Core factor metrics
    information_coefficient: Optional[Dict[str, float]] = None
    information_ratio: Optional[Dict[str, float]] = None

    # Alpha predictiveness metrics (forward-looking)
    alpha_metrics: Optional[Dict[str, Any]] = None
    quantile_analysis: Optional[Dict[str, Any]] = None

    # Risk attribution metrics (backward-looking)
    risk_attribution: Optional[Dict[str, Any]] = None
    factor_loadings: Optional[Dict[str, float]] = None
    factor_returns: Optional[Dict[str, float]] = None

    # Statistical test results
    statistical_tests: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Robustness tests
    bootstrap_results: Optional[Dict[str, Any]] = None
    jackknife_results: Optional[Dict[str, Any]] = None

    # Model diagnostics
    model_diagnostics: Optional[Dict[str, Any]] = None

    # Generated artifacts
    artifacts: Dict[str, str] = Field(default_factory=dict)

    error_message: Optional[str] = None


# =============================================================================
# Core Factor Analysis Functions
# =============================================================================

def analyze_factor(request: FactorAnalysisRequest) -> FactorAnalysisResponse:
    """Perform comprehensive factor and alpha analysis.

    This is the main entry point that orchestrates different types of analysis
    based on the request configuration.
    """
    try:
        logger.info(f"Starting {request.analysis_type.value} analysis")

        # Convert input data to pandas
        factor_df = pd.DataFrame(request.factor_data)
        returns_df = pd.DataFrame(request.returns_data)

        if factor_df.empty or returns_df.empty:
            return FactorAnalysisResponse(
                success=False,
                analysis_type=request.analysis_type.value,
                error_message="Factor data or returns data is empty"
            )

        # Standardize date columns
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        returns_df['date'] = pd.to_datetime(returns_df['date'])

        response = FactorAnalysisResponse(
            success=True,
            analysis_type=request.analysis_type.value
        )

        # Perform analysis based on type
        if request.analysis_type in [FactorAnalysisType.ALPHA_PREDICTIVENESS, FactorAnalysisType.COMBINED_ANALYSIS]:
            alpha_results = _analyze_alpha_predictiveness(factor_df, returns_df, request)
            response.alpha_metrics = alpha_results.get('alpha_metrics')
            response.quantile_analysis = alpha_results.get('quantile_analysis')
            response.information_coefficient = alpha_results.get('information_coefficient')
            response.information_ratio = alpha_results.get('information_ratio')

        if request.analysis_type in [FactorAnalysisType.RISK_ATTRIBUTION, FactorAnalysisType.COMBINED_ANALYSIS]:
            risk_results = _analyze_risk_attribution(factor_df, returns_df, request)
            if 'error' not in risk_results:
                response.risk_attribution = risk_results.get('risk_attribution')
                response.factor_loadings = risk_results.get('factor_loadings')
                response.factor_returns = risk_results.get('factor_returns')
            else:
                # Set empty results for missing factor data
                response.risk_attribution = {'error': risk_results['error']}
                response.factor_loadings = {}
                response.factor_returns = {}

        # Perform statistical tests
        if HAS_STATSMODELS:
            response.statistical_tests = _perform_statistical_tests(factor_df, returns_df, request)

        # Robustness testing
        if StatisticalTest.BOOTSTRAP in request.statistical_tests:
            response.bootstrap_results = _bootstrap_analysis(factor_df, returns_df, request)

        if StatisticalTest.JACKKNIFE in request.statistical_tests:
            response.jackknife_results = _jackknife_analysis(factor_df, returns_df, request)

        # Model diagnostics
        response.model_diagnostics = _calculate_model_diagnostics(factor_df, returns_df, request)

        # Generate artifacts if requested
        if request.save_charts and HAS_PLOTTING:
            response.artifacts = _generate_factor_artifacts(factor_df, returns_df, request, response)

        logger.info(f"Factor analysis completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in factor analysis: {e}")
        return FactorAnalysisResponse(
            success=False,
            analysis_type=request.analysis_type.value,
            error_message=str(e)
        )


def _analyze_alpha_predictiveness(factor_df: pd.DataFrame,
                                returns_df: pd.DataFrame,
                                request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Analyze alpha predictiveness using alphalens integration."""
    results = {}

    if not HAS_ALPHALENS:
        logger.warning("Alphalens not available - using basic IC analysis")
        return _basic_ic_analysis(factor_df, returns_df, request)

    try:
        # Prepare alphalens data format
        factor_data = _prepare_alphalens_data(factor_df, returns_df, request)

        if factor_data is None or factor_data.empty:
            logger.warning("Could not prepare alphalens data format")
            return _basic_ic_analysis(factor_df, returns_df, request)

        # Calculate IC and IR by period
        ic_data = al.performance.factor_information_coefficient(factor_data)

        results['information_coefficient'] = {
            f"ic_period_{period}": {
                "mean": float(ic_data.loc[period].mean()),
                "std": float(ic_data.loc[period].std()),
                "ir": float(ic_data.loc[period].mean() / ic_data.loc[period].std()) if ic_data.loc[period].std() > 0 else 0.0
            }
            for period in ic_data.index
        }

        # Information ratio
        results['information_ratio'] = {
            f"ir_period_{period}": float(ic_data.loc[period].mean() / ic_data.loc[period].std()) if ic_data.loc[period].std() > 0 else 0.0
            for period in ic_data.index
        }

        # Quantile analysis
        quantile_rets = al.performance.mean_return_by_quantile(factor_data, by_date=False)

        results['quantile_analysis'] = {
            f"period_{period}": {
                f"quantile_{i+1}": float(quantile_rets.loc[period, i+1])
                for i in range(request.quantiles)
            }
            for period in quantile_rets.index
        }

        # Alpha decay analysis
        results['alpha_metrics'] = _calculate_alpha_decay(factor_data)

        logger.info("Alpha predictiveness analysis completed using alphalens")

    except Exception as e:
        logger.warning(f"Alphalens analysis failed: {e}, falling back to basic IC")
        return _basic_ic_analysis(factor_df, returns_df, request)

    return results


def _basic_ic_analysis(factor_df: pd.DataFrame,
                      returns_df: pd.DataFrame,
                      request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Basic information coefficient analysis without alphalens."""
    results = {}

    try:
        # Merge factor and returns data
        merged_df = pd.merge(factor_df, returns_df, on=['date', 'symbol'], how='inner')

        if merged_df.empty:
            return {'error': 'No overlapping data between factors and returns'}

        # Calculate IC for each period
        ic_results = {}
        ir_results = {}

        for period in request.periods:
            # Calculate forward returns
            merged_df[f'fwd_ret_{period}'] = merged_df.groupby('symbol')['return'].shift(-period)

            # Calculate IC (Spearman correlation)
            ic_by_date = merged_df.groupby('date').apply(
                lambda x: x['factor'].corr(x[f'fwd_ret_{period}'], method='spearman')
            ).dropna()

            if len(ic_by_date) > 0:
                ic_results[f"ic_period_{period}"] = {
                    "mean": float(ic_by_date.mean()),
                    "std": float(ic_by_date.std()),
                    "ir": float(ic_by_date.mean() / ic_by_date.std()) if ic_by_date.std() > 0 else 0.0,
                    "count": int(len(ic_by_date))
                }

                # Information Ratio
                ir_results[f"ir_period_{period}"] = float(ic_by_date.mean() / ic_by_date.std()) if ic_by_date.std() > 0 else 0.0

        results['information_coefficient'] = ic_results
        results['information_ratio'] = ir_results

        # Basic quantile analysis
        quantile_results = {}
        for period in request.periods:
            fwd_ret_col = f'fwd_ret_{period}'
            if fwd_ret_col in merged_df.columns:
                merged_df['factor_quantile'] = pd.qcut(merged_df['factor'], request.quantiles, labels=False) + 1
                quantile_means = merged_df.groupby('factor_quantile')[fwd_ret_col].mean()

                quantile_results[f"period_{period}"] = {
                    f"quantile_{i+1}": float(quantile_means.iloc[i]) if i < len(quantile_means) else 0.0
                    for i in range(request.quantiles)
                }

        results['quantile_analysis'] = quantile_results

        logger.info("Basic IC analysis completed")

    except Exception as e:
        logger.error(f"Error in basic IC analysis: {e}")
        results['error'] = str(e)

    return results


def _analyze_risk_attribution(factor_df: pd.DataFrame,
                             returns_df: pd.DataFrame,
                             request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Analyze risk factor attribution using statistical factor models."""
    results = {}

    try:
        # Load or create factor model
        if request.factor_model == FactorModel.CUSTOM and request.custom_factors:
            factor_returns = pd.DataFrame(request.custom_factors)
        else:
            factor_returns = _load_factor_model_data(request.factor_model, returns_df)

        if factor_returns is None or factor_returns.empty:
            logger.warning("Could not load factor model data")
            return {'error': 'Could not load factor model data'}

        # Prepare return series
        returns_series = _prepare_returns_series(returns_df)

        if returns_series is None or returns_series.empty:
            return {'error': 'Could not prepare returns series'}

        # Align dates
        common_dates = returns_series.index.intersection(factor_returns.index)
        if len(common_dates) < 20:  # Minimum observations
            return {'error': 'Insufficient overlapping observations for factor model'}

        returns_aligned = returns_series.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]

        # Run factor regression
        factor_results = _run_factor_regression(returns_aligned, factors_aligned, request)

        # Calculate factor contributions
        factor_contributions = _calculate_factor_contributions(returns_aligned, factors_aligned, factor_results)

        # Risk decomposition
        risk_decomp = _calculate_risk_decomposition(returns_aligned, factors_aligned, factor_results)

        # Consolidate all risk attribution results
        results['risk_attribution'] = {
            'factor_regression': factor_results,
            'factor_contributions': factor_contributions,
            'risk_decomposition': risk_decomp
        }

        # Also provide at top level for backward compatibility
        results['factor_loadings'] = factor_results.get('factor_loadings', {})
        results['factor_returns'] = factor_contributions

        logger.info("Risk attribution analysis completed")

    except Exception as e:
        logger.error(f"Error in risk attribution: {e}")
        results['error'] = str(e)

    return results


def _perform_statistical_tests(factor_df: pd.DataFrame,
                              returns_df: pd.DataFrame,
                              request: FactorAnalysisRequest) -> Dict[str, Dict[str, Any]]:
    """Perform various statistical tests for robustness."""
    test_results = {}

    if not HAS_STATSMODELS or not HAS_SCIPY:
        logger.warning("Statistical libraries not available - skipping advanced tests")
        return test_results

    try:
        # Merge data for testing
        merged_df = pd.merge(factor_df, returns_df, on=['date', 'symbol'], how='inner')

        if merged_df.empty:
            return {'error': 'No data for statistical testing'}

        # Calculate forward returns for different periods
        for period in request.periods:
            merged_df[f'fwd_ret_{period}'] = merged_df.groupby('symbol')['return'].shift(-period)

        # T-test for mean IC
        if StatisticalTest.T_TEST in request.statistical_tests:
            test_results['t_test'] = _perform_t_tests(merged_df, request)

        # Newey-West HAC standard errors
        if StatisticalTest.NEWEY_WEST in request.statistical_tests:
            test_results['newey_west'] = _perform_newey_west_tests(merged_df, request)

        # Additional robustness tests
        if StatisticalTest.HAC_ROBUST in request.statistical_tests:
            test_results['hac_robust'] = _perform_hac_tests(merged_df, request)

        logger.info("Statistical tests completed")

    except Exception as e:
        logger.error(f"Error in statistical tests: {e}")
        test_results['error'] = str(e)

    return test_results


def _bootstrap_analysis(factor_df: pd.DataFrame,
                       returns_df: pd.DataFrame,
                       request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Perform bootstrap analysis for stability testing."""
    if not HAS_STATSMODELS:
        return {'error': 'Arch library not available for bootstrap'}

    try:
        # Merge data
        merged_df = pd.merge(factor_df, returns_df, on=['date', 'symbol'], how='inner')

        if merged_df.empty:
            return {'error': 'No data for bootstrap analysis'}

        # Calculate IC time series
        ic_series = []
        for period in request.periods:
            merged_df[f'fwd_ret_{period}'] = merged_df.groupby('symbol')['return'].shift(-period)

            daily_ic = merged_df.groupby('date').apply(
                lambda x: x['factor'].corr(x[f'fwd_ret_{period}'], method='spearman')
            ).dropna()

            if len(daily_ic) > 0:
                ic_series.append(daily_ic.values)

        if not ic_series:
            return {'error': 'Could not calculate IC series for bootstrap'}

        # Perform block bootstrap
        results = {}
        for i, period in enumerate(request.periods):
            if i < len(ic_series):
                ic_data = ic_series[i]

                # Determine block size
                block_size = request.block_size or max(1, int(np.sqrt(len(ic_data))))

                # Block bootstrap
                bootstrap = CircularBlockBootstrap(block_size, ic_data)
                bootstrap_means = []

                for bootstrap_sample in bootstrap.bootstrap(request.bootstrap_samples):
                    bootstrap_means.append(np.mean(bootstrap_sample[0][0]))

                bootstrap_means = np.array(bootstrap_means)

                results[f'period_{period}'] = {
                    'mean': float(np.mean(bootstrap_means)),
                    'std': float(np.std(bootstrap_means)),
                    'confidence_interval': [
                        float(np.percentile(bootstrap_means, (1 - request.confidence_level) * 50)),
                        float(np.percentile(bootstrap_means, (1 + request.confidence_level) * 50))
                    ],
                    'samples': int(request.bootstrap_samples)
                }

        logger.info("Bootstrap analysis completed")
        return results

    except Exception as e:
        logger.error(f"Error in bootstrap analysis: {e}")
        return {'error': str(e)}


def _jackknife_analysis(factor_df: pd.DataFrame,
                       returns_df: pd.DataFrame,
                       request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Perform jackknife analysis for stability testing."""
    try:
        # Merge data
        merged_df = pd.merge(factor_df, returns_df, on=['date', 'symbol'], how='inner')

        if merged_df.empty:
            return {'error': 'No data for jackknife analysis'}

        unique_dates = merged_df['date'].unique()

        if len(unique_dates) < 10:  # Need enough dates for meaningful jackknife
            return {'error': 'Insufficient dates for jackknife analysis'}

        results = {}

        for period in request.periods:
            merged_df[f'fwd_ret_{period}'] = merged_df.groupby('symbol')['return'].shift(-period)

            # Calculate IC for each jackknife sample (leave-one-date-out)
            jackknife_ics = []

            for date_to_exclude in unique_dates:
                jackknife_data = merged_df[merged_df['date'] != date_to_exclude]

                if not jackknife_data.empty:
                    daily_ic = jackknife_data.groupby('date').apply(
                        lambda x: x['factor'].corr(x[f'fwd_ret_{period}'], method='spearman')
                    ).dropna()

                    if len(daily_ic) > 0:
                        jackknife_ics.append(daily_ic.mean())

            if jackknife_ics:
                jackknife_ics = np.array(jackknife_ics)

                results[f'period_{period}'] = {
                    'mean': float(np.mean(jackknife_ics)),
                    'std': float(np.std(jackknife_ics)),
                    'bias': float(np.mean(jackknife_ics) - merged_df.groupby('date').apply(
                        lambda x: x['factor'].corr(x[f'fwd_ret_{period}'], method='spearman')
                    ).dropna().mean()),
                    'samples': len(jackknife_ics)
                }

        logger.info("Jackknife analysis completed")
        return results

    except Exception as e:
        logger.error(f"Error in jackknife analysis: {e}")
        return {'error': str(e)}


# =============================================================================
# Helper Functions
# =============================================================================

def _prepare_alphalens_data(factor_df: pd.DataFrame,
                           returns_df: pd.DataFrame,
                           request: FactorAnalysisRequest) -> Optional[pd.DataFrame]:
    """Prepare data in alphalens format."""
    try:
        # Handle price data
        if request.price_data:
            price_df = pd.DataFrame(request.price_data)
            price_df['date'] = pd.to_datetime(price_df['date'])
            prices = price_df.set_index('date').pivot(columns='symbol', values='price')
        else:
            # Create synthetic prices from returns
            returns_pivot = returns_df.set_index('date').pivot(columns='symbol', values='return')
            prices = (1 + returns_pivot).cumprod().fillna(method='ffill')

        # Prepare factor data
        factor_pivot = factor_df.set_index('date').pivot(columns='symbol', values='factor')

        # Create alphalens factor data
        factor_data = al.utils.get_clean_factor_and_forward_returns(
            factor_pivot.stack(),
            prices,
            quantiles=request.quantiles,
            periods=request.periods,
            max_loss=0.35  # Allow up to 35% data loss
        )

        return factor_data

    except Exception as e:
        logger.warning(f"Could not prepare alphalens data: {e}")
        return None


def _calculate_alpha_decay(factor_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate alpha decay metrics."""
    try:
        if not HAS_ALPHALENS:
            return {}

        # Alpha decay analysis
        ic_decay = al.performance.factor_information_coefficient(factor_data)

        decay_metrics = {}
        for period in ic_decay.index:
            decay_metrics[f'period_{period}'] = {
                'ic_mean': float(ic_decay.loc[period].mean()),
                'ic_std': float(ic_decay.loc[period].std()),
                'ic_ir': float(ic_decay.loc[period].mean() / ic_decay.loc[period].std()) if ic_decay.loc[period].std() > 0 else 0.0
            }

        return {'alpha_decay': decay_metrics}

    except Exception as e:
        logger.warning(f"Could not calculate alpha decay: {e}")
        return {}


def _load_factor_model_data(factor_model: FactorModel, returns_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Load factor model data (Fama-French, etc.)."""
    try:
        if not HAS_DATAREADER:
            logger.warning("pandas-datareader not available - using synthetic factor data")
            return _create_synthetic_factors(returns_df)

        # Determine date range from returns data
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        start_date = returns_df['date'].min()
        end_date = returns_df['date'].max()

        if factor_model == FactorModel.FAMA_FRENCH_3:
            # Try to load Fama-French 3-factor model
            try:
                ff3 = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_date, end_date)[0]
                ff3.index = pd.to_datetime(ff3.index)
                ff3 = ff3 / 100  # Convert from percentages
                return ff3[['Mkt-RF', 'SMB', 'HML']]
            except:
                logger.warning("Could not load Fama-French data - using synthetic factors")
                return _create_synthetic_factors(returns_df)

        elif factor_model == FactorModel.FAMA_FRENCH_5:
            try:
                ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start_date, end_date)[0]
                ff5.index = pd.to_datetime(ff5.index)
                ff5 = ff5 / 100
                return ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
            except:
                logger.warning("Could not load Fama-French 5-factor data - using synthetic factors")
                return _create_synthetic_factors(returns_df)

        elif factor_model == FactorModel.CAPM:
            try:
                market = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_date, end_date)[0]
                market.index = pd.to_datetime(market.index)
                market = market / 100
                return market[['Mkt-RF']]
            except:
                logger.warning("Could not load market factor data - using synthetic factors")
                return _create_synthetic_factors(returns_df)

        return _create_synthetic_factors(returns_df)

    except Exception as e:
        logger.warning(f"Error loading factor model data: {e}")
        return _create_synthetic_factors(returns_df)


def _create_synthetic_factors(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic factor data for testing."""
    try:
        dates = pd.to_datetime(returns_df['date'].unique())
        dates = pd.DatetimeIndex(dates).sort_values()

        # Create synthetic factor returns
        np.random.seed(42)  # For reproducibility
        n_days = len(dates)

        synthetic_factors = pd.DataFrame(index=dates)
        synthetic_factors['Mkt-RF'] = np.random.normal(0.0005, 0.015, n_days)  # Market excess return
        synthetic_factors['SMB'] = np.random.normal(0.0001, 0.008, n_days)     # Size factor
        synthetic_factors['HML'] = np.random.normal(0.0002, 0.008, n_days)     # Value factor

        # Add some realistic autocorrelation
        for col in synthetic_factors.columns:
            for i in range(1, len(synthetic_factors)):
                synthetic_factors.iloc[i][col] += 0.1 * synthetic_factors.iloc[i-1][col]

        logger.info("Created synthetic factor data")
        return synthetic_factors

    except Exception as e:
        logger.error(f"Error creating synthetic factors: {e}")
        return pd.DataFrame()


def _prepare_returns_series(returns_df: pd.DataFrame) -> Optional[pd.Series]:
    """Prepare returns series for factor analysis."""
    try:
        returns_df['date'] = pd.to_datetime(returns_df['date'])

        # If multiple symbols, use equal-weighted portfolio
        if 'symbol' in returns_df.columns:
            portfolio_returns = returns_df.groupby('date')['return'].mean()
        else:
            portfolio_returns = returns_df.set_index('date')['return']

        return portfolio_returns.sort_index()

    except Exception as e:
        logger.error(f"Error preparing returns series: {e}")
        return None


def _run_factor_regression(returns: pd.Series,
                          factors: pd.DataFrame,
                          request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Run factor model regression."""
    try:
        # Align data
        common_index = returns.index.intersection(factors.index)
        y = returns.loc[common_index]
        X = factors.loc[common_index]

        # Add constant for alpha
        X = sm.add_constant(X)

        # Run regression
        model = sm.OLS(y, X).fit()

        # Extract results
        results = {
            'factor_loadings': {
                factor: float(model.params[factor])
                for factor in model.params.index if factor != 'const'
            },
            'alpha': float(model.params.get('const', 0)),
            'alpha_pvalue': float(model.pvalues.get('const', 1)),
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue)
        }

        # Factor significance
        factor_significance = {}
        for factor in factors.columns:
            if factor in model.pvalues.index:
                factor_significance[factor] = {
                    'coefficient': float(model.params[factor]),
                    'pvalue': float(model.pvalues[factor]),
                    'tstat': float(model.tvalues[factor])
                }

        results['factor_significance'] = factor_significance

        return results

    except Exception as e:
        logger.error(f"Error in factor regression: {e}")
        return {'error': str(e)}


def _calculate_factor_contributions(returns: pd.Series,
                                  factors: pd.DataFrame,
                                  regression_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate factor contributions to returns."""
    try:
        contributions = {}
        factor_loadings = regression_results.get('factor_loadings', {})

        for factor_name, loading in factor_loadings.items():
            if factor_name in factors.columns:
                factor_contribution = loading * factors[factor_name].mean() * 252  # Annualized
                contributions[factor_name] = float(factor_contribution)

        # Alpha contribution
        alpha = regression_results.get('alpha', 0)
        contributions['alpha'] = float(alpha * 252)  # Annualized

        return contributions

    except Exception as e:
        logger.error(f"Error calculating factor contributions: {e}")
        return {}


def _calculate_risk_decomposition(returns: pd.Series,
                                factors: pd.DataFrame,
                                regression_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate risk decomposition by factors."""
    try:
        factor_loadings = regression_results.get('factor_loadings', {})

        # Calculate factor covariance matrix
        factor_cov = factors.cov() * 252  # Annualized

        # Portfolio risk from factors
        risk_decomp = {}
        total_factor_risk = 0

        for factor_name, loading in factor_loadings.items():
            if factor_name in factor_cov.index:
                factor_risk = (loading ** 2) * factor_cov.loc[factor_name, factor_name]
                risk_decomp[f'{factor_name}_risk'] = float(factor_risk)
                total_factor_risk += factor_risk

        # Idiosyncratic risk
        total_risk = returns.var() * 252  # Annualized
        idiosyncratic_risk = max(0, total_risk - total_factor_risk)
        risk_decomp['idiosyncratic_risk'] = float(idiosyncratic_risk)
        risk_decomp['total_risk'] = float(total_risk)

        return risk_decomp

    except Exception as e:
        logger.error(f"Error in risk decomposition: {e}")
        return {}


def _perform_t_tests(merged_df: pd.DataFrame, request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Perform t-tests for IC significance."""
    results = {}

    try:
        for period in request.periods:
            fwd_ret_col = f'fwd_ret_{period}'
            if fwd_ret_col in merged_df.columns:
                # Calculate daily IC
                daily_ic = merged_df.groupby('date').apply(
                    lambda x: x['factor'].corr(x[fwd_ret_col], method='spearman')
                ).dropna()

                if len(daily_ic) > 1:
                    # T-test for non-zero mean
                    t_stat, p_value = stats.ttest_1samp(daily_ic, 0)

                    results[f'period_{period}'] = {
                        'ic_mean': float(daily_ic.mean()),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < (1 - request.confidence_level)),
                        'observations': int(len(daily_ic))
                    }

    except Exception as e:
        logger.error(f"Error in t-tests: {e}")
        results['error'] = str(e)

    return results


def _perform_newey_west_tests(merged_df: pd.DataFrame, request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Perform Newey-West HAC tests."""
    results = {}

    if not HAS_STATSMODELS:
        return {'error': 'Statsmodels not available'}

    try:
        from statsmodels.tsa.stattools import acf
        from statsmodels.stats.sandwich_covariance import cov_hac

        for period in request.periods:
            fwd_ret_col = f'fwd_ret_{period}'
            if fwd_ret_col in merged_df.columns:
                # Calculate daily IC series
                daily_ic = merged_df.groupby('date').apply(
                    lambda x: x['factor'].corr(x[fwd_ret_col], method='spearman')
                ).dropna()

                if len(daily_ic) > 10:  # Need sufficient observations
                    # Determine optimal lag length using autocorrelation
                    acf_result = acf(daily_ic, nlags=min(20, len(daily_ic)//4), fft=False)
                    optimal_lags = max(1, int(4 * (len(daily_ic)/100)**(2/9)))  # Newey-West rule

                    # Create design matrix for HAC
                    X = np.ones((len(daily_ic), 1))  # Just constant term

                    # Regular OLS
                    beta = np.linalg.lstsq(X, daily_ic.values, rcond=None)[0]
                    residuals = daily_ic.values - X @ beta

                    # HAC standard errors
                    try:
                        hac_cov = cov_hac(X, residuals, nlags=optimal_lags)
                        hac_se = np.sqrt(np.diag(hac_cov))

                        # HAC t-statistic
                        hac_t_stat = beta[0] / hac_se[0] if hac_se[0] > 0 else 0
                        hac_p_value = 2 * (1 - stats.t.cdf(abs(hac_t_stat), len(daily_ic) - 1))

                        results[f'period_{period}'] = {
                            'ic_mean': float(daily_ic.mean()),
                            'hac_t_statistic': float(hac_t_stat),
                            'hac_p_value': float(hac_p_value),
                            'hac_significant': bool(hac_p_value < (1 - request.confidence_level)),
                            'optimal_lags': int(optimal_lags),
                            'autocorrelation': float(acf_result[1]) if len(acf_result) > 1 else 0.0
                        }

                    except Exception as hac_error:
                        logger.warning(f"HAC calculation failed for period {period}: {hac_error}")
                        results[f'period_{period}'] = {'error': str(hac_error)}

    except Exception as e:
        logger.error(f"Error in Newey-West tests: {e}")
        results['error'] = str(e)

    return results


def _perform_hac_tests(merged_df: pd.DataFrame, request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Perform additional HAC robustness tests."""
    results = {}

    try:
        # Runs test for randomness
        for period in request.periods:
            fwd_ret_col = f'fwd_ret_{period}'
            if fwd_ret_col in merged_df.columns:
                daily_ic = merged_df.groupby('date').apply(
                    lambda x: x['factor'].corr(x[fwd_ret_col], method='spearman')
                ).dropna()

                if len(daily_ic) > 10:
                    # Convert to binary sequence (above/below median)
                    binary_sequence = (daily_ic > daily_ic.median()).astype(int)

                    # Runs test
                    if HAS_STATSMODELS:
                        try:
                            runs_stat, runs_pvalue = runstest_1samp(binary_sequence)
                            results[f'period_{period}_runs'] = {
                                'runs_statistic': float(runs_stat),
                                'runs_p_value': float(runs_pvalue),
                                'is_random': bool(runs_pvalue > 0.05)
                            }
                        except:
                            pass

                    # Ljung-Box test for autocorrelation
                    if HAS_SCIPY:
                        from statsmodels.stats.diagnostic import acorr_ljungbox
                        try:
                            ljung_box = acorr_ljungbox(daily_ic, lags=min(10, len(daily_ic)//5), return_df=True)
                            results[f'period_{period}_ljung_box'] = {
                                'lb_statistic': float(ljung_box['lb_stat'].iloc[-1]),
                                'lb_p_value': float(ljung_box['lb_pvalue'].iloc[-1]),
                                'no_autocorr': bool(ljung_box['lb_pvalue'].iloc[-1] > 0.05)
                            }
                        except:
                            pass

    except Exception as e:
        logger.error(f"Error in HAC tests: {e}")
        results['error'] = str(e)

    return results


def _calculate_model_diagnostics(factor_df: pd.DataFrame,
                               returns_df: pd.DataFrame,
                               request: FactorAnalysisRequest) -> Dict[str, Any]:
    """Calculate model diagnostics and goodness-of-fit measures."""
    diagnostics = {}

    try:
        # Basic data quality metrics
        diagnostics['data_quality'] = {
            'factor_observations': len(factor_df),
            'returns_observations': len(returns_df),
            'factor_missing_rate': float(factor_df['factor'].isna().mean()) if 'factor' in factor_df.columns else 0.0,
            'returns_missing_rate': float(returns_df['return'].isna().mean()) if 'return' in returns_df.columns else 0.0
        }

        # Merge for analysis
        merged_df = pd.merge(factor_df, returns_df, on=['date', 'symbol'], how='inner')

        if not merged_df.empty:
            # Factor distribution diagnostics
            factor_stats = merged_df['factor'].describe()
            diagnostics['factor_distribution'] = {
                'mean': float(factor_stats['mean']),
                'std': float(factor_stats['std']),
                'skewness': float(stats.skew(merged_df['factor'].dropna())),
                'kurtosis': float(stats.kurtosis(merged_df['factor'].dropna())),
                'normality_test_p': float(stats.jarque_bera(merged_df['factor'].dropna())[1])
            }

            # Cross-sectional analysis
            unique_dates = merged_df['date'].nunique()
            avg_cross_section = len(merged_df) / unique_dates if unique_dates > 0 else 0

            diagnostics['cross_sectional'] = {
                'unique_dates': int(unique_dates),
                'avg_cross_section_size': float(avg_cross_section),
                'min_cross_section': int(merged_df.groupby('date').size().min()),
                'max_cross_section': int(merged_df.groupby('date').size().max())
            }

            # Factor stability over time
            monthly_ic = []
            for month in pd.to_datetime(merged_df['date']).dt.to_period('M').unique():
                month_data = merged_df[pd.to_datetime(merged_df['date']).dt.to_period('M') == month]
                if len(month_data) > 5:  # Minimum observations per month
                    ic = month_data['factor'].corr(month_data['return'], method='spearman')
                    if not pd.isna(ic):
                        monthly_ic.append(ic)

            if monthly_ic:
                diagnostics['stability'] = {
                    'monthly_ic_mean': float(np.mean(monthly_ic)),
                    'monthly_ic_std': float(np.std(monthly_ic)),
                    'monthly_ic_min': float(np.min(monthly_ic)),
                    'monthly_ic_max': float(np.max(monthly_ic)),
                    'months_analyzed': len(monthly_ic)
                }

    except Exception as e:
        logger.error(f"Error calculating model diagnostics: {e}")
        diagnostics['error'] = str(e)

    return diagnostics


def _generate_factor_artifacts(factor_df: pd.DataFrame,
                             returns_df: pd.DataFrame,
                             request: FactorAnalysisRequest,
                             response: FactorAnalysisResponse) -> Dict[str, str]:
    """Generate visualization artifacts for factor analysis."""
    artifacts = {}

    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available - skipping chart generation")
        return artifacts

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # IC time series plot
        if response.information_coefficient:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot IC by period
            for period_key, ic_data in response.information_coefficient.items():
                if isinstance(ic_data, dict) and 'mean' in ic_data:
                    period = period_key.replace('ic_period_', '')
                    ax.axhline(y=ic_data['mean'], label=f'Period {period} (IC={ic_data["mean"]:.3f})')

            ax.set_title('Information Coefficient by Period')
            ax.set_ylabel('IC')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ic_path = os.path.join(request.output_dir, f"ic_analysis_{timestamp}.png")
            fig.savefig(ic_path, dpi=180, bbox_inches="tight")
            artifacts['ic_analysis'] = ic_path
            plt.close(fig)

        # Quantile analysis heatmap
        if response.quantile_analysis:
            periods = list(response.quantile_analysis.keys())
            if periods:
                period_data = response.quantile_analysis[periods[0]]
                quantiles = [int(k.replace('quantile_', '')) for k in period_data.keys()]

                # Create heatmap data
                heatmap_data = []
                for period_key in periods:
                    period_returns = []
                    period_data = response.quantile_analysis[period_key]
                    for q in sorted(quantiles):
                        period_returns.append(period_data.get(f'quantile_{q}', 0))
                    heatmap_data.append(period_returns)

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

                # Labels
                ax.set_xticks(range(len(quantiles)))
                ax.set_xticklabels([f'Q{q}' for q in sorted(quantiles)])
                ax.set_yticks(range(len(periods)))
                ax.set_yticklabels([p.replace('period_', 'Period ') for p in periods])

                # Add colorbar
                plt.colorbar(im, ax=ax, label='Forward Return')
                ax.set_title('Factor Quantile Returns Heatmap')

                quantile_path = os.path.join(request.output_dir, f"quantile_analysis_{timestamp}.png")
                fig.savefig(quantile_path, dpi=180, bbox_inches="tight")
                artifacts['quantile_analysis'] = quantile_path
                plt.close(fig)

        # Statistical test results
        if response.statistical_tests:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0
            for test_name, test_results in response.statistical_tests.items():
                if plot_idx < 4 and isinstance(test_results, dict):
                    ax = axes[plot_idx]

                    # Extract p-values for plotting
                    periods = []
                    p_values = []

                    for key, value in test_results.items():
                        if isinstance(value, dict) and 'p_value' in value:
                            periods.append(key.replace('period_', ''))
                            p_values.append(value['p_value'])

                    if periods and p_values:
                        bars = ax.bar(periods, p_values)
                        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% threshold')
                        ax.set_title(f'{test_name.title()} P-Values')
                        ax.set_ylabel('P-Value')
                        ax.set_xlabel('Period')
                        ax.legend()

                        # Color bars based on significance
                        for bar, p_val in zip(bars, p_values):
                            bar.set_color('green' if p_val < 0.05 else 'gray')

                    plot_idx += 1

            # Hide unused subplots
            for i in range(plot_idx, 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            stats_path = os.path.join(request.output_dir, f"statistical_tests_{timestamp}.png")
            fig.savefig(stats_path, dpi=180, bbox_inches="tight")
            artifacts['statistical_tests'] = stats_path
            plt.close(fig)

        logger.info(f"Generated {len(artifacts)} visualization artifacts")

    except Exception as e:
        logger.error(f"Error generating artifacts: {e}")

    return artifacts


# =============================================================================
# Plugin Wrapper for Tool Registry Integration
# =============================================================================

@dataclass
class FactorAnalysisPlugin:
    """Plugin wrapper for factor analysis tool."""

    name: str = "analyze_factor"
    description: str = "Comprehensive factor and alpha analysis with statistical robustness testing"
    semantic_key: str = "factor_analysis"
    rate_limit_cps: float = 0.2
    timeout_sec: float = 300.0
    max_retries: int = 2

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_data": {
                        "type": "array",
                        "description": "Factor exposures or alpha signals with date, symbol, factor columns"
                    },
                    "returns_data": {
                        "type": "array",
                        "description": "Asset returns data with date, symbol, return columns"
                    },
                    "price_data": {
                        "type": "array",
                        "description": "Price data for quantile analysis (optional)"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["alpha_predictiveness", "risk_attribution", "combined_analysis"],
                        "default": "combined_analysis"
                    },
                    "statistical_tests": {
                        "type": "array",
                        "items": {"enum": ["t_test", "newey_west", "hac_robust", "bootstrap", "jackknife"]},
                        "default": ["t_test", "newey_west"]
                    },
                    "confidence_level": {
                        "type": "number",
                        "default": 0.95
                    },
                    "bootstrap_samples": {
                        "type": "integer",
                        "default": 1000
                    },
                    "factor_model": {
                        "type": "string",
                        "enum": ["capm", "fama_french_3", "fama_french_5", "custom"],
                        "default": "fama_french_3"
                    },
                    "quantiles": {
                        "type": "integer",
                        "default": 5
                    },
                    "periods": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": [1, 5, 10]
                    },
                    "save_charts": {
                        "type": "boolean",
                        "default": True
                    },
                    "output_dir": {
                        "type": "string",
                        "default": "artifacts"
                    }
                },
                "required": ["factor_data", "returns_data"]
            }
        }

    def validate(self, params: Dict[str, Any]):
        try:
            request = FactorAnalysisRequest(**params)
            return True, request.model_dump(), []
        except Exception as e:
            return False, params, [str(e)]

    def execute(self, params: Dict[str, Any]):
        request = FactorAnalysisRequest(**params)
        result = analyze_factor(request)
        return result.model_dump()


__all__ = [
    # Request/Response models
    "FactorAnalysisRequest",
    "FactorAnalysisResponse",

    # Core functions
    "analyze_factor",

    # Plugin classes
    "FactorAnalysisPlugin",

    # Enums
    "FactorAnalysisType",
    "StatisticalTest",
    "FactorModel"
]