import streamlit as st
import pandas as pd
import altair as alt  # Ensure Altair is imported
from datetime import timedelta

# Import functions from our modules.
from optimizer import run_optimizers  
from utils import dynamic_backtest_portfolio  
from user_input import get_backtest_settings, get_asset_selection, get_optimization_methods
from plots import (
    plot_cumulative_returns, 
    plot_rolling_sharpe, 
    plot_drawdowns, 
    plot_allocations_per_method,
    plot_asset_returns, 
    plot_asset_prices,
    pie_chart_allocation
)

st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide")
st.title("🔀 Crypto Portfolio Optimizer (Dynamic HRB, MVO, EW)")

# Load merged price data from Parquet.
@st.cache_data
def load_data():
    return pd.read_parquet("Data/prices.parquet")

data = load_data()
available_dates = data.index.sort_values()

# Get user inputs from the sidebar.
start_date, end_date, lookback_days, rebalance_days, nonnegative_toggle = get_backtest_settings(available_dates)
selected_coins = get_asset_selection(data)
data = data[selected_coins]

# Define simulation data based on the selected dates.
simulation_data = data.loc[start_date:end_date]
if simulation_data.empty:
    st.error("No data available for the selected backtest period.")
    st.stop()

# ----- Underlying Asset Plots: Plot Assets Button -----
st.markdown("## Underlying Asset Data")
plot_assets_button = st.button("Plot Assets")
price_scale_option = st.sidebar.radio("Select Price Scale", ("Linear", "Log"))

if plot_assets_button:
    st.markdown("### Daily Returns by Asset")
    st.altair_chart(plot_asset_returns(simulation_data, selected_coins), use_container_width=True)
    
    st.markdown("### Asset Prices")
    st.altair_chart(plot_asset_prices(simulation_data, selected_coins, log_scale=(price_scale_option=="Log")), use_container_width=True)
    
# ----- Dynamic Backtesting -----
st.markdown("## Dynamic Backtest Results")
optimize_button = st.button("Optimize Portfolio")

if optimize_button:
    try:
        # Compute initial allocations using the lookback window before the start date.
        lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
        initial_allocations = run_optimizers(lookback_window, nonnegative_mvo=nonnegative_toggle)

        # Let the user select which optimization methods to include.
        selected_methods = get_optimization_methods(initial_allocations)

        st.markdown("### Initial Allocations (Pie Charts)")
        pie_charts = []
        for method in selected_methods:
            chart = pie_chart_allocation(initial_allocations[method].round(4), method)
            pie_charts.append(chart)
        st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

        st.write(f"Backtest period: {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}")
        st.write(f"Rebalance Frequency: Every {rebalance_days} days")
        st.write(f"Dynamic reoptimization uses the past {lookback_days} days of data with exponential weighting.")

        # Run dynamic backtest for each selected optimization method.
        results_dict = {}
        for method in selected_methods:
            res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative_toggle)
            results_dict[method] = res

        st.markdown("### Cumulative Returns (starting at 0)")
        st.altair_chart(plot_cumulative_returns(results_dict), use_container_width=True)

        st.markdown("### Rolling Annualized Sharpe Ratio")
        st.altair_chart(plot_rolling_sharpe(results_dict), use_container_width=True)

        st.markdown("### Rolling Maximum Drawdown")
        st.altair_chart(plot_drawdowns(results_dict), use_container_width=True)

        st.markdown("### Dynamic Asset Allocations Per Method")
        for method in selected_methods:
            st.altair_chart(plot_allocations_per_method(results_dict[method]["allocations"], method), use_container_width=True)

        st.markdown("### Summary Metrics by Method")
        for method, res in results_dict.items():
            st.subheader(method)
            st.write(f"**Final Annualized Sharpe Ratio:** {res['sharpe']:.2f}")
            st.write(f"**Maximum Drawdown:** {res['drawdown']:.2%}")

    except Exception as e:
        st.error("An error occurred during dynamic backtesting. Underlying asset plots are still displayed.")
        st.error(f"Error details: {e}")
else:
    st.info("Click the 'Optimize Portfolio' button to run the dynamic backtest and view optimization results.")
