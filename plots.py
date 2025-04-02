import altair as alt
import pandas as pd

def plot_allocations_per_method(allocations, method):
    # allocations: DataFrame with date as index and asset columns for one method.
    df_reset = allocations.reset_index()
    date_column = df_reset.columns[0]
    alloc_df = df_reset.melt(id_vars=date_column, var_name="Asset", value_name="Allocation")
    alloc_df = alloc_df.rename(columns={date_column: "date"})
    chart = alt.Chart(alloc_df).mark_line().encode(
        x="date:T",
        y=alt.Y("Allocation:Q", title="Allocation"),
        color="Asset:N",
        tooltip=["date:T", "Asset:N", alt.Tooltip("Allocation:Q", format=".2%")]
    ).properties(title=f"Asset Allocations Over Time ({method})", width=700, height=400)
    return chart

def pie_chart_allocation(initial_weights, method):
    # initial_weights: a pandas Series with asset weights.
    df = pd.DataFrame({'Asset': initial_weights.index, 'Weight': initial_weights.values})
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Weight", type="quantitative", stack=True),
        color=alt.Color(field="Asset", type="nominal"),
        tooltip=[alt.Tooltip("Asset:N"), alt.Tooltip("Weight:Q", format=".2%")]
    ).properties(
        title=f"Initial Allocation ({method})",
        width=200,
        height=200
    )
    return chart


def plot_cumulative_returns(results_dict):
    cumul_df_list = []
    for method, res in results_dict.items():
        temp = res["cumulative"].reset_index()
        temp.columns = ["date", "cumulative"]
        # Adjust cumulative returns to start at 0 (subtract 1)
        temp["cumulative"] = temp["cumulative"] - 1
        temp["Method"] = method
        cumul_df_list.append(temp)
    cumul_df = pd.concat(cumul_df_list)
    chart = alt.Chart(cumul_df).mark_line().encode(
        x="date:T",
        y=alt.Y("cumulative:Q", title="Cumulative Return (starting at 0)"),
        color="Method:N"
    ).properties(width=700, height=400, title="Cumulative Returns by Optimization Method")
    return chart

def plot_rolling_sharpe(results_dict):
    sharpe_df_list = []
    for method, res in results_dict.items():
        temp = res["rolling_sharpe"].reset_index()
        temp.columns = ["date", "rolling_sharpe"]
        temp["Method"] = method
        sharpe_df_list.append(temp)
    sharpe_df = pd.concat(sharpe_df_list)
    chart = alt.Chart(sharpe_df).mark_line().encode(
        x="date:T",
        y=alt.Y("rolling_sharpe:Q", title="Rolling Annualized Sharpe Ratio"),
        color="Method:N"
    ).properties(width=700, height=400, title="Rolling Annualized Sharpe Ratio")
    return chart

def plot_drawdowns(results_dict):
    drawdown_df_list = []
    for method, res in results_dict.items():
        temp = res["drawdowns"].reset_index()
        temp.columns = ["date", "drawdown"]
        temp["Method"] = method
        drawdown_df_list.append(temp)
    drawdown_df = pd.concat(drawdown_df_list)
    chart = alt.Chart(drawdown_df).mark_line().encode(
        x="date:T",
        y=alt.Y("drawdown:Q", title="Rolling Maximum Drawdown"),
        color="Method:N"
    ).properties(width=700, height=400, title="Rolling Maximum Drawdown")
    return chart

def plot_allocations(results_dict):
    alloc_df_list = []
    for method, res in results_dict.items():
        # Reset the index; this will add a column for the dates.
        df_reset = res["allocations"].reset_index()
        # Use the first column name as the date column.
        date_column = df_reset.columns[0]
        # Melt the DataFrame using that column as id_vars.
        alloc_df = df_reset.melt(id_vars=date_column, var_name="Asset", value_name="Allocation")
        # Rename the date column to "date" for consistency.
        alloc_df = alloc_df.rename(columns={date_column: "date"})
        alloc_df["Method"] = method
        alloc_df_list.append(alloc_df)
    alloc_df_all = pd.concat(alloc_df_list)
    alloc_df_all["Method_Asset"] = alloc_df_all["Method"] + " - " + alloc_df_all["Asset"]
    
    chart = alt.Chart(alloc_df_all).mark_line().encode(
        x="date:T",
        y=alt.Y("Allocation:Q", title="Asset Allocation (%)"),
        color=alt.Color("Method_Asset:N", title="Method - Asset")
    ).properties(width=700, height=400, title="Asset Allocation Over Time")
    return chart


def plot_asset_returns(simulation_data, selected_assets):
    # Filter data to selected assets and compute daily returns (%)
    data_filtered = simulation_data[selected_assets].copy()
    returns = data_filtered.pct_change() * 100
    returns = returns.reset_index().melt(id_vars="date", var_name="Asset", value_name="Daily Return (%)")
    chart = alt.Chart(returns).mark_line().encode(
         x="date:T",
         y=alt.Y("Daily Return (%):Q", title="Daily Return (%)"),
         color="Asset:N"
    ).properties(width=700, height=400, title="Daily Returns by Asset")
    return chart

def plot_asset_prices(simulation_data, selected_assets, log_scale=False):
    # Plot absolute prices for the selected assets.
    data_filtered = simulation_data[selected_assets].reset_index().melt(id_vars="date", var_name="Asset", value_name="Price")
    scale_type = "log" if log_scale else "linear"
    chart = alt.Chart(data_filtered).mark_line().encode(
         x="date:T",
         y=alt.Y("Price:Q", title="Price", scale=alt.Scale(type=scale_type)),
         color="Asset:N"
    ).properties(width=700, height=400, title="Asset Prices")
    return chart
