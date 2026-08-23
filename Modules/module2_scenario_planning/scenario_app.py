# Streamlit UI for Module 2: Scenario Planning (What-If)
from __future__ import annotations

# Import Libraries
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st


# =========================
# Helper Functions
# =========================

# Waterfall Chart
def _waterfall_chart(adjusted_sales: float, adjusted_costs: float, adjusted_interest: float, profit: float) -> go.Figure:
    """Waterfall showing how adjustments impact profit."""
    fig = go.Figure(
        go.Waterfall(
            name="Scenario",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Sales", "Costs", "Interest", "Profit"],
            #textposition="outside",
            text=[
                f"{adjusted_sales:,.2f}",
                f"{-adjusted_costs:,.2f}",
                f"{-adjusted_interest:,.2f}",
                f"{profit:,.2f}"
            ],
            y=[adjusted_sales, -adjusted_costs, -adjusted_interest, profit],
        )
    )
    # Plot Big Title
    st.subheader("Scenario Waterfall Analysis")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Tornado Sensitivity
def _tornado_sensitivity(base_sales: float, base_costs: float, base_interest: float, inflation: float, sales_change: float, interest_rate: float) -> go.Figure:
    """Compute simple +/-10% sensitivity on each driver and plot as tornado bars."""

    def profit_fn(infl, s_change, i_rate):
        # Calculate Scenario Profit based on adjusted values, costs, and interest rate.
        adj_sales = base_sales * (1 + s_change / 100)
        adj_costs = base_costs * (1 + infl / 100)
        adj_interest = base_interest * (1 + i_rate / 100)
        profit = adj_sales - adj_costs - adj_interest
        return profit

    drivers = [
        ("Inflation", inflation, lambda v: profit_fn(v, sales_change, interest_rate)),
        ("Sales Change", sales_change, lambda v: profit_fn(inflation, v, interest_rate)),
        ("Interest Rate", interest_rate, lambda v: profit_fn(inflation, sales_change, v)),
    ]

    base_profit = profit_fn(inflation, sales_change, interest_rate)

    names, lows, highs = [], [], []
    for name, cur, fn in drivers:
        p_low = fn(cur - 10)
        p_high = fn(cur + 10)
        names.append(name)
        lows.append(p_low - base_profit)
        highs.append(p_high - base_profit)


    fig = go.Figure()
    fig.add_bar(y=names, x=lows, orientation="h", name="-10%")
    fig.add_bar(y=names, x=highs, orientation="h", name="+10%")

    # Plot Big Title
    #st.subheader("Tornado Sensitivity Analysis (±10%)")
    fig.update_layout(
        title="Tornado Sensitivity (profit delta vs. base)",
        barmode="overlay",
        xaxis_title="Δ Profit",
        margin=dict(l=80, r=20, t=40, b=30)
    )
    return fig


# Monte Carlo Simulation
"""def _monte_carlo(base_sales: float, base_costs: float, base_interest: float, inflation: float, sales_change: float, interest_rate: float, runs: int = 1000, seed: int | None = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    infl_samp = rng.normal(loc=inflation, scale=2.0, size=runs)
    sales_samp = rng.normal(loc=sales_change, scale=5.0, size=runs)
    rate_samp = rng.normal(loc=interest_rate, scale=1.0, size=runs)
    
    sales = base_sales * (1 + sales_samp / 100)
    costs = base_costs * (1 + infl_samp / 100)
    interest = base_interest * (1 + rate_samp / 100)
    profit = sales - costs - interest
    
    return pd.DataFrame({
        "inflation": infl_samp,
        "sales_change": sales_samp,
        "interest_rate": rate_samp,
        "sales": sales,
        "costs": costs,
        "interest": interest,
        "profit": profit,
    })
"""

# =========================
# Main Page with Tabs
# =========================

def run_scenario_module(capex_pct, da_pct, nwc_change):
    """Run the Scenario Planning app with tabs for What-If, Monte Carlo, and Sensitivity Analysis."""
    
    # Set up Streamlit page config
    st.header("📊 Scenario Planning (What‑If & Risk)")
    st.subheader("Tweak drivers, compare scenarios, run a quick Monte Carlo, and see what moves the needle")
    st.caption("Adjust inflation, sales change and interest rates; see KPI impacts and stress tests.")

    # Basic Inputs: Base Annual Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        base_sales = st.number_input("Base Annual Sales ($)", min_value=0.0, value=100000.0, step=1000.0, format="%.0f", help="e.g., revenue")
    with col2:
        base_costs = st.number_input("Base Annual Costs ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.0f")
    with col3:
        base_interest = st.number_input("Base Interest Expense ($)", min_value=0.0, value=2000.0, step=100.0, format="%.0f")

    st.subheader("Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        inflation = st.slider("Inflation (%)", -10.0, 15.0, 2.0, step=0.5, help="Impact on costs and prices.")
    with col2:
        sales_change = st.slider("Sales Change (%)", -100.0, 100.0, 0.0, step=1.0, help="Impact on sales volume or price changes.")
    with col3:
        interest_rate = st.slider("Interest Rate (%)", -20.0, 20.0, 2.0, step=0.5, help="Impact on financing costs.")

    years = st.slider("Projection Horizon (years)", 1, 10, 3)
    discount_rate = st.slider("Discount Rate (%)", 0.0, 30.0, 10.0, step=0.5, help="For Net Present Value. Use 0% to disable NPV.")

    # Compute Adjusted Annual Values
    adj_sales = base_sales * (1 + sales_change / 100)
    adj_costs = base_costs * (1 + inflation / 100)
    adj_interest = base_interest * (1 + interest_rate / 100)
    profit = adj_sales - adj_costs - adj_interest
    margin = (profit / adj_sales * 100) if adj_sales > 0 else 0.0

    st.metric("Adjusted Profit", f"${profit:,.0f}")
    st.metric("Profit Margin (%)", f"{margin:,.1f}%")

    # Visual Waterfall
    st.plotly_chart(_waterfall_chart(adj_sales, adj_costs, adj_interest, profit), use_container_width=True)


    # Multi-Year Projection with Compounding
    df = pd.DataFrame(index=range(1, years + 1), columns=["Sales", "Costs", "Interest", "Profit", "FreeCF", "Discount Factor", "PV of FCF"])

    sales, costs, interest = base_sales, base_costs, base_interest

    for year in range(1, years + 1):
        sales = sales * (1 + sales_change / 100)
        costs = costs * (1 + inflation / 100)
        interest = interest * (1 + interest_rate / 100)
        profit = sales - costs - interest
        #free_cf = profit
        capex = adj_sales * (capex_pct / 100)
        d_and_a = adj_sales * (da_pct / 100)
        free_cf = profit + d_and_a - capex - nwc_change
        # CHECK FORMULAS FOR FCF AND NPV
        df.loc[year, ["Sales", "Costs", "Interest", "Profit", "FreeCF"]] = [sales, costs, interest, profit, free_cf]
        df.loc[year, "Discount Factor"] = 1 / ((1 + discount_rate / 100) ** year) if discount_rate > 0 else 1.0
        df.loc[year, "PV of FCF"] = df.loc[year, "FreeCF"] * df.loc[year, "Discount Factor"]

    # Projections
    st.subheader("Projection Table")
    
    styled_df = df.style.format({
        "Sales": "${:,.0f}",
        "Costs": "${:,.0f}",
        "Interest": "${:,.0f}",
        "Profit": "${:,.0f}",
        "FreeCF": "${:,.0f}",
        "Discount Factor": "{:.3f}",
        "PV of FCF": "${:,.0f}"
    })
    st.dataframe(styled_df)

    st.write(f"**Net Present Value (NPV) of Free Cash Flows (FCF) over {years} years @ {discount_rate:.1f}%:** ${df['PV of FCF'].sum():,.0f}")


    # Tornado Sensitivity Chart
    with st.expander("🌪️ Sensitivity: What moves profit most? (±10% on each driver)", expanded=False):
        st.plotly_chart(_tornado_sensitivity(base_sales, base_costs, base_interest, inflation, sales_change, interest_rate), use_container_width=True)


    # Monte Carlo Stress Test
    
    with st.expander("🎲 Monte Carlo Stress Test", expanded=False):
        runs = st.slider("Simulation Runs", 100, 5000, 1000, step=100)

        # Simple Assumption: Normal noise around the selected % changes
        #st.status("Running simulation…", state= 'running', expanded=False)
        with st.status("Running simulation...", expanded=False) as s:
            s.update(label="Calculating Profits...", state='running')
            s.update(label="Simulation Completed!", state='complete')
                
        sales_mu, sales_sd = sales_change, max(1.0, abs(sales_change)/3)
        infl_mu, infl_sd = inflation, max(1.0, abs(inflation)/3)
        interest_rate_mu, interest_rate_sd = interest_rate, max(1.0, abs(interest_rate)/3)

        time_range = np.random.default_rng(42)
        sales_draws = time_range.normal(loc=sales_mu, scale=sales_sd, size=runs)
        infl_draws = time_range.normal(loc=infl_mu, scale=infl_sd, size=runs)
        interest_rate_draws = time_range.normal(loc=interest_rate_mu, scale=interest_rate_sd, size=runs)

        sim_profits = base_sales * (1 + sales_draws / 100) - base_costs * (1 + infl_draws / 100) - base_interest * (1 + interest_rate_draws / 100)
        st.markdown("*(Note: This is a simplified Monte Carlo simulation for demonstration purposes only.)*")
        st.write(f"**Mean Profit:** {sim_profits.mean():,.2f}$")
        st.write(f"**Median:** {np.median(sim_profits):,.2f}$")
        st.write(f"**5th Percentile:** {np.percentile(sim_profits, 5):,.2f}$")
        st.write(f"**95th Percentile:** {np.percentile(sim_profits, 95):,.2f}$")
        st.write(f"**Loss Probability:** {(sim_profits < 0).mean():.2%}")
        st.subheader("\n**Profit Distribution Histogram of Monte Carlo Simulations:**")
        st.bar_chart(pd.Series(sim_profits, name="Profit Distribution"), use_container_width=True)


        # Executive Summary Download
        summary_md = f"""
        ## Executive Summary
        - Base Profit: {profit:,.0f} | Margin: {margin:.1f}%
        - Mean (Monte Carlo): ${sim_profits.mean():,.0f} | Median (Monte Carlo): ${np.median(sim_profits):,.0f}
        - 5th- 95th Pct (Monte Carlo): ${np.percentile(sim_profits, 5):,.0f} - {np.percentile(sim_profits, 95):,.0f}
        - Loss Probability: {(sim_profits < 0).mean():.1%}

        - Base Sales: ${base_sales:,.0f} | Base Costs: ${base_costs:,.0f} | Base Interest: ${base_interest:,.0f}
        - Inflation: {inflation:.1f}% | Sales Δ: {sales_change:.1f}% | Rate Δ: {interest_rate:.1f}%
        - Adjusted Profit: ${profit:,.0f} | Margin: {margin:.1f}%
        - NPV of FCF over {years} years @ {discount_rate:.1f}%: ${df['PV of FCF'].sum():,.0f} 
        """
        st.download_button("📥 Download Executive Summary (MD)", summary_md, "scenario_executive_summary.md")    
        # Consider only doing MC things for the summary      

