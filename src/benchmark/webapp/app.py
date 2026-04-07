"""Streamlit web application for benchmark visualization."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark.config import RESULTS_DIR

# Page configuration
st.set_page_config(
    page_title="Benchmarking Tree Regressions",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Available options
DATA_MODELS = {
    "Friedman": "sim_friedman",
    "Checkerboard": "sim_checkerboard",
    "Linear": "sim_linear",
    "Max": "sim_max",
    "Single Index": "sim_single_index",
}

X_STRUCTURES = {
    "Independent": "indep",
    "AR(1)": "ar1",
    "AR(1)+": "ar1+",
    "Factor": "factor",
}


def load_results():
    """Load benchmark results from results directory."""
    synthetic_path = RESULTS_DIR / "benchmark_synthetic.csv"
    real_path = RESULTS_DIR / "benchmark_real.csv"

    df_synthetic = None
    df_real = None

    if synthetic_path.exists():
        df_synthetic = pd.read_csv(synthetic_path)

    if real_path.exists():
        df_real = pd.read_csv(real_path)

    return df_synthetic, df_real


def get_data_model_formula(data_model: str) -> str:
    """Get LaTeX formula for the data model."""
    formulas = {
        "sim_friedman": r"$f(x) = 10\sin(\pi x_1x_2) + 20(x_3-0.5)^2 + 10x_4 + 5x_5$",
        "sim_checkerboard": r"$f(x) = 2x_1x_2 + 2x_3x_4$",
        "sim_linear": r"$f(x) = 2x_1 + 2x_2 + 4x_3$",
        "sim_max": r"$f(x) = \max(x_1, x_2, x_3)$",
        "sim_single_index": r"$f(x) = 10\sqrt{a} + \sin(5a); a = \sum_{j=1}^{10}(x_j - \gamma_j)^2$",
    }
    return formulas.get(data_model, "")


def get_x_structure_formula(structure: str) -> str:
    """Get LaTeX formula for the X structure."""
    formulas = {
        "indep": r"$X_{ij} \sim_{i.i.d.} N(0, 1)$",
        "ar1": r"$X \sim N(0_p, \Sigma), \Sigma_{jk} = 0.9^{|j-k|}$",
        "ar1+": r"$X \sim N(0_p, \Sigma), \Sigma_{jk} = 0.5^{|j-k|} + 0.2I(j \neq k)$",
        "factor": r"$\mathbf{X = (BF)^\top + \epsilon}$",
    }
    return formulas.get(structure, "")


def plot_cv_error(df: pd.DataFrame, x_col: str, title: str) -> None:
    """Plot CV error comparison."""
    if df is None or df.empty:
        st.warning("No synthetic benchmark results found. Run benchmarks first.")
        return

    fig = px.line(
        df,
        x=x_col,
        y="cverr",
        color="method",
        markers=True,
        title=title,
        labels={"cverr": "5-fold CV MSE", x_col: x_col},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.5),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_runtime(df: pd.DataFrame, x_col: str, title: str) -> None:
    """Plot runtime comparison."""
    if df is None or df.empty:
        st.warning("No synthetic benchmark results found.")
        return

    fig = px.line(
        df,
        x=x_col,
        y="runtime",
        color="method",
        markers=True,
        title=title,
        labels={"runtime": "Runtime (seconds)", x_col: x_col},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.5),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_real_data_results(df: pd.DataFrame, metric: str, title: str) -> None:
    """Plot real data benchmark results."""
    if df is None or df.empty:
        st.warning("No real data benchmark results found.")
        return

    fig = px.bar(
        df,
        x="method",
        y=metric,
        color="method",
        title=title,
        labels={metric: f"5-fold CV {metric.upper()}"},
    )
    fig.update_layout(showlegend=False, height=500)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Benchmarking Tree Regressions")

    # Load results
    df_synthetic, df_real = load_results()

    # Sidebar for navigation
    page = st.sidebar.radio(
        "Select Section",
        ["Synthetic Data", "Real Data", "About"],
    )

    if page == "Synthetic Data":
        st.header("Synthetic Data Benchmarks")

        # Data model selection
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox(
                "Data Model",
                options=list(DATA_MODELS.keys()),
                index=0,
            )
        with col2:
            selected_structure = st.selectbox(
                "X Structure",
                options=list(X_STRUCTURES.keys()),
                index=0,
            )

        data_model = DATA_MODELS[selected_model]
        x_structure = X_STRUCTURES[selected_structure]

        # Display formulas
        formula = get_data_model_formula(data_model)
        if formula:
            st.latex(formula)

        x_formula = get_x_structure_formula(x_structure)
        if x_formula:
            st.latex(x_formula)

        # Filter data
        if df_synthetic is not None:
            df_filtered = df_synthetic[
                (df_synthetic["data_model"] == data_model)
                & (df_synthetic["structure"] == x_structure)
            ]
        else:
            df_filtered = None

        # X-axis selection
        x_axis = st.radio(
            "X-axis variable",
            options=["n (sample size)", "p (dimension)"],
            horizontal=True,
        )

        if x_axis == "n (sample size)":
            x_col = "n"
            other_col = "p"
        else:
            x_col = "p"
            other_col = "n"

        # Allow selecting fixed value of other variable
        if df_filtered is not None and not df_filtered.empty:
            fixed_values = sorted(df_filtered[other_col].unique())
            fixed_value = st.selectbox(
                f"Fixed {other_col}",
                options=fixed_values,
                index=len(fixed_values) // 2,
            )
            df_plot = df_filtered[df_filtered[other_col] == fixed_value]
        else:
            df_plot = df_filtered

        # Plots
        st.subheader("5-fold CV MSE")
        plot_cv_error(
            df_plot,
            x_col,
            f"CV Error: {selected_model} with {selected_structure} structure",
        )

        st.subheader("Running Time")
        plot_runtime(
            df_plot,
            x_col,
            f"Runtime: {selected_model} with {selected_structure} structure",
        )

        # Warnings for failed methods
        if df_plot is not None and "error" in df_plot.columns:
            failed = df_plot[df_plot["error"].notna()]["method"].unique()
            if len(failed) > 0:
                st.warning(f"Failed methods: {', '.join(failed)}")

    elif page == "Real Data":
        st.header("Real Data Benchmarks")

        if df_real is not None and not df_real.empty:
            dataset_names = df_real["data_model"].unique()
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=dataset_names,
            )

            df_dataset = df_real[df_real["data_model"] == selected_dataset]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("5-fold CV MSE")
                plot_real_data_results(df_dataset, "cverr", f"MSE - {selected_dataset}")
            with col2:
                st.subheader("Running Time")
                plot_real_data_results(df_dataset, "runtime", f"Runtime - {selected_dataset}")
        else:
            st.info("No real data results available. Run benchmarks first.")

    elif page == "About":
        st.header("About")
        st.markdown("""
        **Benchmarking Tree Regressions**

        This project benchmarks various tree-based regression methods on both
        synthetic and real datasets.

        **Methods Compared:**
        - **BART** - Bayesian Additive Regression Trees (via sklearn approximation)
        - **XBART** - Accelerated BART
        - **MARS** - Multivariate Adaptive Regression Splines
        - **XGBoost** - Extreme Gradient Boosting
        - **CatBoost** - Categorical Boosting
        - **Random Forest** - Breiman's Random Forest

        **Performance Metrics:**
        - 5-fold Cross-Validation MSE
        - Running Time

        **Data Generating Processes:**
        - Friedman
        - Checkerboard
        - Linear
        - Max
        - Single Index

        **Covariance Structures:**
        - Independent
        - AR(1)
        - AR(1)+
        - Factor
        """)


if __name__ == "__main__":
    main()
