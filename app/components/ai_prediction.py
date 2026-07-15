import streamlit as st
import sys
import os
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Ensure ml/ directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource(show_spinner=False)
def _load_prediction_service():
    """Load PredictionService once and cache it across Streamlit reruns."""
    try:
        from ml_config import MLConfig
        from predict_service import PredictionService
        config = MLConfig()
        service = PredictionService(config)
        return service, config, None
    except Exception as e:
        return None, None, str(e)


def _probability_gauge(probability: float, threshold: float) -> go.Figure:
    """Renders a Plotly arc-gauge for layoff recurrence probability."""
    pct = round(probability * 100, 1)

    # Colour zones based on threshold
    t = threshold * 100
    if pct < t * 0.4:
        bar_color = "#22c55e"   # green
    elif pct < t:
        bar_color = "#f59e0b"   # amber
    elif pct < t + (100 - t) * 0.5:
        bar_color = "#f97316"   # orange
    else:
        bar_color = "#ef4444"   # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        delta={"reference": round(threshold * 100, 1), "suffix": "%",
               "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#f3f4f6"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#6b7280", "tickfont": {"color": "#9ca3af"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  round(threshold * 40)],   "color": "rgba(34,197,94,0.08)"},
                {"range": [round(threshold * 40), round(threshold * 100)], "color": "rgba(245,158,11,0.08)"},
                {"range": [round(threshold * 100), 100], "color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#6366f1", "width": 3},
                "thickness": 0.8,
                "value": round(threshold * 100, 1)
            }
        },
        title={"text": "Layoff Recurrence Probability", "font": {"size": 14, "color": "#9ca3af"}}
    ))

    fig.update_layout(
        height=260,
        margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f3f4f6"}
    )
    return fig


def _risk_badge_html(category: str) -> str:
    colours = {
        "Very Low": ("#22c55e", "rgba(34,197,94,0.12)"),
        "Low":      ("#86efac", "rgba(134,239,172,0.10)"),
        "Medium":   ("#f59e0b", "rgba(245,158,11,0.12)"),
        "High":     ("#f97316", "rgba(249,115,22,0.12)"),
        "Critical": ("#ef4444", "rgba(239,68,68,0.15)"),
    }
    text_col, bg_col = colours.get(category, ("#9ca3af", "rgba(156,163,175,0.10)"))
    return (
        f'<span style="display:inline-block;padding:4px 14px;border-radius:20px;'
        f'background:{bg_col};border:1px solid {text_col};color:{text_col};'
        f'font-weight:700;font-size:0.9rem;letter-spacing:0.05em;">{category}</span>'
    )


def _shap_bar_chart(factors: list, title: str, color: str) -> go.Figure:
    """Horizontal bar chart for top SHAP contributors."""
    if not factors:
        return None
    names  = [f.get("feature_clean", f.get("feature", ""))[:28] for f in factors]
    values = [abs(f["shap_value"]) for f in factors]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=color,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#d1d5db", "size": 11}
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 12, "color": "#9ca3af"}, "x": 0},
        height=180,
        margin=dict(t=36, b=0, l=0, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False, "visible": False},
        yaxis={"showgrid": False, "tickfont": {"color": "#d1d5db", "size": 11}},
        font={"color": "#f3f4f6"}
    )
    return fig


def render_ai_prediction_tab(company_name: str):
    """Renders the full AI Risk Forecast panel for a selected company."""

    service, config, load_error = _load_prediction_service()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h3 style="margin:0;color:#f3f4f6;font-size:1.25rem;font-weight:700;">
            🤖 AI Workforce Risk Forecast
        </h3>
        <p style="margin:0.25rem 0 0;color:#6b7280;font-size:0.85rem;">
            Powered by SoftVotingEnsemble · Optuna HPO · SHAP Explainability
        </p>
    </div>
    """, unsafe_allow_html=True)

    if load_error:
        st.error(f"AI Prediction Service unavailable: {load_error}")
        st.info("Run `python ml/run_pipeline.py` to train the models first.")
        return

    if not service.model:
        st.warning("No trained model found. Please run `python ml/run_pipeline.py` first.")
        return

    # ── Run Prediction ────────────────────────────────────────────────────────
    with st.spinner("Running AI inference…"):
        result = service.predict_recurrence(company_name)

    if "error" in result:
        st.error(result["error"])
        return

    if not result.get("has_history", False):
        st.info(result.get("message", "No layoff history found for this company."))
        return

    prob       = result["probability"]
    prob_pct   = result["probability_percentage"]
    risk_cat   = result["risk_category"]
    threshold  = result["decision_threshold"]
    confidence = result["confidence_percentage"]
    label      = result["prediction_label"]
    shap_exp   = result.get("shap_explanation", {})
    ts         = result["prediction_timestamp"]

    # ── Top Row: Gauge + Summary ──────────────────────────────────────────────
    col_gauge, col_summary = st.columns([1, 1], gap="large")

    with col_gauge:
        st.plotly_chart(
            _probability_gauge(prob, threshold),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col_summary:
        st.markdown("<br>", unsafe_allow_html=True)

        verdict = "⚠️ Layoff Likely" if label == 1 else "✅ Stable Outlook"
        verdict_color = "#ef4444" if label == 1 else "#22c55e"

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:0.75rem;">
            <div style="font-size:1.4rem;font-weight:800;color:{verdict_color};margin-bottom:0.5rem;">
                {verdict}
            </div>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">Risk Category</div>
                    <div style="margin-top:4px;">{_risk_badge_html(risk_cat)}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">Probability</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#f3f4f6;margin-top:2px;">{prob_pct}%</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">Model Confidence</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#818cf8;margin-top:2px;">{confidence}%</div>
                </div>
            </div>
        </div>
        <div style="font-size:0.76rem;color:#4b5563;margin-top:0.4rem;">
            Decision threshold: {round(threshold*100,1)}% &nbsp;·&nbsp; Predicted: {ts}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── SHAP Explanation ─────────────────────────────────────────────────────
    if "error" not in shap_exp:
        st.markdown("#### 🔍 AI Explanation — Why This Prediction?")

        pos = shap_exp.get("positive_factors", [])
        neg = shap_exp.get("negative_factors", [])

        col_pos, col_neg = st.columns(2, gap="medium")

        with col_pos:
            fig_pos = _shap_bar_chart(pos, "Risk-Increasing Factors", "#f97316")
            if fig_pos:
                st.plotly_chart(fig_pos, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("No significant risk-increasing factors.")

        with col_neg:
            fig_neg = _shap_bar_chart(neg, "Risk-Reducing Factors", "#22c55e")
            if fig_neg:
                st.plotly_chart(fig_neg, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("No significant risk-reducing factors.")

        # Waterfall explanation plot
        waterfall_path = result.get("waterfall_plot_path")
        if waterfall_path and os.path.exists(waterfall_path):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📈 SHAP Prediction Breakdown (Waterfall Chart)")
            st.image(waterfall_path, caption=f"SHAP Waterfall plot detailing features contributing to recurrence risk of {company_name}", use_container_width=True)

        st.divider()

    # ── Global SHAP Plots ─────────────────────────────────────────────────────
    shap_bar_path = PROJECT_ROOT / "ml" / "outputs" / "shap" / "shap_summary_bar.png"
    shap_dot_path = PROJECT_ROOT / "ml" / "outputs" / "shap" / "shap_summary_dot.png"

    if shap_bar_path.exists() or shap_dot_path.exists():
        with st.expander("📊 Global Feature Importance (SHAP — All Companies)", expanded=False):
            g1, g2 = st.columns(2)
            with g1:
                if shap_bar_path.exists():
                    st.image(str(shap_bar_path), caption="SHAP Summary Bar", use_container_width=True)
            with g2:
                if shap_dot_path.exists():
                    st.image(str(shap_dot_path), caption="SHAP Summary Dot", use_container_width=True)

    # ── Model Metadata ────────────────────────────────────────────────────────
    with st.expander("⚙️ Model & Pipeline Metadata", expanded=False):
        meta_path = PROJECT_ROOT / "ml" / "models" / "production_model_meta.json"
        hpo_path  = PROJECT_ROOT / "ml" / "models" / "xgboost_hpo_metadata.json"
        thr_path  = PROJECT_ROOT / "ml" / "models" / "optimized_threshold.json"

        c1, c2, c3 = st.columns(3)

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            with c1:
                st.metric("Production Model", meta.get("model_name", "—"))
                st.metric("ROC-AUC", f"{meta.get('metrics', {}).get('roc_auc', 0):.3f}")

        if hpo_path.exists():
            with open(hpo_path) as f:
                hpo = json.load(f)
            with c2:
                st.metric("HPO Best CV AUC", f"{hpo.get('best_roc_auc', 0):.4f}")
                st.metric("Optuna Trials", hpo.get("n_trials", "—"))

        if thr_path.exists():
            with open(thr_path) as f:
                thr = json.load(f)
            with c3:
                st.metric("Optimal Threshold", f"{thr.get('optimized_threshold', 0.5):.2f}")
                st.metric("Threshold F1", f"{thr.get('f1_score', 0):.3f}")

        # Benchmark comparison table
        bench_csv = PROJECT_ROOT / "ml" / "outputs" / "model_benchmark_comparison.csv"
        if bench_csv.exists():
            st.markdown("**Model Benchmark Comparison**")
            df_bench = pd.read_csv(bench_csv)
            display_cols = ["model_name", "roc_auc", "f1", "precision", "recall", "brier_score"]
            display_cols = [c for c in display_cols if c in df_bench.columns]
            st.dataframe(
                df_bench[display_cols].sort_values("roc_auc", ascending=False)
                         .style.format({c: "{:.3f}" for c in display_cols if c != "model_name"})
                         .highlight_max(subset=[c for c in display_cols if c != "model_name"],
                                        color="rgba(99,102,241,0.18)"),
                use_container_width=True, hide_index=True
            )

    # ── Prediction History ────────────────────────────────────────────────────
    history_path = PROJECT_ROOT / "ml" / "outputs" / "prediction_history.json"
    if history_path.exists():
        with st.expander("🕐 Recent Prediction History", expanded=False):
            with open(history_path) as f:
                history = json.load(f)
            if history:
                df_hist = pd.DataFrame(reversed(history[-10:]))
                df_hist["probability"] = df_hist["probability"].map("{:.1%}".format)
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.caption("No predictions logged yet.")
