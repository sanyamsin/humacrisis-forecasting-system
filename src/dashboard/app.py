import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ── Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="HumaCrisis Forecasting System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background-color: #1a1a2e; border-radius: 10px; padding: 10px; }
    .critical { color: #e74c3c; font-weight: bold; }
    .high { color: #e67e22; font-weight: bold; }
    .medium { color: #f39c12; font-weight: bold; }
    .low { color: #2ecc71; font-weight: bold; }
    h1 { color: #3498db; }
    h2 { color: #ecf0f1; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "SOM": "#e74c3c", "SSD": "#e67e22", "ETH": "#f39c12",
    "CAR": "#9b59b6", "COD": "#3498db", "MLI": "#1abc9c",
    "NER": "#2ecc71", "TCD": "#e91e63", "MRT": "#00bcd4",
    "SEN": "#8bc34a"
}

RISK_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH": "#e67e22",
    "MEDIUM": "#f39c12",
    "LOW": "#2ecc71"
}

API_URL = "http://127.0.0.1:8000"

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/merged_dataset.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_features():
    try:
        df = pd.read_csv("data/processed/features_dataset.csv", parse_dates=["date"])
        return df
    except:
        return None

def get_risk_level(severity):
    if severity >= 0.75: return "CRITICAL"
    elif severity >= 0.60: return "HIGH"
    elif severity >= 0.40: return "MEDIUM"
    else: return "LOW"

def compute_severity(ipc, events, displaced):
    return float(np.clip(
        (ipc / 5) * 0.4 +
        np.log1p(events) / 10 * 0.3 +
        np.log1p(displaced) / 20 * 0.3,
        0, 1
    ))

# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Earthlights_dmsp.jpg/320px-Earthlights_dmsp.jpg",
        use_column_width=True
    )
    st.sidebar.title("🌍 HumaCrisis")
    st.sidebar.caption("Humanitarian Crisis Forecasting System")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "🗺️ Country Analysis", "🔮 Forecasting", "📊 Model Insights"]
    )

    st.sidebar.divider()
    st.sidebar.caption("Data Sources")
    st.sidebar.markdown("🌾 FEWS NET — Food Security")
    st.sidebar.markdown("⚔️ ACLED — Conflict Data")
    st.sidebar.markdown("🚶 UNHCR — Displacement")
    st.sidebar.divider()
    st.sidebar.caption("v1.0.0 | HumaCrisis Forecasting System")

    return page

# ── Page : Overview ──────────────────────────────────────────
def page_overview(df):
    st.title("🌍 HumaCrisis Forecasting System")
    st.caption("Multi-dimensional humanitarian crisis forecasting for Sub-Saharan Africa")
    st.divider()

    # KPI Cards
    latest = df.sort_values("date").groupby("country").last().reset_index()
    latest["severity"] = latest.apply(
        lambda r: compute_severity(r["ipc_phase"], r["total_events"], r["total_displaced"]), axis=1
    )
    latest["risk"] = latest["severity"].apply(get_risk_level)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        critical = (latest["risk"] == "CRITICAL").sum()
        st.metric("🔴 Critical", critical, delta="countries")
    with col2:
        high = (latest["risk"] == "HIGH").sum()
        st.metric("🟠 High Risk", high, delta="countries")
    with col3:
        total_displaced = latest["total_displaced"].sum()
        st.metric("🚶 Total Displaced", f"{total_displaced/1e6:.1f}M")
    with col4:
        avg_ipc = latest["ipc_phase"].mean()
        st.metric("🌾 Avg IPC Phase", f"{avg_ipc:.1f}/5")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Crisis Severity by Country")
        fig = px.bar(
            latest.sort_values("severity", ascending=True),
            x="severity", y="country", orientation="h",
            color="severity",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            labels={"severity": "Crisis Severity Index", "country": "Country"},
        )
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="white", height=400,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Distribution")
        risk_counts = latest["risk"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=[RISK_COLORS.get(r, "gray") for r in risk_counts.index],
            hole=0.4
        ))
        fig2.update_layout(
            paper_bgcolor="#0f1117", font_color="white", height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Timeline
    st.subheader("📈 Crisis Evolution — All Countries")
    fig3 = go.Figure()
    for country in df["country"].unique():
        cdf = df[df["country"] == country].copy()
        cdf["severity"] = cdf.apply(
            lambda r: compute_severity(r["ipc_phase"], r["total_events"], r["total_displaced"]), axis=1
        )
        fig3.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["severity"],
            name=country, mode="lines",
            line=dict(color=COLORS.get(country, "white"), width=2)
        ))

    fig3.add_hline(y=0.6, line_dash="dash", line_color="#e74c3c",
                   annotation_text="High Risk Threshold")
    fig3.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="white", height=400,
        xaxis_title="Date", yaxis_title="Crisis Severity Index"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Page : Country Analysis ──────────────────────────────────
def page_country(df):
    st.title("🗺️ Country Analysis")
    st.divider()

    country = st.selectbox(
        "Select Country",
        options=df["country"].unique(),
        format_func=lambda x: {
            "SOM": "🇸🇴 Somalia", "SSD": "🇸🇸 South Sudan",
            "ETH": "🇪🇹 Ethiopia", "CAR": "🇨🇫 Central African Republic",
            "COD": "🇨🇩 DR Congo", "MLI": "🇲🇱 Mali",
            "NER": "🇳🇪 Niger", "TCD": "🇹🇩 Chad",
            "MRT": "🇲🇷 Mauritania", "SEN": "🇸🇳 Senegal"
        }.get(x, x)
    )

    cdf = df[df["country"] == country].copy().sort_values("date")
    cdf["severity"] = cdf.apply(
        lambda r: compute_severity(r["ipc_phase"], r["total_events"], r["total_displaced"]), axis=1
    )

    latest = cdf.iloc[-1]
    severity = compute_severity(latest["ipc_phase"], latest["total_events"], latest["total_displaced"])
    risk = get_risk_level(severity)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌾 IPC Phase", f"{latest['ipc_phase']:.1f}/5")
    with col2:
        st.metric("⚔️ Monthly Events", int(latest["total_events"]))
    with col3:
        st.metric("🚶 Displaced", f"{int(latest['total_displaced'])/1e6:.2f}M")
    with col4:
        color = RISK_COLORS.get(risk, "white")
        st.metric("🚨 Risk Level", risk)

    st.divider()

    # Charts
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Food Insecurity (IPC Phase)", "Conflict Events", "Displaced Population"],
        vertical_spacing=0.1
    )

    color = COLORS.get(country, "#3498db")

    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["ipc_phase"],
        fill="tozeroy", line=dict(color=color, width=2),
        name="IPC Phase"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=cdf["date"], y=cdf["total_events"],
        marker_color=color, opacity=0.8, name="Events"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["total_displaced"]/1e6,
        fill="tozeroy", line=dict(color=color, width=2),
        name="Displaced (M)"
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="white", height=700, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Page : Forecasting ───────────────────────────────────────
def page_forecasting(df):
    st.title("🔮 Crisis Forecasting")
    st.caption("Generate predictions using the HumaCrisis ML model")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        country = st.selectbox("Country", df["country"].unique())
        date = st.date_input("Reference Date", value=pd.Timestamp("2024-01-01"))
        horizon = st.slider("Forecast Horizon (months)", 1, 6, 3)

        cdf = df[df["country"] == country].sort_values("date").iloc[-1]
        ipc = st.slider("IPC Phase", 1.0, 5.0, float(round(cdf["ipc_phase"], 1)), 0.1)
        events = st.number_input("Monthly Conflict Events", 0, 500, int(cdf["total_events"]))
        fatalities = st.number_input("Monthly Fatalities", 0, 2000, int(cdf["total_fatalities"]))
        displaced = st.number_input("Total Displaced", 0, 10000000, int(cdf["total_displaced"]), 10000)

        predict_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

    with col2:
        st.subheader("Forecast Results")
        if predict_btn:
            with st.spinner("Running forecast model..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json={
                        "country": country,
                        "date": str(date),
                        "ipc_phase": ipc,
                        "total_events": events,
                        "total_fatalities": fatalities,
                        "total_displaced": displaced,
                        "forecast_horizon": horizon
                    })
                    result = response.json()

                    risk = result["risk_level"]
                    color = RISK_COLORS.get(risk, "white")

                    st.markdown(f"### Forecast for {country} — {result['forecast_date']}")
                    st.markdown(f"**Crisis Severity Index:** `{result['crisis_severity_index']:.4f}`")
                    st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk}</span>",
                               unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** `{result['confidence']*100:.0f}%`")

                    st.divider()
                    st.markdown("**📋 Recommendations:**")
                    for rec in result["recommendations"]:
                        st.markdown(f"- {rec}")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["crisis_severity_index"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Crisis Severity Index", "font": {"color": "white"}},
                        gauge={
                            "axis": {"range": [0, 1], "tickcolor": "white"},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 0.4], "color": "#1a4a1a"},
                                {"range": [0.4, 0.6], "color": "#4a3a00"},
                                {"range": [0.6, 0.75], "color": "#4a1a00"},
                                {"range": [0.75, 1], "color": "#3a0000"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 2},
                                "thickness": 0.75,
                                "value": result["crisis_severity_index"]
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor="#0f1117", font_color="white", height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"API Error: {e}. Make sure the API is running!")
        else:
            st.info("👈 Set parameters and click **Generate Forecast**")

# ── Page : Model Insights ────────────────────────────────────
def page_insights():
    st.title("📊 Model Insights")
    st.divider()

    col1, col2 = st.columns(2)

    figures = [
        ("reports/figures/06_xgboost_importance.png", "XGBoost Feature Importance"),
        ("reports/figures/07_lstm_training.png", "LSTM Training Curves"),
        ("reports/figures/08_model_comparison.png", "Model Comparison"),
        ("reports/figures/04_correlation_heatmap.png", "Correlation Heatmap"),
    ]

    for i, (path, title) in enumerate(figures):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.subheader(title)
            try:
                st.image(path, use_column_width=True)
            except:
                st.warning(f"Figure not found: {path}")

# ── Main ─────────────────────────────────────────────────────
def main():
    df = load_data()
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "🗺️ Country Analysis":
        page_country(df)
    elif page == "🔮 Forecasting":
        page_forecasting(df)
    elif page == "📊 Model Insights":
        page_insights()

if __name__ == "__main__":
    main()