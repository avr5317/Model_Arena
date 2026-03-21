import json
import os
import glob as glob_mod

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Arena", page_icon="🏟️", layout="wide")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stMetric label { color: #a0a0b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e94560 !important; font-size: 1.8rem !important; }
    h1 {
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    .model-card {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_latest_results():
    """Load the most recent evaluation results."""
    if not os.path.exists(RESULTS_DIR):
        return None
    files = sorted(glob_mod.glob(os.path.join(RESULTS_DIR, "eval_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_all_result_files():
    """Get list of all result files."""
    if not os.path.exists(RESULTS_DIR):
        return []
    return sorted(glob_mod.glob(os.path.join(RESULTS_DIR, "eval_*.json")), reverse=True)


def normalize_series(s):
    """Min-max normalize a pandas Series to [0, 1]."""
    if s.max() == s.min():
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# --- Header ---
st.title("🏟️ Model Arena")
st.caption("Local LLM Evaluation Platform — Compare models side-by-side with automated metrics")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")

    # Run evaluation button
    if st.button("🚀 Run New Evaluation", use_container_width=True):
        with st.status("Running evaluation...", expanded=True) as status:
            st.write("Checking model availability...")
            from evaluate import wait_for_models, evaluate_all, save_results
            available = wait_for_models(timeout=300)
            if not available:
                st.error("No models available. Make sure containers are running.")
            else:
                st.write(f"Found {len(available)} models: {', '.join(available)}")
                progress_bar = st.progress(0)
                progress_text = st.empty()

                def update_progress(done, total, model, prompt_id):
                    progress_bar.progress(done / total)
                    progress_text.text(f"{done}/{total}: {model} | {prompt_id}")

                results = evaluate_all(available, progress_callback=update_progress)
                filepath = save_results(results)
                status.update(label="Evaluation complete!", state="complete")
                st.success(f"Saved {len(results)} results")
                st.rerun()

    st.divider()

    # Result file selector
    result_files = load_all_result_files()
    if result_files:
        selected_file = st.selectbox(
            "Load Results",
            result_files,
            format_func=lambda x: os.path.basename(x),
        )
    else:
        selected_file = None

    st.divider()
    st.markdown("**Architecture**")
    st.code("3x Ollama Containers\n→ Evaluation Pipeline\n→ Streamlit Dashboard", language=None)

# --- Load Data ---
if selected_file:
    with open(selected_file) as f:
        raw_data = json.load(f)
else:
    raw_data = load_latest_results()

if not raw_data:
    st.info("No evaluation results found. Click **Run New Evaluation** in the sidebar to get started.")
    st.markdown("""
    ### Quick Start
    1. Make sure Docker containers are running: `docker compose up -d`
    2. Pull models: `docker exec ollama-phi3 ollama pull phi3:mini`
    3. Click **Run New Evaluation** in the sidebar
    """)
    st.stop()

df = pd.DataFrame(raw_data)

# Filter out errors
df_valid = df[df["response"].astype(str).str.len() > 0].copy()

if df_valid.empty:
    st.error("All evaluations returned errors. Check that models are pulled and containers are healthy.")
    st.stop()

# --- Hero Metrics ---
model_stats = df_valid.groupby("model").agg({
    "tokens_per_second": "mean",
    "keyword_accuracy": "mean",
    "latency_seconds": "mean",
    "memory_mb": "mean",
}).reset_index()

# Composite score
model_stats["norm_speed"] = normalize_series(model_stats["tokens_per_second"])
model_stats["norm_accuracy"] = normalize_series(model_stats["keyword_accuracy"])
model_stats["norm_memory"] = 1 - normalize_series(model_stats["memory_mb"])  # Lower is better
bleu_avg = df_valid.groupby("model")["bleu1_score"].mean()
model_stats = model_stats.merge(bleu_avg.rename("avg_bleu"), left_on="model", right_index=True)
model_stats["norm_bleu"] = normalize_series(model_stats["avg_bleu"])
model_stats["composite_score"] = (
    0.3 * model_stats["norm_speed"]
    + 0.4 * model_stats["norm_accuracy"]
    + 0.2 * model_stats["norm_bleu"]
    + 0.1 * model_stats["norm_memory"]
)
model_stats = model_stats.sort_values("composite_score", ascending=False)

fastest = model_stats.loc[model_stats["tokens_per_second"].idxmax()]
most_accurate = model_stats.loc[model_stats["keyword_accuracy"].idxmax()]
best_overall = model_stats.iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⚡ Fastest", fastest["model"], f"{fastest['tokens_per_second']:.1f} tok/s")
with col2:
    st.metric("🎯 Most Accurate", most_accurate["model"], f"{most_accurate['keyword_accuracy']:.0%}")
with col3:
    st.metric("🏆 Best Overall", best_overall["model"], f"Score: {best_overall['composite_score']:.2f}")
with col4:
    st.metric("📊 Evaluations", f"{len(df_valid)}", f"{df_valid['model'].nunique()} models")

st.divider()

# --- Leaderboard ---
st.subheader("📋 Leaderboard")
leaderboard = model_stats[["model", "tokens_per_second", "keyword_accuracy", "avg_bleu", "latency_seconds", "memory_mb", "composite_score"]].copy()
leaderboard.columns = ["Model", "Avg Tokens/s", "Keyword Accuracy", "BLEU-1", "Avg Latency (s)", "Avg Memory (MB)", "Composite Score"]
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
leaderboard = leaderboard.reset_index(drop=True)

st.dataframe(
    leaderboard.style.format({
        "Avg Tokens/s": "{:.1f}",
        "Keyword Accuracy": "{:.1%}",
        "BLEU-1": "{:.4f}",
        "Avg Latency (s)": "{:.2f}",
        "Avg Memory (MB)": "{:.0f}",
        "Composite Score": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- Charts Row 1: Speed & Accuracy ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("⚡ Speed Comparison")
    fig_speed = px.bar(
        df_valid.groupby("model")["tokens_per_second"].mean().reset_index(),
        x="model", y="tokens_per_second",
        color="model",
        color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
        labels={"tokens_per_second": "Tokens/sec", "model": "Model"},
    )
    fig_speed.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0a0b0",
        showlegend=False,
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with chart_col2:
    st.subheader("🎯 Accuracy by Category")
    # Radar chart
    categories = df_valid["category"].unique().tolist()
    models = df_valid["model"].unique().tolist()
    colors = ["#e94560", "#0f3460", "#533483"]

    fig_radar = go.Figure()
    for i, model in enumerate(models):
        model_data = df_valid[df_valid["model"] == model]
        values = [model_data[model_data["category"] == cat]["keyword_accuracy"].mean()
                  for cat in categories]
        values.append(values[0])  # Close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=model,
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0a0b0",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Charts Row 2: Latency & Resources ---
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("⏱️ Latency by Prompt")
    fig_latency = px.bar(
        df_valid,
        x="prompt_id", y="latency_seconds",
        color="model",
        barmode="group",
        color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
        labels={"latency_seconds": "Latency (s)", "prompt_id": "Prompt"},
    )
    fig_latency.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0a0b0",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_latency, use_container_width=True)

with chart_col4:
    st.subheader("💾 Resource Usage")
    resource_data = model_stats[["model", "memory_mb"]].copy()
    # Also get CPU data
    cpu_avg = df_valid.groupby("model")["cpu_percent"].mean().reset_index()
    resource_data = resource_data.merge(cpu_avg, on="model")

    fig_resources = go.Figure()
    fig_resources.add_trace(go.Bar(
        name="Memory (MB)",
        x=resource_data["model"],
        y=resource_data["memory_mb"],
        marker_color="#e94560",
    ))
    fig_resources.add_trace(go.Bar(
        name="CPU %",
        x=resource_data["model"],
        y=resource_data["cpu_percent"],
        marker_color="#0f3460",
    ))
    fig_resources.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#a0a0b0",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_resources, use_container_width=True)

st.divider()

# --- Side-by-Side Response Viewer ---
st.subheader("💬 Side-by-Side Response Comparison")

categories_list = ["All"] + sorted(df_valid["category"].unique().tolist())
selected_category = st.selectbox("Filter by category", categories_list)

if selected_category != "All":
    filtered = df_valid[df_valid["category"] == selected_category]
else:
    filtered = df_valid

prompts_in_view = filtered["prompt_id"].unique()
models_in_view = sorted(filtered["model"].unique())

for prompt_id in prompts_in_view:
    prompt_data = filtered[filtered["prompt_id"] == prompt_id]
    prompt_text = prompt_data.iloc[0].get("prompt_id", prompt_id)

    with st.expander(f"📝 {prompt_id}", expanded=False):
        cols = st.columns(len(models_in_view))
        for i, model in enumerate(models_in_view):
            with cols[i]:
                row = prompt_data[prompt_data["model"] == model]
                if row.empty:
                    st.warning(f"{model}: No data")
                    continue
                row = row.iloc[0]
                st.markdown(f"**{model}**")
                st.caption(f"⏱️ {row['latency_seconds']:.2f}s | 🚀 {row['tokens_per_second']:.1f} tok/s")
                st.caption(f"🎯 Keywords: {row['keyword_accuracy']:.0%} | BLEU: {row['bleu1_score']:.4f}")
                st.text_area(
                    "Response",
                    value=row["response"],
                    height=200,
                    key=f"{prompt_id}_{model}",
                    disabled=True,
                )

# --- Footer ---
st.divider()
st.caption("Model Arena — Built with Docker Compose, Ollama, and Streamlit")
