import json
import os
import glob as glob_mod
import uuid as uuid_mod

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Arena", page_icon="🏟️", layout="wide")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

# Initialize session state for onboarding pipeline
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = "input"  # input -> analyzing -> selecting -> downloading -> ready
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "dynamic_models_dict" not in st.session_state:
    st.session_state.dynamic_models_dict = None

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
</style>
""", unsafe_allow_html=True)


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def save_prompts(prompts):
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2)


def load_latest_results():
    if not os.path.exists(RESULTS_DIR):
        return None
    files = sorted(glob_mod.glob(os.path.join(RESULTS_DIR, "eval_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_all_result_files():
    if not os.path.exists(RESULTS_DIR):
        return []
    return sorted(glob_mod.glob(os.path.join(RESULTS_DIR, "eval_*.json")), reverse=True)


def normalize_series(s):
    if s.max() == s.min():
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# ---------------------------------------------------------------------------
# Onboarding Flow — GitHub URL → Analysis → Model Selection → Download
# ---------------------------------------------------------------------------

if st.session_state.pipeline_state == "input":
    st.title("🏟️ Model Arena")
    st.caption("Find the best local LLM for your project — powered by Claude Sonnet + HuggingFace")

    st.markdown("---")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("### Paste your GitHub repository URL")
        st.markdown("We'll analyze your codebase with Claude Sonnet, find the best matching LLMs on HuggingFace, and evaluate them locally.")

        github_url = st.text_input("GitHub URL", placeholder="https://github.com/user/project", label_visibility="collapsed")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔍 Analyze Repository", use_container_width=True, type="primary", disabled=not github_url):
                with st.spinner("Fetching repo and analyzing with Claude Sonnet..."):
                    try:
                        from repo_analyzer import analyze_github_url
                        analysis = analyze_github_url(github_url)
                        st.session_state.analysis = analysis
                        st.session_state.github_url = github_url
                        st.session_state.pipeline_state = "analyzing"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
        with col_b:
            if st.button("⏩ Skip to Dashboard", use_container_width=True):
                st.session_state.pipeline_state = "ready"
                st.rerun()

    st.stop()

elif st.session_state.pipeline_state == "analyzing":
    st.title("🏟️ Model Arena")
    analysis = st.session_state.analysis

    st.markdown("### Analysis Results")
    st.info(f"**Repository:** {st.session_state.get('github_url', '')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected Task", analysis.get("llm_task", "unknown"))
    with col2:
        st.metric("Primary Language", analysis.get("primary_language", "unknown"))
    with col3:
        st.metric("Generated Prompts", len(analysis.get("suggested_prompts", [])))

    st.markdown(f"**Summary:** {analysis.get('project_summary', '')}")
    st.markdown(f"**Why this task:** {analysis.get('task_rationale', '')}")

    with st.expander("View Generated Prompts", expanded=False):
        for p in analysis.get("suggested_prompts", []):
            st.markdown(f"- **[{p.get('category', '')}]** {p.get('prompt', '')[:120]}")

    st.markdown("---")
    st.markdown("### Recommended Models")
    st.write("Searching HuggingFace for the best models...")

    from model_discovery import search_models
    task = analysis.get("llm_task", "text-generation")
    size = analysis.get("size_constraint", "medium")
    models = search_models(task, size, count=3)

    st.session_state.discovered_models = models

    for i, m in enumerate(models):
        st.markdown(f"**{i+1}. {m['name']}** — {m.get('params', '?')} params ({m['source']})")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ Proceed with these models", use_container_width=True, type="primary"):
            st.session_state.selected_models = models
            st.session_state.pipeline_state = "downloading"
            st.rerun()
    with col_b:
        if st.button("🔄 Re-analyze", use_container_width=True):
            st.session_state.pipeline_state = "input"
            st.rerun()
    with col_c:
        if st.button("⏩ Skip to Dashboard", use_container_width=True):
            st.session_state.pipeline_state = "ready"
            st.rerun()

    st.stop()

elif st.session_state.pipeline_state == "downloading":
    st.title("🏟️ Model Arena")
    st.markdown("### Setting up Docker containers and pulling models...")

    models = st.session_state.selected_models
    analysis = st.session_state.analysis

    # Write generated prompts to prompts.json
    suggested = analysis.get("suggested_prompts", [])
    if suggested:
        # Ensure each prompt has all required fields
        for p in suggested:
            p.setdefault("system_prompt", None)
            p.setdefault("negated_keywords", [])
            p.setdefault("test_cases", [])
            p.setdefault("tags", ["auto-generated"])
            p.setdefault("notes", p.get("rubric_hints", ""))
        save_prompts(suggested)
        st.success(f"Saved {len(suggested)} tailored prompts to prompts.json")

    progress_area = st.empty()
    status_container = st.container()

    with status_container:
        with st.status("Starting containers and pulling models...", expanded=True) as status:
            from container_manager import setup_containers

            st.write(f"Setting up {len(models)} containers...")
            model_pull_names = [m["pull_name"] for m in models]

            models_dict = setup_containers(model_pull_names)

            if models_dict:
                st.session_state.dynamic_models_dict = models_dict
                status.update(label="Containers ready! Models downloaded.", state="complete")
                st.success(f"Ready: {list(models_dict.keys())}")

                # Set up evaluate.py with dynamic models
                from evaluate import set_models
                set_models(models_dict)

                st.session_state.pipeline_state = "ready"
                st.rerun()
            else:
                status.update(label="Container setup failed", state="error")
                st.error("Failed to set up containers. Check Docker Desktop is running.")
                if st.button("🔄 Retry"):
                    st.rerun()

    st.stop()


# ---------------------------------------------------------------------------
# Main Dashboard (shown when pipeline_state == "ready")
# ---------------------------------------------------------------------------

# --- Header ---
st.title("🏟️ Model Arena")

# Detect if we came through the onboarding pipeline
_from_pipeline = bool(st.session_state.dynamic_models_dict)
_pipeline_models = st.session_state.dynamic_models_dict or {}
_analysis = st.session_state.analysis

if _from_pipeline:
    from evaluate import set_models
    set_models(_pipeline_models)
    st.caption(f"Evaluating models for: **{st.session_state.get('github_url', '')}**")
else:
    st.caption("Local LLM Evaluation Platform — Compare models side-by-side with automated metrics")

# --- Show loaded models + run evaluation banner (if from pipeline and no results yet) ---
if _from_pipeline and "eval_ran" not in st.session_state:
    st.markdown("---")
    st.subheader("Loaded Models")
    model_cols = st.columns(len(_pipeline_models))
    for i, (model_name, info) in enumerate(_pipeline_models.items()):
        with model_cols[i]:
            st.metric(f"Model {i+1}", model_name)
            st.caption(f"Container: `{info['container']}` | Port: {info['url'].split(':')[-1]}")

    if _analysis:
        with st.expander("Repository Analysis", expanded=False):
            st.markdown(f"**Summary:** {_analysis.get('project_summary', '')}")
            st.markdown(f"**Task:** {_analysis.get('llm_task', '')} — {_analysis.get('task_rationale', '')}")
            st.markdown(f"**Prompts:** {len(_analysis.get('suggested_prompts', []))} tailored prompts generated")

    st.markdown("---")
    # Determine which categories to evaluate based on detected task
    _task = _analysis.get("llm_task", "") if _analysis else ""
    from repo_analyzer import TASK_TO_CATEGORIES
    _eval_categories = TASK_TO_CATEGORIES.get(_task, None)
    if _eval_categories:
        st.info(f"Task: **{_task}** — evaluating categories: {', '.join(_eval_categories)}")

    pipeline_run_name = st.text_input("Run name (optional)", placeholder="Baseline", key="pipeline_run_name")

    if st.button("🚀 Run Evaluation on These Models", use_container_width=True, type="primary"):
        with st.status("Running evaluation...", expanded=True) as status:
            from evaluate import evaluate_all, save_results
            model_names = list(_pipeline_models.keys())
            st.write(f"Evaluating {len(model_names)} models: {', '.join(model_names)}")
            if _eval_categories:
                st.write(f"Filtering to categories: {', '.join(_eval_categories)}")
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def update_progress(done, total, model, prompt_id):
                progress_bar.progress(done / total)
                progress_text.text(f"{done}/{total}: {model} | {prompt_id}")

            results = evaluate_all(model_names, progress_callback=update_progress, categories=_eval_categories)
            filepath = save_results(results, run_name=pipeline_run_name, repository_url=st.session_state.get("github_url"))
            status.update(label="Evaluation complete!", state="complete")
            st.success(f"Saved {len(results)} results")
            st.session_state.eval_ran = True
            st.rerun()
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 New Repository", use_container_width=True):
        # Reset pipeline state to start fresh
        st.session_state.pipeline_state = "input"
        st.session_state.analysis = None
        st.session_state.selected_models = []
        st.session_state.dynamic_models_dict = None
        if "eval_ran" in st.session_state:
            del st.session_state.eval_ran
        st.rerun()

    sidebar_run_name = st.text_input("Run name (optional)", placeholder="Prompt revision", key="sidebar_run_name")

    if st.button("🚀 Run New Evaluation", use_container_width=True):
        with st.status("Running evaluation...", expanded=True) as status:
            st.write("Checking model availability...")
            from evaluate import wait_for_models, evaluate_all, save_results, set_models
            if _from_pipeline:
                set_models(_pipeline_models)
                available = list(_pipeline_models.keys())
            else:
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

                # Filter by task categories if from pipeline
                _sidebar_categories = None
                if _from_pipeline and _analysis:
                    from repo_analyzer import TASK_TO_CATEGORIES
                    _sidebar_categories = TASK_TO_CATEGORIES.get(_analysis.get("llm_task", ""), None)
                results = evaluate_all(available, progress_callback=update_progress, categories=_sidebar_categories)
                filepath = save_results(results, run_name=sidebar_run_name, repository_url=st.session_state.get("github_url") if _from_pipeline else None)
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

    # --- Add Custom Prompt (from Syed) ---
    with st.expander("➕ Add Custom Prompt"):
        with st.form("add_prompt_form", clear_on_submit=True):
            p_category = st.selectbox("Category", ["Factual", "Reasoning", "Creative", "Coding", "Custom"])
            p_difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
            p_method = st.selectbox("Scoring method", ["keyword", "coherence_only", "exact"])
            p_prompt = st.text_area("Prompt text *", placeholder="Enter your prompt here...")
            p_ref = st.text_area("Reference answer", placeholder="Expected answer (not needed for coherence_only)")
            p_keywords = st.text_input("Expected keywords (comma-separated)", placeholder="word1, word2")
            p_negated = st.text_input("Negated keywords (comma-separated)", placeholder="wrong1, wrong2")
            p_tokens = st.slider("Max tokens", 50, 500, 200)
            p_notes = st.text_input("Notes (optional)", placeholder="Scoring intent or hints")
            submitted = st.form_submit_button("💾 Save Prompt")

            if submitted:
                if not p_prompt.strip():
                    st.error("Prompt text is required.")
                else:
                    short_id = uuid_mod.uuid4().hex[:6]
                    new_prompt = {
                        "id": f"{p_category.lower()}_{short_id}",
                        "category": p_category,
                        "difficulty": p_difficulty,
                        "scoring_method": p_method,
                        "prompt": p_prompt.strip(),
                        "system_prompt": None,
                        "reference_answer": p_ref.strip(),
                        "expected_keywords": [k.strip() for k in p_keywords.split(",") if k.strip()],
                        "negated_keywords": [k.strip() for k in p_negated.split(",") if k.strip()],
                        "test_cases": [],
                        "max_tokens": p_tokens,
                        "tags": ["user-defined"],
                        "rubric_hints": p_notes.strip(),
                        "notes": p_notes.strip(),
                    }
                    prompts = load_prompts()
                    prompts.append(new_prompt)
                    save_prompts(prompts)
                    st.success(f"Saved: {new_prompt['id']}")

    # --- Manage Existing Prompts (from Syed) ---
    with st.expander("📋 Manage Prompts"):
        prompts = load_prompts()
        prompt_df = pd.DataFrame([
            {"id": p["id"], "category": p["category"], "difficulty": p.get("difficulty", ""),
             "prompt": p["prompt"][:60] + "…" if len(p["prompt"]) > 60 else p["prompt"]}
            for p in prompts
        ])
        st.dataframe(prompt_df, use_container_width=True, hide_index=True)
        delete_id = st.selectbox("Delete prompt by ID", [""] + [p["id"] for p in prompts])
        if st.button("🗑️ Delete Selected", disabled=not delete_id):
            remaining = [p for p in prompts if p["id"] != delete_id]
            save_prompts(remaining)
            st.success(f"Deleted {delete_id}")
            st.rerun()

    st.divider()
    st.markdown("**Architecture**")
    st.code("3x Ollama Containers\n→ Evaluation Pipeline\n→ Claude Judge (Haiku)\n→ Streamlit Dashboard", language=None)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY not set — judge scores disabled.")

# Comparison runs before legacy leaderboard early exits, including failed runs.
view = st.radio("Dashboard view", ["Results", "Compare runs"], horizontal=True)
if view == "Compare runs":
    from arena.comparison_ui import render_comparison
    render_comparison(result_files)
    st.stop()

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
df_valid = df[df["response"].astype(str).str.len() > 0].copy()

if df_valid.empty:
    st.error("All evaluations returned errors. Check that models are pulled and containers are healthy.")
    st.stop()

# --- Composite Score ---
model_stats = df_valid.groupby("model").agg({
    "tokens_per_second": "mean",
    "keyword_accuracy": "mean",
    "latency_seconds": "mean",
    "memory_mb": "mean",
}).reset_index()

model_stats["norm_speed"] = normalize_series(model_stats["tokens_per_second"])
model_stats["norm_accuracy"] = normalize_series(model_stats["keyword_accuracy"])
model_stats["norm_memory"] = 1 - normalize_series(model_stats["memory_mb"])
bleu_avg = df_valid.groupby("model")["bleu1_score"].mean()
model_stats = model_stats.merge(bleu_avg.rename("avg_bleu"), left_on="model", right_index=True)
model_stats["norm_bleu"] = normalize_series(model_stats["avg_bleu"])

has_judge_scores = (
    "judge_score" in df_valid.columns
    and df_valid["judge_score"].notna().any()
)

if has_judge_scores:
    judge_avg = df_valid.groupby("model")["judge_score"].mean()
    model_stats = model_stats.merge(judge_avg.rename("avg_judge"), left_on="model", right_index=True)
    model_stats["norm_judge"] = normalize_series(model_stats["avg_judge"])
    model_stats["composite_score"] = (
        0.40 * model_stats["norm_judge"]
        + 0.30 * model_stats["norm_speed"]
        + 0.20 * model_stats["norm_bleu"]
        + 0.10 * model_stats["norm_memory"]
    )
else:
    model_stats["avg_judge"] = None
    model_stats["composite_score"] = (
        0.3 * model_stats["norm_speed"]
        + 0.4 * model_stats["norm_accuracy"]
        + 0.2 * model_stats["norm_bleu"]
        + 0.1 * model_stats["norm_memory"]
    )

model_stats = model_stats.sort_values("composite_score", ascending=False)

# --- Hero Metrics ---
fastest = model_stats.loc[model_stats["tokens_per_second"].idxmax()]
best_overall = model_stats.iloc[0]

if has_judge_scores:
    most_accurate = model_stats.loc[model_stats["avg_judge"].idxmax()]
    accuracy_label = f"{most_accurate['avg_judge']:.0%} judge"
else:
    most_accurate = model_stats.loc[model_stats["keyword_accuracy"].idxmax()]
    accuracy_label = f"{most_accurate['keyword_accuracy']:.0%}"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⚡ Fastest", fastest["model"], f"{fastest['tokens_per_second']:.1f} tok/s")
with col2:
    st.metric("🎯 Most Accurate", most_accurate["model"], accuracy_label)
with col3:
    st.metric("🏆 Best Overall", best_overall["model"], f"Score: {best_overall['composite_score']:.2f}")
with col4:
    st.metric("📊 Evaluations", f"{len(df_valid)}", f"{df_valid['model'].nunique()} models")

st.divider()

# --- Leaderboard ---
st.subheader("📋 Leaderboard")
if has_judge_scores:
    leaderboard = model_stats[["model", "tokens_per_second", "keyword_accuracy", "avg_bleu", "avg_judge", "latency_seconds", "memory_mb", "composite_score"]].copy()
    leaderboard.columns = ["Model", "Avg Tokens/s", "Keyword Acc", "BLEU-1", "Judge Score", "Avg Latency (s)", "Memory (MB)", "Composite"]
    fmt = {"Avg Tokens/s": "{:.1f}", "Keyword Acc": "{:.1%}", "BLEU-1": "{:.4f}", "Judge Score": "{:.1%}", "Avg Latency (s)": "{:.2f}", "Memory (MB)": "{:.0f}", "Composite": "{:.3f}"}
else:
    leaderboard = model_stats[["model", "tokens_per_second", "keyword_accuracy", "avg_bleu", "latency_seconds", "memory_mb", "composite_score"]].copy()
    leaderboard.columns = ["Model", "Avg Tokens/s", "Keyword Acc", "BLEU-1", "Avg Latency (s)", "Memory (MB)", "Composite"]
    fmt = {"Avg Tokens/s": "{:.1f}", "Keyword Acc": "{:.1%}", "BLEU-1": "{:.4f}", "Avg Latency (s)": "{:.2f}", "Memory (MB)": "{:.0f}", "Composite": "{:.3f}"}

leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
leaderboard = leaderboard.reset_index(drop=True)
st.dataframe(leaderboard.style.format(fmt), use_container_width=True, hide_index=True)

st.divider()

# --- Charts Row 1: Speed & Accuracy Radar ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("⚡ Speed Comparison")
    fig_speed = px.bar(
        df_valid.groupby("model")["tokens_per_second"].mean().reset_index(),
        x="model", y="tokens_per_second", color="model",
        color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
        labels={"tokens_per_second": "Tokens/sec", "model": "Model"},
    )
    fig_speed.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0b0", showlegend=False)
    st.plotly_chart(fig_speed, use_container_width=True)

with chart_col2:
    st.subheader("🎯 Accuracy by Category")
    radar_metric = "judge_score" if has_judge_scores else "keyword_accuracy"
    categories = df_valid["category"].unique().tolist()
    models = df_valid["model"].unique().tolist()
    colors = ["#e94560", "#0f3460", "#533483"]

    fig_radar = go.Figure()
    for i, model in enumerate(models):
        model_data = df_valid[df_valid["model"] == model]
        values = [model_data[model_data["category"] == cat][radar_metric].mean() for cat in categories]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=model, line_color=colors[i % len(colors)], opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"), angularaxis=dict(gridcolor="#333")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0b0",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Charts Row 2: Latency & Resources ---
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("⏱️ Latency by Prompt")
    fig_latency = px.bar(
        df_valid, x="prompt_id", y="latency_seconds", color="model", barmode="group",
        color_discrete_sequence=["#e94560", "#0f3460", "#533483"],
        labels={"latency_seconds": "Latency (s)", "prompt_id": "Prompt"},
    )
    fig_latency.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0b0", xaxis_tickangle=-45, legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    st.plotly_chart(fig_latency, use_container_width=True)

with chart_col4:
    st.subheader("💾 Resource Usage")
    resource_data = model_stats[["model", "memory_mb"]].copy()
    cpu_avg = df_valid.groupby("model")["cpu_percent"].mean().reset_index()
    resource_data = resource_data.merge(cpu_avg, on="model")

    fig_resources = go.Figure()
    fig_resources.add_trace(go.Bar(name="Memory (MB)", x=resource_data["model"], y=resource_data["memory_mb"], marker_color="#e94560"))
    fig_resources.add_trace(go.Bar(name="CPU %", x=resource_data["model"], y=resource_data["cpu_percent"], marker_color="#0f3460"))
    fig_resources.update_layout(barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0b0", legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    st.plotly_chart(fig_resources, use_container_width=True)

st.divider()

# --- Claude Judge Heatmap (from Igor) ---
if has_judge_scores:
    st.subheader("🤖 Claude Judge Scores")
    models_list = sorted(df_valid["model"].unique().tolist())
    categories_hm = sorted(df_valid["category"].unique().tolist())

    heatmap_data = []
    for model in models_list:
        row = []
        for cat in categories_hm:
            subset = df_valid[(df_valid["model"] == model) & (df_valid["category"] == cat)]
            val = subset["judge_score"].mean() if not subset.empty and subset["judge_score"].notna().any() else None
            row.append(val)
        heatmap_data.append(row)

    z_matrix = [[v if v is not None else 0 for v in row] for row in heatmap_data]
    text_matrix = [[f"{v:.0%}" if v is not None else "N/A" for v in row] for row in heatmap_data]

    fig_heatmap = px.imshow(
        z_matrix, x=categories_hm, y=models_list,
        color_continuous_scale="RdYlGn", zmin=0, zmax=1,
        text_auto=False, labels={"color": "Judge Score"},
    )
    fig_heatmap.update_traces(text=text_matrix, texttemplate="%{text}")
    fig_heatmap.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0b0",
        xaxis_title="Category", yaxis_title="Model", coloraxis_colorbar=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
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

                # Judge details (from Igor)
                judge_score = row.get("judge_score") if "judge_score" in row.index else None
                if judge_score is not None and not (isinstance(judge_score, float) and np.isnan(judge_score)):
                    st.caption(f"🤖 Judge: {judge_score:.0%}")
                    judge_explanation = row.get("judge_explanation", "")
                    judge_breakdown = row.get("judge_rubric_breakdown", {})
                    if judge_explanation or judge_breakdown:
                        with st.expander("Judge Details", expanded=False):
                            if judge_explanation:
                                st.write(judge_explanation)
                            if judge_breakdown and isinstance(judge_breakdown, dict):
                                for criterion, score in judge_breakdown.items():
                                    st.progress(float(score), text=f"{criterion}: {score:.0%}")

                st.text_area(
                    "Response", value=row["response"], height=200,
                    key=f"{prompt_id}_{model}", disabled=True,
                )

# --- Footer ---
st.divider()
st.caption("Model Arena — Built with Docker Compose, Ollama, Claude Judge, and Streamlit")