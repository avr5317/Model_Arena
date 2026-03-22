"""Imported lazily by the existing Streamlit entry point."""
from pathlib import Path
import pandas as pd
import streamlit as st
from .comparison import compare_runs
from .reporting import to_csv, to_markdown
from .run_store import load_run


def render_comparison(result_files):
    st.subheader("Compare evaluation runs")
    st.caption("Choose a baseline and a later run. Deltas use paired raw metrics, not leaderboard scores.")
    if len(result_files) < 2:
        st.info("Save at least two evaluations to compare runs. Existing JSON results are supported.")
        return
    runs = {}
    for path in result_files:
        try:
            runs[path] = load_run(path)
        except (OSError, ValueError) as exc:
            st.warning(f"Skipped {Path(path).name}: {exc}")
    if len(runs) < 2:
        st.info("At least two valid result files are needed.")
        return
    paths = list(runs)
    def label(path):
        return f"{runs[path]['name']} · {Path(path).name}"
    col1, col2 = st.columns(2)
    with col1:
        baseline = st.selectbox("Baseline run", paths, index=1, format_func=label, key="compare_baseline")
    with col2:
        current = st.selectbox("Current run", [p for p in paths if p != baseline], format_func=label, key="compare_current")
    report = compare_runs(runs[baseline], runs[current])
    for warning in report["warnings"]:
        st.warning(warning)
    st.markdown("**Coverage and failures**")
    st.dataframe(pd.DataFrame(report["summaries"]), hide_index=True, use_container_width=True)
    st.markdown("**Paired metric means**")
    st.caption("Delta = current − baseline. Missing metrics stay blank. Failed responses are counted above and excluded from paired means. Sample counts can differ by metric.")
    st.dataframe(pd.DataFrame(report["metrics"]), hide_index=True, use_container_width=True)
    left, middle, right = st.columns(3)
    with left:
        st.download_button("Download metrics CSV", to_csv(report["metrics"]), "comparison-metrics.csv", "text/csv", key="compare_csv")
    with middle:
        st.download_button("Download report", to_markdown(report), "comparison.md", "text/markdown", key="compare_md")
    with right:
        st.download_button("Download failures CSV", to_csv(report["summaries"]), "comparison-coverage.csv", "text/csv", key="compare_coverage")
    if report["pairs"]:
        st.markdown("**Inspect matched responses**")
        selected = st.selectbox("Model / prompt", range(len(report["pairs"])),
                                format_func=lambda i: " / ".join(report["pairs"][i][0]), key="compare_pair")
        _, before, after = report["pairs"][selected]
        for col, name, row in zip(st.columns(2), ["Baseline", "Current"], [before, after]):
            with col:
                st.markdown(f"**{name}**")
                if row.get("error"):
                    st.error(str(row["error"]))
                st.text_area("Response", str(row.get("response") or ""), height=220,
                             disabled=True, key=f"compare_response_{name}_{baseline}_{current}_{selected}")
                st.write({"judge_score": row.get("judge_score"), "judge_explanation": row.get("judge_explanation", "")})
    with st.expander("Recorded run metadata"):
        st.json({"baseline": runs[baseline]["metadata"], "current": runs[current]["metadata"]})
