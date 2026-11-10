"""Streamlit app: upload and triage a photo, review deferred cases, compare sites, watch the
service's health and read the model card.

    streamlit run src/retinaguard/app/main.py      (or ``make app``; API at ``API_URL``)

Live data comes from the API; per-site results come from ``site_summary.json``
(``python -m retinaguard.app.summary``).
"""

import base64
import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from retinaguard import DISCLAIMER, __version__
from retinaguard.app.client import ApiClient, ApiError
from retinaguard.app.views import (
    DECISION_COLOURS,
    GRADE_NAMES,
    OUTCOMES,
    daily_decisions,
    decision_text,
    health_stats,
    load_summary,
    records_frame,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
OUTCOME_COLOURS = ["#b4b2ab", "#8c8a83", "#d3d1c7", "#86b6ef", "#1c5cab"]


@st.cache_resource
def client() -> ApiClient:
    return ApiClient(API_URL)


def api_call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call the API; on failure show the error and return None."""
    try:
        return getattr(client(), name)(*args, **kwargs)
    except ApiError as error:
        st.error(f"The API at {API_URL} did not answer: {error}")
        return None


def percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def predict_page() -> None:
    st.header("Upload & predict")
    info = api_call("models")
    upload = st.file_uploader("Fundus photo", type=["jpg", "jpeg", "png"])
    left, right = st.columns(2)
    site = left.text_input("Site (optional)", placeholder="e.g. clinic-a")
    lesions_available = bool(info and info["lesions_available"])
    lesions = right.checkbox(
        "Show lesion overlay",
        value=lesions_available,
        disabled=not lesions_available,
        help=None if lesions_available else "no lesion model is loaded",
    )
    if upload is None or not st.button("Triage", type="primary"):
        return
    with st.spinner("Running quality check, grader and policy..."):
        result = api_call("predict", upload.getvalue(), upload.name, site=site, lesions=lesions)
    if result is None:
        return
    decision = result["decision"]
    st.markdown(
        f"<div style='border-left: 8px solid {DECISION_COLOURS.get(decision, '#d3d1c7')};"
        f"padding: 0.4rem 0.8rem; margin: 0.5rem 0'><b>{decision}</b><br>"
        f"{decision_text(decision, result['reason'])}</div>",
        unsafe_allow_html=True,
    )
    photo, details = st.columns([3, 2])
    overlay = result.get("overlay_png_base64")
    if overlay:
        photo.image(base64.b64decode(overlay), caption="Predicted lesions", width="stretch")
        photo.caption(
            "Blue: microaneurysms · green: haemorrhages · yellow: hard exudates · "
            "purple: soft exudates"
        )
        if result.get("lesions"):
            photo.dataframe(
                pd.Series(result["lesions"], name="share of image").map(percent),
                width="stretch",
            )
    else:
        photo.image(upload.getvalue(), caption=upload.name, width="stretch")
    if result["p_referable"] is not None:
        details.metric("Referable probability (calibrated)", percent(result["p_referable"]))
        grades = pd.DataFrame(
            {"grade": GRADE_NAMES, "probability": result["grade_probs"]}
        ).set_index("grade")
        details.bar_chart(grades, horizontal=True, height=220)
    details.metric("Quality", result["quality_label"] or "n/a")
    if result["ood_score"] is not None:
        details.metric("Out-of-distribution score", f"{result['ood_score']:.1f}")
    details.caption(
        f"{result['latency_ms']:.0f} ms · prediction {result['prediction_id']} · "
        + ", ".join(f"{k} {v}" for k, v in result["model_versions"].items())
    )


def review_page() -> None:
    st.header("Review queue")
    st.write(
        "Photos the system did not decide on its own. Uploaded images are not stored, so a "
        "reviewer finds the photo by its time, site and SHA-256 in the record."
    )
    retakes = st.toggle("Include photos sent back for retake")
    records = api_call("predictions", "DEFER_TO_HUMAN", 500) or []
    if retakes:
        records += api_call("predictions", "RETAKE", 500) or []
    frame = records_frame(records)
    st.metric("Waiting for review", len(frame))
    st.dataframe(frame, width="stretch", hide_index=True)


def sites_page() -> None:
    st.header("Site comparison")
    summary = load_summary()
    if summary is None:
        st.info("No site summary yet: run `python -m retinaguard.app.summary`.")
        return
    sites = pd.DataFrame(summary["sites"])
    st.write(
        "Trained on EyePACS (US) only; APTOS, IDRiD and DDR were never used for training or "
        "model selection. Intervals are 95% bootstrap CIs over patients or duplicate groups."
    )
    points = (
        alt.Chart(sites)
        .mark_point(filled=True, size=80, color="#1c5cab")
        .encode(
            x=alt.X("auroc:Q", title="referable AUROC", scale=alt.Scale(zero=False)),
            y=alt.Y("site:N", title=None, sort=None),
            tooltip=["site", alt.Tooltip("auroc:Q", format=".3f"), "images", "referable"],
        )
    )
    bars = points.mark_rule(color="#1c5cab", strokeWidth=2).encode(
        x="auroc_low:Q", x2="auroc_high:Q"
    )
    st.altair_chart(bars + points, width="stretch")

    mix = sites.melt(
        id_vars="site", value_vars=list(OUTCOMES), var_name="outcome", value_name="share"
    )
    chart = (
        alt.Chart(mix)
        .mark_bar(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("share:Q", title="share of photos", axis=alt.Axis(format="%")),
            y=alt.Y("site:N", title=None, sort=None),
            color=alt.Color(
                "outcome:N",
                sort=list(OUTCOMES),
                scale=alt.Scale(domain=list(OUTCOMES), range=OUTCOME_COLOURS),
                legend=alt.Legend(orient="top", title=None),
            ),
            order=alt.Order("outcome_order:Q"),
            tooltip=["site", "outcome", alt.Tooltip("share:Q", format=".1%")],
        )
        .transform_calculate(outcome_order=f"indexof({list(OUTCOMES)}, datum.outcome)")
    )
    st.subheader("What happens to each photo")
    st.altair_chart(chart, width="stretch")
    table = sites.set_index("site")[
        ["images", "referable", "auroc", "fnr", "human_workload", "referable_ece_calibrated"]
    ].rename(
        columns={
            "fnr": "referable eyes cleared",
            "human_workload": "seen by a human",
            "referable_ece_calibrated": "calibration error",
        }
    )
    st.dataframe(table.style.format(precision=3), width="stretch")
    st.caption(
        f"Clearing threshold set for at most {percent(summary['alpha'])} referable eyes cleared "
        "on average, guaranteed only for EyePACS-like images; other sites are measured."
    )


def health_page() -> None:
    st.header("Data health")
    status = api_call("health")
    if status:
        left, right = st.columns(2)
        left.metric("Models loaded", "yes" if status["models_loaded"] else "no")
        right.metric("Database", "ok" if status["database"] else "down")
    records = api_call("predictions", None, 500) or []
    stats = health_stats(records)
    if not stats:
        st.info("No predictions yet.")
        return
    columns = st.columns(4)
    columns[0].metric("Predictions (last 500)", int(stats["predictions"]))
    columns[1].metric("Retake rate", percent(stats["retake_rate"]))
    columns[2].metric("Deferral rate", percent(stats["defer_rate"]))
    columns[3].metric("Latency p95", f"{stats['latency_p95_ms']:.0f} ms")
    daily = daily_decisions(records)
    chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("predictions:Q", title="predictions"),
            color=alt.Color(
                "decision:N",
                scale=alt.Scale(
                    domain=list(DECISION_COLOURS), range=list(DECISION_COLOURS.values())
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["day:T", "decision", "predictions"],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.subheader("Drift reports")
    reports = pd.DataFrame(api_call("drift") or [])
    if reports.empty:
        st.info(
            "No drift reports yet: `python -m retinaguard.monitoring.batches report` (or the daily "
            "Dagster schedule) writes one per site and day."
        )
    else:
        alerted = reports["alert"].notna()
        if alerted.any():
            for _, row in reports[alerted].iterrows():
                st.error(f"{row['day']} · {row['site']}: {row['alert']}")
        else:
            st.success(
                "No batch above the alert rule (retake, deferral or OOD rate over 2x reference)."
            )
        st.dataframe(
            reports.drop(columns=["created_at"]).style.format(precision=3),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "A rising retake, deferral or error rate, or a shift in the mean referable probability "
        f"(now {stats['mean_p_referable']:.2f}), is the first sign that incoming photos changed."
    )


def model_card_page() -> None:
    st.header("Model card")
    st.warning(DISCLAIMER)
    summary = load_summary()
    info = api_call("models")
    st.markdown(
        """
**Intended use.** Research and teaching: triage of colour fundus photos for referable diabetic
retinopathy (ICDR grade 2 or worse), with a human in the loop for every uncertain case.
Not for diagnosis or any clinical decision.

**Chain.** A quality model (EfficientNet-B0, EyeQ labels) sends ungradable photos back; a
grader (ConvNeXt-Tiny, EyePACS) scores referable retinopathy; its probability is temperature
scaled; an out-of-distribution check and two thresholds decide REFER, NO_REFER or
DEFER_TO_HUMAN. The clearing threshold comes from conformal risk control. Any error defers.

**Training data.** EyePACS (US) for the grader, EyeQ quality labels on EyePACS for the quality
model. APTOS (India), IDRiD (India) and DDR (China) are external tests only.

**Known limitations.** End-stage disease (vitreous haemorrhage, fibrosis) is sometimes graded
as no retinopathy. Calibration is fitted on EyePACS and does not transfer to every site. The
quality check asks for more retakes on cameras it has not seen. Photos of one eye of one
patient at one time; no follow-up, no other eye diseases.
"""
    )
    if summary:
        st.subheader("Headline results")
        sites = pd.DataFrame(summary["sites"]).set_index("site")
        st.dataframe(
            sites[
                ["images", "auroc", "auroc_low", "auroc_high", "fnr", "human_workload"]
            ].style.format(precision=3),
            width="stretch",
        )
        quality = summary["quality"]
        st.write(
            f"Quality model: macro-F1 {quality['macro_f1']:.3f} "
            f"({quality['macro_f1_low']:.3f} to {quality['macro_f1_high']:.3f}) on the EyePACS "
            f"test split; {percent(quality['ddr_ungradable_recall'])} of DDR's ungradable photos "
            "flagged."
        )
    if info:
        st.subheader("Loaded models")
        st.json({"versions": info["versions"], "thresholds": info["thresholds"]})
    st.caption(f"retinaguard {__version__}")


def main() -> None:
    st.set_page_config(page_title="RetinaGuard", page_icon=":material/visibility:", layout="wide")
    st.sidebar.caption(DISCLAIMER)
    pages = [
        st.Page(
            predict_page, title="Upload & predict", icon=":material/upload:", url_path="predict"
        ),
        st.Page(review_page, title="Review queue", icon=":material/fact_check:", url_path="review"),
        st.Page(sites_page, title="Site comparison", icon=":material/public:", url_path="sites"),
        st.Page(
            health_page, title="Data health", icon=":material/monitor_heart:", url_path="health"
        ),
        st.Page(
            model_card_page, title="Model card", icon=":material/description:", url_path="card"
        ),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":  # streamlit runs this file as __main__
    main()
