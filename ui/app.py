"""Streamlit UI for the bean leaf disease pipeline.

Run:  streamlit run ui/app.py      (API_URL defaults to http://localhost:8000)
"""
import json
import os
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Bean Leaf Doctor", page_icon="🌱", layout="wide")


def _setting(name, default):
    """Environment variable first (Docker), then Streamlit secrets (Community Cloud)."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml locally
        return default


API_URL = str(_setting("API_URL", "http://localhost:8000")).rstrip("/")
EDA_STATS_PATH = Path(os.getenv("EDA_STATS_PATH", Path(__file__).resolve().parents[1] / "docs" / "eda_stats.json"))
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
PRETTY = {"angular_leaf_spot": "Angular leaf spot", "bean_rust": "Bean rust", "healthy": "Healthy"}
TIMEOUT = 90  # free hosts can take a while to wake up


# ---------------------------------------------------------------- API helpers
def _report(method, path, exc):
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        st.warning("The API is not responding. On the free hosting tier it sleeps when idle and takes "
                   "about 30-60 s to wake up. Please retry shortly.")
    else:
        st.error(f"API error on {method} {path}: {exc}")


def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        _report("GET", path, exc)
        return None


def api_post(path, **kwargs):
    try:
        return requests.post(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        _report("POST", path, exc)
        return None


def metric_row(metrics, label=""):
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(f"{label}{key.replace('_', '-').upper() if key == 'roc_auc' else key.title()}",
                   f"{metrics[key]:.3f}" if metrics and key in metrics else "–")


@st.cache_data
def load_eda():
    if not EDA_STATS_PATH.exists():
        return None
    return pd.DataFrame(json.loads(EDA_STATS_PATH.read_text()))


# ---------------------------------------------------------------- pages
def page_predict():
    st.header("Predict a leaf")
    st.write("Upload a photo of a single bean leaf. The model classifies it as angular leaf spot, bean rust or healthy.")
    file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if not file:
        return
    left, right = st.columns([1, 1])
    left.image(file, caption=file.name, width="stretch")
    with right:
        with st.spinner("Classifying..."):
            r = api_post("/predict", files={"file": (file.name, file.getvalue(), file.type)})
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
            return
        res = r.json()
        st.subheader(PRETTY.get(res["label"], res["label"]))
        st.progress(res["confidence"], text=f"Confidence {res['confidence']:.1%}")
        df = pd.DataFrame({"class": [PRETTY[c] for c in res["probabilities"]], "probability": list(res["probabilities"].values())})
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("class:N", sort="-x", title=None),
            color=alt.Color("class:N", legend=None),
            tooltip=["class", alt.Tooltip("probability:Q", format=".1%")],
        ).properties(height=160)
        st.altair_chart(chart, width="stretch")
        if res["confidence"] < 0.6:
            st.warning("Low confidence: the photo may be blurry, contain several leaves, or show something the model has not seen.")


def page_visualizations():
    st.header("Dataset visualizations")

    st.subheader("1. Class distribution (live from the API)")
    m = api_get("/metrics")
    if m:
        dist = pd.DataFrame({"class": [PRETTY[c] for c in m["class_distribution"]], "images": list(m["class_distribution"].values())})
        st.altair_chart(alt.Chart(dist).mark_bar().encode(x=alt.X("class:N", title=None), y="images:Q", color=alt.Color("class:N", legend=None), tooltip=["class", "images"]).properties(height=280), width="stretch")
        counts = list(m["class_distribution"].values())
        ratio = max(counts) / max(min(counts), 1)
        st.markdown(
            f"**Interpretation.** The largest class has {ratio:.2f}x the images of the smallest. "
            "The original dataset is almost perfectly balanced (about 390 images per class), so the model has no reason to favour one class "
            "and recall should be similar for all three. Uploads change this chart: if one class is uploaded much more than the others, "
            "the training code applies inverse-frequency class weights so minority-class recall does not collapse."
        )

    eda = load_eda()
    if eda is None:
        st.info("EDA stats not found. Run `python -m src.preprocessing` to generate docs/eda_stats.json.")
        return
    eda["class"] = eda["label"].map(PRETTY)

    st.subheader("2. Leaf colour: mean R, G, B per class")
    long = eda.melt(id_vars="class", value_vars=["mean_r", "mean_g", "mean_b"], var_name="channel", value_name="mean value")
    long["channel"] = long["channel"].map({"mean_r": "Red", "mean_g": "Green", "mean_b": "Blue"})
    st.altair_chart(
        alt.Chart(long).mark_boxplot(extent="min-max").encode(
            x=alt.X("class:N", title=None), y=alt.Y("mean value:Q", title="mean pixel value (0-255)"),
            color=alt.Color("channel:N", scale=alt.Scale(domain=["Red", "Green", "Blue"], range=["#d1495b", "#3b9a57", "#3d6fb6"])),
            xOffset="channel:N",
        ).properties(height=320),
        width="stretch",
    )
    g = eda.groupby("label")[["mean_g", "hue"]].mean()
    st.markdown(
        f"**Interpretation.** Healthy leaves are the greenest (mean green {g.loc['healthy', 'mean_g']:.0f} vs "
        f"{g.loc['bean_rust', 'mean_g']:.0f} for rust) and have the highest average hue "
        f"({g.loc['healthy', 'hue']:.0f}° vs about {g.loc['bean_rust', 'hue']:.0f}° for rust), because disease replaces green tissue "
        "with brown/yellow lesions. Bean rust pulls the green channel down the most; angular leaf spot keeps more green but adds red. "
        "Colour is therefore a genuine signal, but the boxes overlap heavily, so colour alone cannot separate the classes; "
        "the CNN has to learn lesion shape and texture too."
    )

    st.subheader("3. Lighting: brightness and contrast per class")
    c1, c2 = st.columns(2)
    for col, feat, title in ((c1, "brightness", "Brightness (mean grey level)"), (c2, "contrast", "Contrast (grey-level std)")):
        col.altair_chart(
            alt.Chart(eda).transform_density(feat, groupby=["class"], as_=[feat, "density"]).mark_area(opacity=0.45).encode(
                x=alt.X(f"{feat}:Q", title=title), y="density:Q", color="class:N"
            ).properties(height=260),
            width="stretch",
        )
    b = eda.groupby("label")[["brightness", "contrast"]].mean()
    st.markdown(
        f"**Interpretation.** Contrast is almost identical across classes ({b['contrast'].min():.0f}-{b['contrast'].max():.0f}), "
        f"and brightness differs only modestly (rust {b.loc['bean_rust', 'brightness']:.0f}, healthy {b.loc['healthy', 'brightness']:.0f}) "
        "with wide overlap. The photos were taken in similar field conditions, so there is no strong lighting shortcut the model could "
        "exploit. Rust images being slightly darker is consistent with dark rust pustules, not with camera settings. "
        "Random contrast augmentation during training further guards against lighting bias."
    )


def page_upload_retrain():
    st.header("Upload data and retrain")
    status = api_get("/status") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending uploads", status.get("pending_uploads", "–"))
    c2.metric("Auto-retrain threshold", status.get("retrain_threshold", "–"))
    c3.metric("Model version", status.get("model_version", "–"))

    st.subheader("Bulk upload")
    with st.form("upload", clear_on_submit=True):
        label = st.selectbox("Class of these images", CLASSES, format_func=PRETTY.get)
        files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Upload")
    if submitted:
        if not files:
            st.warning("Choose at least one image.")
        else:
            with st.spinner(f"Uploading {len(files)} image(s)..."):
                r = api_post("/upload", files=[("files", (f.name, f.getvalue(), f.type)) for f in files], data={"label": label})
            if r is not None and r.status_code == 200:
                body = r.json()
                st.success(f"Saved {body['saved']} image(s) as {PRETTY[label]}. Pending uploads: {body['pending_uploads']}.")
                for msg in body.get("rejected", []):
                    st.warning(f"Rejected: {msg}")
                if body.get("auto_retrain_job_id"):
                    st.info("Upload threshold reached: automatic retraining started.")
            elif r is not None:
                st.error(r.json().get("detail", r.text))

    st.subheader("Retrain")
    epochs = st.slider("Epochs", 1, 10, 2)
    if st.button("Retrain model", type="primary"):
        r = api_post("/retrain", params={"epochs": epochs})
        if r is not None and r.status_code == 200:
            st.session_state["watch_job"] = r.json()["job_id"]
        elif r is not None and r.status_code == 409:
            st.warning("A retrain is already running; showing its progress.")
            st.session_state["watch_job"] = True
        elif r is not None:
            st.error(r.json().get("detail", r.text))

    job = api_get("/retrain/status") or {}
    if job.get("status") == "running" or st.session_state.get("watch_job"):
        show_job_progress(job)
    elif job.get("status") in ("done", "failed"):
        show_job_result(job)


def show_job_progress(job):
    bar = st.progress(0.0, text="Starting retrain...")
    while job.get("status") == "running":
        p = job.get("progress", {})
        total = max(p.get("total", 1), 1)
        acc = f" · val acc {p['val_accuracy']:.3f}" if "val_accuracy" in p else ""
        bar.progress(min(p.get("epoch", 0) / total, 1.0), text=f"Epoch {p.get('epoch', 0)}/{total}{acc}")
        time.sleep(2)
        job = api_get("/retrain/status") or {}
    bar.progress(1.0, text="Finished")
    st.session_state.pop("watch_job", None)
    show_job_result(job)


def show_job_result(job):
    if job.get("status") == "failed":
        st.error(f"Last retrain failed: {job.get('error')}")
        return
    st.success(f"Last retrain ({job.get('trigger')}) finished at {job.get('finished_at', '')[:19]}: model v{job.get('version')}.")
    st.markdown("**Before**")
    metric_row(job.get("metrics_before"))
    st.markdown("**After**")
    metric_row(job.get("metrics"))


def page_system():
    st.header("System status")
    t0 = time.time()
    status = api_get("/status")
    latency = (time.time() - t0) * 1000
    if not status:
        st.error("🔴 API unreachable")
        return
    st.success(f"🟢 API healthy · {latency:.0f} ms · served by {status['hostname']}")
    c1, c2, c3, c4 = st.columns(4)
    up = int(status["uptime_seconds"])
    c1.metric("Uptime", f"{up // 3600}h {up % 3600 // 60}m {up % 60}s")
    c2.metric("Model version", status.get("model_version"))
    c3.metric("Predictions served", status.get("prediction_count"))
    c4.metric("Requests (this replica)", status.get("request_count"))
    st.write(f"**Last trained:** {status.get('trained_at', '–')}")
    st.write(f"**Retrain status:** {status.get('retrain_status')} · **training enabled on this replica:** {status.get('training_enabled')}")

    m = api_get("/metrics")
    if m and m.get("metrics"):
        st.subheader("Current model metrics (test set)")
        metric_row(m["metrics"])
        cm = pd.DataFrame(m["metrics"]["confusion_matrix"], index=[PRETTY[c] for c in CLASSES], columns=[PRETTY[c] for c in CLASSES])
        cm_long = cm.reset_index(names="actual").melt(id_vars="actual", var_name="predicted", value_name="count")
        base = alt.Chart(cm_long).encode(x="predicted:N", y="actual:N")
        st.altair_chart(
            (base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="greens"))) + base.mark_text(fontSize=16).encode(text="count:Q")).properties(height=280),
            width="stretch",
        )
    if st.button("Refresh"):
        st.rerun()


PAGES = {
    "Predict": page_predict,
    "Visualizations": page_visualizations,
    "Upload & Retrain": page_upload_retrain,
    "System": page_system,
}

st.sidebar.title("🌱 Bean Leaf Doctor")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(f"API: {API_URL}")
PAGES[choice]()
