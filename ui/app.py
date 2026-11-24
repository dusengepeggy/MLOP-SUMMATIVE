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
if not API_URL.startswith(("http://", "https://")):  # e.g. Render's fromService gives a bare host
    API_URL = f"https://{API_URL}"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "samples"
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


METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
                "roc_auc": "ROC-AUC", "loss": "Loss"}


def metric_row(metrics, reference=None):
    """Show the test-set metrics; with `reference`, also show the change versus it."""
    cols = st.columns(len(METRIC_NAMES))
    for col, (key, name) in zip(cols, METRIC_NAMES.items()):
        value = metrics.get(key) if metrics else None
        delta = None
        if value is not None and reference and reference.get(key) is not None:
            delta = f"{value - reference[key]:+.3f}"
        col.metric(name, f"{value:.3f}" if value is not None else "–", delta,
                   delta_color="inverse" if key == "loss" else "normal")


@st.cache_data
def load_eda():
    if not EDA_STATS_PATH.exists():
        return None
    return pd.DataFrame(json.loads(EDA_STATS_PATH.read_text()))


# ---------------------------------------------------------------- pages
def page_predict():
    st.header("Predict a leaf")
    st.write("Upload a photo of a single bean leaf. The model classifies it as angular leaf spot, bean rust or healthy.")
    source = st.radio("Image source", ["Upload a photo", "Use a held-out test image"], horizontal=True)
    if source == "Upload a photo":
        file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
        if not file:
            return
        name, data, mime, true_label = file.name, file.getvalue(), file.type, None
    else:
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        label_of = lambda p: next((c for c in CLASSES if p.stem.startswith(c)), None)  # noqa: E731
        pick = st.selectbox("Test image (true class in brackets)", samples,
                            format_func=lambda p: f"{p.name}  [{PRETTY.get(label_of(p), '?')}]")
        name, data, mime, true_label = pick.name, pick.read_bytes(), "image/jpeg", label_of(pick)
    left, right = st.columns([1, 1])
    left.image(data, caption=name, width="stretch")
    with right:
        with st.spinner("Classifying..."):
            r = api_post("/predict", files={"file": (name, data, mime)})
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
            return
        res = r.json()
        st.subheader(PRETTY.get(res["label"], res["label"]))
        if true_label:
            if res["label"] == true_label:
                st.success(f"Correct: the true class is {PRETTY[true_label]}.")
            else:
                st.error(f"Incorrect: the true class is {PRETTY[true_label]}.")
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
    st.header("Data insights")
    st.caption("Three features of the dataset, what they show, and how they shaped the pipeline (summary at the bottom).")

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

    st.subheader("How these features drove downstream decisions")
    st.markdown(
        "| Finding | Decision in the pipeline |\n|---|---|\n"
        "| Classes are balanced (ratio ≈ 1.02) | Plain accuracy is meaningful, but macro precision/recall/F1 are reported too; "
        "inverse-frequency **class weights** are applied at every (re)training because user uploads can unbalance the data |\n"
        "| Colour separates healthy from diseased *on average* but overlaps heavily | A **CNN on raw pixels** (MobileNetV2 transfer "
        "learning) instead of hand-crafted colour features; texture and lesion shape are needed |\n"
        "| Disease cues are small lesions | Input kept at **224×224** (MobileNetV2's native size) rather than smaller, and the top "
        "backbone layers are **fine-tuned** to adapt ImageNet features to lesion textures |\n"
        "| No lighting shortcut, but leaves appear at every angle and zoom | **Augmentation**: flips, rotation, zoom and contrast jitter |\n"
        "| Leaves are photographed in the field with varied backgrounds | Uploads are validated, converted to RGB and resized "
        "before they enter retraining |"
    )


def page_upload():
    st.header("Upload training data")
    st.write("Add labelled leaf images for the next retraining. Each file is validated, **saved to the database** "
             "(SQLite) and **preprocessed** right away (RGB → 224×224 → feature vector from the trained backbone).")
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

    st.subheader("Uploads in the database")
    up = api_get("/uploads")
    if up and up["uploads"]:
        df = pd.DataFrame(up["uploads"])
        df["label"] = df["label"].map(PRETTY)
        st.dataframe(df[["id", "filename", "label", "status", "width", "height", "uploaded_at", "retrain_id"]],
                     hide_index=True, width="stretch")
        st.caption("Status: preprocessed = waiting for the next retrain; used = consumed by retrain #retrain_id.")
    elif up is not None:
        st.info("No uploads yet.")


def page_retrain():
    st.header("Retrain the model")
    st.write(
        "Retraining fine-tunes **our trained model** (its classifier head, starting from the current weights) on the "
        "original training data plus every uploaded image, with Adam, L2 regularisation, class weights and early "
        "stopping. The **held-out test set** (128 images, never trained on) is scored before and after, so the "
        "effect of the new data is measured on the same images."
    )
    status = api_get("/status") or {}
    c1, c2 = st.columns(2)
    c1.metric("New uploads waiting", status.get("pending_uploads", "–"))
    c2.metric("Current model version", status.get("model_version", "–"))
    epochs = st.slider("Maximum epochs (early stopping may finish sooner)", 5, 200, 30, step=5)
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

    st.subheader("Retraining history")
    hist = api_get("/retrain/history")
    if hist and hist["retrains"]:
        rows = [{
            "run": r["id"], "trigger": r["trigger"], "status": r["status"], "new images": r["n_new"],
            "version": f"v{r['version_before']} → v{r['version_after']}" if r["version_after"] else f"v{r['version_before']}",
            "accuracy before": (r["metrics_before"] or {}).get("accuracy"),
            "accuracy after": (r["metrics_after"] or {}).get("accuracy"),
            "loss before": (r["metrics_before"] or {}).get("loss"),
            "loss after": (r["metrics_after"] or {}).get("loss"),
            "finished": (r["finished_at"] or "")[:19],
        } for r in hist["retrains"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif hist is not None:
        st.info("No retraining runs yet.")


def show_job_progress(job):
    bar = st.progress(0.0, text="Starting retrain...")
    while job.get("status") == "running":
        p = job.get("progress", {})
        total = max(p.get("total", 1), 1)
        loss = f" · validation loss {p['val_loss']:.4f}" if "val_loss" in p else ""
        bar.progress(min(p.get("epoch", 0) / total, 1.0), text=f"Epoch {p.get('epoch', 0)}/{total}{loss}")
        time.sleep(1)
        job = api_get("/retrain/status") or {}
    bar.progress(1.0, text="Finished")
    st.session_state.pop("watch_job", None)
    show_job_result(job)


def show_job_result(job):
    if job.get("status") == "failed":
        st.error(f"Last retrain failed: {job.get('error')}")
        return
    st.success(f"Last retrain ({job.get('trigger')}) finished at {job.get('finished_at', '')[:19]}: "
               f"model v{job.get('version')}, {job.get('n_new_uploads', 0)} new image(s), "
               f"{job.get('epochs_run', '?')} epoch(s) run (early stopping on validation loss).")
    st.markdown("**Held-out test set: before retraining**")
    metric_row(job.get("metrics_before"))
    st.markdown("**Held-out test set: after retraining** (change vs. before)")
    metric_row(job.get("metrics"), reference=job.get("metrics_before"))


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
    "Data insights": page_visualizations,
    "Upload data": page_upload,
    "Retrain": page_retrain,
    "System": page_system,
}

st.sidebar.title("🌱 Bean Leaf Doctor")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(f"API: {API_URL}")
PAGES[choice]()
