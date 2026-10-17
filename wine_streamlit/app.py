import os

import pandas as pd
import streamlit as st

from wine_client import (CULTIVARS, FEATURE_NAMES, PRESETS, feature_ranges,
                         predict_cloud, predict_local, train_local_model)

st.set_page_config(page_title="Wine cultivar classifier", page_icon="🍷", layout="wide")

RANGES = feature_ranges()
LABELS = {name: name.replace("_", " ").capitalize() for name in FEATURE_NAMES}
LABELS["od280/od315_of_diluted_wines"] = "OD280/OD315 ratio"


@st.cache_resource
def local_model():
    return train_local_model()


def apply_preset():
    values = PRESETS.get(st.session_state.preset)
    if values:
        for name, value in zip(FEATURE_NAMES, values):
            st.session_state[name] = float(value)


# First run: start from the cultivar 1 sample.
if "preset" not in st.session_state:
    st.session_state.preset = "Sample of cultivar 1"
    apply_preset()

# Sidebar: where predictions come from
with st.sidebar:
    st.header("Prediction source")
    mode = st.radio("Run the model on", ["Cloud Function", "This computer"],
                    help="'This computer' trains the same SVM locally, "
                         "so you can use the app without GCP.")
    url, token = "", ""
    if mode == "Cloud Function":
        url = st.text_input("Function URL", value=os.getenv("WINE_FUNCTION_URL", ""),
                            placeholder="https://us-central1-PROJECT.cloudfunctions.net/wine-classifier")
        with st.expander("Function requires authentication?"):
            st.caption("Only needed if you deployed with --no-allow-unauthenticated. "
                       "In Cloud Shell, run `gcloud auth print-identity-token` and paste "
                       "the result. Tokens expire after about an hour.")
            token = st.text_input("Identity token", type="password")

st.title("🍷 Wine cultivar classifier")
st.write("Enter the chemical analysis of a wine to predict which of three "
         "cultivars it came from. The model is an RBF SVM trained on the "
         "UCI Wine dataset.")

st.selectbox("Start from", list(PRESETS) + ["Custom values"],
             key="preset", on_change=apply_preset)

cols = st.columns(3)
for i, name in enumerate(FEATURE_NAMES):
    lo, hi = RANGES[name]
    with cols[i % 3]:
        st.number_input(LABELS[name], key=name, format="%.2f",
                        help=f"Dataset range: {lo:.2f} to {hi:.2f}")

features = [float(st.session_state[name]) for name in FEATURE_NAMES]

if st.button("Predict cultivar", type="primary"):
    result, error = None, None
    if mode == "Cloud Function":
        if not url:
            error = "Add the function URL in the sidebar first."
        else:
            with st.spinner("Calling the Cloud Function (the first call can take ~20s)..."):
                try:
                    status, body = predict_cloud(url, features, token or None)
                    if status == 200 and "predicted_class" in body:
                        result = body
                    elif status in (401, 403):
                        error = (f"HTTP {status}: the function rejected the request. "
                                 "Paste an identity token in the sidebar, or redeploy "
                                 "with --allow-unauthenticated.")
                    else:
                        error = f"HTTP {status}: {body.get('error', body)}"
                except Exception as e:
                    error = f"Couldn't reach the function: {e}"
    else:
        scaler, model = local_model()
        result = predict_local(scaler, model, features)

    if error:
        st.error(error)
    if result:
        probs = result["prediction_confidences"][0]
        pred = result["predicted_class"]
        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted", CULTIVARS[pred], f"{probs[pred]:.1%} confidence",
                      delta_color="off")
        with right:
            st.bar_chart(pd.DataFrame({"Probability": probs}, index=CULTIVARS),
                         horizontal=True)
        with st.expander("Raw response"):
            st.json(result)
