import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models import classification_metrics, roc_points
from src.ui import get_models, hbar, page_setup, sidebar_note, style

page_setup("Classification", "🎯")
sidebar_note()
M, _ = get_models()
y, p_lr, p_rf = M.cls_test["y"], M.cls_test["p_logit"], M.cls_test["p_forest"]

st.title("🎯 Classification: will this person subscribe?")
st.markdown(f"Target: **interested** = 1 when Q29 is 4 or 5 ({y.mean():.0%} of respondents). Inputs: Q1–Q28. "
            f"Models are tested on {len(y)} held-out responses (25%).")

thr = st.slider("Decision threshold: predict 'will subscribe' when probability is at least", 0.10, 0.90, 0.50, 0.05)
m_lr, m_rf = classification_metrics(y, p_lr, thr), classification_metrics(y, p_rf, thr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy (logistic)", f"{m_lr['Accuracy']:.1%}")
c2.metric("Precision", f"{m_lr['Precision']:.1%}", help="Of people predicted to subscribe, the share who said 4–5")
c3.metric("Recall", f"{m_lr['Recall']:.1%}", help="Of people who said 4–5, the share the model found")
c4.metric("ROC-AUC", f"{m_lr['ROC-AUC']:.3f}")

left, mid, right = st.columns([1, 1, 1.2])
with left:
    st.markdown("**Model comparison**")
    keys = ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]
    st.dataframe(pd.DataFrame({"Metric": keys, "Logistic regression": [m_lr[k] for k in keys],
                               "Random forest": [m_rf[k] for k in keys]}).round(3),
                 hide_index=True, use_container_width=True)
    st.markdown("**Confusion matrix (logistic)**")
    cm = m_lr["confusion"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted no", "Predicted yes"], y=["Actually no", "Actually yes"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
                               hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 260, legend=False), use_container_width=True)
with mid:
    st.markdown("**ROC curves**")
    fig = go.Figure()
    for name, p, col in [("Logistic", p_lr, "#2a78d6"), ("Random forest", p_rf, "#eb6834")]:
        fpr, tpr = roc_points(y, p)
        auc = classification_metrics(y, p)["ROC-AUC"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {auc:.2f})", line=dict(color=col, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color="#7a8491", dash="dash")))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    st.plotly_chart(style(fig, 420), use_container_width=True)
with right:
    st.markdown("**What drives subscription** (random-forest importance; ▲/▼ = direction in logistic model)")
    top = M.cls_drivers.head(12)
    st.plotly_chart(hbar(top.feature.tolist(), top.importance.round(4).tolist(),
                         text=[("▲ " if c > 0 else "▼ ") + f"{i:.3f}" for i, c in zip(top.importance, top.logit_coef)]),
                    use_container_width=True)

st.markdown("**Predicted probability by actual answer**")
dist = pd.DataFrame({"Probability": p_lr, "Actual": ["Said 4–5 (interested)" if v else "Said 1–3" for v in y]})
fig = go.Figure()
for lab, col in [("Said 1–3", "#7a8491"), ("Said 4–5 (interested)", "#2a78d6")]:
    fig.add_trace(go.Histogram(x=dist[dist.Actual == lab].Probability, name=lab, marker_color=col, opacity=0.75,
                               xbins=dict(start=0, end=1, size=0.05)))
fig.add_vline(x=thr, line_dash="dash", line_color="#d03b3b", annotation_text=f"threshold {thr:.2f}")
fig.update_layout(barmode="overlay")
fig.update_xaxes(title="Predicted probability of subscribing")
fig.update_yaxes(title="Respondents")
st.plotly_chart(style(fig, 300), use_container_width=True)
