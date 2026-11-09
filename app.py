"""MediConnect – AI-powered medication & health reminder service: analytics and live simulation app."""
import streamlit as st

from src.config import LIVE_CSV, SURVEY_CSV
from src.models import classification_metrics, regression_metrics, segment_profiles
from src.ui import get_models, page_setup, sidebar_note

page_setup("Home")
sidebar_note()
M, live = get_models()
df = M.survey

st.title("💊 MediConnect Customer Intelligence")
st.markdown(
    "A WhatsApp-first medication and health reminder service. This app applies four machine-learning algorithms "
    "to a 1,500-response consumer survey. It then runs two live simulations: new survey respondents scored as they "
    "arrive, and the service itself operating hour by hour with costs and profit.")

interested = df.interested.mean()
nps = ((df.recommend_score >= 9).mean() - (df.recommend_score <= 6).mean()) * 100
c = st.columns(5)
c[0].metric("Survey respondents", f"{len(df):,}", f"{df.response_time.min()[:10]} → {df.response_time.max()[:10]}", delta_color="off")
c[1].metric("Would subscribe (Q29 = 4–5)", f"{interested:.1%}")
c[2].metric("Avg spend if subscribing", f"{df.loc[df.interested == 1, 'wtp_monthly_aed'].mean():.0f} AED/mo")
c[3].metric("Services wanted (of 10)", f"{df.n_services_selected.mean():.1f}")
c[4].metric("Net promoter score", f"{nps:.0f}")

st.subheader("The four algorithms at a glance")
cls = classification_metrics(M.cls_test["y"], M.cls_test["p_logit"])
reg = regression_metrics(M.reg_test["y"], M.reg_test["p_gbr"])
prof = segment_profiles(df)
strong = M.rules[(M.rules.support >= 0.15) & (M.rules.confidence >= 0.6) & (M.rules.lift >= 1.15)]
g = st.columns(4)
with g[0]:
    st.markdown("**1 · Classification**  \nWill this person subscribe?")
    st.metric("Accuracy (logistic)", f"{cls['Accuracy']:.1%}", f"ROC-AUC {cls['ROC-AUC']:.2f}", delta_color="off")
with g[1]:
    st.markdown("**2 · Clustering**  \nMost, medium and least probable customers")
    st.metric("High-probable segment subscribes", f"{prof.iloc[0]['Would subscribe (Q29 4–5)']:.0%}",
              f"{prof.iloc[0]['Share']:.0%} of respondents", delta_color="off")
with g[2]:
    st.markdown("**3 · Association rules**  \nIf they want 2 services, which 3rd?")
    st.metric("Strong rules found", f"{len(strong)}", "support ≥15%, confidence ≥60%, lift ≥1.15", delta_color="off")
with g[3]:
    st.markdown("**4 · Regression**  \nHow much will they spend?")
    st.metric("R² (gradient boosting)", f"{reg['R²']:.2f}", f"±{reg['MAE (AED)']:.0f} AED mean error", delta_color="off")

st.subheader("Pages")
st.markdown("""
| Page | What it does |
|---|---|
| **Questionnaire** | The 36-question survey, the column each answer fills, and the algorithm it feeds |
| **Classification** | Logistic regression vs random forest, threshold slider, ROC curves, drivers |
| **Clustering** | K-means segments, choose k, silhouette and elbow, segment profiles |
| **Association Rules** | Apriori with adjustable support / confidence / lift, cross-sell finder |
| **Regression** | Willingness to pay: ridge vs gradient boosting, drivers, price-acceptance curve |
| **Live Survey Stream** | 300 new respondents scored live by all four models, checked against their answers |
| **Live Operations** | The service running hour by hour: sign-ups, reminders, refills, churn, costs and profit |
| **Score a Respondent** | Fill in the questionnaire and get all four predictions instantly |
""")

st.subheader("Data")
d1, d2 = st.columns(2)
d1.download_button("Download survey responses (CSV, 1,500 rows)", SURVEY_CSV.read_bytes(),
                   file_name=SURVEY_CSV.name, mime="text/csv", use_container_width=True)
d2.download_button("Download live-stream respondents (CSV, 300 rows)", LIVE_CSV.read_bytes(),
                   file_name=LIVE_CSV.name, mime="text/csv", use_container_width=True)
with st.expander("Preview the survey data"):
    st.dataframe(df.drop(columns=["cluster", "pc1", "pc2"]).head(50), use_container_width=True, hide_index=True)
