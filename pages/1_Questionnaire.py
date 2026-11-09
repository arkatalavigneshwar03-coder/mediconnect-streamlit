import pandas as pd
import streamlit as st

from src.questionnaire import ALGORITHM_MAP, QUESTIONS, SECTIONS, to_markdown
from src.ui import page_setup, sidebar_note

page_setup("Questionnaire", "📋")
sidebar_note()

st.title("📋 Survey questionnaire")
st.markdown("36 questions plus a 10-service Yes/No list, about 8 minutes. Every answer fills one dataset column, "
            "so responses go straight into the four models.")
st.info("Screening: *Do you, or someone you help care for, take any medicine regularly or have recurring doctor "
        "or lab appointments?* Yes / No — the survey ends if No.")

st.subheader("Question-to-algorithm map")
st.dataframe(pd.DataFrame(ALGORITHM_MAP, columns=["Algorithm", "Target", "Questions used as inputs", "Method"]),
             hide_index=True, use_container_width=True)

for code, title, why in SECTIONS:
    with st.expander(f"Section {code} – {title}", expanded=(code == "A")):
        st.caption(why)
        rows = [{"#": q["id"], "Question": q["text"],
                 "Answer options": " · ".join(lbl for _, lbl in q.get("options", [])) or "Number",
                 "Column": q["col"]} for q in QUESTIONS if q["sec"] == code]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.download_button("Download questionnaire (Markdown)", to_markdown(), file_name="mediconnect_questionnaire.md",
                   mime="text/markdown")
