"""The MediConnect survey questionnaire. Each question maps to one dataset column.

kind: 'cat' (stored as text), 'ord' (stored as the code number), 'num' (free number),
      'multi' (several 1/0 columns), 'yesno' (1/0).
"""

SECTIONS = [
    ("A", "Respondent profile", "Who is buying and who is the patient; drives segmentation and spending power."),
    ("B", "Health profile", "How much the patient needs reminders."),
    ("C", "Current behaviour and pain points", "How often the problem happens and how well it is solved today."),
    ("D", "Digital readiness and channel", "Whether a WhatsApp-first service can reach the respondent."),
    ("E", "Interest in MediConnect services", "Yes/No per service; the basket used for association rules."),
    ("F", "Purchase intent, spend and buying behaviour", "Classification target (Q29), regression target (Q31) and how people buy."),
]

L5 = [(1, "1 – Very low"), (2, "2"), (3, "3"), (4, "4"), (5, "5 – Very high")]

QUESTIONS = [
    # A
    dict(sec="A", id="Q1", col="respondent_role", text="Who are you answering for?", kind="cat",
         options=[("Myself", "Myself"), ("Caregiver for parent/relative", "A parent or elderly relative I care for"), ("Both", "Both")]),
    dict(sec="A", id="Q2", col="patient_age_group", text="Age of the person who takes the medicines", kind="ord",
         options=[(1, "18–29"), (2, "30–44"), (3, "45–59"), (4, "60–74"), (5, "75+")]),
    dict(sec="A", id="Q3", col="gender", text="Gender of the patient", kind="cat",
         options=[(v, v) for v in ["Male", "Female", "Prefer not to say"]]),
    dict(sec="A", id="Q4", col="emirate", text="Emirate of residence", kind="cat",
         options=[(v, v) for v in ["Dubai", "Abu Dhabi", "Sharjah", "Northern Emirates"]]),
    dict(sec="A", id="Q5", col="income_band", text="Monthly household income (AED)", kind="ord",
         options=[(1, "Under 5,000"), (2, "5,000–9,999"), (3, "10,000–19,999"), (4, "20,000–34,999"), (5, "35,000+")]),
    dict(sec="A", id="Q6", col="employment", text="Your employment status", kind="cat",
         options=[(v, v) for v in ["Employed", "Self-employed", "Retired", "Homemaker", "Other"]]),
    dict(sec="A", id="Q7", col="living_arrangement", text="The patient lives…", kind="cat",
         options=[(v, v) for v in ["Alone", "With spouse only", "With family", "With hired caregiver"]]),
    dict(sec="A", id="Q8", col="insurance", text="Health insurance", kind="cat",
         options=[(v, v) for v in ["No insurance", "Basic", "Enhanced/premium"]]),
    # B
    dict(sec="B", id="Q9", col="n_conditions", text="How many long-term conditions does the patient have?", kind="ord",
         options=[(0, "0"), (1, "1"), (2, "2"), (3, "3 or more")]),
    dict(sec="B", id="Q10", col="cond_*", text="Which conditions? (select all)", kind="multi",
         options=[("cond_diabetes", "Diabetes"), ("cond_hypertension", "Hypertension"), ("cond_heart", "Heart disease"),
                  ("cond_cholesterol", "High cholesterol"), ("cond_thyroid", "Thyroid"), ("cond_asthma", "Asthma/COPD"),
                  ("cond_other", "Other")]),
    dict(sec="B", id="Q11", col="n_medicines", text="How many different medicines are taken daily?", kind="ord",
         options=[(0, "0"), (1, "1–2"), (2, "3–4"), (3, "5–6"), (4, "7+")]),
    dict(sec="B", id="Q12", col="doses_per_day", text="How many times a day are doses taken?", kind="ord",
         options=[(1, "1"), (2, "2"), (3, "3"), (4, "4+")]),
    dict(sec="B", id="Q13", col="doctor_visits", text="Doctor visits in the last 12 months", kind="ord",
         options=[(0, "0"), (1, "1–2"), (2, "3–5"), (3, "6+")]),
    dict(sec="B", id="Q14", col="lab_tests", text="Lab tests in the last 12 months", kind="ord",
         options=[(0, "0"), (1, "1–2"), (2, "3–5"), (3, "6+")]),
    dict(sec="B", id="Q15", col="med_spend_aed", text="Monthly out-of-pocket spend on medicines (AED)", kind="num"),
    # C
    dict(sec="C", id="Q16", col="missed_doses_week", text="In a typical week, how often is a dose missed or taken late?", kind="ord",
         options=[(0, "Never"), (1, "1–2 times"), (2, "3–5 times"), (3, "6+ times")]),
    dict(sec="C", id="Q17", col="missed_appointments", text="Doctor appointments missed or forgotten in the last 12 months", kind="ord",
         options=[(0, "0"), (1, "1"), (2, "2"), (3, "3+")]),
    dict(sec="C", id="Q18", col="ran_out_refill", text="Times medicine ran out before a refill in the last 6 months", kind="ord",
         options=[(0, "Never"), (1, "Once"), (2, "2–3 times"), (3, "4+")]),
    dict(sec="C", id="Q19", col="current_method", text="Main way you remember medicines today", kind="cat",
         options=[(v, v) for v in ["Memory only", "Phone alarm", "Family member reminds", "Pill organiser box", "Another app"]]),
    dict(sec="C", id="Q20", col="current_satisfaction", text="How satisfied are you with that method?", kind="ord", options=L5),
    dict(sec="C", id="Q21", col="worry_level", text="How worried are you about the effects of missed doses?", kind="ord", options=L5),
    # D
    dict(sec="D", id="Q22", col="smartphone", text="Does the patient use a smartphone?", kind="ord",
         options=[(2, "Own phone"), (1, "Shared family phone"), (0, "No")]),
    dict(sec="D", id="Q23", col="whatsapp_use", text="How often does the patient use WhatsApp?", kind="ord",
         options=[(3, "Several times a day"), (2, "Daily"), (1, "Weekly"), (0, "Rarely or never")]),
    dict(sec="D", id="Q24", col="tech_comfort", text="Comfort using apps and chat services", kind="ord", options=L5),
    dict(sec="D", id="Q25", col="preferred_channel", text="Preferred reminder channel", kind="cat",
         options=[(v, v) for v in ["WhatsApp text", "WhatsApp voice note", "SMS", "App notification", "Automated phone call"]]),
    dict(sec="D", id="Q26", col="preferred_language", text="Preferred language", kind="cat",
         options=[(v, v) for v in ["English", "Arabic", "Hindi/Urdu", "Malayalam", "Tagalog", "Other"]]),
    dict(sec="D", id="Q27", col="data_trust", text="Comfort sharing medicine and health details with a digital service", kind="ord", options=L5),
    dict(sec="D", id="Q28", col="used_health_app", text="Used any health app in the last 12 months?", kind="yesno",
         options=[(1, "Yes"), (0, "No")]),
    # E
    dict(sec="E", id="E1–E10", col="svc_*", text="Would you use this service if MediConnect offered it?", kind="multi",
         options=[("svc_med_reminder", "E1 Medicine dose reminders on WhatsApp"),
                  ("svc_appt_reminder", "E2 Doctor appointment reminders"),
                  ("svc_lab_reminder", "E3 Lab test reminders"),
                  ("svc_refill_alert", "E4 Prescription refill alerts"),
                  ("svc_rx_scan", "E5 Send a prescription photo; reminders set up automatically"),
                  ("svc_caregiver_alert", "E6 Alert a family member when a dose is missed"),
                  ("svc_pharmacy_delivery", "E7 Refill ordering with home delivery from a partner pharmacy"),
                  ("svc_vitals_log", "E8 Log blood pressure or sugar by WhatsApp, with trend charts"),
                  ("svc_teleconsult", "E9 Teleconsultation booking"),
                  ("svc_doctor_report", "E10 Monthly adherence report shared with your doctor")]),
    # F
    dict(sec="F", id="Q29", col="purchase_intent", text="If MediConnect cost a fair monthly fee, how likely are you to subscribe?", kind="ord",
         options=[(1, "1 – Very unlikely"), (2, "2"), (3, "3"), (4, "4"), (5, "5 – Very likely")]),
    dict(sec="F", id="Q30", col="recommend_score", text="How likely are you to recommend it to a friend? (0–10)", kind="num"),
    dict(sec="F", id="Q31", col="wtp_monthly_aed", text="The most you would pay per month for the services you chose (AED)", kind="num"),
    dict(sec="F", id="Q32", col="preferred_plan", text="Which plan would you choose?", kind="cat",
         options=[(v, v) for v in ["Free basic + paid add-ons", "Monthly", "Annual (discounted)", "Family plan", "Pay per service"]]),
    dict(sec="F", id="Q33", col="purchase_channel", text="How would you prefer to sign up?", kind="cat",
         options=[(v, v) for v in ["Directly on WhatsApp", "Through pharmacy", "Doctor/clinic recommendation", "Through insurer", "Through employer"]]),
    dict(sec="F", id="Q34", col="payer", text="Who would pay?", kind="cat",
         options=[(v, v) for v in ["Myself", "Children/family", "Insurance", "Employer"]]),
    dict(sec="F", id="Q35", col="payment_method", text="Preferred payment method", kind="cat",
         options=[(v, v) for v in ["Card", "Apple Pay / Google Pay", "Added to pharmacy bill", "Bank transfer"]]),
    dict(sec="F", id="Q36", col="top_factor", text="What matters most when choosing?", kind="cat",
         options=[(v, v) for v in ["Price", "Ease of use", "Data privacy", "Doctor endorsement", "Arabic support", "Family involvement"]]),
]

ALGORITHM_MAP = [
    ("Classification", "interested (Q29 = 4 or 5)", "Q1–Q28", "Logistic regression and random forest"),
    ("Clustering", "none (unsupervised)", "Q11, Q16–Q18, Q20, Q21, Q23, Q24, Q27", "K-means; segments ranked by subscribe rate"),
    ("Association rules", "E1–E10 basket", "Section E", "Apriori: support, confidence, lift"),
    ("Regression", "wtp_monthly_aed (Q31)", "Q1–Q28, Section E, Q29, Q32", "Ridge and gradient boosting"),
]


def to_markdown() -> str:
    out = ["# MediConnect – Consumer Survey Questionnaire", "",
           "Screening: *Do you, or someone you help care for, take any medicine regularly or have recurring doctor "
           "or lab appointments?* Yes / No (end survey if No).", ""]
    for code, title, why in SECTIONS:
        out += [f"## Section {code} – {title}", "", why, "", "| # | Question | Answer options | Column |", "|---|---|---|---|"]
        for q in QUESTIONS:
            if q["sec"] != code:
                continue
            opts = " · ".join(lbl for _, lbl in q.get("options", [])) or "Number"
            out.append(f"| {q['id']} | {q['text']} | {opts} | `{q['col']}` |")
        out.append("")
    out += ["## Question-to-algorithm map", "", "| Algorithm | Target | Inputs | Method |", "|---|---|---|---|"]
    out += [f"| {a} | {t} | {i} | {m} |" for a, t, i, m in ALGORITHM_MAP]
    return "\n".join(out)
