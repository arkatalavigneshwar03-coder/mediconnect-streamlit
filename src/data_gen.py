"""
MediConnect – synthetic consumer survey generator.

Every column matches a question in the MediConnect survey questionnaire.
Answers are generated from latent traits (health need, digital readiness,
trust, spending power) so the data carries realistic patterns for:
  - classification  (interested = Q29 >= 4)
  - clustering      (need x digital-readiness segments)
  - association     (service baskets E1-E10 with built-in co-purchase links)
  - regression      (wtp_monthly_aed, Q31)

Usage (from the repo root):  python -m src.data_gen
  -> data/mediconnect_survey_responses.csv (1,500 respondents)
  -> data/mediconnect_live_stream.csv     (300 new respondents for the live stream)
"""
import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)


def reset_rng(seed=SEED):
    global rng
    rng = np.random.default_rng(seed)


def pick(options, probs, size=None):
    p = np.asarray(probs, dtype=float)
    return rng.choice(options, size=size, p=p / p.sum())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip_int(x, lo, hi):
    return int(np.clip(round(x), lo, hi))


CONDITIONS = ["diabetes", "hypertension", "heart", "cholesterol", "thyroid", "asthma", "other"]
COND_W = [0.34, 0.40, 0.14, 0.30, 0.12, 0.10, 0.10]
SERVICES = ["svc_med_reminder", "svc_appt_reminder", "svc_lab_reminder", "svc_refill_alert",
            "svc_rx_scan", "svc_caregiver_alert", "svc_pharmacy_delivery", "svc_vitals_log",
            "svc_teleconsult", "svc_doctor_report"]


def one_respondent(i, start, days):
    r = {"respondent_id": f"MC{i:05d}"}
    # timestamp: weekday-heavy, daytime/evening
    d = int(rng.integers(0, days))
    ts = start + pd.Timedelta(days=d) + pd.Timedelta(minutes=int(rng.integers(8 * 60, 23 * 60)))
    r["response_time"] = ts.strftime("%Y-%m-%d %H:%M")

    # ---------- Section A: profile ----------
    role = pick(["Myself", "Caregiver for parent/relative", "Both"], [0.55, 0.30, 0.15])
    r["respondent_role"] = role
    age_p = {"Myself": [0.12, 0.25, 0.30, 0.23, 0.10],
             "Caregiver for parent/relative": [0.0, 0.0, 0.08, 0.47, 0.45],
             "Both": [0.0, 0.10, 0.30, 0.35, 0.25]}[role]
    age = int(pick([1, 2, 3, 4, 5], age_p))
    r["patient_age_group"] = age
    r["gender"] = pick(["Male", "Female", "Prefer not to say"], [0.52, 0.45, 0.03])
    emirate = pick(["Dubai", "Abu Dhabi", "Sharjah", "Northern Emirates"], [0.40, 0.25, 0.22, 0.13])
    r["emirate"] = emirate
    inc_shift = {"Dubai": 0.35, "Abu Dhabi": 0.4, "Sharjah": -0.25, "Northern Emirates": -0.45}[emirate]
    income = clip_int(rng.normal(2.85 + inc_shift, 1.1), 1, 5)
    r["income_band"] = income
    if role == "Myself" and age >= 4:
        emp = pick(["Employed", "Self-employed", "Retired", "Homemaker", "Other"], [0.25, 0.10, 0.45, 0.15, 0.05])
    else:
        emp = pick(["Employed", "Self-employed", "Retired", "Homemaker", "Other"], [0.62, 0.15, 0.03, 0.15, 0.05])
    r["employment"] = emp
    if age >= 5:
        living = pick(["Alone", "With spouse only", "With family", "With hired caregiver"], [0.12, 0.20, 0.50, 0.18])
    elif age == 4:
        living = pick(["Alone", "With spouse only", "With family", "With hired caregiver"], [0.12, 0.33, 0.50, 0.05])
    else:
        living = pick(["Alone", "With spouse only", "With family", "With hired caregiver"], [0.25, 0.20, 0.54, 0.01])
    r["living_arrangement"] = living
    ins_lat = income + rng.normal(0, 1)
    r["insurance"] = "No insurance" if ins_lat < 1.6 else ("Basic" if ins_lat < 3.9 else "Enhanced/premium")

    # ---------- Section B: health ----------
    n_cond = clip_int(rng.poisson(0.35 + 0.45 * (age - 1)), 0, 3)
    r["n_conditions"] = n_cond
    chosen = set()
    if n_cond > 0:
        k = n_cond if n_cond < 3 else int(rng.integers(3, 5))
        chosen = set(rng.choice(CONDITIONS, size=k, replace=False, p=np.array(COND_W) / sum(COND_W)))
    for c in CONDITIONS:
        r[f"cond_{c}"] = int(c in chosen)
    n_med = clip_int(0.6 + 0.95 * n_cond + rng.normal(0, 0.6), 0, 4)
    if n_med == 0 and rng.random() < 0.7:
        n_med = 1  # screener: regular medicine or appointments
    r["n_medicines"] = n_med
    r["doses_per_day"] = clip_int(1 + 0.6 * n_med + rng.normal(0, 0.55), 1, 4)
    r["doctor_visits"] = clip_int(0.5 + 0.7 * n_cond + rng.normal(0, 0.6), 0, 3)
    r["lab_tests"] = clip_int(0.3 + 0.55 * n_cond + 0.6 * r["cond_diabetes"] + rng.normal(0, 0.6), 0, 3)
    ins_factor = {"No insurance": 1.7, "Basic": 1.0, "Enhanced/premium": 0.55}[r["insurance"]]
    spend = rng.lognormal(np.log(45 * (1 + n_med) * ins_factor), 0.45) if n_med > 0 else rng.uniform(0, 30)
    r["med_spend_aed"] = int(round(spend / 5) * 5)

    # ---------- Section D: digital (generated before C, C depends on it) ----------
    sp_p = {1: [0.97, 0.02, 0.01], 2: [0.96, 0.03, 0.01], 3: [0.90, 0.07, 0.03],
            4: [0.72, 0.17, 0.11], 5: [0.42, 0.30, 0.28]}[age]
    smartphone = int(pick([2, 1, 0], sp_p))
    if role != "Myself":  # caregiver answers; channel can reach the caregiver's phone
        smartphone = max(smartphone, int(pick([2, 1, 0], [0.5, 0.4, 0.1])))
    r["smartphone"] = smartphone
    wa = 0 if smartphone == 0 else clip_int(1.2 + 0.8 * smartphone - 0.25 * (age - 3) + rng.normal(0, 0.7), 0, 3)
    r["whatsapp_use"] = wa
    tech = clip_int(4.4 - 0.45 * (age - 1) + 0.2 * (income - 3) + 0.5 * (role != "Myself") + rng.normal(0, 0.8), 1, 5)
    r["tech_comfort"] = tech
    if tech <= 2 or wa <= 1:
        ch = pick(["WhatsApp text", "WhatsApp voice note", "SMS", "App notification", "Automated phone call"],
                  [0.22, 0.28, 0.18, 0.04, 0.28])
    else:
        ch = pick(["WhatsApp text", "WhatsApp voice note", "SMS", "App notification", "Automated phone call"],
                  [0.58, 0.12, 0.10, 0.17, 0.03])
    r["preferred_channel"] = ch
    r["preferred_language"] = pick(["English", "Arabic", "Hindi/Urdu", "Malayalam", "Tagalog", "Other"],
                                   [0.40, 0.22, 0.18, 0.10, 0.07, 0.03])
    trust = clip_int(rng.normal(3.1 + 0.15 * (income - 3) + 0.15 * (tech - 3), 1.0), 1, 5)
    r["data_trust"] = trust
    r["used_health_app"] = int(rng.random() < sigmoid(-1.6 + 0.6 * (tech - 3) + 0.3 * n_cond))

    # ---------- Section C: behaviour ----------
    if tech >= 4 and rng.random() < 0.35:
        method = pick(["Phone alarm", "Another app"], [0.6, 0.4])
    else:
        method = pick(["Memory only", "Phone alarm", "Family member reminds", "Pill organiser box", "Another app"],
                      [0.34, 0.22, 0.22 + 0.1 * (living == "With family"), 0.16, 0.04])
    r["current_method"] = method
    forget = (0.35 * (r["doses_per_day"] - 1) + 0.35 * (age >= 5) + 0.15 * (living == "Alone")
              - 0.3 * (living in ["With family", "With hired caregiver"]) + 0.45 * (method == "Memory only")
              - 0.3 * (method in ["Another app", "Pill organiser box"]) + rng.normal(0.4, 0.75))
    r["missed_doses_week"] = clip_int(forget, 0, 3)
    r["missed_appointments"] = clip_int(-0.2 + 0.4 * forget + 0.2 * r["doctor_visits"] + rng.normal(0, 0.6), 0, 3)
    r["ran_out_refill"] = clip_int(-0.1 + 0.35 * forget + 0.25 * n_med + rng.normal(0, 0.6), 0, 3)
    r["current_satisfaction"] = clip_int(4.0 - 0.7 * r["missed_doses_week"] - 0.2 * r["ran_out_refill"]
                                         + 0.3 * (method == "Another app") + rng.normal(0, 0.7), 1, 5)
    r["worry_level"] = clip_int(1.6 + 0.55 * n_cond + 0.35 * r["missed_doses_week"]
                                + 0.4 * (role != "Myself") + rng.normal(0, 0.7), 1, 5)

    # ---------- latent purchase intent ----------
    need = (0.35 * n_cond + 0.25 * n_med + 0.45 * r["missed_doses_week"] + 0.25 * r["missed_appointments"]
            + 0.3 * r["ran_out_refill"] + 0.3 * (r["worry_level"] - 3) - 0.3 * (r["current_satisfaction"] - 3))
    digital = 0.35 * smartphone + 0.3 * wa + 0.3 * (tech - 3)
    intent_lat = (-1.55 + 0.75 * need + 0.55 * digital + 0.35 * (trust - 3) + 0.55 * (role != "Myself")
                  + 0.12 * (income - 3) + 0.25 * r["used_health_app"] - 0.6 * (smartphone == 0)
                  + rng.normal(0, 0.75))
    r["_intent_lat"] = intent_lat

    # ---------- Section E: service baskets (with built-in co-purchase links) ----------
    il = intent_lat
    caregiver = role != "Myself"
    s = {}
    s["svc_med_reminder"] = rng.random() < sigmoid(0.9 + 0.9 * il + 0.3 * n_med)
    s["svc_refill_alert"] = rng.random() < sigmoid(-1.6 + 0.6 * il + 0.4 * n_med + 0.35 * r["ran_out_refill"]
                                                    + 1.6 * s["svc_med_reminder"])
    s["svc_pharmacy_delivery"] = rng.random() < sigmoid(-2.6 + 0.5 * il + 0.3 * (age - 3) + 0.4 * caregiver
                                                         + 2.2 * s["svc_refill_alert"] + 0.6 * s["svc_med_reminder"])
    s["svc_rx_scan"] = rng.random() < sigmoid(-2.0 + 0.5 * il + 0.4 * (tech - 3) + 1.3 * s["svc_med_reminder"])
    s["svc_appt_reminder"] = rng.random() < sigmoid(-1.0 + 0.6 * il + 0.5 * r["doctor_visits"]
                                                     + 0.35 * r["missed_appointments"])
    s["svc_lab_reminder"] = rng.random() < sigmoid(-2.4 + 0.5 * il + 0.45 * r["lab_tests"]
                                                    + 1.9 * s["svc_appt_reminder"])
    s["svc_doctor_report"] = rng.random() < sigmoid(-3.0 + 0.5 * il + 0.3 * n_cond
                                                     + 1.4 * s["svc_appt_reminder"] + 1.3 * s["svc_lab_reminder"])
    s["svc_caregiver_alert"] = rng.random() < sigmoid(-2.2 + 0.5 * il + 1.7 * caregiver + 0.5 * (age >= 5)
                                                       + 0.5 * (living == "Alone") + 0.4 * s["svc_med_reminder"])
    s["svc_vitals_log"] = rng.random() < sigmoid(-2.2 + 0.45 * il + 1.1 * r["cond_diabetes"]
                                                  + 0.9 * r["cond_hypertension"] + 0.3 * (tech - 3))
    s["svc_teleconsult"] = rng.random() < sigmoid(-2.4 + 0.5 * il + 0.3 * (tech - 3)
                                                   + 0.5 * (r["insurance"] == "Enhanced/premium")
                                                   + 1.2 * s["svc_vitals_log"])
    for k in SERVICES:
        r[k] = int(s[k])
    n_svc = sum(r[k] for k in SERVICES)
    r["n_services_selected"] = n_svc

    # ---------- Section F: intent, spend, buying ----------
    pi_lat = intent_lat + 0.12 * (n_svc - 4) + rng.normal(0, 0.35)
    r["_pi_lat"] = pi_lat
    r["recommend_score"] = None  # filled after cut points
    r["_n_svc"] = n_svc

    if caregiver and rng.random() < 0.45:
        plan = "Family plan"
    elif income <= 2 and rng.random() < 0.55:
        plan = "Free basic + paid add-ons"
    else:
        plan = pick(["Free basic + paid add-ons", "Monthly", "Annual (discounted)", "Family plan", "Pay per service"],
                    [0.22, 0.34, 0.20 + 0.05 * (income - 3), 0.10, 0.14])
    r["preferred_plan"] = plan
    chan_w = np.array([1.8 * (tech >= 4) + 0.5, 0.6 + 0.35 * (age - 1), 0.4 + 0.45 * n_cond,
                       0.2 + 1.3 * (r["insurance"] == "Enhanced/premium"), 0.1 + 0.6 * (emp == "Employed")])
    r["purchase_channel"] = pick(["Directly on WhatsApp", "Through pharmacy", "Doctor/clinic recommendation",
                                  "Through insurer", "Through employer"], chan_w)
    if caregiver:
        payer = pick(["Myself", "Children/family", "Insurance", "Employer"], [0.25, 0.65, 0.07, 0.03])
    else:
        payer = pick(["Myself", "Children/family", "Insurance", "Employer"],
                     [0.66, 0.12 + 0.05 * (age >= 4), 0.12 + 0.08 * (r["insurance"] == "Enhanced/premium"), 0.06])
    r["payer"] = payer
    r["payment_method"] = pick(["Card", "Apple Pay / Google Pay", "Added to pharmacy bill", "Bank transfer"],
                               [0.40, 0.10 + 0.08 * tech, 0.10 + 0.07 * (age - 1), 0.10])
    fw = np.array([0.4 + 0.5 * (income <= 2), 0.3 + 0.12 * (age - 1), 0.2 + 0.3 * (trust <= 2), 0.25 + 0.08 * n_cond,
                   0.05 + 0.6 * (r["preferred_language"] == "Arabic"), 0.1 + 0.5 * caregiver])
    r["top_factor"] = pick(["Price", "Ease of use", "Data privacy", "Doctor endorsement", "Arabic support",
                            "Family involvement"], fw)
    r["_income"] = income
    r["_caregiver"] = int(caregiver)
    return r


def finalize(df):
    # Q29 purchase intent on 1-5 from latent, fixed cut points (same scale for train and live data)
    cuts = [-0.95, 0.10, 1.30, 2.50]
    df["purchase_intent"] = np.digitize(df["_pi_lat"], cuts) + 1
    df["interested"] = (df["purchase_intent"] >= 4).astype(int)
    df["recommend_score"] = np.clip(np.round(5.6 + 1.6 * df["_pi_lat"] + rng.normal(0, 1.3, len(df))), 0, 10).astype(int)

    # Q31 willingness to pay (AED / month)
    wtp = []
    for _, r in df.iterrows():
        pi = r["purchase_intent"]
        if pi == 1 or (pi == 2 and rng.random() < 0.6):
            wtp.append(0)
            continue
        base = (6 + 3.2 * r["_n_svc"] + 5.5 * (r["_income"] - 1) + 7 * (pi - 3) + 0.025 * r["med_spend_aed"]
                + 9 * (r["preferred_plan"] == "Family plan") + 6 * r["_caregiver"]
                + 4 * (r["insurance"] == "Enhanced/premium"))
        val = max(5, base) * rng.lognormal(0, 0.22)
        wtp.append(int(min(200, round(val / 5) * 5)))
    df["wtp_monthly_aed"] = wtp
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")] + ["n_services_selected"])
    df["n_services_selected"] = df[SERVICES].sum(axis=1)
    return df


ORDER = (["respondent_id", "response_time", "respondent_role", "patient_age_group", "gender", "emirate",
          "income_band", "employment", "living_arrangement", "insurance", "n_conditions"]
         + [f"cond_{c}" for c in CONDITIONS]
         + ["n_medicines", "doses_per_day", "doctor_visits", "lab_tests", "med_spend_aed",
            "missed_doses_week", "missed_appointments", "ran_out_refill", "current_method",
            "current_satisfaction", "worry_level", "smartphone", "whatsapp_use", "tech_comfort",
            "preferred_channel", "preferred_language", "data_trust", "used_health_app"]
         + SERVICES + ["n_services_selected", "purchase_intent", "interested", "recommend_score",
                       "wtp_monthly_aed", "preferred_plan", "purchase_channel", "payer", "payment_method",
                       "top_factor"])

def generate(n_survey=1500, n_live=300, seed=SEED):
    """Return (survey_df, live_df). Same seed -> identical data."""
    reset_rng(seed)
    survey = pd.DataFrame([one_respondent(i + 1, pd.Timestamp("2026-10-01"), 36) for i in range(n_survey)])
    survey = finalize(survey)[ORDER].sort_values("response_time").reset_index(drop=True)
    live = pd.DataFrame([one_respondent(i + 1, pd.Timestamp("2026-11-06"), 1)
                         for i in range(n_survey, n_survey + n_live)])
    live = finalize(live)[ORDER]
    return survey, live


if __name__ == "__main__":
    from src.config import SURVEY_CSV, LIVE_CSV
    survey, live = generate()
    SURVEY_CSV.parent.mkdir(parents=True, exist_ok=True)
    survey.to_csv(SURVEY_CSV, index=False)
    live.to_csv(LIVE_CSV, index=False)
    print(survey.shape, live.shape, "interested rate:", round(survey["interested"].mean(), 3))
