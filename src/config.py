"""Shared constants: column groups, labels and colours used across the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SURVEY_CSV = DATA_DIR / "mediconnect_survey_responses.csv"
LIVE_CSV = DATA_DIR / "mediconnect_live_stream.csv"

SEED = 42

SERVICES = ["svc_med_reminder", "svc_appt_reminder", "svc_lab_reminder", "svc_refill_alert", "svc_rx_scan",
            "svc_caregiver_alert", "svc_pharmacy_delivery", "svc_vitals_log", "svc_teleconsult", "svc_doctor_report"]
SVC_LABEL = {"svc_med_reminder": "Medicine reminders", "svc_appt_reminder": "Appointment reminders",
             "svc_lab_reminder": "Lab test reminders", "svc_refill_alert": "Refill alerts",
             "svc_rx_scan": "Prescription photo setup", "svc_caregiver_alert": "Caregiver alerts",
             "svc_pharmacy_delivery": "Pharmacy delivery", "svc_vitals_log": "Vitals logging",
             "svc_teleconsult": "Teleconsult booking", "svc_doctor_report": "Doctor adherence report"}
CONDS = ["cond_diabetes", "cond_hypertension", "cond_heart", "cond_cholesterol", "cond_thyroid", "cond_asthma",
         "cond_other"]

# Inputs for classification (Q1-Q28): ordinal/binary numbers + one-hot categories
NUMERIC_AD = (["patient_age_group", "income_band", "n_conditions"] + CONDS +
              ["n_medicines", "doses_per_day", "doctor_visits", "lab_tests", "med_spend_aed", "missed_doses_week",
               "missed_appointments", "ran_out_refill", "current_satisfaction", "worry_level", "smartphone",
               "whatsapp_use", "tech_comfort", "data_trust", "used_health_app"])
CATEG_AD = ["respondent_role", "gender", "emirate", "employment", "living_arrangement", "insurance",
            "current_method", "preferred_channel", "preferred_language"]

# Inputs for regression: Q1-Q28 + Section E + Q29 + Q32
NUMERIC_REG = NUMERIC_AD + SERVICES + ["n_services_selected", "purchase_intent"]
CATEG_REG = CATEG_AD + ["preferred_plan"]

# Clustering: pain (how often the problem happens) x readiness (can and will they use WhatsApp)
CLU = ["n_medicines", "missed_doses_week", "missed_appointments", "ran_out_refill", "worry_level",
       "current_satisfaction", "whatsapp_use", "tech_comfort", "data_trust"]

SEGS = ["High probable", "Medium probable", "Low probable"]
SEG_COLORS = {"High probable": "#2a78d6", "Medium probable": "#eb6834", "Low probable": "#1baf7a"}
GOOD, BAD, WARN, ACCENT, MUTED = "#0ca30c", "#d03b3b", "#fab219", "#2a78d6", "#7a8491"

CHANNELS = ["WhatsApp text", "WhatsApp voice note", "SMS", "App notification", "Automated phone call"]
LANGS = ["English", "Arabic", "Hindi/Urdu", "Malayalam", "Tagalog", "Other"]
AGE_LABELS = {1: "18–29", 2: "30–44", 3: "45–59", 4: "60–74", 5: "75+"}

PRETTY = {"patient_age_group": "Patient age group", "income_band": "Income band", "n_conditions": "Chronic conditions",
          "n_medicines": "Medicines per day", "doses_per_day": "Dose times per day", "doctor_visits": "Doctor visits",
          "lab_tests": "Lab tests", "med_spend_aed": "Medicine spend (AED)", "missed_doses_week": "Missed doses / week",
          "missed_appointments": "Missed appointments", "ran_out_refill": "Ran out before refill",
          "current_satisfaction": "Satisfaction with current method", "worry_level": "Worry about missed doses",
          "smartphone": "Smartphone access", "whatsapp_use": "WhatsApp use", "tech_comfort": "Tech comfort",
          "data_trust": "Comfort sharing data", "used_health_app": "Used a health app",
          "n_services_selected": "Services selected", "purchase_intent": "Purchase intent (Q29)"}


def pretty(name: str) -> str:
    """Readable label for an encoded feature name such as 'insurance=Basic'."""
    if name in SVC_LABEL:
        return "Wants " + SVC_LABEL[name].lower()
    if "=" in name:
        col, val = name.split("=", 1)
        return f"{col.replace('_', ' ').capitalize()}: {val}"
    if name.startswith("cond_"):
        return "Condition: " + name[5:]
    return PRETTY.get(name, name.replace("_", " ").capitalize())
