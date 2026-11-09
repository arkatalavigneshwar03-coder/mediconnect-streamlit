# MediConnect – Customer Intelligence & Live Simulation (Streamlit)

MediConnect is a WhatsApp-first medication and health reminder service. This Streamlit app:

- applies **four machine-learning algorithms** to a 1,500-response consumer survey: classification, clustering, association rules and regression
- runs **two live simulations**:
  - new survey respondents scored by all four models as they arrive
  - the service operating hour by hour, with sign-ups, reminders, refills, churn, **costs and profit**

> The survey data is **synthetic**: it was generated to match the questionnaire design. The figures show the method, not the real market.

## Pages

| Page | What it does |
|---|---|
| Home | Survey headline figures, the four algorithms at a glance, data downloads |
| Questionnaire | The 36-question survey, the column each answer fills, and the algorithm it feeds |
| Classification | Logistic regression vs random forest, threshold slider, ROC curves, confusion matrix, drivers |
| Clustering | K-means segments (High / Medium / Low probable), choose k, silhouette and elbow, profiles |
| Association Rules | Apriori with adjustable support / confidence / lift, "2 services → 3rd" rules, cross-sell finder |
| Regression | Willingness to pay (Q31): ridge vs gradient boosting, drivers, price-acceptance curve |
| Live Survey Stream | 300 new respondents scored live by all four models and checked against their own answers |
| Live Operations | The service running hour by hour: patients, messages, adherence, growth, **costs & profit**, services |
| Score a Respondent | Fill in the questionnaire and get all four predictions instantly |

## Run it on your computer

```bash
cd mediconnect-streamlit
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The first load takes about 5 seconds while the models train; after that they are cached.

## Put it on GitHub

**Option A – in the browser (no Git needed)**
1. Sign in at github.com and click **New repository**. Name it `mediconnect-streamlit`, choose **Public**, and click **Create repository**.
2. On the new repository page, click **uploading an existing file**.
3. Unzip `mediconnect-streamlit.zip`, open the folder, select **everything inside it** (`app.py`, `requirements.txt`, `README.md`, and the `pages`, `src`, `data` and `.streamlit` folders) and drag it onto the upload page.
   - `.streamlit` is a hidden folder. If your computer hides it, the app still works; you just get Streamlit's default theme.
4. Click **Commit changes**.

**Option B – with Git**
```bash
cd mediconnect-streamlit
git init
git add .
git commit -m "MediConnect Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/mediconnect-streamlit.git
git push -u origin main
```

## Deploy free on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app** and choose to deploy from a GitHub repository.
3. Fill in:
   - Repository: `<your-username>/mediconnect-streamlit`
   - Branch: `main`
   - Main file path: `app.py`
4. Open **Advanced settings** and choose **Python 3.12**.
5. Click **Deploy**. The first build installs the packages and takes a few minutes. You then get a public `*.streamlit.app` link.

Each time you push changes to GitHub, the app updates automatically.

## Using the live simulations

- **Live Survey Stream:** starts automatically. Use **Pause / Run**, change the speed (1, 3 or 10 respondents a second) or click **Restart**.
- **Live Operations:** starts automatically after a three-week warm-up.
  - Change the price, leads per day, reminder effect, follow-up and caregiver alerts in the sidebar.
  - Edit unit costs (fixed costs, cost per lead, WhatsApp/SMS/call rates, payment and support) under **Unit costs**. Changes apply straight away.
  - Speeds go up to one simulated week per second.

## Project structure

```
mediconnect-streamlit/
├── app.py                    # Home page
├── pages/                    # One file per page (Streamlit multipage app)
├── src/
│   ├── config.py             # Column groups, labels, colours
│   ├── data_gen.py           # Synthetic survey generator (python -m src.data_gen)
│   ├── models.py             # Training + scoring for the four algorithms
│   ├── apriori.py            # Apriori itemsets and rules (no extra dependency)
│   ├── ops_sim.py            # Hour-by-hour operations simulation with costs
│   ├── questionnaire.py      # The questionnaire, mapped to dataset columns
│   └── ui.py                 # Cached models and chart helpers
├── data/
│   ├── mediconnect_survey_responses.csv   # 1,500 survey responses (generated on first run if missing)
│   └── mediconnect_live_stream.csv        # 300 new respondents (generated on first run if missing)
├── .streamlit/config.toml    # Theme
└── requirements.txt
```

## Regenerate the data

```bash
python -m src.data_gen
```
This rewrites both CSVs in `data/` (same seed, so you get the same data). Change `SEED` in `src/data_gen.py` for a different sample.

## Replacing synthetic data with real survey responses

Export your real responses with the same column names as `data/mediconnect_survey_responses.csv`. The Questionnaire page lists every column and its coding. Save the export over that file and redeploy. All models retrain automatically on the next load.

## Key assumptions in the operations simulation

| Assumption | Value |
|---|---|
| Qualified leads who finish WhatsApp onboarding | 62% |
| Unconfirmed doses saved by the follow-up reminder | 30% |
| Doses saved by a caregiver alert | 40% |
| Pharmacy delivery commission | AED 10 per order |
| Teleconsult booking fee | AED 15 |
| WhatsApp message cost | AED 0.069 (Meta's UAE utility rate of AED 0.0576 + 20% provider markup) |
| SMS | AED 0.10 |
| Automated call | AED 0.30 |
| Payment fee | 2.9% + AED 1 |
| Support | AED 3 per subscriber per month |
| Fixed costs | AED 75,000 per month |
| Cost per lead | AED 20 |

All costs can be changed in the app. Replace the estimates with real quotes and pilot data.
