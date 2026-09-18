# AI Credit Risk & Loan Intelligence Platform

An end-to-end credit risk analytics platform — data entry, MySQL storage, ML-based
default prediction (XGBoost), SHAP explainability, financial loss calculation
(PD × LGD × EAD), a What-If scenario simulator, and a GenAI assistant (Groq)
that answers portfolio questions in plain English.

**Live demo:** _add your Streamlit Community Cloud link here after deploying_

## Tech stack
Python · Streamlit · MySQL · Scikit-learn · XGBoost · SHAP · Plotly · Groq (LLM)

## Repository contents
| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (5 pages) |
| `generate_dataset.py` | Synthetic data generator |
| `customers.csv`, `credit_history.csv`, `loans.csv` | Generated dataset |
| `load_to_mysql.py` | Loads the CSVs into MySQL |
| `train_models.py` | Trains & compares ML models, saves model artifacts |
| `credit_risk_model.pkl`, `model_features.pkl`, `shap_explainer.pkl` | Trained model artifacts used by the app |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml.example` | Template for required secrets (copy to `secrets.toml`, never commit the real one) |

## Run it locally
1. Install dependencies: `pip install -r requirements.txt`
2. Create a MySQL database using the schema (see project manual / `MYSQL_SCHEMA.sql`)
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your
   own MySQL credentials and Groq API key (free at console.groq.com)
4. Load data: `python load_to_mysql.py`
5. Train models: `python train_models.py`
6. Run the app: `streamlit run app.py`

## Deploy it live (so anyone can try it)
1. Create a free MySQL database on [Aiven](https://aiven.io) (always-free tier, no card required)
2. Point `load_to_mysql.py` / `train_models.py` at that database once, to populate it
   and generate the model artifacts
3. Push this repo to GitHub (`secrets.toml` is git-ignored automatically — never commit it)
4. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub repo,
   set `app.py` as the entry point
5. In the app's "Secrets" settings, paste the same contents as
   `.streamlit/secrets.toml.example`, filled in with your real Aiven + Groq credentials
6. Deploy — you'll get a public link like `your-app.streamlit.app` to share on
   LinkedIn / your resume

## Note on shared data
Because the deployed app uses one shared cloud database, any data entered by a
visitor (e.g. a recruiter testing the "New Loan Application" form) is saved to
that same database and visible to all other visitors on the dashboard. This is
expected behavior for a shared demo.
