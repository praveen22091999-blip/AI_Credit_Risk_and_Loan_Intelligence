"""
Day 3: Streamlit App
Page 1: New Loan Application (data entry -> MySQL)
Page 2: AI Risk Prediction (loads saved model, predicts, shows result, saves to MySQL)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import mysql.connector
import joblib
import plotly.express as px
from groq import Groq
from datetime import date

# ---------- CONFIG ----------
# Reads from Streamlit secrets (used when deployed on Streamlit Community Cloud).
# Falls back to the local values below when running on your own machine
# WITHOUT a .streamlit/secrets.toml file (e.g. first-time local testing).
try:
    DB_CONFIG = {
        "host": st.secrets["mysql"]["host"],
        "port": st.secrets["mysql"].get("port", 3306),
        "user": st.secrets["mysql"]["user"],
        "password": st.secrets["mysql"]["password"],
        "database": st.secrets["mysql"]["database"],
    }
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
except (KeyError, FileNotFoundError):
    # LOCAL DEVELOPMENT FALLBACK — never commit real secrets here.
    DB_CONFIG = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "YOUR_MYSQL_PASSWORD_HERE",
        "database": "credit_risk_db"
    }
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
# -------------------------------------------

st.set_page_config(page_title="AI Credit Risk Platform", layout="wide", page_icon="◆")


# ============================================================
# DESIGN SYSTEM — colors, type, and reusable UI components
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --navy: #1B2432;
        --navy-panel: #232D3F;
        --navy-active: #2E3A50;
        --teal: #1FA98C;
        --teal-dark: #17836C;
        --bg: #F3F5F9;
        --surface: #FFFFFF;
        --border: #EAEDF2;
        --text: #172433;
        --text-muted: #6B7688;
        --low: #1E8E5A;
        --medium: #C08A1E;
        --high: #C1392B;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .stApp { background-color: var(--bg); }

    /* ---------- Sidebar shell ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #C7CFDB !important; }

    /* ---------- Sidebar nav buttons (secondary = inactive, primary = active) ---------- */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        color: #9AA6B8 !important;
        padding: 0.55rem 0.8rem !important;
        border-radius: 8px !important;
        width: 100% !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.06) !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--navy-active) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-left: 3px solid var(--teal) !important;
    }

    /* Sidebar CTA button (special teal, matched via widget key) */
    div[class*="st-key-nav_cta"] button {
        background-color: var(--teal) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        justify-content: center !important;
        padding: 0.6rem !important;
    }
    div[class*="st-key-nav_cta"] button:hover {
        background-color: var(--teal-dark) !important;
    }

    /* ---------- Main-area buttons ---------- */
    .main .stButton > button, .stFormSubmitButton > button {
        background-color: var(--teal); color: white; border: none;
        border-radius: 6px; font-weight: 600; padding: 0.5rem 1.2rem;
    }
    .main .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: var(--teal-dark); color: white;
    }

    /* ---------- Breadcrumb + page title ---------- */
    .breadcrumb { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; margin-bottom: 0.15rem; }
    .page-title { font-size: 1.7rem; font-weight: 700; color: var(--navy); margin: 0 0 1.3rem 0;
                  font-family: 'Space Grotesk', sans-serif; }

    /* ---------- KPI cards (icon badge style) ---------- */
    .kpi-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(16,36,61,0.04);
    }
    .kpi-icon {
        width: 38px; height: 38px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; margin-bottom: 0.6rem;
    }
    .kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--navy); }
    .kpi-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; margin-top: 0.15rem; }

    /* ---------- Risk badges ---------- */
    .risk-badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem; color: white;
    }
    .risk-low { background-color: var(--low); }
    .risk-medium { background-color: var(--medium); }
    .risk-high { background-color: var(--high); }

    /* ---------- Chat bubbles ---------- */
    .chat-question {
        background-color: var(--navy); color: white; padding: 0.6rem 1rem;
        border-radius: 10px 10px 2px 10px; margin-bottom: 0.5rem; display: inline-block;
        font-weight: 600;
    }
    .chat-answer {
        background-color: var(--surface); border: 1px solid var(--border);
        padding: 0.8rem 1rem; border-radius: 10px 10px 10px 2px; color: var(--text);
    }

    /* ---------- Input fields (text, number, select) ---------- */
    .stTextInput input, .stNumberInput input,
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 1px var(--teal) !important;
    }

    /* ---------- Cards for form sections ---------- */
    div[data-testid="stForm"] {
        background-color: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 2px rgba(16,36,61,0.04);
    }
    </style>
    """, unsafe_allow_html=True)


ICON_COLORS = {
    "teal":  ("#1FA98C", "#E4F5F1"),
    "navy":  ("#1B2432", "#E9EBEF"),
    "amber": ("#C08A1E", "#FBF0DC"),
    "red":   ("#C1392B", "#FBE7E4"),
    "blue":  ("#2D6CA6", "#E4EDF6"),
}


def format_inr(amount):
    """Format a number using the Indian numbering system (lakh/crore grouping).
    Example: 5002707667 -> '5,00,27,07,667' instead of the Western '5,002,707,667'."""
    amount = round(float(amount))
    negative = amount < 0
    amount = abs(int(amount))
    s = str(amount)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ",".join(parts) + "," + last3
    return ("-" if negative else "") + formatted


def page_title(breadcrumb, title):
    st.markdown(f"""
    <div class="breadcrumb">{breadcrumb}</div>
    <div class="page-title">{title}</div>
    """, unsafe_allow_html=True)


def kpi_row(items):
    """items: list of dicts with keys label, value, color(optional, key into ICON_COLORS)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        fg, _ = ICON_COLORS.get(item.get("color", "teal"), ICON_COLORS["teal"])
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {fg};">
                <div class="kpi-value">{item["value"]}</div>
                <div class="kpi-label">{item["label"]}</div>
            </div>
            """, unsafe_allow_html=True)


def risk_badge(category):
    css_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(category, "risk-medium")
    return f'<span class="risk-badge {css_class}">{category} Risk</span>'


inject_custom_css()

PAGE_META = {
    "Executive Dashboard": ("Overview", "Executive Dashboard"),
    "New Loan Application": ("Underwriting", "New Loan Application"),
    "AI Risk Assessment": ("Underwriting", "AI Risk Assessment"),
    "What-If Simulator": ("Underwriting", "What-If Simulator"),
    "AI Risk Assistant": ("Insights", "AI Risk Assistant"),
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


DB_SCHEMA_DESCRIPTION = """
Tables:
1. customers(customer_id, age, income, employment_type, dependents)
2. credit_history(history_id, customer_id, credit_score, credit_utilization, debt_ratio,
   open_credit_lines, past_due_30_59, past_due_60_89, past_due_90plus, real_estate_loans)
3. loans(loan_id, customer_id, loan_amount, tenure_months, interest_rate, loan_type,
   application_date, default_flag)
4. risk_predictions(prediction_id, customer_id, probability_default, risk_category,
   expected_loss, prediction_date)

All tables join on customer_id.
"""


def ask_ai_assistant(question, client):
    """
    Two-step flow:
    1. Ask Groq to write a SQL SELECT query for the question.
    2. Run the query on MySQL, then ask Groq to summarize the result in plain language.
    """
    sql_prompt = f"""You are a SQL expert. Given this MySQL schema:
{DB_SCHEMA_DESCRIPTION}

Write ONE MySQL SELECT query (no explanations, no markdown fences, just the raw SQL)
to answer this question: "{question}"

Rules:
- Only SELECT statements. Never write INSERT/UPDATE/DELETE/DROP.
- Use JOINs on customer_id where needed.
- Add LIMIT 20 unless the question asks for an aggregate (like COUNT, AVG, SUM).
"""

    sql_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0
    )
    sql_query = sql_response.choices[0].message.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # Safety check: only allow SELECT
    if not sql_query.lower().startswith("select"):
        return "I can only answer questions that involve reading data (SELECT queries).", None, sql_query

    conn = get_connection()
    try:
        result_df = pd.read_sql(sql_query, conn)
    except Exception as e:
        conn.close()
        return f"I generated a query but it failed to run: {e}", None, sql_query
    conn.close()

    # Step 2: summarize results in natural language
    summary_prompt = f"""The user asked: "{question}"
Here is the query result (as a table):
{result_df.to_string(index=False) if not result_df.empty else "No rows returned."}

Write a short, clear natural-language answer (2-4 sentences) for a bank loan officer,
based only on this data. Mention specific numbers where relevant.

IMPORTANT: All monetary amounts in this data (loan amounts, expected loss, income, etc.)
are in Indian Rupees. Always use the ₹ symbol (never $), and group digits using the
Indian numbering system (lakh/crore), e.g. ₹1,50,00,000 for 1.5 crore — not the
Western ₹15,000,000.
"""
    summary_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3
    )
    answer = summary_response.choices[0].message.content.strip()

    return answer, result_df, sql_query


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("credit_risk_model.pkl")
    feature_cols = joblib.load("model_features.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, feature_cols, explainer


def insert_application(data):
    """Insert a new customer + credit_history + loan record into MySQL."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO customers (customer_id, age, income, employment_type, dependents)
        VALUES (%s, %s, %s, %s, %s)
    """, (data["customer_id"], data["age"], data["income"],
          data["employment_type"], data["dependents"]))

    cursor.execute("""
        INSERT INTO credit_history
        (customer_id, credit_score, credit_utilization, debt_ratio, open_credit_lines,
         past_due_30_59, past_due_60_89, past_due_90plus, real_estate_loans)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (data["customer_id"], data["credit_score"], data["credit_utilization"],
          data["debt_ratio"], data["open_credit_lines"], data["past_due_30_59"],
          data["past_due_60_89"], data["past_due_90plus"], data["real_estate_loans"]))

    cursor.execute("""
        INSERT INTO loans
        (customer_id, loan_amount, tenure_months, interest_rate, loan_type,
         application_date, default_flag)
        VALUES (%s, %s, %s, %s, %s, %s, NULL)
    """, (data["customer_id"], data["loan_amount"], data["tenure_months"],
          data["interest_rate"], data["loan_type"], date.today()))

    conn.commit()
    cursor.close()
    conn.close()


def save_prediction(customer_id, prob_default, risk_category, expected_loss):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_predictions
        (customer_id, probability_default, risk_category, expected_loss)
        VALUES (%s, %s, %s, %s)
    """, (customer_id, float(prob_default), risk_category, float(expected_loss)))
    conn.commit()
    cursor.close()
    conn.close()


def prepare_features(data, feature_cols):
    """Build a single-row dataframe matching the training feature columns exactly."""
    row = {
        "age": data["age"],
        "income": data["income"],
        "dependents": data["dependents"],
        "credit_score": data["credit_score"],
        "credit_utilization": data["credit_utilization"],
        "debt_ratio": data["debt_ratio"],
        "open_credit_lines": data["open_credit_lines"],
        "past_due_30_59": data["past_due_30_59"],
        "past_due_60_89": data["past_due_60_89"],
        "past_due_90plus": data["past_due_90plus"],
        "real_estate_loans": data["real_estate_loans"],
        "loan_amount": data["loan_amount"],
        "tenure_months": data["tenure_months"],
        "interest_rate": data["interest_rate"],
    }

    # One-hot flags for employment_type and loan_type (must match training's drop_first=True)
    for emp in ["Salaried", "Self-Employed", "Unemployed"]:  # Business Owner was dropped (reference category)
        row[f"employment_type_{emp}"] = 1 if data["employment_type"] == emp else 0
    for lt in ["Business", "Education", "Home", "Personal"]:  # Auto was dropped (reference category)
        row[f"loan_type_{lt}"] = 1 if data["loan_type"] == lt else 0

    df = pd.DataFrame([row])

    # Ensure every training column exists (fill missing with 0), in the right order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #323E52; margin-bottom: 1rem;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: white;">
        ◆ CreditIQ
    </div>
    <div style="font-size: 0.75rem; color: #8792A3; margin-top: 0.1rem;">
        AI Credit Risk & Loan Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state.current_page = "Executive Dashboard"

if st.sidebar.button("+ New Application", key="nav_cta", use_container_width=True):
    st.session_state.current_page = "New Loan Application"

st.sidebar.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

for pname in PAGE_META:
    is_active = st.session_state.current_page == pname
    if st.sidebar.button(
        pname,
        key=f"nav_{pname.replace(' ', '_')}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.current_page = pname

page = st.session_state.current_page
page_title(*PAGE_META[page])

# ============================================================
# PAGE 1: NEW LOAN APPLICATION
# ============================================================
if page == "New Loan Application":

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            customer_id = st.text_input("Customer ID", value="", placeholder="e.g. CUST100045")
            age = st.number_input("Age", min_value=18, max_value=100, value=None, placeholder="e.g. 32")
            income = st.number_input("Annual Income (INR)", min_value=0, value=None, placeholder="e.g. 600000")
            employment_type = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
                index=None, placeholder="e.g. Salaried"
            )
            dependents = st.number_input("Number of Dependents", min_value=0, value=None, placeholder="e.g. 1")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=None, placeholder="e.g. 700")
            credit_utilization = st.number_input("Credit Utilization Ratio (0-1)", min_value=0.0, max_value=1.0,
                                                   value=None, placeholder="e.g. 0.30")
            debt_ratio = st.number_input("Debt Ratio (0-1)", min_value=0.0, max_value=1.0,
                                         value=None, placeholder="e.g. 0.30")

        with col2:
            open_credit_lines = st.number_input("Open Credit Lines", min_value=0, value=None, placeholder="e.g. 4")
            past_due_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, value=None, placeholder="e.g. 0")
            past_due_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, value=None, placeholder="e.g. 0")
            past_due_90plus = st.number_input("Times 90+ Days Late", min_value=0, value=None, placeholder="e.g. 0")
            real_estate_loans = st.number_input("Real Estate Loans", min_value=0, value=None, placeholder="e.g. 0")
            loan_amount = st.number_input("Loan Amount (INR)", min_value=0, value=None, placeholder="e.g. 600000")
            tenure_months = st.selectbox("Tenure (months)", [12, 24, 36, 48, 60, 84, 120],
                                         index=None, placeholder="e.g. 36")
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=None, placeholder="e.g. 11.0")
            loan_type = st.selectbox("Loan Type", ["Personal", "Home", "Auto", "Business", "Education"],
                                     index=None, placeholder="e.g. Personal")

        submitted = st.form_submit_button("Submit Application")

        if submitted:
            data = dict(
                customer_id=customer_id, age=age, income=income,
                employment_type=employment_type, dependents=dependents,
                credit_score=credit_score, credit_utilization=credit_utilization,
                debt_ratio=debt_ratio, open_credit_lines=open_credit_lines,
                past_due_30_59=past_due_30_59, past_due_60_89=past_due_60_89,
                past_due_90plus=past_due_90plus, real_estate_loans=real_estate_loans,
                loan_amount=loan_amount, tenure_months=tenure_months,
                interest_rate=interest_rate, loan_type=loan_type
            )

            missing = [k for k, v in data.items() if v is None or v == ""]
            if missing:
                st.error(f"Please fill in all fields before submitting. Missing: {', '.join(missing)}")
            else:
                try:
                    insert_application(data)
                    st.session_state["last_application"] = data
                    st.success(f"Application for {customer_id} saved successfully! "
                               f"Go to 'AI Risk Assessment' to see the prediction.")
                except mysql.connector.Error as e:
                    st.error(f"Database error: {e}")

# ============================================================
# PAGE 2: AI RISK ASSESSMENT
# ============================================================
elif page == "AI Risk Assessment":

    if "last_application" not in st.session_state:
        st.warning("No application submitted yet in this session. "
                   "Go to 'New Loan Application' first, or enter a Customer ID below "
                   "to re-run assessment on an existing record.")
        customer_id_lookup = st.text_input("Customer ID to assess")
        run_lookup = st.button("Fetch and Assess")

        if run_lookup and customer_id_lookup:
            conn = get_connection()
            query = """
                SELECT c.*, ch.*, l.*
                FROM customers c
                JOIN credit_history ch ON c.customer_id = ch.customer_id
                JOIN loans l ON c.customer_id = l.customer_id
                WHERE c.customer_id = %s
                ORDER BY l.loan_id DESC LIMIT 1
            """
            row = pd.read_sql(query, conn, params=(customer_id_lookup,))
            conn.close()
            if row.empty:
                st.error("Customer ID not found.")
            else:
                r = row.iloc[0]
                st.session_state["last_application"] = {
                    "customer_id": r["customer_id"], "age": r["age"], "income": r["income"],
                    "employment_type": r["employment_type"], "dependents": r["dependents"],
                    "credit_score": r["credit_score"], "credit_utilization": r["credit_utilization"],
                    "debt_ratio": r["debt_ratio"], "open_credit_lines": r["open_credit_lines"],
                    "past_due_30_59": r["past_due_30_59"], "past_due_60_89": r["past_due_60_89"],
                    "past_due_90plus": r["past_due_90plus"], "real_estate_loans": r["real_estate_loans"],
                    "loan_amount": r["loan_amount"], "tenure_months": r["tenure_months"],
                    "interest_rate": r["interest_rate"], "loan_type": r["loan_type"]
                }

    if "last_application" in st.session_state:
        data = st.session_state["last_application"]
        model, feature_cols, explainer = load_model_artifacts()

        X = prepare_features(data, feature_cols)
        prob_default = model.predict_proba(X)[0][1]

        # Risk category thresholds
        if prob_default < 0.10:
            risk_category = "Low"
        elif prob_default < 0.30:
            risk_category = "Medium"
        else:
            risk_category = "High"

        # Expected Loss = PD x LGD x EAD (assume LGD = 45% as industry default)
        LGD = 0.45
        EAD = data["loan_amount"]
        expected_loss = prob_default * LGD * EAD

        recommendation = "APPROVE" if risk_category == "Low" else (
            "REVIEW MANUALLY" if risk_category == "Medium" else "REJECT / HIGH SCRUTINY"
        )

        rec_color = "teal" if recommendation == "APPROVE" else ("amber" if "REVIEW" in recommendation else "red")
        risk_color_map = {"Low": "teal", "Medium": "amber", "High": "red"}

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="kpi-card" style="border-top: 3px solid {ICON_COLORS['blue'][0]};">
                <div class="kpi-value">{prob_default*100:.1f}%</div>
                <div class="kpi-label">Default Probability</div></div>""", unsafe_allow_html=True)
        with col2:
            fg, _ = ICON_COLORS[risk_color_map[risk_category]]
            st.markdown(f"""<div class="kpi-card" style="border-top: 3px solid {fg};">
                <div style="margin-top:0.1rem;">{risk_badge(risk_category)}</div>
                <div class="kpi-label" style="margin-top:0.4rem;">Risk Category</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="kpi-card" style="border-top: 3px solid {ICON_COLORS['red'][0]};">
                <div class="kpi-value">₹{format_inr(expected_loss)}</div>
                <div class="kpi-label">Expected Loss</div></div>""", unsafe_allow_html=True)
        with col4:
            fg, _ = ICON_COLORS[rec_color]
            st.markdown(f"""<div class="kpi-card" style="border-top: 3px solid {fg};">
                <div class="kpi-value" style="font-size:1.05rem;">{recommendation}</div>
                <div class="kpi-label">Recommendation</div></div>""", unsafe_allow_html=True)

        st.write("")

        if st.button("Save Prediction to Database"):
            try:
                save_prediction(data["customer_id"], prob_default, risk_category, expected_loss)
                st.success("Prediction saved to risk_predictions table.")
            except mysql.connector.Error as e:
                st.error(f"Database error: {e}")

        st.subheader("Why this prediction? (SHAP explanation)")
        shap_values = explainer.shap_values(X)

        # Handle all possible SHAP output shapes across library versions:
        # - list of arrays (one per class): shap_values[1][0]
        # - 3D array (samples, features, classes): shap_values[0, :, 1]
        # - 2D array (samples, features): shap_values[0]
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            values = shap_values[0, :, 1]
        else:
            values = shap_values[0]

        values = pd.Series(values).to_numpy().flatten()

        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "impact": values
        }).sort_values("impact", key=abs, ascending=False).head(8)

        st.bar_chart(shap_df.set_index("feature")["impact"])
        st.caption("Positive bars push risk UP, negative bars push risk DOWN.")

# ============================================================
# PAGE 3: EXECUTIVE DASHBOARD
# ============================================================
elif page == "Executive Dashboard":
    conn = get_connection()
    portfolio = pd.read_sql("""
        SELECT c.customer_id, c.age, c.income, c.employment_type,
               l.loan_amount, l.loan_type, l.interest_rate, l.default_flag,
               ch.credit_score
        FROM customers c
        JOIN loans l ON c.customer_id = l.customer_id
        JOIN credit_history ch ON c.customer_id = ch.customer_id
    """, conn)

    predictions = pd.read_sql("""
        SELECT customer_id, probability_default, risk_category, expected_loss, prediction_date
        FROM risk_predictions
        ORDER BY prediction_date DESC
    """, conn)
    conn.close()

    # ---------- Top KPI row ----------
    total_el = f"₹{format_inr(predictions['expected_loss'].sum())}" if not predictions.empty else "No predictions yet"
    kpi_row([
        {"label": "Total Customers", "value": f"{len(portfolio):,}", "color": "teal"},
        {"label": "Total Loan Book", "value": f"₹{format_inr(portfolio['loan_amount'].sum())}", "color": "navy"},
        {"label": "Historical Default Rate", "value": f"{portfolio['default_flag'].mean()*100:.1f}%", "color": "amber"},
        {"label": "Total Predicted Expected Loss", "value": total_el, "color": "red"},
    ])

    st.divider()

    CHART_TEMPLATE = "plotly_white"
    FONT = dict(family="Inter, sans-serif", color="#172433")
    RISK_COLORS = {0: "#1B7A76", 1: "#C1392B"}  # 0 = no default (teal), 1 = default (red)

    # ---------- Charts ----------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Loan Type Distribution")
        loan_type_counts = portfolio["loan_type"].value_counts().reset_index()
        loan_type_counts.columns = ["loan_type", "count"]
        fig1 = px.pie(loan_type_counts, names="loan_type", values="count", hole=0.55,
                      color_discrete_sequence=["#10243D", "#1B7A76", "#4A90A4", "#8FB8C9", "#C08A1E"])
        fig1.update_layout(template=CHART_TEMPLATE, font=FONT, margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Default Rate by Employment Type")
        emp_default = portfolio.groupby("employment_type")["default_flag"].mean().reset_index()
        emp_default["default_flag"] = emp_default["default_flag"] * 100
        fig2 = px.bar(emp_default, x="employment_type", y="default_flag",
                      labels={"default_flag": "Default Rate (%)", "employment_type": ""},
                      color_discrete_sequence=["#1B7A76"])
        fig2.update_layout(template=CHART_TEMPLATE, font=FONT, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Credit Score Distribution")
        fig3 = px.histogram(portfolio, x="credit_score", nbins=30, color="default_flag",
                            labels={"default_flag": "Defaulted", "credit_score": "Credit Score"},
                            color_discrete_map=RISK_COLORS)
        fig3.update_layout(template=CHART_TEMPLATE, font=FONT, margin=dict(t=10, b=10), bargap=0.05)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Loan Amount vs Interest Rate")
        fig4 = px.scatter(portfolio, x="loan_amount", y="interest_rate",
                          color="default_flag", opacity=0.65,
                          labels={"default_flag": "Defaulted", "loan_amount": "Loan Amount (₹)",
                                  "interest_rate": "Interest Rate (%)"},
                          color_discrete_map=RISK_COLORS)
        fig4.update_layout(template=CHART_TEMPLATE, font=FONT, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---------- Recent predictions table ----------
    st.subheader("Recent AI Risk Predictions")
    if predictions.empty:
        st.info("No predictions saved yet. Go to 'AI Risk Assessment' and save a prediction.")
    else:
        st.dataframe(predictions.head(20), use_container_width=True)

# ============================================================
# PAGE 4: WHAT-IF SIMULATOR
# ============================================================
elif page == "What-If Simulator":
    model, feature_cols, explainer = load_model_artifacts()

    if "last_application" not in st.session_state:
        st.warning("Submit or fetch a customer first (in 'New Loan Application' or "
                   "'AI Risk Assessment') before running the simulator.")
    else:
        base_data = st.session_state["last_application"].copy()

        st.subheader(f"Simulating for: {base_data['customer_id']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            sim_loan_amount = st.number_input(
                "Loan Amount (INR)", min_value=0, value=int(base_data["loan_amount"])
            )
        with col2:
            sim_interest_rate = st.number_input(
                "Interest Rate (%)", min_value=0.0, value=float(base_data["interest_rate"])
            )
        with col3:
            sim_tenure = st.selectbox(
                "Tenure (months)", [12, 24, 36, 48, 60, 84, 120],
                index=[12, 24, 36, 48, 60, 84, 120].index(base_data["tenure_months"])
                if base_data["tenure_months"] in [12, 24, 36, 48, 60, 84, 120] else 2
            )

        # ---------- Original scenario ----------
        X_original = prepare_features(base_data, feature_cols)
        pd_original = model.predict_proba(X_original)[0][1]
        el_original = pd_original * 0.45 * base_data["loan_amount"]

        # ---------- New scenario ----------
        sim_data = base_data.copy()
        sim_data["loan_amount"] = sim_loan_amount
        sim_data["interest_rate"] = sim_interest_rate
        sim_data["tenure_months"] = sim_tenure

        X_sim = prepare_features(sim_data, feature_cols)
        pd_sim = model.predict_proba(X_sim)[0][1]
        el_sim = pd_sim * 0.45 * sim_loan_amount

        st.divider()

        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Current Scenario")
            kpi_row([
                {"label": "Default Probability", "value": f"{pd_original*100:.1f}%", "color": "navy"},
                {"label": "Expected Loss", "value": f"₹{format_inr(el_original)}", "color": "navy"},
            ])

        with colB:
            st.markdown("##### New Scenario")
            arrow_pd = "▲" if pd_sim > pd_original else ("▼" if pd_sim < pd_original else "▬")
            arrow_el = "▲" if el_sim > el_original else ("▼" if el_sim < el_original else "▬")
            pd_color = "red" if pd_sim > pd_original else ("teal" if pd_sim < pd_original else "amber")
            el_color = "red" if el_sim > el_original else ("teal" if el_sim < el_original else "amber")
            kpi_row([
                {"label": "Default Probability", "value": f"{pd_sim*100:.1f}% {arrow_pd}", "color": pd_color},
                {"label": "Expected Loss", "value": f"₹{format_inr(el_sim)} {arrow_el}", "color": el_color},
            ])

        st.divider()
        if el_sim < el_original:
            st.success(f"This change reduces expected loss by ₹{format_inr(el_original-el_sim)}.")
        elif el_sim > el_original:
            st.error(f"This change increases expected loss by ₹{format_inr(el_sim-el_original)}.")
        else:
            st.info("No meaningful change in expected loss.")

# ============================================================
# PAGE 5: AI RISK ASSISTANT
# ============================================================
elif page == "AI Risk Assistant":

    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        st.warning("Add your free Groq API key in app.py (GROQ_API_KEY variable) "
                   "to enable this feature. Get one at console.groq.com")
    else:
        client = Groq(api_key=GROQ_API_KEY)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.markdown("**Try asking:**")
        example_cols = st.columns(3)
        examples = [
            "Show me the top 10 high-risk customers",
            "Which loan type has the highest default rate?",
            "What is the average expected loss for high risk customers?"
        ]
        for col, ex in zip(example_cols, examples):
            if col.button(ex):
                st.session_state["pending_question"] = ex

        question = st.text_input(
            "Ask a question",
            value=st.session_state.pop("pending_question", "")
        )

        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                answer, result_df, sql_query = ask_ai_assistant(question, client)
                st.session_state["chat_history"].insert(0, {
                    "question": question, "answer": answer,
                    "df": result_df, "sql": sql_query
                })

        for chat in st.session_state["chat_history"]:
            st.markdown(f'<div class="chat-question">{chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-answer">{chat["answer"]}</div>', unsafe_allow_html=True)
            if chat["df"] is not None and not chat["df"].empty:
                with st.expander("View data"):
                    st.dataframe(chat["df"], use_container_width=True)
                    st.caption(f"SQL used: `{chat['sql']}`")
            st.write("")
