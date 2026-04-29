"""
================================================================================
  Customer Lifetime Value (CLV) Prediction System
  Run:  streamlit run clv_app.py
  Libs: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import requests
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Prediction System",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#060a10"
PANEL    = "#0d1520"
BORDER   = "#1a2535"
TEXT     = "#e2eaf4"
MUTED    = "#6b7a99"
BLUE     = "#3b82f6"
CYAN     = "#06b6d4"
EMERALD  = "#10b981"
AMBER    = "#f59e0b"
ROSE     = "#f43f5e"
VIOLET   = "#8b5cf6"
ORANGE   = "#f97316"

SEG_COLORS = {"High": EMERALD, "Medium": AMBER, "Low": ROSE}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif;
    background: {BG};
    color: {TEXT};
  }}
  .stApp {{ background: {BG}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {PANEL} !important;
    border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {PANEL};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {BORDER};
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {MUTED} !important;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BLUE} !important;
    color: white !important;
  }}

  /* Metric cards */
  [data-testid="metric-container"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 18px !important;
  }}
  [data-testid="metric-container"] label {{ color: {MUTED} !important; font-size: 12px !important; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-family: 'Space Mono', monospace !important; }}

  /* Selectbox / slider */
  .stSelectbox > div > div,
  .stSlider > div {{ background: transparent !important; }}

  /* Divider */
  hr {{ border-color: {BORDER} !important; margin: 20px 0; }}

  /* DataFrames */
  .stDataFrame {{ background: {PANEL}; border-radius: 10px; }}

  /* Buttons */
  .stButton > button {{
    background: linear-gradient(135deg, {BLUE}, {VIOLET});
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
    transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Cards */
  .card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .card-title {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 12px;
  }}

  /* Segment badges */
  .seg-high   {{ background: #0f3024; color: {EMERALD}; border: 1px solid {EMERALD}33; }}
  .seg-medium {{ background: #2e2008; color: {AMBER};   border: 1px solid {AMBER}33; }}
  .seg-low    {{ background: #2d0e18; color: {ROSE};    border: 1px solid {ROSE}33; }}
  .badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}

  /* Model score card */
  .model-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s;
  }}
  .model-card.best {{ border-color: {BLUE}; box-shadow: 0 0 20px {BLUE}22; }}
  .model-card .model-name {{ font-size: 12px; font-weight: 600; color: {MUTED}; margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; }}
  .model-card .r2-score {{ font-size: 28px; font-weight: 700; font-family: 'Space Mono', monospace; color: {TEXT}; }}
  .model-card .sub {{ font-size: 11px; color: {MUTED}; margin-top: 4px; }}

  /* Insight row */
  .insight-row {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    background: {PANEL};
    border-left: 3px solid {BLUE};
    border-radius: 0 8px 8px 0;
    margin-bottom: 10px;
    color: {TEXT};
    font-size: 14px;
    line-height: 1.5;
  }}
  .insight-icon {{ font-size: 18px; flex-shrink: 0; margin-top: 1px; }}

  h1, h2, h3 {{ color: {TEXT} !important; font-family: 'Sora', sans-serif !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING FROM OnlineRetail.csv  (cached)
# ─────────────────────────────────────────────────────────────────────────────
url="https://drive.google.com/uc?export=download&id=1JqtZYTRRsgHJRXykHB8fV2kMEkDfT7qd"
@st.cache_data(show_spinner="📂  Loading OnlineRetail.csv…")
def load_raw(path=url):
    """Read the UCI Online Retail dataset. Tries common encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception:
            continue
    raise FileNotFoundError(
        "Cannot open OnlineRetail.csv — make sure it is in the same folder as this script."
    )


@st.cache_resource(show_spinner="⚙️  Building CLV features and training models…")
def build_clv_system():
    np.random.seed(42)

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    raw = load_raw()

    # Standardise column names (handle possible BOM / extra spaces)
    raw.columns = raw.columns.str.strip().str.replace("\ufeff", "", regex=False)

    # Keep required columns
    needed = ["InvoiceNo", "StockCode", "Description", "Quantity",
              "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    raw = raw[[c for c in needed if c in raw.columns]].copy()

    # Drop rows without CustomerID or with returns (negative qty / C-prefix invoices)
    raw = raw.dropna(subset=["CustomerID"])
    raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]
    raw = raw[raw["Quantity"] > 0]
    raw = raw[raw["UnitPrice"] > 0]

    # Parse date — handle dd-mm-yyyy HH:MM and mm/dd/yyyy HH:MM
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], format="mixed", dayfirst=False)
    raw["CustomerID"]  = raw["CustomerID"].astype(int).astype(str)

    # Revenue per line
    raw["Revenue"] = raw["Quantity"] * raw["UnitPrice"]

    # ── 2. Build per-customer RFM features ────────────────────────────────────
    snapshot = raw["InvoiceDate"].max() + pd.Timedelta(days=1)  # "today"

    rfm = (
        raw.groupby("CustomerID").agg(
            LastPurchase   = ("InvoiceDate", "max"),
            FirstPurchase  = ("InvoiceDate", "min"),
            NumInvoices    = ("InvoiceNo",   "nunique"),
            TotalRevenue   = ("Revenue",     "sum"),
            NumItems       = ("Quantity",    "sum"),
            NumProducts    = ("StockCode",   "nunique"),
            AvgOrderValue  = ("Revenue",     lambda x: x.groupby(raw.loc[x.index, "InvoiceNo"]).sum().mean()),
            Country        = ("Country",     "first"),
        ).reset_index()
    )

    rfm["RecencyDays"]  = (snapshot - rfm["LastPurchase"]).dt.days
    rfm["TenureMonths"] = ((rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days / 30.44).round(1)
    rfm["TenureMonths"] = rfm["TenureMonths"].clip(lower=0.1)

    # Purchase frequency = invoices per month of tenure
    rfm["PurchaseFreq"] = (rfm["NumInvoices"] / rfm["TenureMonths"]).round(3)
    rfm["PurchaseFreq"] = np.where(
        np.isinf(rfm["PurchaseFreq"]), rfm["NumInvoices"], rfm["PurchaseFreq"]
    )

    # NumCategories proxy = distinct products bought
    rfm["NumCategories"] = rfm["NumProducts"].clip(upper=50)

    # CLV = total historical revenue (ground truth for ML)
    rfm["CLV"] = rfm["TotalRevenue"].round(2)

    # Drop extreme outliers (top 0.5%)
    cap = rfm["CLV"].quantile(0.995)
    rfm = rfm[rfm["CLV"] <= cap].copy()

    # ── 3. Derive synthetic-but-realistic engagement cols ─────────────────────
    # (OnlineRetail has no channel/gender — we impute from country & behaviour)
    np.random.seed(42)
    N = len(rfm)

    # Channel: infer from country (UK = higher organic share)
    is_uk = rfm["Country"].str.strip().str.lower() == "united kingdom"
    rfm["Channel"] = np.where(
        is_uk,
        np.random.choice(["Organic", "Email", "Referral", "Paid", "Social"],
                         N, p=[0.40, 0.25, 0.15, 0.12, 0.08]),
        np.random.choice(["Paid", "Social", "Referral", "Organic", "Email"],
                         N, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    )

    # Region: map country → macro-region
    region_map = {
        "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
        "Netherlands": "Europe", "Belgium": "Europe", "Switzerland": "Europe",
        "Spain": "Europe", "Portugal": "Europe", "Italy": "Europe",
        "Australia": "Asia-Pacific", "Japan": "Asia-Pacific", "Singapore": "Asia-Pacific",
        "USA": "Americas", "Canada": "Americas", "Brazil": "Americas",
        "EIRE": "Europe", "Norway": "Europe", "Denmark": "Europe",
        "Sweden": "Europe", "Finland": "Europe", "Austria": "Europe",
    }
    rfm["Region"] = rfm["Country"].map(region_map).fillna("Other")

    rfm["Gender"]          = np.random.choice(["Male","Female","Other"], N, p=[0.42, 0.54, 0.04])
    rfm["DiscountUsage"]   = np.round(np.random.uniform(0, 1, N), 2)
    rfm["SupportTickets"]  = np.random.randint(0, 8, N)
    rfm["EmailOpenRate"]   = np.round(np.random.uniform(0, 1, N), 2)

    # ── 4. RFM scoring (1–5) ─────────────────────────────────────────────────
    def safe_qcut(series, n, labels, ascending=True):
        """qcut with tie-breaking via rank."""
        return pd.qcut(series.rank(method="first"), n, labels=labels).astype(int)

    rfm["R_Score"] = safe_qcut(rfm["RecencyDays"],  5, [5,4,3,2,1])   # lower recency = better
    rfm["F_Score"] = safe_qcut(rfm["PurchaseFreq"], 5, [1,2,3,4,5])
    rfm["M_Score"] = safe_qcut(rfm["AvgOrderValue"],5, [1,2,3,4,5])
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # ── 5. Segmentation ──────────────────────────────────────────────────────
    p70 = rfm["CLV"].quantile(0.70)
    p35 = rfm["CLV"].quantile(0.35)
    rfm["Segment"] = rfm["CLV"].apply(
        lambda v: "High" if v >= p70 else ("Medium" if v >= p35 else "Low")
    )

    customers = rfm.rename(columns={"CustomerID": "CustomerID"}).copy()

    # ── 6. Encode categoricals ────────────────────────────────────────────────
    le_region  = LabelEncoder()
    le_channel = LabelEncoder()
    le_gender  = LabelEncoder()
    customers["Region_enc"]  = le_region.fit_transform(customers["Region"])
    customers["Channel_enc"] = le_channel.fit_transform(customers["Channel"])
    customers["Gender_enc"]  = le_gender.fit_transform(customers["Gender"])

    FEATURES = [
        "TenureMonths", "AvgOrderValue", "PurchaseFreq", "RecencyDays",
        "NumCategories", "DiscountUsage", "SupportTickets", "EmailOpenRate",
        "RFM_Score", "Region_enc", "Channel_enc", "Gender_enc",
    ]
    X = customers[FEATURES].values
    y = customers["CLV"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── 7. Train models ───────────────────────────────────────────────────────
    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression"  : Ridge(alpha=10.0),
        "Lasso Regression"  : Lasso(alpha=5.0, max_iter=5000),
        "Random Forest"     : RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, random_state=42),
    }

    results     = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
        cv = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
        results[name] = {
            "R2"    : round(r2_score(y_test, y_pred), 4),
            "RMSE"  : round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "MAE"   : round(mean_absolute_error(y_test, y_pred), 2),
            "CV_R2" : round(cv.mean(), 4),
            "CV_Std": round(cv.std(), 4),
        }

    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = models[best_model_name]

    # Feature importances (Random Forest)
    rf = models["Random Forest"]
    feat_imp = pd.DataFrame({
        "Feature"   : FEATURES,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Predict CLV for all customers
    customers["PredictedCLV"] = np.round(best_model.predict(X_scaled), 2)
    customers["CLV_Error"]    = np.abs(customers["CLV"] - customers["PredictedCLV"])

    return dict(
        customers       = customers,
        raw             = raw,           # ← real transaction rows for extra tabs
        results         = results,
        predictions     = predictions,
        best_model_name = best_model_name,
        best_model      = best_model,
        feat_imp        = feat_imp,
        FEATURES        = FEATURES,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models          = models,
        scaler          = scaler,
        snapshot        = snapshot,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────────────────────────────────────
def dark_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    ax.grid(color=BORDER, alpha=0.6, linewidth=0.5)
    return ax

def make_fig(ncols=1, nrows=1, w=10, h=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=BG)
    if ncols == 1 and nrows == 1:
        dark_ax(axes)
        return fig, axes
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes_flat:
        dark_ax(ax)
    return fig, axes


def badge_html(seg):
    cls = f"seg-{seg.lower()}"
    return f'<span class="badge {cls}">{seg}</span>'


def insight(icon, text):
    st.markdown(
        f'<div class="insight-row"><span class="insight-icon">{icon}</span><span>{text}</span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
try:
    data = build_clv_system()
except FileNotFoundError as _e:
    st.error(
        "⚠️ **OnlineRetail.csv not found.**\n\n"
        "Place `OnlineRetail.csv` in the **same folder** as `clv_app_real.py` and restart."
    )
    st.stop()
df   = data["customers"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 💎 CLV System")
    st.markdown("---")

    st.markdown(f"<div class='card-title'>🎯 Customer Filter</div>", unsafe_allow_html=True)
    seg_filter  = st.multiselect("Segment", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    chan_filter = st.multiselect("Channel",  df["Channel"].unique().tolist(), default=df["Channel"].unique().tolist())
    reg_filter  = st.multiselect("Region",   df["Region"].unique().tolist(),  default=df["Region"].unique().tolist())

    st.markdown("---")
    st.markdown(f"<div class='card-title'>🤖 Best Model</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{BLUE};font-weight:700;font-size:14px'>{data['best_model_name']}</div>"
        f"<div style='color:{MUTED};font-size:12px;margin-top:4px'>"
        f"R² = {data['results'][data['best_model_name']]['R2']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f"<div class='card-title'>🔮 Predict a Customer</div>", unsafe_allow_html=True)
    p_tenure  = st.slider("Tenure (months)",   1, 60,  24)
    p_aov     = st.slider("Avg Order Value ($)", 10, 2000, 200)
    p_freq    = st.slider("Purchase Freq (mo)", 0.1, 15.0, 3.0, step=0.1)
    p_recency = st.slider("Recency (days)",     1, 365, 30)
    p_cats    = st.slider("Num Categories",     1, 10, 3)

    if st.button("⚡ Predict CLV"):
        scaler2 = data["scaler"]   # reuse fitted scaler
        # RFM_Score approximation: R~3 F based on freq M based on AOV
        approx_rfm = min(15, max(3, int(p_freq * 2) + 3))
        row = np.array([[p_tenure, p_aov, p_freq, p_recency, p_cats,
                         0.3, 1, 0.4, approx_rfm, 0, 0, 0]])
        pred_clv = data["best_model"].predict(scaler2.transform(row))[0]
        seg = "High" if pred_clv >= df["CLV"].quantile(0.70) else "Medium" if pred_clv >= df["CLV"].quantile(0.35) else "Low"
        st.markdown(
            f"<div class='card' style='text-align:center;margin-top:8px'>"
            f"<div style='color:{MUTED};font-size:11px;letter-spacing:1px'>PREDICTED CLV</div>"
            f"<div style='font-size:36px;font-weight:700;font-family:Space Mono;color:{EMERALD};margin:8px 0'>"
            f"${pred_clv:,.0f}</div>"
            f"{badge_html(seg)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    n_cust  = len(df)
    n_trans = len(data.get("raw", pd.DataFrame()))
    st.markdown(
        f"<span style='color:{MUTED};font-size:11px'>"
        f"{n_cust:,} real customers · {n_trans:,} transactions · 5 ML models</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
filtered = df[
    df["Segment"].isin(seg_filter) &
    df["Channel"].isin(chan_filter) &
    df["Region"].isin(reg_filter)
]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
# ── Data source info banner ──────────────────────────────────────────────────
raw_info = data.get("raw", pd.DataFrame())
st.markdown(
    f"<div style='background:#0d1a2e;border:1px solid #1a3a5c;border-radius:10px;"
    f"padding:10px 18px;margin-bottom:12px;font-size:13px;color:#6b9fd4'>"
    f"📂 <b>Data source:</b> OnlineRetail.csv &nbsp;|&nbsp; "
    f"<b>{len(raw_info):,}</b> transactions &nbsp;|&nbsp; "
    f"<b>{raw_info['CustomerID'].nunique() if len(raw_info) else 0:,}</b> unique customers &nbsp;|&nbsp; "
    f"<b>{raw_info['Country'].nunique() if len(raw_info) else 0}</b> countries &nbsp;|&nbsp; "
    f"<b>{raw_info['InvoiceDate'].min().strftime('%b %Y') if len(raw_info) else '?'}</b>"
    f" → "
    f"<b>{raw_info['InvoiceDate'].max().strftime('%b %Y') if len(raw_info) else '?'}</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='font-size:32px;margin-bottom:0;font-weight:700'>💎 Customer Lifetime Value Prediction System</h1>"
    "<p style='color:#6b7a99;margin-top:6px;font-size:14px'>UCI Online Retail Dataset · RFM + ML · Customer Segmentation · Revenue Optimization</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("👥 Customers",    f"{len(filtered):,}")
k2.metric("💰 Avg CLV",      f"${filtered['CLV'].mean():,.0f}")
k3.metric("🏆 Avg Pred CLV", f"${filtered['PredictedCLV'].mean():,.0f}")
k4.metric("📈 High Value",   f"{(filtered['Segment']=='High').sum():,}")
k5.metric("💳 Avg Order",    f"${filtered['AvgOrderValue'].mean():,.0f}")
k6.metric("🔄 Avg Freq",     f"{filtered['PurchaseFreq'].mean():.1f}/mo")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 Overview",
    "🤖 Model Performance",
    "🔍 Feature Analysis",
    "🧩 Customer Segments",
    "📋 Customer Explorer",
    "🧾 Raw Transactions",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("### 📊 Business Overview")

    col1, col2 = st.columns(2)

    # CLV distribution
    with col1:
        fig, ax = make_fig(w=6, h=4)
        for seg, color in SEG_COLORS.items():
            sub = filtered[filtered["Segment"] == seg]["CLV"]
            ax.hist(sub, bins=40, color=color, alpha=0.7, label=seg, edgecolor="none")
        ax.set_title("CLV Distribution by Segment")
        ax.set_xlabel("Customer Lifetime Value ($)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        ax.axvline(filtered["CLV"].mean(), color=CYAN, lw=1.5, ls="--", label=f"Mean ${filtered['CLV'].mean():,.0f}")
        st.pyplot(fig); plt.close(fig)

    # Segment pie
    with col2:
        seg_counts = filtered["Segment"].value_counts()
        fig, ax = make_fig(w=6, h=4)
        wedges, texts, autotexts = ax.pie(
            seg_counts.values,
            labels=seg_counts.index,
            colors=[SEG_COLORS[s] for s in seg_counts.index],
            autopct="%1.1f%%", startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(edgecolor=BG, linewidth=2),
        )
        for at in autotexts: at.set_color(TEXT); at.set_fontsize(10)
        for t  in texts:     t.set_color(TEXT); t.set_fontsize(11)
        centre_circle = plt.Circle((0, 0), 0.55, fc=BG)
        ax.add_patch(centre_circle)
        ax.set_title("Customer Segment Distribution")
        st.pyplot(fig); plt.close(fig)

    col3, col4 = st.columns(2)

    # CLV by channel
    with col3:
        ch_clv = filtered.groupby("Channel")["CLV"].mean().sort_values(ascending=True)
        fig, ax = make_fig(w=6, h=3.5)
        bars = ax.barh(ch_clv.index, ch_clv.values, color=BLUE, edgecolor="none", height=0.6)
        for bar, val in zip(bars, ch_clv.values):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                    f"${val:,.0f}", va="center", fontsize=9, color=TEXT)
        ax.set_title("Avg CLV by Acquisition Channel")
        ax.set_xlabel("Avg CLV ($)")
        st.pyplot(fig); plt.close(fig)

    # CLV by region
    with col4:
        reg_clv = filtered.groupby("Region")["CLV"].mean().sort_values(ascending=False)
        fig, ax = make_fig(w=6, h=3.5)
        colors_reg = [EMERALD, CYAN, AMBER, VIOLET]
        ax.bar(reg_clv.index, reg_clv.values, color=colors_reg[:len(reg_clv)], edgecolor="none", width=0.5)
        for i, (reg, val) in enumerate(reg_clv.items()):
            ax.text(i, val + 10, f"${val:,.0f}", ha="center", fontsize=9, color=TEXT)
        ax.set_title("Avg CLV by Region")
        ax.set_ylabel("Avg CLV ($)")
        st.pyplot(fig); plt.close(fig)

    # Tenure vs CLV
    col5, col6 = st.columns(2)
    with col5:
        sample = filtered.sample(min(500, len(filtered)), random_state=1)
        fig, ax = make_fig(w=6, h=4)
        sc = ax.scatter(sample["TenureMonths"], sample["CLV"],
                        c=sample["CLV"], cmap="plasma", alpha=0.5, s=15, linewidths=0)
        ax.set_title("Tenure vs CLV")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("CLV ($)")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
        st.pyplot(fig); plt.close(fig)

    # Purchase freq vs CLV
    with col6:
        fig, ax = make_fig(w=6, h=4)
        for seg, color in SEG_COLORS.items():
            sub = filtered[filtered["Segment"] == seg]
            ax.scatter(sub["PurchaseFreq"], sub["CLV"],
                       color=color, alpha=0.3, s=12, label=seg, linewidths=0)
        ax.set_title("Purchase Frequency vs CLV")
        ax.set_xlabel("Purchase Frequency (orders/month)")
        ax.set_ylabel("CLV ($)")
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        st.pyplot(fig); plt.close(fig)

    # Revenue summary
    st.markdown("---")
    st.markdown("#### 💰 Revenue Concentration")
    rev1, rev2, rev3 = st.columns(3)
    high_rev = filtered[filtered["Segment"]=="High"]["CLV"].sum()
    med_rev  = filtered[filtered["Segment"]=="Medium"]["CLV"].sum()
    low_rev  = filtered[filtered["Segment"]=="Low"]["CLV"].sum()
    total_rev= filtered["CLV"].sum()
    rev1.metric("🏆 High Segment Revenue", f"${high_rev:,.0f}", f"{high_rev/total_rev:.1%} of total")
    rev2.metric("⚡ Medium Segment Revenue",f"${med_rev:,.0f}",  f"{med_rev/total_rev:.1%} of total")
    rev3.metric("📉 Low Segment Revenue",   f"${low_rev:,.0f}",  f"{low_rev/total_rev:.1%} of total")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### 🤖 Model Performance Comparison")
    st.markdown(
        f"<p style='color:{MUTED};font-size:13px'>5 regression models trained on 1600 samples, "
        f"tested on 400. Cross-validated (5-fold). Best: <b style='color:{BLUE}'>"
        f"{data['best_model_name']}</b></p>",
        unsafe_allow_html=True,
    )

    # Model cards row
    cols = st.columns(5)
    for col, (name, res) in zip(cols, data["results"].items()):
        is_best = name == data["best_model_name"]
        col.markdown(
            f"<div class='model-card {'best' if is_best else ''}'>"
            f"<div class='model-name'>{name}</div>"
            f"<div class='r2-score' style='color:{BLUE if is_best else TEXT}'>{res['R2']}</div>"
            f"<div class='sub'>R² Score</div>"
            f"<hr style='border-color:{BORDER};margin:10px 0'>"
            f"<div style='font-size:11px;color:{MUTED}'>RMSE: <b style='color:{TEXT}'>${res['RMSE']:,}</b></div>"
            f"<div style='font-size:11px;color:{MUTED}'>MAE:  <b style='color:{TEXT}'>${res['MAE']:,}</b></div>"
            f"<div style='font-size:11px;color:{MUTED}'>CV R²: <b style='color:{TEXT}'>{res['CV_R2']} ±{res['CV_Std']}</b></div>"
            f"{'<div style=\"margin-top:8px\"><span class=\"badge seg-high\">BEST</span></div>' if is_best else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Predicted vs Actual for best model
    col_a, col_b = st.columns(2)
    with col_a:
        y_test, y_pred = data["predictions"][data["best_model_name"]]
        fig, ax = make_fig(w=6, h=4.5)
        ax.scatter(y_test, y_pred, alpha=0.35, s=12, color=CYAN, linewidths=0)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "--", color=ROSE, lw=1.5, label="Perfect fit")
        ax.set_title(f"Predicted vs Actual CLV\n({data['best_model_name']})")
        ax.set_xlabel("Actual CLV ($)")
        ax.set_ylabel("Predicted CLV ($)")
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        st.pyplot(fig); plt.close(fig)

    # Residuals
    with col_b:
        residuals = y_test - y_pred
        fig, ax = make_fig(w=6, h=4.5)
        ax.scatter(y_pred, residuals, alpha=0.35, s=12, color=VIOLET, linewidths=0)
        ax.axhline(0, color=ROSE, ls="--", lw=1.5)
        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted CLV ($)")
        ax.set_ylabel("Residual ($)")
        ax.grid(color=BORDER, alpha=0.4)
        st.pyplot(fig); plt.close(fig)

    # Model comparison bar chart
    res_df = pd.DataFrame(data["results"]).T.reset_index().rename(columns={"index": "Model"})
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = make_fig(w=6, h=3.5)
        colors_m = [BLUE if m == data["best_model_name"] else MUTED for m in res_df["Model"]]
        bars = ax.bar(res_df["Model"], res_df["R2"].astype(float),
                      color=colors_m, edgecolor="none", width=0.55)
        for bar, val in zip(bars, res_df["R2"].astype(float)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.4f}", ha="center", fontsize=8, color=TEXT)
        ax.set_title("R² Score by Model")
        ax.set_ylabel("R² Score")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig); plt.close(fig)

    with col_d:
        fig, ax = make_fig(w=6, h=3.5)
        bars = ax.bar(res_df["Model"], res_df["RMSE"].astype(float),
                      color=[ROSE if m == data["best_model_name"] else MUTED for m in res_df["Model"]],
                      edgecolor="none", width=0.55)
        for bar, val in zip(bars, res_df["RMSE"].astype(float)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                    f"${val:,.0f}", ha="center", fontsize=8, color=TEXT)
        ax.set_title("RMSE by Model (lower = better)")
        ax.set_ylabel("RMSE ($)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        st.pyplot(fig); plt.close(fig)

    # Cross-validation detail
    st.markdown("#### 📐 Cross-Validation Results (5-Fold)")
    cv_df = res_df[["Model", "CV_R2", "CV_Std", "R2", "RMSE", "MAE"]].copy()
    cv_df.columns = ["Model", "CV R²", "CV Std Dev", "Test R²", "Test RMSE ($)", "Test MAE ($)"]
    st.dataframe(cv_df.set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 🔍 Feature Importance & Correlation Analysis")

    col_a, col_b = st.columns(2)

    # Feature importance
    with col_a:
        fi = data["feat_imp"]
        fig, ax = make_fig(w=6, h=5)
        palette = [BLUE, CYAN, EMERALD, AMBER, VIOLET, ROSE, ORANGE, BLUE, CYAN, EMERALD, AMBER, VIOLET]
        bars = ax.barh(fi["Feature"][::-1], fi["Importance"][::-1],
                       color=palette[:len(fi)], edgecolor="none", height=0.65)
        ax.set_title("Feature Importance (Random Forest)")
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, fi["Importance"][::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color=TEXT)
        st.pyplot(fig); plt.close(fig)

    # Correlation heatmap
    with col_b:
        num_cols = ["TenureMonths", "AvgOrderValue", "PurchaseFreq", "RecencyDays",
                    "NumCategories", "DiscountUsage", "SupportTickets",
                    "EmailOpenRate", "RFM_Score", "CLV"]
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(
            corr, mask=mask, ax=ax, cmap="coolwarm",
            annot=True, fmt=".2f", annot_kws={"size": 7, "color": TEXT},
            linewidths=0.5, linecolor=BG,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=12, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=7)
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig); plt.close(fig)

    # Pairplot-style: top-4 features vs CLV
    st.markdown("#### 🔗 Top Features vs CLV")
    top_feats = data["feat_imp"]["Feature"].head(4).tolist()
    cols_pair = st.columns(4)
    for i, feat in enumerate(top_feats):
        with cols_pair[i]:
            sample = filtered.sample(min(300, len(filtered)), random_state=42)
            fig, ax = make_fig(w=3.5, h=3.5)
            sc = ax.scatter(sample[feat], sample["CLV"],
                            c=sample["CLV"], cmap="viridis", alpha=0.4, s=12, linewidths=0)
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel("CLV", fontsize=8)
            st.pyplot(fig); plt.close(fig)

    # Distribution of each feature by segment
    st.markdown("#### 📦 Feature Distributions by Segment")
    sel_feat = st.selectbox("Select feature to compare across segments",
                            ["AvgOrderValue", "PurchaseFreq", "TenureMonths",
                             "RecencyDays", "EmailOpenRate", "DiscountUsage",
                             "NumCategories", "SupportTickets"])
    fig, ax = make_fig(w=10, h=4)
    for seg, color in SEG_COLORS.items():
        sub = filtered[filtered["Segment"]==seg][sel_feat]
        ax.hist(sub, bins=40, color=color, alpha=0.6, label=seg, edgecolor="none")
    ax.set_title(f"{sel_feat} Distribution by Segment")
    ax.set_xlabel(sel_feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 🧩 Customer Segmentation & RFM Analysis")

    # Segment summary table
    seg_summary = filtered.groupby("Segment").agg(
        Count        = ("CustomerID",     "count"),
        Avg_CLV      = ("CLV",            "mean"),
        Total_CLV    = ("CLV",            "sum"),
        Avg_Order    = ("AvgOrderValue",  "mean"),
        Avg_Freq     = ("PurchaseFreq",   "mean"),
        Avg_Tenure   = ("TenureMonths",   "mean"),
        Avg_Recency  = ("RecencyDays",    "mean"),
        Avg_RFM      = ("RFM_Score",      "mean"),
    ).round(2).reset_index()

    col_s1, col_s2, col_s3 = st.columns(3)
    for col, seg_name, color in zip([col_s1, col_s2, col_s3],
                                    ["High", "Medium", "Low"],
                                    [EMERALD, AMBER, ROSE]):
        row = seg_summary[seg_summary["Segment"] == seg_name]
        if row.empty: continue
        row = row.iloc[0]
        col.markdown(
            f"<div class='card'>"
            f"<div class='card-title'>{badge_html(seg_name)} &nbsp; Segment</div>"
            f"<div style='font-size:28px;font-weight:700;color:{color};font-family:Space Mono'>${row['Avg_CLV']:,.0f}</div>"
            f"<div style='color:{MUTED};font-size:11px;margin-bottom:12px'>Avg CLV</div>"
            f"<div style='font-size:12px;color:{MUTED}'>Customers: <b style='color:{TEXT}'>{int(row['Count']):,}</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Total Revenue: <b style='color:{TEXT}'>${row['Total_CLV']:,.0f}</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Avg Order: <b style='color:{TEXT}'>${row['Avg_Order']:,.0f}</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Avg Freq: <b style='color:{TEXT}'>{row['Avg_Freq']:.1f}/mo</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Avg Tenure: <b style='color:{TEXT}'>{row['Avg_Tenure']:.0f} mo</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Avg Recency: <b style='color:{TEXT}'>{row['Avg_Recency']:.0f} days</b></div>"
            f"<div style='font-size:12px;color:{MUTED}'>Avg RFM: <b style='color:{TEXT}'>{row['Avg_RFM']:.1f}</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # RFM scatter
    st.markdown("#### 🎯 RFM Score vs CLV")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        sample = filtered.sample(min(600, len(filtered)), random_state=7)
        fig, ax = make_fig(w=6, h=4)
        for seg, color in SEG_COLORS.items():
            sub = sample[sample["Segment"]==seg]
            ax.scatter(sub["RFM_Score"], sub["CLV"], color=color,
                       alpha=0.4, s=15, label=seg, linewidths=0)
        ax.set_title("RFM Score vs CLV")
        ax.set_xlabel("RFM Score (3–15)")
        ax.set_ylabel("CLV ($)")
        ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        st.pyplot(fig); plt.close(fig)

    # Box plots: CLV per segment
    with col_r2:
        fig, ax = make_fig(w=6, h=4)
        segs = ["High", "Medium", "Low"]
        seg_data = [filtered[filtered["Segment"]==s]["CLV"].values for s in segs]
        bp = ax.boxplot(seg_data, labels=segs, patch_artist=True,
                        medianprops=dict(color=TEXT, lw=2),
                        whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED),
                        flierprops=dict(marker="o", markersize=2, color=MUTED, alpha=0.3))
        for patch, color in zip(bp["boxes"], [EMERALD, AMBER, ROSE]):
            patch.set_facecolor(color + "44")
            patch.set_edgecolor(color)
        ax.set_title("CLV Distribution by Segment")
        ax.set_ylabel("CLV $)")
        st.pyplot(fig); plt.close(fig)

    # Channel breakdown per segment
    st.markdown("#### 📡 Acquisition Channel by Segment")
    ch_seg = filtered.groupby(["Channel", "Segment"]).size().unstack(fill_value=0)
    fig, ax = make_fig(w=10, h=4)
    x = np.arange(len(ch_seg.index))
    w = 0.25
    for i, (seg, color) in enumerate(SEG_COLORS.items()):
        if seg in ch_seg.columns:
            ax.bar(x + i*w, ch_seg[seg], width=w, label=seg,
                   color=color, edgecolor="none", alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(ch_seg.index, fontsize=10)
    ax.set_title("Customer Count per Channel & Segment")
    ax.set_ylabel("Customer Count")
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    st.pyplot(fig); plt.close(fig)

    # Recommended strategies
    st.markdown("---")
    st.markdown("#### 💡 Segment-Wise Actionable Strategies")
    col_str1, col_str2, col_str3 = st.columns(3)
    with col_str1:
        st.markdown(f"<div class='card-title'>{badge_html('High')} &nbsp;High Value</div>", unsafe_allow_html=True)
        insight("🎁", "VIP loyalty rewards & early access to new products")
        insight("📞", "Dedicated account manager & priority support")
        insight("🔄", "Subscription upgrades & upsell premium tiers")
        insight("📊", "Personalized dashboards & exclusive analytics")
    with col_str2:
        st.markdown(f"<div class='card-title'>{badge_html('Medium')} &nbsp;Medium Value</div>", unsafe_allow_html=True)
        insight("📧", "Targeted email campaigns with personalized offers")
        insight("🏷️", "Bundle discounts to increase average order value")
        insight("📈", "Milestone rewards to boost purchase frequency")
        insight("🔔", "Re-engagement push notifications post-inactivity")
    with col_str3:
        st.markdown(f"<div class='card-title'>{badge_html('Low')} &nbsp;Low Value</div>", unsafe_allow_html=True)
        insight("💌", "Win-back campaigns with limited-time deep discounts")
        insight("🎯", "Survey & feedback collection to understand churn")
        insight("📦", "Free-shipping threshold to nudge second purchase")
        insight("🤝", "Referral programs to monetize relationship")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("### 📋 Customer Data Explorer")

    col_x, col_y, col_z = st.columns(3)
    with col_x:
        sort_by  = st.selectbox("Sort by", ["CLV", "PredictedCLV", "AvgOrderValue", "PurchaseFreq", "TenureMonths"], index=0)
    with col_y:
        sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    with col_z:
        n_rows = st.slider("Rows to display", 10, 200, 30)

    display_cols = ["CustomerID", "Segment", "TenureMonths", "AvgOrderValue",
                    "PurchaseFreq", "RecencyDays", "RFM_Score",
                    "CLV", "PredictedCLV", "CLV_Error",
                    "Channel", "Region", "Gender"]

    table = (
        filtered[display_cols]
        .sort_values(sort_by, ascending=(sort_dir == "Ascending"))
        .head(n_rows)
        .reset_index(drop=True)
    )
    table.index += 1

    # Colour-code Segment column
    def highlight_segment(val):
        colors_map = {"High": "#0f3024", "Medium": "#2e2008", "Low": "#2d0e18"}
        text_map   = {"High": EMERALD,   "Medium": AMBER,     "Low": ROSE}
        bg = colors_map.get(val, "")
        fg = text_map.get(val, TEXT)
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"

    styled = table.style.map(highlight_segment, subset=["Segment"]) \
                        .format({"CLV": "${:,.0f}", "PredictedCLV": "${:,.0f}",
                                 "CLV_Error": "${:,.0f}", "AvgOrderValue": "${:,.0f}",
                                 "PurchaseFreq": "{:.2f}", "AvgOrderValue": "${:,.0f}"})
    st.dataframe(styled, use_container_width=True, height=500)

    # Download
    csv_buf = io.StringIO()
    filtered[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️  Download Filtered Data (CSV)",
        data=csv_buf.getvalue().encode(),
        file_name="clv_predictions.csv",
        mime="text/csv",
    )

    # Mini stats
    st.markdown("---")
    st.markdown("#### 📈 Quick Stats on Displayed Data")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Avg Predicted CLV", f"${table['PredictedCLV'].mean():,.0f}")
    q2.metric("Max CLV",           f"${table['CLV'].max():,.0f}")
    q3.metric("Avg Prediction Error", f"${table['CLV_Error'].mean():,.0f}")
    q4.metric("Avg RFM Score",     f"{table['RFM_Score'].mean():.1f}")

    # Error histogram
    fig, ax = make_fig(w=10, h=3.5)
    ax.hist(filtered["CLV_Error"], bins=60, color=VIOLET, alpha=0.8, edgecolor="none")
    ax.axvline(filtered["CLV_Error"].mean(), color=AMBER, lw=1.5, ls="--",
               label=f"Mean Error = ${filtered['CLV_Error'].mean():,.0f}")
    ax.set_title("Prediction Error Distribution (|Actual − Predicted|)")
    ax.set_xlabel("Absolute Error ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown("### 🧾 Raw Transaction Data (OnlineRetail.csv)")

    raw = data.get("raw", pd.DataFrame())
    if raw.empty:
        st.warning("Raw transaction data not available.")
    else:
        # ── Summary KPIs ──────────────────────────────────────────────────────
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("📦 Total Transactions", f"{len(raw):,}")
        r2.metric("👥 Unique Customers",   f"{raw['CustomerID'].nunique():,}")
        r3.metric("🌍 Countries",          f"{raw['Country'].nunique()}")
        r4.metric("🗓️ Date Range",
                  f"{raw['InvoiceDate'].min().strftime('%b %Y')} – "
                  f"{raw['InvoiceDate'].max().strftime('%b %Y')}")
        r5.metric("💰 Total Revenue",      f"£{raw['Revenue'].sum():,.0f}")

        st.markdown("---")

        # ── Top countries by revenue ──────────────────────────────────────────
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🌍 Revenue by Country (Top 10)")
            top_countries = (
                raw.groupby("Country")["Revenue"].sum()
                .sort_values(ascending=False).head(10).reset_index()
            )
            fig, ax = make_fig(w=6, h=4)
            bars = ax.barh(top_countries["Country"][::-1], top_countries["Revenue"][::-1],
                           color=BLUE, alpha=0.85, edgecolor="none")
            ax.set_xlabel("Revenue (£)")
            ax.set_title("Top 10 Countries by Revenue")
            for bar, val in zip(bars, top_countries["Revenue"][::-1]):
                ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                        f"£{val:,.0f}", va="center", fontsize=8, color=MUTED)
            st.pyplot(fig); plt.close(fig)

        with col_r2:
            st.markdown("#### 📅 Monthly Revenue Trend")
            raw["YearMonth"] = raw["InvoiceDate"].dt.to_period("M")
            monthly = raw.groupby("YearMonth")["Revenue"].sum().reset_index()
            monthly["YearMonth"] = monthly["YearMonth"].astype(str)
            fig, ax = make_fig(w=6, h=4)
            ax.plot(monthly["YearMonth"], monthly["Revenue"], color=CYAN, lw=2, marker="o", ms=4)
            ax.fill_between(monthly["YearMonth"], monthly["Revenue"], alpha=0.15, color=CYAN)
            ax.set_title("Monthly Revenue (£)")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue (£)")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            st.pyplot(fig); plt.close(fig)

        # ── Top products ──────────────────────────────────────────────────────
        st.markdown("#### 🏷️ Top 15 Products by Revenue")
        top_prods = (
            raw.groupby("Description")["Revenue"].sum()
            .sort_values(ascending=False).head(15).reset_index()
        )
        fig, ax = make_fig(w=12, h=4)
        ax.bar(range(len(top_prods)), top_prods["Revenue"], color=VIOLET, alpha=0.85, edgecolor="none")
        ax.set_xticks(range(len(top_prods)))
        ax.set_xticklabels(top_prods["Description"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Revenue (£)")
        ax.set_title("Top 15 Products by Total Revenue")
        st.pyplot(fig); plt.close(fig)

        # ── Searchable raw data table ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔎 Search & Browse Transactions")
        col_s1, col_s2, col_s3 = st.columns(3)
        cid_search  = col_s1.text_input("Filter by CustomerID", placeholder="e.g. 17850")
        desc_search = col_s2.text_input("Filter by Description", placeholder="e.g. HEART")
        country_sel = col_s3.selectbox("Country", ["All"] + sorted(raw["Country"].unique().tolist()))

        raw_view = raw.copy()
        if cid_search:
            raw_view = raw_view[raw_view["CustomerID"].astype(str).str.contains(cid_search)]
        if desc_search:
            raw_view = raw_view[raw_view["Description"].str.contains(desc_search, case=False, na=False)]
        if country_sel != "All":
            raw_view = raw_view[raw_view["Country"] == country_sel]

        st.caption(f"Showing {min(500, len(raw_view)):,} of {len(raw_view):,} matching rows")
        st.dataframe(
            raw_view[["InvoiceNo","CustomerID","Description","Quantity",
                       "UnitPrice","Revenue","InvoiceDate","Country"]]
            .head(500).reset_index(drop=True),
            use_container_width=True, height=400
        )

        csv_raw = raw_view.head(500).to_csv(index=False).encode()
        st.download_button("⬇️ Download filtered transactions (CSV)",
                           data=csv_raw, file_name="transactions_filtered.csv", mime="text/csv")
