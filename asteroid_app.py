import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score
)
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Asteroid Hazard Predictor",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #020b18;
    color: #c8e6ff;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px;
}

.stApp {
    background: radial-gradient(ellipse at top left, #051428 0%, #020b18 60%);
}

.metric-card {
    background: linear-gradient(135deg, #0a1f35 0%, #071525 100%);
    border: 1px solid #1a4a6e;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(30, 120, 200, 0.15);
}

.hazard-badge {
    background: linear-gradient(135deg, #7f0000, #c0392b);
    color: white;
    padding: 12px 28px;
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    border: 1px solid #ff4444;
    box-shadow: 0 0 30px rgba(255, 50, 50, 0.4);
    letter-spacing: 2px;
}

.safe-badge {
    background: linear-gradient(135deg, #003d00, #1a7a1a);
    color: #aaffaa;
    padding: 12px 28px;
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    border: 1px solid #44ff44;
    box-shadow: 0 0 30px rgba(50, 255, 50, 0.3);
    letter-spacing: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #0d3b6e, #1a6bb5);
    color: #ffffff;
    border: 1px solid #2a8fd4;
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 10px 20px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1a6bb5, #2a8fd4);
    box-shadow: 0 0 20px rgba(42, 143, 212, 0.5);
    border-color: #4ab0f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030f1f 0%, #020b18 100%);
    border-right: 1px solid #1a4a6e;
}

.stSlider [data-baseweb="slider"] {
    accent-color: #2a8fd4;
}

div[data-testid="metric-container"] {
    background: #0a1f35;
    border: 1px solid #1a4a6e;
    border-radius: 8px;
    padding: 12px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}

hr {
    border-color: #1a4a6e;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4ab0f0; margin-bottom:0;'>
    ☄️ ASTEROID HAZARD PREDICTOR
</h1>
<p style='text-align:center; color:#5a8ab0; font-family:Share Tech Mono; font-size:0.85rem; letter-spacing:3px;'>
    POWERED BY NASA DATA · XGBOOST MODEL · REAL-TIME ANALYSIS
</p>
<hr>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'model_xgb' not in st.session_state:
    st.session_state.model_xgb = None
if 'model_rf' not in st.session_state:
    st.session_state.model_rf = None
if 'model_lr' not in st.session_state:
    st.session_state.model_lr = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#4ab0f0;'>⚙ MISSION CONTROL</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**STEP 1 — Upload Dataset**")
    uploaded_file = st.file_uploader("Upload NASA asteroid CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**STEP 2 — Train Models**")
    train_btn = st.button("🚀 Train All Models", use_container_width=True)

    st.markdown("---")
    st.markdown("<small style='color:#5a8ab0;'>Models: Logistic Regression · Random Forest · XGBoost</small>", unsafe_allow_html=True)

# ── Helper: preprocess ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=['pha'])

    numerical_cols = [c for c in ['diameter', 'albedo', 'H_sigma', 'e', 'a', 'i', 'moid', 'condition_code'] if c in df.columns]
    if numerical_cols:
        imputer_median = SimpleImputer(strategy='median')
        df[numerical_cols] = imputer_median.fit_transform(df[numerical_cols])

    if 'class' in df.columns:
        imputer_mode = SimpleImputer(strategy='most_frequent')
        df[['class']] = imputer_mode.fit_transform(df[['class']])

    if 'diameter' in df.columns and 'albedo' in df.columns and 'H_sigma' in df.columns:
        df['diameter'] = df.apply(
            lambda row: (1329 / (row['albedo']**0.5)) * (10**(-0.2 * row['H_sigma']))
            if pd.isna(row['diameter']) else row['diameter'], axis=1
        )

    df['pha'] = df['pha'].map({'Y': 1, 'N': 0, True: 1, False: 0})
    df = df.dropna(subset=['pha'])
    df['pha'] = df['pha'].astype(int)

    if 'class' in df.columns:
        df = pd.get_dummies(df, columns=['class'])

    X = df.drop(columns=['full_name', 'neo', 'pha', 'class'], errors='ignore')
    y = df['pha']
    return X, y

# ── Train models ──────────────────────────────────────────────────────────────
def train_models(df):
    with st.spinner("Preprocessing data..."):
        X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    with st.spinner("Training Logistic Regression..."):
        model_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        model_lr.fit(X_train_s, y_train)
        y_pred_lr = model_lr.predict(X_test_s)

    with st.spinner("Training Random Forest..."):
        model_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
        model_rf.fit(X_train_s, y_train)
        y_pred_rf = model_rf.predict(X_test_s)

    with st.spinner("Training XGBoost..."):
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        model_xgb = XGBClassifier(random_state=42, scale_pos_weight=scale, eval_metric='logloss', use_label_encoder=False)
        model_xgb.fit(X_train_s, y_train)
        y_pred_xgb = model_xgb.predict(X_test_s)

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_rf),
            accuracy_score(y_test, y_pred_xgb)
        ],
        'Hazardous Recall': [
            recall_score(y_test, y_pred_lr, zero_division=0),
            recall_score(y_test, y_pred_rf, zero_division=0),
            recall_score(y_test, y_pred_xgb, zero_division=0)
        ],
        'F1 Score': [
            f1_score(y_test, y_pred_lr, zero_division=0),
            f1_score(y_test, y_pred_rf, zero_division=0),
            f1_score(y_test, y_pred_xgb, zero_division=0)
        ]
    })

    st.session_state.model_lr   = model_lr
    st.session_state.model_rf   = model_rf
    st.session_state.model_xgb  = model_xgb
    st.session_state.scaler     = scaler
    st.session_state.feature_cols = X.columns.tolist()
    st.session_state.trained    = True
    st.session_state.results_df = results
    st.session_state.X_test     = X_test_s
    st.session_state.y_test     = y_test
    st.session_state.y_pred_lr  = y_pred_lr
    st.session_state.y_pred_rf  = y_pred_rf
    st.session_state.y_pred_xgb = y_pred_xgb

    return results

# ── Train on button click ─────────────────────────────────────────────────────
if train_btn:
    if uploaded_file is None:
        st.sidebar.error("Please upload a CSV first!")
    else:
        df_raw = pd.read_csv(uploaded_file)
        train_models(df_raw)
        st.sidebar.success("✅ All models trained!")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 MODEL RESULTS", "🔭 PREDICT ASTEROID", "🌐 LIVE NASA FEED", "📈 FEATURE IMPORTANCE"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Model Results
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.trained:
        st.info("⬅️ Upload your CSV and click **Train All Models** to get started.")
    else:
        st.markdown("<h3 style='color:#4ab0f0;'>MODEL COMPARISON</h3>", unsafe_allow_html=True)

        results = st.session_state.results_df
        cols = st.columns(3)
        colors = ['#f39c12', '#27ae60', '#2980b9']
        icons  = ['📉', '🌲', '⚡']
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:2rem;'>{icons[i]}</div>
                    <div style='font-family:Orbitron;color:#4ab0f0;font-size:0.7rem;letter-spacing:1px;margin:8px 0;'>{row['Model'].upper()}</div>
                    <div style='font-size:1.4rem;font-weight:700;color:{colors[i]};'>{row['Accuracy']*100:.1f}%</div>
                    <div style='font-size:0.75rem;color:#5a8ab0;'>Accuracy</div>
                    <hr style='border-color:#1a4a6e;margin:10px 0;'>
                    <div style='display:flex;justify-content:space-between;font-size:0.75rem;'>
                        <span>Recall<br><b style='color:#e74c3c;'>{row['Hazardous Recall']*100:.1f}%</b></span>
                        <span>F1<br><b style='color:#9b59b6;'>{row['F1 Score']:.3f}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrices
        st.markdown("<h3 style='color:#4ab0f0;'>CONFUSION MATRICES</h3>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#020b18')
        models_cm = [
            ('Logistic Regression', st.session_state.y_pred_lr),
            ('Random Forest',       st.session_state.y_pred_rf),
            ('XGBoost',             st.session_state.y_pred_xgb)
        ]
        for ax, (name, y_pred) in zip(axes, models_cm):
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap='Blues',
                        xticklabels=['Safe', 'Hazardous'],
                        yticklabels=['Safe', 'Hazardous'],
                        linewidths=1, linecolor='#1a4a6e')
            ax.set_title(name, color='#4ab0f0', fontsize=10, pad=10)
            ax.set_xlabel('Predicted', color='#5a8ab0')
            ax.set_ylabel('Actual', color='#5a8ab0')
            ax.tick_params(colors='#c8e6ff')
            ax.set_facecolor('#0a1f35')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Bar chart comparison
        st.markdown("<h3 style='color:#4ab0f0;'>METRIC COMPARISON</h3>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#020b18')
        x = np.arange(3)
        w = 0.25
        ax2.bar(x - w, results['Accuracy'],         w, label='Accuracy',         color='#2980b9')
        ax2.bar(x,     results['Hazardous Recall'],  w, label='Hazardous Recall', color='#e74c3c')
        ax2.bar(x + w, results['F1 Score'],          w, label='F1 Score',         color='#9b59b6')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results['Model'], color='#c8e6ff')
        ax2.set_facecolor('#0a1f35')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(colors='#c8e6ff')
        ax2.legend(facecolor='#0a1f35', labelcolor='#c8e6ff', edgecolor='#1a4a6e')
        ax2.spines[:].set_color('#1a4a6e')
        fig2.patch.set_facecolor('#020b18')
        st.pyplot(fig2)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Manual Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.trained:
        st.info("⬅️ Train the models first.")
    else:
        st.markdown("<h3 style='color:#4ab0f0;'>ASTEROID PARAMETERS</h3>", unsafe_allow_html=True)
        st.markdown("Enter asteroid data below to get an instant hazard prediction from XGBoost.")

        feature_cols = st.session_state.feature_cols

        # Build input form
        col1, col2, col3 = st.columns(3)
        user_input = {}

        # Map common features to friendly labels
        feature_labels = {
            'diameter':       ('Diameter (km)',           0.1,   50.0,  1.0),
            'albedo':         ('Albedo (reflectivity)',   0.0,    1.0,  0.15),
            'H_sigma':        ('H Magnitude sigma',       0.0,   10.0,  0.5),
            'e':              ('Eccentricity',             0.0,    1.0,  0.3),
            'a':              ('Semi-major axis (AU)',    0.1,   10.0,  1.5),
            'i':              ('Inclination (deg)',        0.0,  180.0, 10.0),
            'moid':           ('MOID (AU)',                0.0,    1.0,  0.05),
            'condition_code': ('Condition code',          0.0,    9.0,  0.0),
        }

        all_cols = [col1, col2, col3]
        for idx, feat in enumerate(feature_cols):
            c = all_cols[idx % 3]
            with c:
                if feat in feature_labels:
                    label, mn, mx, default = feature_labels[feat]
                    user_input[feat] = st.number_input(label, min_value=mn, max_value=mx, value=default, step=0.01, key=f"inp_{feat}")
                else:
                    # Boolean dummy columns (one-hot class)
                    user_input[feat] = st.selectbox(feat, [0, 1], key=f"inp_{feat}")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("☄️ ANALYZE ASTEROID", use_container_width=False)

        if predict_btn:
            input_df = pd.DataFrame([user_input])
            input_scaled = st.session_state.scaler.transform(input_df)
            pred = st.session_state.model_xgb.predict(input_scaled)[0]
            prob = st.session_state.model_xgb.predict_proba(input_scaled)[0][1]

            st.markdown("<br>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"<div class='hazard-badge'>⚠️ POTENTIALLY HAZARDOUS &nbsp;|&nbsp; {prob*100:.1f}% PROBABILITY</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='safe-badge'>✅ NOT HAZARDOUS &nbsp;|&nbsp; {prob*100:.1f}% HAZARD PROBABILITY</div>", unsafe_allow_html=True)

            # Gauge
            fig3, ax3 = plt.subplots(figsize=(5, 2.5), facecolor='#020b18')
            bar_color = '#e74c3c' if pred == 1 else '#27ae60'
            ax3.barh(['Hazard %'], [prob * 100], color=bar_color, height=0.4)
            ax3.barh(['Hazard %'], [100], color='#1a4a6e', height=0.4, zorder=0)
            ax3.set_xlim(0, 100)
            ax3.set_facecolor('#0a1f35')
            ax3.tick_params(colors='#c8e6ff')
            ax3.spines[:].set_color('#1a4a6e')
            ax3.set_xlabel('Probability (%)', color='#5a8ab0')
            fig3.patch.set_facecolor('#020b18')
            st.pyplot(fig3)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Live NASA Feed
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<h3 style='color:#4ab0f0;'>LIVE NASA NEAR-EARTH OBJECTS</h3>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        api_key = st.text_input("NASA API Key", value="DEMO_KEY", type="password")
    with col_b:
        start_date = st.date_input("Start Date")
    with col_c:
        end_date = st.date_input("End Date")

    fetch_btn = st.button("🛰️ FETCH LIVE DATA", use_container_width=False)

    if fetch_btn:
        url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}"
        with st.spinner("Connecting to NASA API..."):
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()

                if 'near_earth_objects' not in data:
                    st.error(f"API error: {data.get('error_message', 'Unknown error')}")
                else:
                    asteroids = []
                    for date, objects in data['near_earth_objects'].items():
                        for obj in objects:
                            asteroids.append({
                                'Name': obj['name'],
                                'Date': date,
                                'Diameter (km)': round(obj['estimated_diameter']['kilometers']['estimated_diameter_max'], 4),
                                'Velocity (km/s)': round(float(obj['close_approach_data'][0]['relative_velocity']['kilometers_per_second']), 2),
                                'Miss Distance (AU)': round(float(obj['close_approach_data'][0]['miss_distance']['astronomical']), 5),
                                'NASA Hazardous': obj['is_potentially_hazardous_asteroid']
                            })

                    live_df = pd.DataFrame(asteroids)

                    if st.session_state.trained:
                        feature_cols = st.session_state.feature_cols
                        live_features = pd.DataFrame(0.0, index=np.arange(len(live_df)), columns=feature_cols)
                        if 'diameter' in feature_cols:
                            live_features['diameter'] = live_df['Diameter (km)'].values
                        if 'moid' in feature_cols:
                            live_features['moid'] = live_df['Miss Distance (AU)'].values

                        live_scaled = st.session_state.scaler.transform(live_features)
                        preds = st.session_state.model_xgb.predict(live_scaled)
                        probs = st.session_state.model_xgb.predict_proba(live_scaled)[:, 1]

                        live_df['Model Prediction'] = ['⚠️ Hazardous' if p == 1 else '✅ Safe' for p in preds]
                        live_df['Hazard Probability'] = (probs * 100).round(1).astype(str) + '%'

                    st.markdown(f"**{len(live_df)} asteroids retrieved**")
                    st.dataframe(live_df, use_container_width=True, height=400)

                    # Summary
                    if st.session_state.trained:
                        h_count = int(preds.sum())
                        s_count = len(preds) - h_count
                        c1, c2 = st.columns(2)
                        c1.metric("⚠️ Predicted Hazardous", h_count)
                        c2.metric("✅ Predicted Safe", s_count)

            except Exception as e:
                st.error(f"Failed to fetch data: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if not st.session_state.trained:
        st.info("⬅️ Train the models first.")
    else:
        st.markdown("<h3 style='color:#4ab0f0;'>FEATURE IMPORTANCE — XGBOOST</h3>", unsafe_allow_html=True)

        importances = st.session_state.model_xgb.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': st.session_state.feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(15)

        fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='#020b18')
        bars = ax4.barh(feat_df['Feature'], feat_df['Importance'],
                        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_df))))
        ax4.set_facecolor('#0a1f35')
        ax4.tick_params(colors='#c8e6ff')
        ax4.spines[:].set_color('#1a4a6e')
        ax4.set_xlabel('Importance Score', color='#5a8ab0')
        ax4.set_title('Top 15 Features for Hazard Detection', color='#4ab0f0', pad=15)
        fig4.patch.set_facecolor('#020b18')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        st.markdown("<h3 style='color:#4ab0f0;'>LOGISTIC REGRESSION COEFFICIENTS</h3>", unsafe_allow_html=True)
        coefs = st.session_state.model_lr.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': st.session_state.feature_cols,
            'Coefficient': coefs
        }).sort_values('Coefficient', ascending=True)

        fig5, ax5 = plt.subplots(figsize=(10, 6), facecolor='#020b18')
        colors_coef = ['#e74c3c' if c > 0 else '#2980b9' for c in coef_df['Coefficient']]
        ax5.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef)
        ax5.axvline(0, color='#5a8ab0', linewidth=1, linestyle='--')
        ax5.set_facecolor('#0a1f35')
        ax5.tick_params(colors='#c8e6ff')
        ax5.spines[:].set_color('#1a4a6e')
        ax5.set_xlabel('Coefficient Value', color='#5a8ab0')
        ax5.set_title('Feature Coefficients (Red = increases hazard risk)', color='#4ab0f0', pad=15)

        red_patch  = mpatches.Patch(color='#e74c3c', label='Increases hazard risk')
        blue_patch = mpatches.Patch(color='#2980b9', label='Decreases hazard risk')
        ax5.legend(handles=[red_patch, blue_patch], facecolor='#0a1f35',
                   labelcolor='#c8e6ff', edgecolor='#1a4a6e')
        fig5.patch.set_facecolor('#020b18')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
