
import streamlit as st
import pandas as pd
import joblib

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="AgriSmart AI | Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# LOAD EXISTING MODEL
# ==================================================
@st.cache_resource
def load_model():
    model = joblib.load("model/crop_model.pkl")
    features = joblib.load("model/features.pkl")
    return model, features

model, features = load_model()

# ==================================================
# CUSTOM CSS - COMPACT FULL-WIDTH DASHBOARD
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #f3f7f1;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewBlockContainer"],
.block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0.5rem 1.3rem 0.3rem !important;
    box-sizing: border-box;
}

/* Remove excessive vertical gaps */
[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}

[data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: #183c2a !important;
}

h3 {
    font-size: 17px !important;
    margin: 0.25rem 0 !important;
}

h4 {
    font-size: 14px !important;
    margin: 0.25rem 0 !important;
}

/* Header banner */
.hero {
    background: linear-gradient(120deg, #123d2b, #267147);
    padding: 14px 23px;
    border-radius: 14px;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(18, 61, 43, 0.08);
}

.hero-tag {
    color: #c8edce;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.3px;
}

.hero h1 {
    color: white !important;
    font-size: 27px !important;
    margin: 4px 0 !important;
}

.hero p {
    color: #e1f0e4;
    font-size: 12px;
    margin: 0;
}

/* Summary cards */
.stat-card {
    background: white;
    border: 1px solid #e1e9df;
    border-radius: 10px;
    padding: 9px 13px;
    min-height: 66px;
    box-sizing: border-box;
}

.stat-label {
    color: #718075;
    font-size: 11px;
}

.stat-value {
    color: #205c38;
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
}

/* Panel headings */
.section-description {
    color: #778479;
    font-size: 11px;
    margin-bottom: 5px;
}

/* Compact input form */
div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e0e8dd;
    border-radius: 12px;
    padding: 10px 13px 12px;
    margin: 0;
}

div[data-testid="stNumberInput"] {
    width: 100%;
}

div[data-testid="stNumberInput"] label {
    font-size: 11px;
    color: #4d6252;
}

div[data-testid="stNumberInput"] input {
    font-size: 12px;
}

div[data-testid="stFormSubmitButton"] button {
    background: #216b40;
    color: white;
    border: 1px solid #216b40;
    border-radius: 8px;
    min-height: 38px;
    font-weight: 700;
    font-size: 13px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #174c2e;
    color: white;
    border-color: #174c2e;
}

/* Result area */
.result-card {
    background: linear-gradient(120deg, #e6f5e9, #f5fbf3);
    border: 1px solid #cce8d1;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0 10px;
}

.result-label {
    color: #35754a;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-crop {
    color: #17482b;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin: 3px 0;
}

.result-note {
    color: #52735b;
    font-size: 11px;
}

/* Top three recommendations */
.recommendation-card {
    background: white;
    border: 1px solid #e1e9df;
    border-radius: 9px;
    padding: 8px 11px;
    margin: 3px 0;
}

.crop-name {
    color: #244b31;
    font-size: 13px;
    font-weight: 700;
}

.confidence {
    color: #627466;
    font-size: 11px;
}

[data-testid="stProgressBar"] {
    margin-top: -5px;
    margin-bottom: 5px;
}

[data-testid="stProgressBar"] > div > div {
    background-color: #2c8a50;
}

/* Empty result state */
.empty-result {
    background: white;
    border: 1px dashed #c8d9c8;
    border-radius: 12px;
    padding: 22px 15px;
    text-align: center;
    margin-top: 6px;
}

.empty-result .plant {
    font-size: 34px;
}

.empty-result h4 {
    color: #28633d !important;
    font-size: 17px !important;
}

.empty-result p {
    color: #748477;
    font-size: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #879287;
    font-size: 10px;
    margin-top: 10px;
    padding: 7px;
    border-top: 1px solid #e1e9df;
}

/* Responsive layout */
@media (max-width: 750px) {
    [data-testid="stAppViewBlockContainer"],
    .block-container {
        padding: 0.5rem 0.65rem !important;
    }

    .hero {
        padding: 12px 15px;
    }

    .hero h1 {
        font-size: 23px !important;
    }

    .result-crop {
        font-size: 23px;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">
        🌿 SMART AGRICULTURE · MACHINE LEARNING
    </div>
    <h1>AgriSmart AI</h1>
    <p>
        Grow smarter. Harvest better.
        Discover suitable crops using soil and weather data.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# DASHBOARD SUMMARY
# ==================================================
crop_count = len(model.classes_) if hasattr(model, "classes_") else "—"

summary1, summary2, summary3 = st.columns(3, gap="small")

with summary1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🌾 Crop Categories</div>
        <div class="stat-value">{crop_count}</div>
    </div>
    """, unsafe_allow_html=True)

with summary2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">🧠 ML Algorithm</div>
        <div class="stat-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with summary3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">📊 Input Parameters</div>
        <div class="stat-value">{len(features)}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================
if "crop_prediction" not in st.session_state:
    st.session_state.crop_prediction = None

if "crop_results" not in st.session_state:
    st.session_state.crop_results = None

# ==================================================
# TWO-COLUMN DASHBOARD
# ==================================================
left_panel, right_panel = st.columns(
    [1.08, 1],
    gap="medium"
)

# ---------------- LEFT: INPUTS ----------------
with left_panel:
    st.markdown("### 🧪 Field Parameters")
    st.markdown(
        '<div class="section-description">'
        'Enter your soil nutrients and weather measurements.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("crop_form"):
        st.markdown("#### Soil Nutrients")

        col1, col2 = st.columns(2, gap="small")

        with col1:
            N = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=1.0
            )

            P = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                max_value=200.0,
                value=42.0,
                step=1.0
            )

        with col2:
            K = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                max_value=250.0,
                value=43.0,
                step=1.0
            )

            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1
            )

        st.markdown("#### 🌤️ Weather Conditions")

        col3, col4 = st.columns(2, gap="small")

        with col3:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=25.0,
                step=0.5
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

        with col4:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0
            )

        submitted = st.form_submit_button(
            "🌱 Recommend Suitable Crop",
            use_container_width=True
        )

# ---------------- RIGHT: RESULTS ----------------
with right_panel:
    st.markdown("### 🌿 AI Recommendation")
    st.markdown(
        '<div class="section-description">'
        'Your predicted crop and top alternatives.'
        '</div>',
        unsafe_allow_html=True
    )

    # Generate prediction when the form is submitted
    if submitted:
        values = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        if not all(feature in values for feature in features):
            st.error(
                "Input features do not match the trained model. "
                "Check model/features.pkl."
            )
        else:
            input_df = pd.DataFrame(
                [[values[feature] for feature in features]],
                columns=features
            )

            try:
                prediction = model.predict(input_df)[0]
                st.session_state.crop_prediction = str(prediction)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]

                    results = pd.DataFrame({
                        "Crop": model.classes_,
                        "Confidence (%)": probabilities * 100
                    }).sort_values(
                        "Confidence (%)",
                        ascending=False
                    ).head(3).reset_index(drop=True)

                    st.session_state.crop_results = results.to_dict(
                        orient="records"
                    )
                else:
                    st.session_state.crop_results = []

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    # Display results on every rerun after prediction
    if st.session_state.crop_prediction is not None:
        crop = st.session_state.crop_prediction.title()

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">✦ AI Prediction Result</div>
            <div class="result-crop">🌱 {crop}</div>
            <div class="result-note">
                Top predicted crop for your entered conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🏆 Top 3 Recommendations")

        medals = ["🥇", "🥈", "🥉"]

        for index, row in enumerate(
            st.session_state.crop_results or []
        ):
            crop_name = str(row["Crop"]).title()
            confidence = float(row["Confidence (%)"])

            st.markdown(f"""
            <div class="recommendation-card">
                <div class="crop-name">
                    {medals[index]} {crop_name}
                </div>
                <div class="confidence">
                    {confidence:.2f}% model probability
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

        st.caption(
            "Model probabilities are estimates, not guarantees of "
            "crop suitability under real farming conditions."
        )

    else:
        st.markdown("""
        <div class="empty-result">
            <div class="plant">🌱</div>
            <h4>Ready to Grow?</h4>
            <p>
                Enter your field measurements on the left and click
                Recommend Suitable Crop to see your results here.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    AgriSmart AI · Crop Recommendation System · Powered by Random Forest ML
</div>
""", unsafe_allow_html=True)
