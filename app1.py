import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import plotly.express as px

# -------------------- Page config --------------------
# st.set_page_config(
#     page_title="🚀 Exoplanet Classifier",
#     page_icon="🪐",
#     layout="wide"
# )

# Optional: Background image (space theme)
# -------------------- Custom CSS Styling --------------------
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background-image: url('https://static.vecteezy.com/system/resources/previews/029/163/762/large_2x/3d-cg-rendering-of-space-ship-high-resolution-image-gallery-an-intergalactic-modern-spaceship-orbiting-a-distant-planet-on-a-black-background-ai-generated-free-photo.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Title styling */
    .stTitle {{
        color: #00FFFF;  /* gold */
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 6px #000000;
    }}

    /* Subheaders styling */
    h2, .stSubheader {{
        color: #00FFFF;  /* cyan */
        text-shadow: 1px 1px 4px #000000;
    }}

    /* Paragraphs / normal text */
    p, .stMarkdown {{
        color: #FFFFFF;
        font-size: 1rem;
        text-shadow: 1px 1px 2px #000000;
    }}

    /* Tables */
    .dataframe, table {{
        border-collapse: collapse;
        width: 100%;
        color: #FFFFFF;
        font-size: 0.9rem;
        background-color: rgba(0,0,0,0.6);
        border: 1px solid #444;
    }}

    .dataframe th, table th {{
        background-color: rgba(34,34,34,0.8);
        color: #FFD700;
        padding: 8px;
        text-align: center;
    }}

    .dataframe td, table td {{
        border: 1px solid #444;
        padding: 6px;
        text-align: center;
    }}

    /* Form labels and input numbers */
    label {{
        color: #FF69B4; /* pink */
        font-weight: bold;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background-color: #00CED1;
        color: black;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border-radius: 10px;
    }}

    /* Tabs background (optional) */
    .css-1v3fvcr {{
        background-color: rgba(0,0,0,0.5);
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Colored Title --------------------
st.markdown('<h1 class="stTitle">🚀 Exoplanet Candidate Classifier</h1>', unsafe_allow_html=True)

# -------------------- Optional Subheader --------------------
st.subheader("Upload NASA Kepler dataset and predict exoplanet candidates")




# -------------------- Header --------------------
# st.title("🚀 Exoplanet Candidate Classifier")
st.markdown("""
Welcome to the Exoplanet Classifier!  
Upload NASA Kepler dataset and predict which stars might host planets.  
✨ Powered by ML models: Gradient Boosting, Random Forest, SVM, and Logistic Regression.
""")

# -------------------- Upload CSV --------------------
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, comment='#', on_bad_lines='skip', engine='python')

    # Drop 'CANDIDATE'
    df = df[df['koi_disposition'] != 'CANDIDATE']
    y = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

    # Drop unwanted columns and keep numeric only
    drop_cols = ['epid','kepoi_name','kepler_name','koi_disposition','ra','dec','koi_kepmag']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X_clean = pd.DataFrame(X_imputed, columns=X.columns[:X_imputed.shape[1]], index=X.index)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------- Train models --------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    results = {}
    reports = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

    # -------------------- Tabs --------------------
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔥 Feature Importance", "🔮 Predict Candidate"])

    # -------------------- Tab 1: Model Comparison --------------------
    with tab1:
        st.subheader("Model Accuracies")
        st.write(results)

        # Plot interactive bar chart
        fig = px.bar(
            x=list(results.keys()),
            y=list(results.values()),
            color=list(results.keys()),
            text=list(results.values()),
            title="Model Accuracies",
            labels={"x":"Model", "y":"Accuracy"}
        )
        fig.update_layout(yaxis=dict(range=[0.9, 1.0]))
        st.plotly_chart(fig)

    # -------------------- Tab 2: Feature Importance --------------------
    best_model = models["Gradient Boosting"]
    importance_df = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    with tab2:
        st.subheader("Top 15 Important Features (Gradient Boosting)")
        st.write(importance_df.head(15))

        # Horizontal bar chart
        fig2 = px.bar(
            importance_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importances",
            text="importance"
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2)

    # -------------------- Tab 3: Predict New Candidate --------------------
    with tab3:
        st.subheader("Predict New Candidate")

        candidate_data = {col: float(X_clean[col].mean()) for col in X_clean.columns}

        # Display form
        with st.form("candidate_form"):
            cols = st.columns(2)
            top5_features = importance_df.head(5)['feature'].tolist()
            for i, feat in enumerate(top5_features):
                col_idx = i % 2
                candidate_data[feat] = cols[col_idx].number_input(
                    f"Enter value for {feat}",
                    value=float(X_clean[feat].mean())
                )
            submitted = st.form_submit_button("Predict")

        if submitted:
            new_candidate = pd.DataFrame([candidate_data], columns=X_clean.columns)

            # Ensure all required columns are present
            for col in imputer.feature_names_in_:
                if col not in new_candidate.columns:
                    new_candidate[col] = np.nan

            # Reorder columns
            new_candidate = new_candidate[imputer.feature_names_in_]

            # Apply imputer and scaler
            new_candidate_imputed = imputer.transform(new_candidate)
            new_candidate_scaled = scaler.transform(new_candidate_imputed)

            # Predict
            prediction = best_model.predict(new_candidate_scaled)
            prob = best_model.predict_proba(new_candidate_scaled)

            label = "CONFIRMED" if prediction[0] == 1 else "FALSE POSITIVE"

            st.success(f"✅ Prediction: {label}")
            st.write("Prediction probabilities:", prob)

            if prediction[0] == 1:
                st.balloons()
