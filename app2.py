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

# -------------------- Custom CSS Styling (KEPT AS REQUESTED) --------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://static.vecteezy.com/system/resources/previews/029/163/762/large_2x/3d-cg-rendering-of-space-ship-high-resolution-image-gallery-an-intergalactic-modern-spaceship-orbiting-a-distant-planet-on-a-black-background-ai-generated-free-photo.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }}
    .stTitle {{
        color: #00FFFF;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 6px #000000;
    }}
    h2, .stSubheader {{
        color: #00FFFF;
        text-shadow: 1px 1px 4px #000000;
    }}
    p, .stMarkdown {{
        color: #FFFFFF;
        font-size: 1rem;
        text-shadow: 1px 1px 2px #000000;
    }}
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
    label {{
        color: #FF69B4;
        font-weight: bold;
    }}
    div.stButton > button:first-child {{
        background-color: #00CED1;
        color: black;
        font-weight: bold;
        height: 3em;
        width: 100%;
        border-radius: 10px;
    }}
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
st.subheader("Upload NASA Kepler dataset and predict exoplanet candidates")

st.markdown("""
Welcome to the Exoplanet Classifier! 
Upload NASA Kepler dataset and predict which stars might host planets. 
✨ Powered by ML models: Gradient Boosting, Random Forest, SVM, and Logistic Regression.
""")

# -------------------- Upload CSV --------------------
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file:
    # Robust data loading for NASA files (uses comment='#')
    df = pd.read_csv(uploaded_file, comment='#', on_bad_lines='skip', engine='python')
    df.dropna(how='all', inplace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Identify target column
    dispo_col = next((col for col in df.columns if 'disposition' in col), None)

    if dispo_col is None:
        st.error("❌ Error: No disposition column found in uploaded dataset. Expected a column containing 'disposition'.")
    else:
        # Filter out 'CANDIDATE' and map to binary target
        df = df[df[dispo_col].str.upper() != 'CANDIDATE'].copy()
        
        # Create a temporary target column for robust alignment and dropping NaNs
        df.loc[:, 'y'] = df[dispo_col].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})
        df.dropna(subset=['y'], inplace=True)
        y = df['y']

        # --- IMPROVEMENT: Feature Selection and Cleaning ---
        X_all_numeric = df.select_dtypes(include=[np.number]).drop(columns=['y'], errors='ignore')

        # 1. Drop features with 100% missing values (e.g., koi_teq_err1, koi_teq_err2)
        null_counts = X_all_numeric.isnull().sum()
        cols_to_drop_null = null_counts[null_counts == len(X_all_numeric)].index.tolist()
        X_numeric = X_all_numeric.drop(columns=cols_to_drop_null, errors='ignore')

        # 2. Drop non-informative ID columns
        cols_to_drop_id = ['kepid'] # Kepler ID is an identifier, not a feature
        X_numeric = X_numeric.drop(columns=cols_to_drop_id, errors='ignore')
        
        # Impute missing values with mean
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_numeric)
        X_clean = pd.DataFrame(X_imputed, columns=X_numeric.columns, index=X_numeric.index)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Align X and y after cleaning/dropping NaNs
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
        y = y[y.index.isin(X_scaled_df.index)]
        X_scaled_final = X_scaled_df.loc[y.index].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_final, y, test_size=0.2, random_state=42, stratify=y
        )

        # -------------------- Train models --------------------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
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

        # --- Dynamic Best Model Selection ---
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        # -------------------- Tabs --------------------
        tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔥 Feature Importance", "🔮 Predict Candidate"])

        # -------------------- Tab 1: Model Comparison --------------------
        with tab1:
            st.subheader("Model Accuracies")
            st.write(results)

            fig = px.bar(
                x=list(results.keys()),
                y=list(results.values()),
                color=list(results.keys()),
                text=[f'{v:.4f}' for v in results.values()],
                title="Model Accuracies",
                labels={"x":"Model", "y":"Accuracy"}
            )
            # Ensure y-axis is sensible for high accuracy models
            fig.update_layout(yaxis=dict(range=[0.9, max(results.values()) * 1.05]))
            st.plotly_chart(fig)

        # -------------------- Tab 2: Feature Importance --------------------
        importance_df = None
        
        # Check if the best model has a feature_importances_ attribute
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': best_model.feature_importances_
            }).sort_values(by="importance", ascending=False)
            
            with tab2:
                st.subheader(f"Top 15 Important Features ({best_model_name})")
                st.dataframe(importance_df.head(15))

                fig2 = px.bar(
                    importance_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importances",
                    text=[f'{v:.4f}' for v in importance_df.head(15)['importance'].values]
                )
                fig2.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2)
        else:
            with tab2:
                 st.subheader("Feature Importance")
                 st.warning(f"Feature importance is not available for the best model: {best_model_name} (e.g., SVM).")

        # -------------------- Tab 3: Predict New Candidate --------------------
        with tab3:
            st.subheader("Predict New Candidate")

            # Initialize candidate data with mean of cleaned features
            candidate_data = {col: float(X_clean[col].mean()) for col in X_clean.columns}

            # Display form
            with st.form("candidate_form"):
                cols = st.columns(2)
                
                # Use top 5 features for input if available, otherwise use the first 5
                if importance_df is not None and len(importance_df) >= 5:
                    top5_features = importance_df.head(5)['feature'].tolist()
                else:
                    top5_features = X_clean.columns[:5].tolist()
                    
                for i, feat in enumerate(top5_features):
                    col_idx = i % 2
                    
                    # Determine step based on feature values
                    mean = float(X_clean[feat].mean())
                    std = float(X_clean[feat].std())
                    step_val = max(1.0, std / 100) if std > 0 else 1.0
                    
                    candidate_data[feat] = cols[col_idx].number_input(
                        f"Enter value for {feat}",
                        value=mean,
                        step=step_val,
                        format="%.5f"
                    )
                submitted = st.form_submit_button("Predict")

            if submitted:
                # --- IMPROVEMENT: Robust column alignment for prediction ---
                
                # 1. Prepare a dictionary with all features, using input for the top 5, and mean for the rest
                new_row_input = {
                    col: candidate_data.get(col, X_clean[col].mean())
                    for col in imputer.feature_names_in_
                }
                
                # 2. Convert to DataFrame ensuring the column order matches the imputer's training columns
                new_candidate = pd.DataFrame([new_row_input], columns=imputer.feature_names_in_)

                # 3. Apply Imputer (does nothing if no NaNs, but maintains order)
                new_candidate_imputed = imputer.transform(new_candidate)
                
                # 4. Apply Scaler
                new_candidate_scaled = scaler.transform(new_candidate_imputed)

                # Predict
                prediction = best_model.predict(new_candidate_scaled)
                prob = best_model.predict_proba(new_candidate_scaled) if hasattr(best_model, 'predict_proba') else None

                label = "CONFIRMED" if prediction[0] == 1 else "FALSE POSITIVE"

                st.success(f"✅ Prediction: {label}")
                if prob is not None:
                    st.write("Prediction probabilities (False Positive, Confirmed):", prob)

                if prediction[0] == 1:
                    st.balloons()