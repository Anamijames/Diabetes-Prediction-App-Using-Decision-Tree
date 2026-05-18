import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Diabetes Decision Tree", page_icon="🩺", layout="wide")

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COL = "Outcome"

# From notebook Section 2 — clinical descriptions for each feature
FEATURE_INFO = {
    "Pregnancies":              "Number of times pregnant",
    "Glucose":                  "Plasma glucose concentration (2h oral glucose tolerance test)",
    "BloodPressure":            "Diastolic blood pressure (mm Hg)",
    "SkinThickness":            "Triceps skin fold thickness (mm)",
    "Insulin":                  "2-Hour serum insulin (mu U/ml)",
    "BMI":                      "Body mass index (weight in kg / height in m²)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic risk score)",
    "Age":                      "Age (years)",
}

# From notebook Section 5.7 — key clinical thresholds highlighted in EDA
CLINICAL_THRESHOLDS = {
    "Glucose": ("Diabetic threshold", 126.0),
    "BMI":     ("Obesity threshold",  30.0),
}


@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Match notebook logic: replace medically impossible zeros and impute median.
    df_clean = df.copy()
    df_clean[ZERO_COLS] = df_clean[ZERO_COLS].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    df_clean[ZERO_COLS] = imputer.fit_transform(df_clean[ZERO_COLS])

    return df_clean


@st.cache_resource
def train_model(df_clean: pd.DataFrame):
    x = df_clean.drop(TARGET_COL, axis=1)
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    # — Notebook Section 9: feature importances sorted descending
    importances = pd.Series(best_model.feature_importances_, index=x.columns)
    importances = importances.sort_values(ascending=False)

    # — Notebook Section 7.4: 10-fold CV on full dataset
    cv_scores = cross_val_score(best_model, x, y, cv=10, scoring="accuracy")

    # — Notebook Section 8.1 / 8.2: test-set accuracy + AUC
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    meta = {
        "best_params": grid_search.best_params_,
        "importances": importances,
        "cv_scores":   cv_scores,
        "test_acc":    test_acc,
        "auc":         auc,
        "tree_depth":  best_model.get_depth(),
        "n_leaves":    best_model.get_n_leaves(),
    }
    return best_model, x.columns.tolist(), meta


def app() -> None:
    st.title("Diabetes Prediction App")
    st.caption("Decision Tree classifier · Pima Indians Diabetes Dataset · GridSearchCV tuned")

    try:
        df_clean = load_and_clean_data("diabetes.csv")
    except FileNotFoundError:
        st.error("Could not find diabetes.csv in the project root.")
        st.stop()

    model, features, meta = train_model(df_clean)

    # ── Sidebar: model info (from notebook Sections 7.3, 7.4, 8.2, 9) ──────────
    with st.sidebar:
        st.header("🌳 Model Info")
        st.metric("Test Accuracy",  f"{meta['test_acc']*100:.2f}%")
        st.metric("ROC-AUC",        f"{meta['auc']:.4f}")
        st.metric("CV Accuracy (10-fold)",
                  f"{meta['cv_scores'].mean()*100:.2f}% ± {meta['cv_scores'].std()*100:.2f}%")
        st.metric("Tree Depth",  meta["tree_depth"])
        st.metric("Leaf Nodes",  meta["n_leaves"])

        st.divider()
        st.subheader("Best Hyperparameters")
        for k, v in meta["best_params"].items():
            st.write(f"**{k}:** {v}")

        st.divider()
        # Feature importance bar chart — mirrors notebook Section 9
        st.subheader("Feature Importance")
        imp = meta["importances"]
        top_quartile = imp.quantile(0.75)
        colors = ["#d9534f" if v >= top_quartile else "#5bc0de" for v in imp]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Red = Top Quartile", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Input form ordered by feature importance (notebook Section 9) ─────────
    st.subheader("Predict Diabetes Risk")
    st.write("Enter patient values below. Fields are ordered by predictive importance.")

    # Sort inputs by descending importance so most impactful features appear first
    ordered_features = meta["importances"].index.tolist()

    medians = df_clean[features].median()
    mins    = df_clean[features].min()
    maxs    = df_clean[features].max()

    input_values = {}
    cols = st.columns(2)
    for idx, feature in enumerate(ordered_features):
        with cols[idx % 2]:
            # Show clinical description from notebook column_info (Section 2)
            label = f"**{feature}**"
            desc  = FEATURE_INFO.get(feature, "")

            # Warn if value crosses a clinical threshold (notebook Section 5.7)
            threshold_hint = ""
            if feature in CLINICAL_THRESHOLDS:
                tname, tval = CLINICAL_THRESHOLDS[feature]
                threshold_hint = f"  \n*{tname}: {tval}*"

            st.markdown(label + threshold_hint)
            if desc:
                st.caption(desc)

            input_values[feature] = st.number_input(
                feature,
                min_value=float(mins[feature]),
                max_value=float(maxs[feature]),
                value=float(medians[feature]),
                step=1.0,
                label_visibility="collapsed",
                key=feature,
            )

    if st.button("🔍 Predict", type="primary"):
        input_df = pd.DataFrame([input_values], columns=features)
        prediction          = int(model.predict(input_df)[0])
        probability_diabetic = float(model.predict_proba(input_df)[0][1])

        st.divider()
        if prediction == 1:
            st.error(f"**Prediction: Diabetic** — probability {probability_diabetic:.2%}")
        else:
            st.success(f"**Prediction: Non-Diabetic** — probability of diabetes {probability_diabetic:.2%}")

        # Highlight which inputs are above clinical thresholds
        warnings = []
        if input_values.get("Glucose", 0) >= 126:
            warnings.append("Glucose ≥ 126 mg/dL (diabetic threshold per notebook EDA)")
        if input_values.get("BMI", 0) >= 30:
            warnings.append("BMI ≥ 30 (obesity threshold per notebook EDA)")
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")

        # Show top contributing features for this prediction
        st.subheader("Top Predictive Features (from model training)")
        top3 = meta["importances"].head(3)
        for feat, imp_val in top3.items():
            patient_val = input_values[feat]
            median_val  = medians[feat]
            direction   = "above" if patient_val > median_val else "below"
            st.write(
                f"- **{feat}** (importance {imp_val:.3f}): "
                f"patient value **{patient_val:.1f}** is {direction} dataset median ({median_val:.1f})"
            )


if __name__ == "__main__":
    app()