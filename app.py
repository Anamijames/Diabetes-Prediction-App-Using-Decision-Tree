import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Diabetes Decision Tree", page_icon="🩺", layout="wide")

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COL = "Outcome"


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
    return best_model, x.columns.tolist()


def app() -> None:
    st.title("Diabetes Prediction App")
    st.caption("Decision Tree classifier with notebook-matched preprocessing")

    try:
        df_clean = load_and_clean_data("diabetes.csv")
    except FileNotFoundError:
        st.error("Could not find diabetes.csv in the project root.")
        st.stop()

    model, features = train_model(df_clean)

    st.subheader("Predict Diabetes Risk")
    st.write("Provide patient values and click Predict.")

    medians = df_clean[features].median()
    mins = df_clean[features].min()
    maxs = df_clean[features].max()

    input_values = {}
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            input_values[feature] = st.number_input(
                feature,
                min_value=float(mins[feature]),
                max_value=float(maxs[feature]),
                value=float(medians[feature]),
                step=1.0,
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_values], columns=features)
        prediction = int(model.predict(input_df)[0])
        probability_diabetic = float(model.predict_proba(input_df)[0][1])

        if prediction == 1:
            st.error("Prediction: Diabetic")
        else:
            st.success("Prediction: Non-Diabetic")

        st.write(f"Probability of diabetes: {probability_diabetic:.2%}")


if __name__ == "__main__":
    app()
