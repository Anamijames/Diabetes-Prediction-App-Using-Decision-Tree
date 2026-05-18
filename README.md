# Diabetes Decision Tree Streamlit App

This project packages your notebook workflow into a deployable Streamlit app.

## What the app does
- Loads `diabetes.csv`
- Applies notebook-matched preprocessing:
  - Replaces impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
  - Imputes those values with median using `SimpleImputer`
- Trains a Decision Tree model with `GridSearchCV`
- Displays model metrics and allows interactive patient prediction

## Run locally
1. Open a terminal in this project folder.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and sign in.
3. Click **New app** and connect your repository.
4. Set:
   - Main file path: `app.py`
   - Python version: default is fine
5. Deploy.

If deployment fails, confirm these files are in repo root:
- `app.py`
- `requirements.txt`
- `diabetes.csv`

## Project files
- `diabetes_decision_tree.ipynb`: original analysis notebook
- `diabetes.csv`: dataset
- `app.py`: Streamlit app
- `requirements.txt`: dependencies
