"""
Titanic Passenger Survival Prediction - Streamlit App
======================================================

A minimal web interface around the pre-trained Support Vector Classifier
stored in 'svc_trained_model.pkl'.

The label categories and the feature order below are exactly the ones used
in 'Titanic_Passenger_Survival_Recommended.ipynb', so the encoders are
rebuilt in memory instead of being loaded from separate files.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

MODEL_FILE = "svc_trained_model.pkl"

# Same categories as the notebook (LabelEncoder sorts them alphabetically)
PCLASS_CATEGORIES = ["First", "Second", "Third"]
GENDER_CATEGORIES = ["Male", "Female"]
SIBLING_CATEGORIES = ["Zero", "One", "Two", "Three"]
EMBARKED_CATEGORIES = ["Southampton", "Cherbourg", "Queenstown"]

# Feature order expected by the trained model
FEATURE_ORDER = ["PClass", "Gender", "Sibling", "Embarked"]


@st.cache_resource
def load_model():
    """Load the trained SVC model from disk."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' was not found next to app.py.")
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def build_encoders():
    """Recreate the notebook's LabelEncoders in memory."""
    encoders = {}
    for column, categories in (
        ("PClass", PCLASS_CATEGORIES),
        ("Gender", GENDER_CATEGORIES),
        ("Sibling", SIBLING_CATEGORIES),
        ("Embarked", EMBARKED_CATEGORIES),
    ):
        encoder = LabelEncoder()
        encoder.fit(np.ravel(pd.DataFrame({column: categories})))
        encoders[column] = encoder
    return encoders


def encode_input(encoders, pclass, gender, sibling, embarked):
    """Turn the four text inputs into the encoded feature vector."""
    user_input = pd.DataFrame(
        {
            "PClass": [pclass],
            "Gender": [gender],
            "Sibling": [sibling],
            "Embarked": [embarked],
        }
    )

    encoded = user_input.copy()
    for column in FEATURE_ORDER:
        encoded[column] = encoders[column].transform(user_input[column])

    # Keep the exact column order the model was trained with
    return user_input, encoded[FEATURE_ORDER]


st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Passenger Survival Prediction")
st.write(
    "Enter the passenger details below and the trained Support Vector "
    "Classifier will predict whether that passenger survived."
)

# ---- Load the model (and report clearly if that fails) ----
try:
    model = load_model()
except Exception as error:  # noqa: BLE001 - surface any loading problem to the user
    st.error(f"Could not load the trained model: {error}")
    st.info(
        "Make sure 'svc_trained_model.pkl' is in the same folder as app.py and "
        "that the installed scikit-learn version matches requirements.txt."
    )
    st.stop()

encoders = build_encoders()

# ---- Four inputs ----
with st.form("passenger_form"):
    pclass = st.selectbox("Passenger Class (PClass)", PCLASS_CATEGORIES)
    gender = st.selectbox("Gender", GENDER_CATEGORIES)
    sibling = st.selectbox("Number of Siblings (Sibling)", SIBLING_CATEGORIES)
    embarked = st.selectbox("Port of Embarkation (Embarked)", EMBARKED_CATEGORIES)
    submitted = st.form_submit_button("Predict Survival")

if submitted:
    try:
        user_input, features = encode_input(encoders, pclass, gender, sibling, embarked)
        predicted_survival = int(model.predict(features)[0])
    except ValueError as error:
        st.error(f"Invalid input: {error}")
    except Exception as error:  # noqa: BLE001 - keep the app usable on any failure
        st.error(f"Prediction failed: {error}")
    else:
        if predicted_survival == 1:
            st.success("### SURVIVED")
        else:
            st.error("### NOT SURVIVED")

        with st.expander("See how your input was encoded"):
            st.write("Your input:")
            st.dataframe(user_input, hide_index=True)
            st.write("Encoded feature vector sent to the model:")
            st.dataframe(features, hide_index=True)

st.caption(
    "Model: Support Vector Classifier trained on sample-data.csv "
    "(see Titanic_Passenger_Survival_Recommended.ipynb)."
)
