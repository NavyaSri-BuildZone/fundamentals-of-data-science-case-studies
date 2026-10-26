
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_logistic_model.pkl")

st.title("Titanic Survival Prediction")
st.write("Predict passenger survival using Logistic Regression.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(passenger)[0]
    probability = model.predict_proba(passenger)[0][1]

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.warning("Prediction: Did not survive")

    st.write(f"Estimated survival probability: {probability:.1%}")
