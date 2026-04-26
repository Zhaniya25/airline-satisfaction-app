import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("✈️ Airline Passenger Satisfaction")

# =========================
# INPUTS
# =========================

st.header("Enter passenger details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.slider("Age", 0, 100, 30)
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

with col2:
    flight_distance = st.number_input("Flight Distance", 0, 10000, 1000)
    inflight_wifi = st.slider("Inflight wifi service", 0, 5, 3)
    online_boarding = st.slider("Online boarding", 0, 5, 3)
    seat_comfort = st.slider("Seat comfort", 0, 5, 3)
    inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)

with col3:
    food_drink = st.slider("Food and drink", 0, 5, 3)
    onboard_service = st.slider("On-board service", 0, 5, 3)
    leg_room = st.slider("Leg room service", 0, 5, 3)
    baggage = st.slider("Baggage handling", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)

departure_delay = st.number_input("Departure Delay (min)", 0, 1000, 0)
arrival_delay = st.number_input("Arrival Delay (min)", 0, 1000, 0)

# =========================
# CREATE DATAFRAME
# =========================

input_data = pd.DataFrame([{
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": travel_class,
    "Flight Distance": flight_distance,
    "Inflight wifi service": inflight_wifi,
    "Online boarding": online_boarding,
    "Seat comfort": seat_comfort,
    "Inflight entertainment": inflight_entertainment,
    "Food and drink": food_drink,
    "On-board service": onboard_service,
    "Leg room service": leg_room,
    "Baggage handling": baggage,
    "Cleanliness": cleanliness,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay
}])

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Satisfied (Probability: {proba:.2f})")
    else:
        st.error(f"❌ Not satisfied (Probability: {proba:.2f})")
