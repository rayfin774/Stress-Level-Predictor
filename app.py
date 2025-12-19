import streamlit as st
import pandas as pd
from model import train_model
import os

st.set_page_config(page_title="Stress Level Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>Stress Level Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Predict stress levels using lifestyle patterns</p>",
    unsafe_allow_html=True
)

model, le = train_model()
sleep = st.slider(" Sleep Hours", 0, 12, 6)
work = st.slider(" Work Hours", 0, 12, 6)
activity = st.selectbox(" Physical Activity", ["Yes", "No"])
screen = st.slider(" Screen Time (hrs)", 0, 12, 6)

activity_val = 1 if activity == "Yes" else 0

if st.button("Predict Stress Level"):

    input_df = pd.DataFrame(
        [[sleep, work, activity_val, screen]],
        columns=["sleep_hours", "work_hours", "physical_activity", "screen_time"]
    )

    prediction = model.predict(input_df)
    result = le.inverse_transform(prediction)[0]

    if result == "Low":
        st.markdown(
            """
            <div style="
                background-color:#d4edda;
                color:#155724;
                padding:25px;
                border-radius:12px;
                border-left:8px solid #28a745;
                margin-top:20px;
            ">
                <h1> Low Stress Level</h1>
                <p>You are managing stress well. Maintain your healthy routine.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif result == "Medium":
        st.markdown(
            """
            <div style="
                background-color:#fff3cd;
                color:#856404;
                padding:25px;
                border-radius:12px;
                border-left:8px solid #ffc107;
                margin-top:20px;
            ">
                <h1> Medium Stress Level</h1>
                <p>Your stress level is moderate. Try balancing work and rest.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="
                background-color:#f8d7da;
                color:#721c24;
                padding:25px;
                border-radius:12px;
                border-left:8px solid #dc3545;
                margin-top:20px;
            ">
                <h1> High Stress Level</h1>
                <p>High stress detected. Rest and relaxation are strongly advised.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.write("")
    st.markdown(" Stress Management Tips")
    tips = {
        "Low": [
            "Maintain consistent sleep schedule",
            "Continue physical activity"
        ],
        "Medium": [
            "Reduce screen time",
            "Take short breaks during work",
            "Sleep at least 7–8 hours"
        ],
        "High": [
            "Practice meditation or breathing exercises",
            "Reduce workload temporarily",
            "Take short breaks during work",
            "Seek professional help if needed"
        ]
    }

    for tip in tips[result]:
        st.write("->", tip)

    # saving user data for future usage to train the model ifff

    save_data = input_df.copy()
    save_data["predicted_stress"] = result

    file_exists = os.path.isfile("user_predictions.csv")
    save_data.to_csv(
        "user_predictions.csv",
        mode="a",
        header=not file_exists,
        index=False
    )
    st.write("")
    st.info("Thank You for visiting. Stay Safe and Healthy !! ")

