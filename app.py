
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
xgb3 = pickle.load(open('xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("FreshGuard — Food Risk Predictor")
st.write("Enter the nutritional values of a product to predict its Nutriscore and risk group.")

energy = st.slider("Energy (per 100g)", 0, 4000, 1000)
fat = st.slider("Fat (per 100g)", 0, 100, 10)
sugars = st.slider("Sugars (per 100g)", 0, 100, 10)
proteins = st.slider("Proteins (per 100g)", 0, 100, 10)
salt = st.slider("Salt (per 100g)", 0, 100, 2)
category_map = {
    'Sugary snacks': 0,
    'Cereals and potatoes': 1,
    'Beverages': 2,
    'Fat and sauces': 3,
    'Milk and dairy products': 4,
    'Fish Meat Eggs': 5,
    'Fruits and vegetables': 6,
    'Composite foods': 7,
    'Salty snacks': 8,
    'Alcoholic beverages': 9,
    'Baby foods': 10
}

category = st.selectbox("Food Category", options=list(category_map.keys()))
category_enc = category_map[category]

fat_sugar_ratio = fat / (sugars + 1)
energy_per_protein = energy / (proteins + 1)
unhealthy_score = fat + sugars + salt

if st.button("Predict"):
    input_data = np.array([[energy, fat, sugars, proteins, salt,
                            fat_sugar_ratio, energy_per_protein,
                            unhealthy_score, category_enc]])

    input_df = pd.DataFrame(input_data, columns=['energy_100g', 'fat_100g', 'sugars_100g',
                                                 'proteins_100g', 'salt_100g', 'fat_sugar_ratio',
                                                 'energy_per_protein', 'unhealthy_score', 'category_enc'])

    input_scaled = scaler.transform(input_df)
    prediction = xgb3.predict(input_scaled)
    grade = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Nutriscore: {grade.upper()}")

    grade_info = {
        'a': '🟢 Very healthy product. Low risk.',
        'b': '🟡 Healthy product. Acceptable.',
        'c': '🟠 Moderate. Consume with caution.',
        'd': '🔴 Unhealthy. High risk of spoilage and low demand.',
        'e': '⛔ Very unhealthy. Highest fire risk in store.'
    }

    st.info(grade_info[grade])
