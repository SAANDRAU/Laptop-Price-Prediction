import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder,StandardScaler

with open("laptop_price_model.pkl", "rb") as file:
    model_data = pickle.load(file)


model = model_data["model"]
brand_encoder = model_data["brand_encoder"]
processor_encoder=model_data["processor_encoder"]
pro_name_encoder=model_data["pro_name_encoder"]
pro_gen_encoder=model_data["pro_gen_encoder"]
ram_encoder=model_data["ram_encoder"]
os_encoder=model_data["os_encoder"]
os_bit_encoder=model_data["os_bit_encoder"]
weight_encoder=model_data["weight_encoder"]
warranty_encoder=model_data["warranty_encoder"]
touch_encoder=model_data["touch_encoder"]
ms_encoder=model_data["ms_encoder"]
rating_encoder=model_data["rating_encoder"]
scaler = model_data["scaler"]

# def price_prediction(input_data):

#     input_data = np.array(input_data, dtype=float).reshape(1, -1)
#     prediction = loaded_model.predict(input_data)
#     print(prediction)
def main():
    st.title("Laptop Price Prediction Web App")

    brand = st.selectbox("Brand", brand_encoder.classes_)
    processor_brand = st.selectbox("Processor brand", processor_encoder.classes_)
    processor_name = st.selectbox("Processor name", pro_name_encoder.classes_)
    processor_gnrtn = st.selectbox("Processor generation", pro_gen_encoder.classes_)
    ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=128, value=8, step=2)
    ram_type = st.selectbox("RAM type", ram_encoder.classes_)
    ssd = st.number_input("SSD (GB)", min_value=0, max_value=4000, value=0, step=128)
    hdd = st.number_input("HDD (GB)", min_value=0, max_value=4000, value=0, step=128)
    os = st.selectbox("OS", os_encoder.classes_)
    os_bit = st.selectbox("OS bit", os_bit_encoder.classes_)
    graphic_card_gb = st.number_input("Graphic card (GB)", min_value=0, max_value=32, value=0, step=1)
    weight = st.selectbox("Weight category", weight_encoder.classes_)
    warranty = st.selectbox("Warranty (years)", [0, 1, 2, 3])
    touchscreen = st.selectbox("Touchscreen", touch_encoder.classes_)
    msoffice = st.selectbox("MS Office", ms_encoder.classes_)
    rating = st.selectbox("Rating", rating_encoder.classes_)
    number_of_ratings = st.number_input("Number of Ratings", min_value=0, value=0, step=1)

    # price = ""

    if st.button("Predict Price"):
        input_data = [
          brand_encoder.transform([brand])[0],
          processor_encoder.transform([processor_brand])[0],
          pro_name_encoder.transform([processor_name])[0],
          pro_gen_encoder.transform([processor_gnrtn])[0],
          ram_gb,
          ram_encoder.transform([ram_type])[0],
          ssd,
          hdd,
          os_encoder.transform([os])[0],
          os_bit_encoder.transform([os_bit])[0],
          graphic_card_gb,
          weight_encoder.transform([weight])[0],
          warranty,
          touch_encoder.transform([touchscreen])[0],
          ms_encoder.transform([msoffice])[0],
          rating_encoder.transform([rating])[0],
          number_of_ratings
        ]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        st.success(f"Estimated Price: ₹{prediction[0]:,.1f}")


if __name__ == "__main__":
    main()