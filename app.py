import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import random

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model and encoders
def load_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('fertilizer_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    try:
        with open('feature_encoders.pkl', 'rb') as f:
            feature_encoders = pickle.load(f)
    except:
        feature_encoders = None
    return scaler, label_encoder, feature_encoders, model

# Fetch weather data
def fetch_weather_data(city):
    api_key = "38ca0f847c0913f4c4c11b9c1da06047"  # your API key inserted directly
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            return temp, humidity
        else:
            st.error(f"❌ API Error: {response.json().get('message', 'Unknown error')}")
            return None, None
    except Exception as e:
        st.error(f"❌ Exception Error: {e}")
        return None, None

# Fertilizer remarks
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, good for leafy growth.',
    'DAP': 'High Phosphorus fertilizer, promotes root development.',
    '14-35-14': 'Supports flowering and fruiting stages.',
    '28-28': 'Promotes balanced vegetative growth.',
    '17-17-17': 'General-purpose fertilizer for all crops.',
    '20-20': 'Starter fertilizer for young plants.',
    '10-26-26': 'Helps maturity and improves fruit setting.',
    'General Purpose Fertilizer': 'Maintains plant health through all stages.',
    'NPK 19-19-19': 'Well-balanced fertilizer for strong crops.',
    'Compost': 'Enriches soil naturally with organic matter.',
    'Vermicompost': 'Worm castings to boost soil fertility.',
    'Cow Manure': 'Traditional organic soil improvement.',
    'Potash': 'Enhances quality and disease resistance.',
    'Superphosphate': 'Stimulates strong root development.'
}

# Main App
def main():
    local_css("style.css")

    st.markdown("<h1 style='text-align: center; color: #0077b6;'>🌤️ Smart Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #48cae4;'>Get fertilizer recommendations based on real-time weather!</h5>", unsafe_allow_html=True)
    st.write("")

    scaler, label_encoder, feature_encoders, model = load_files()

    st.subheader("📍 Location Details")
    city = st.text_input("🏡 Enter City Name (Example: Chennai, Delhi, Mumbai)")

    st.subheader("🌾 Crop Selection")
    crop_type_input = st.selectbox('🌱 Choose Your Crop', ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley'])

    if st.button('☁️ Fetch Weather & Recommend Fertilizer'):
        if city:
            temperature, humidity = fetch_weather_data(city)

            if temperature is not None and humidity is not None:
                moisture = humidity * 0.6
                soil_type = random.choice(['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial'])

                st.success(f"📊 Weather in **{city.title()}**")
                st.write(f"🌡 Temperature: **{temperature}°C**")
                st.write(f"💧 Humidity: **{humidity}%**")
                st.write(f"🪴 Assumed Soil Type: **{soil_type}**")
                st.write(f"💦 Estimated Soil Moisture: **{moisture:.2f}%**")

                nitrogen = random.randint(10, 80)
                phosphorus = random.randint(10, 80)
                potassium = random.randint(10, 80)

                try:
                    soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
                    crop_encoded = feature_encoders['Crop Type'].transform([crop_type_input])[0]
                except:
                    soil_encoded = random.randint(0, 5)
                    crop_encoded = random.randint(0, 5)

                input_data = np.array([[
                    temperature, humidity, moisture,
                    soil_encoded, crop_encoded,
                    nitrogen, phosphorus, potassium,
                    random.uniform(5.5, 7.5),
                    random.uniform(100.0, 300.0),
                    random.uniform(50.0, 200.0)
                ]])

                input_scaled = scaler.transform(input_data)
                prediction_encoded = model.predict(input_scaled)
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]

                remark = remarks_dict.get(prediction, "🌿 Fertilizer recommended for balanced plant growth.")

                st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
                st.info(f"💬 Remark: {remark}")

        else:
            st.warning("⚠️ Please enter a valid City Name.")

if __name__ == '__main__':
    main()
