import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import random

# Inject CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load models and encoders
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
def fetch_weather_data(api_key, city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            return temperature, humidity
        else:
            return None, None
    except Exception as e:
        return None, None

# Fertilizer remarks
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, good for leafy growth.',
    'DAP': 'High Phosphorus fertilizer, promotes root development.',
    '14-35-14': 'Balanced fertilizer for flowering and fruiting.',
    '28-28': 'Balanced fertilizer for overall growth.',
    '17-17-17': 'General-purpose fertilizer for all crops.',
    '20-20': 'Starter fertilizer for young plants.',
    '10-26-26': 'High phosphorus and potassium for maturity.',
    'General Purpose Fertilizer': 'Good for maintaining healthy plants.',
    'NPK 19-19-19': 'Balanced nutrients for strong flowering.',
    'Compost': 'Organic material for improving soil health.',
    'Vermicompost': 'Natural worm-based fertilizer.',
    'Cow Manure': 'Organic fertilizer improving soil structure.',
    'Potash': 'Increases disease resistance and quality.',
    'Superphosphate': 'Helps strong root development and flowering.'
}

# Main App
def main():
    local_css("style.css")

    st.markdown("<h1 style='text-align: center;'>🌿 Smart Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: green;'>Get real-time fertilizer recommendations based on your location's weather!</h4>", unsafe_allow_html=True)
    st.write("")

    scaler, label_encoder, feature_encoders, model = load_files()

    st.subheader("🌍 Enter Location Details")
    api_key = st.text_input("🔑 OpenWeatherMap API Key", type="password")
    city = st.text_input("🏡 Enter City Name (Example: Chennai, Mumbai, Delhi)")

    st.subheader("🌾 Select Crop Type")
    crop_type_input = st.selectbox('🌱 Crop Type', ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley'])

    if st.button('🚜 Fetch Weather and Recommend Fertilizer'):
        if city and api_key:
            temperature, humidity = fetch_weather_data(api_key, city)

            if temperature is not None and humidity is not None:
                moisture = humidity * 0.6
                soil_type = random.choice(['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial'])

                st.success(f"📈 Weather Details for **{city.capitalize()}**")
                st.write(f"🌡 Temperature: **{temperature}°C**")
                st.write(f"💧 Humidity: **{humidity}%**")
                st.write(f"🪴 Assumed Soil Type: **{soil_type}**")
                st.write(f"🌊 Estimated Soil Moisture: **{moisture:.2f}%**")

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
                st.error("❌ Could not fetch weather data. Please check your API Key or City Name.")

        else:
            st.warning("⚠️ Please enter both City Name and API Key.")

if __name__ == '__main__':
    main()
