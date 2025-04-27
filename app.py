import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import random

# Inject CSS from style.css
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
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        return temperature, humidity
    else:
        return None, None

# Remarks dictionary
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, excellent for green leafy vegetables.',
    'DAP': 'Rich in Phosphorus, ideal for root development and strong growth.',
    '14-35-14': 'Boosts flowering and fruiting stages effectively.',
    '28-28': 'Ensures balanced growth during vegetative phase.',
    '17-17-17': 'General-purpose fertilizer for various crops and stages.',
    '20-20': 'Great starter fertilizer for young plants and seedlings.',
    '10-26-26': 'Promotes flowering, fruiting, and plant maturity.',
    'General Purpose Fertilizer': 'Perfect for maintaining healthy plants throughout the season.',
    'NPK 19-19-19': 'Balanced NPK fertilizer, supports vigorous plant growth.',
    'Compost': 'Organic fertilizer enriching soil quality sustainably.',
    'Vermicompost': 'Natural worm-processed fertilizer for soil health.',
    'Cow Manure': 'Traditional organic fertilizer improving soil texture.',
    'Potash': 'Boosts resistance to diseases and improves crop quality.',
    'Superphosphate': 'Helps rapid root establishment and flowering.'
}

# Main app
def main():
    local_css("style.css")

    st.title("🌾 Smart Fertilizer Recommendation System")
    st.write("Get real-time fertilizer recommendations based on your location's weather!")

    # Load models
    scaler, label_encoder, feature_encoders, model = load_files()

    # Ask for API key and city
    api_key = st.text_input("🔑 Enter your OpenWeatherMap API Key", type="password")
    city = st.text_input("🏡 Enter Your City Name")

    crop_type_input = st.selectbox('🌱 Select Crop Type', ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley'])

    if st.button('Fetch Weather and Recommend Fertilizer'):
        if city and api_key:
            temperature, humidity = fetch_weather_data(api_key, city)

            if temperature is not None:
                moisture = humidity * 0.6  # estimate soil moisture
                soil_type = random.choice(['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial'])

                st.success(f"📈 Live Data for {city}:")
                st.write(f"🌡 Temperature: {temperature}°C")
                st.write(f"💧 Humidity: {humidity}%")
                st.write(f"🪴 Assumed Soil Type: {soil_type}")
                st.write(f"🌊 Estimated Soil Moisture: {moisture:.2f}%")

                # Random NPK for now (could be inputted too)
                nitrogen = random.randint(10, 80)
                phosphorus = random.randint(10, 80)
                potassium = random.randint(10, 80)

                # Encode soil and crop
                try:
                    soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
                    crop_encoded = feature_encoders['Crop Type'].transform([crop_type_input])[0]
                except:
                    soil_encoded = random.randint(0, 5)
                    crop_encoded = random.randint(0, 5)

                # Create input array
                input_data = np.array([[
                    temperature, humidity, moisture,
                    soil_encoded, crop_encoded,
                    nitrogen, phosphorus, potassium,
                    random.uniform(5.5, 7.5),    # pH
                    random.uniform(100.0, 300.0), # Rainfall
                    random.uniform(50.0, 200.0)   # Elevation
                ]])

                # Scale
                input_scaled = scaler.transform(input_data)

                # Predict
                prediction_encoded = model.predict(input_scaled)
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]

                # Remark
                remark = remarks_dict.get(prediction, "🌿 Fertilizer recommended for promoting balanced plant growth.")

                st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
                st.info(f"💬 Remark: {remark}")

            else:
                st.error("❌ Failed to fetch weather data. Please check your city name or API key.")
        else:
            st.warning("⚠️ Please enter your city and API key.")

if __name__ == '__main__':
    main()
