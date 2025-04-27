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
def fetch_weather_data(city):
    api_key = "YOUR_API_KEY_HERE"  # <<<<< Replace this with your correct API key
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
        st.error(f"❌ Exception Error: {str(e)}")
        return None, None

# Fertilizer remarks
remarks_dict = {
    'Urea': 'High Nitrogen fertilizer, good for leafy growth.',
    'DAP': 'High Phosphorus fertilizer, promotes root development.',
    '14-35-14': 'Helps in flowering and fruiting.',
    '28-28': 'Promotes overall vegetative growth.',
    '17-17-17': 'General-purpose fertilizer for all stages.',
    '20-20': 'Starter fertilizer for young crops.',
    '10-26-26': 'Supports crop maturity and harvest.',
    'General Purpose Fertilizer': 'Maintains healthy crop growth.',
    'NPK 19-19-19': 'Balanced growth throughout crop lifecycle.',
    'Compost': 'Enriches soil naturally with organic matter.',
    'Vermicompost': 'Improves soil fertility organically.',
    'Cow Manure': 'Enhances soil structure naturally.',
    'Potash': 'Boosts disease resistance and yield.',
    'Superphosphate': 'Encourages strong root development.'
}

# Main app
def main():
    local_css("style.css")

    st.markdown("<h1 style='text-align: center; color: #0077b6;'>🌤️ Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #48cae4;'>Get real-time fertilizer suggestions based on your city's weather!</h5>", unsafe_allow_html=True)
    st.write("")

    scaler, label_encoder, feature_encoders, model = load_files()

    st.subheader("📍 Location Details")
    city = st.text_input("🏡 Enter City Name (Example: Chennai, Delhi, Mumbai)")

    st.subheader("🌾 Crop Details")
    crop_type_input = st.selectbox('🌱 Choose Crop Type', ['Wheat', 'Rice', 'Sugarcane', 'Maize', 'Cotton', 'Barley'])

    if st.button('🚀 Fetch Weather & Recommend Fertilizer'):
        if city:
            temperature, humidity = fetch_weather_data(city)

            if temperature is not None and humidity is not None:
                moisture = humidity * 0.6
                soil_type = random.choice(['Loamy', 'Sandy', 'Clayey', 'Black', 'Red', 'Alluvial'])

                st.success(f"📊 Live Weather for **{city.title()}**")
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

                remark = remarks_dict.get(prediction, "🌿 Fertilizer recommended for healthy and sustainable crops.")

                st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
                st.info(f"💬 Remark: {remark}")

        else:
            st.warning("⚠️ Please enter a City Name.")

if __name__ == '__main__':
    main()
