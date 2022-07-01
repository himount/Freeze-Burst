import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
import pickle, folium, gzip

# 📌 ✨ 😊 😄 😆  🎈 ⚠️ 🔑 ℹ️ 📈 🐞 💬 ☝️ 🎓 🎉 ⬇️ 📥 🖱️ 👇 🍴 👈 👋 🏀 🗓️ 🖼️ ⬇ ⏳ 🔔 ⏰

im = Image.open("AI_Lab_logo.jpg")

st.set_page_config(
    page_title="계량기 동파예측",
    page_icon='📈',
    layout="wide",
)

col1, col2 = st.columns([1,8])
col1.image(im, width=100)
col2.write("# 지방상수도 함내 계량기 동파 예측")

st.write("##### 📌 예측을 위한 데이터를 입력해주세요 ?")
with st.form('My Form'):
    col1, col2, col3, col4, col5 = st.columns(5)
    # latitude = col1.text_input("위도", 34.10, 38.00, 37.41465)
    # longitude = col2.text_input("경도", 126.00, 130.00, 127.2876)
    # elevation = col3.number_input("고도", 0.00, 500.00, 62.91475)
    latitude = col1.text_input("위도(34~39)", "37.41465")
    latitude = float(latitude)
    longitude = col2.text_input("경도(126~130)", "127.2876")
    longitude = float(longitude)
    elevation = col3.text_input("고도", "62.91475")
    elevation = float(elevation)
    warmer = col4.selectbox("보온재", ['없음', '많음'])
    nwarmer = 1 if warmer == '많음' else 0
    shade = col5.selectbox("음/양지", ['양지','음지'])
    nshade = 1 if shade == '양지' else 0

    col1, col2, col3, col4 = st.columns(4)
    temperature = col1.slider("기온", -50.0, 50.0, -7.1)
    humidity =  col2.slider("습도", 0.0, 100.0, 71.1)
    wind_speed = col3.slider("풍속", 0.0, 50.0, 1.9)
    rainfall = col4.slider("강수량", 0.0, 100.0, 0.0)

    col1, col2 = st.columns([8,1])
    submit = col2.form_submit_button('🎓 동파예측')
    
    if submit:
        non_scale = np.array([latitude, longitude, elevation, temperature, rainfall, wind_speed, humidity]).reshape(1, -1)
        with open('scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)

        x = scaler.transform(non_scale)
        x = np.append(x, nwarmer)
        x = np.append(x, nshade).reshape(1, -1)
        full_data = pd.DataFrame(x, columns=["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "보온재여부", "양지/음지"])
        x_data = full_data[["고도", "위도", "경도", "보온재여부"]]
        with open('kmeans_model.pkl', 'rb') as f:
             Kmeans_model = pickle.load(f)

        ncluster = Kmeans_model.predict(x_data)[0] # 값이 배열
        # print(ncluster)
        # k 값에 따른 모델을 찾아서 온도 예측
        models = ["RFRegress_Model_Zip0.pkl", "RFRegress_Model_Zip1.pkl", "RFRegress_Model_Zip2.pkl", "RFRegress_Model_Zip3.pkl"]
        #models = ["RFRegress_Model0.pkl", "RFRegress_Model1.pkl", "RFRegress_Model2.pkl", "RFRegress_Model3.pkl"]
        select_model = models[ncluster]
        with gzip.open(select_model, 'rb') as m:
            rf_model = pickle.load(m)
        
        #x = np.array([latitude, longitude, elevation, temperature, rainfall, wind_speed, humidity]).reshape(1, -1)
        # x = np.array([34.3375, 126.8486, 34, 2.4, 0, 4.9, 64]).reshape(1, -1) 
        # loc = pd.DataFrame(x[:, 0:2], columns=["lat", "lon"]) # 지도를 위한 것(정규화 이전값)
        x_data = full_data[["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지"]]
        
        # x_data = pd.DataFrame(x, columns=["위도", "경도", "고도", "기온", "강수량", "풍속", "습도", "양지/음지"])
        y_pred = rf_model.predict(x_data)[0]

        loc_map = folium.Map(location=[latitude, longitude], 
                            zoom_start = 12)
        tooltip_prn = "예측온도는 {0:0.3f}입니다".format(y_pred)
        folium.Marker([latitude, longitude], icon=folium.Icon(color='red'), 
                        popup=tooltip_prn, tooltip=tooltip_prn).add_to(loc_map)
        st_folium(loc_map, width=1400)
        st.balloons()
        col1.write("#### ✨ 예측온도는 {0:0.3f} 입니다 ✨".format(y_pred))