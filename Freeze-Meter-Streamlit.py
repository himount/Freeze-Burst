import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
import pickle, folium, gzip

# π β¨ π π π  π β οΈ π βΉοΈ π π π¬ βοΈ π π β¬οΈ π₯ π±οΈ π π΄ π π π ποΈ πΌοΈ β¬ β³ π β°

im = Image.open("AI_Lab_logo.jpg")

st.set_page_config(
    page_title="κ³λκΈ° λνμμΈ‘",
    page_icon='π',
    layout="wide",
)

col1, col2 = st.columns([1,8])
col1.image(im, width=100)
col2.write("# μ§λ°©μμλ ν¨λ΄ κ³λκΈ° λν μμΈ‘")

st.write("##### π μμΈ‘μ μν λ°μ΄ν°λ₯Ό μλ ₯ν΄μ£ΌμΈμ ?")
with st.form('My Form'):
    col1, col2, col3, col4, col5 = st.columns(5)
    # latitude = col1.text_input("μλ", 34.10, 38.00, 37.41465)
    # longitude = col2.text_input("κ²½λ", 126.00, 130.00, 127.2876)
    # elevation = col3.number_input("κ³ λ", 0.00, 500.00, 62.91475)
    latitude = col1.text_input("μλ(34~39)", "37.41465")
    latitude = float(latitude)
    longitude = col2.text_input("κ²½λ(126~130)", "127.2876")
    longitude = float(longitude)
    elevation = col3.text_input("κ³ λ", "62.91475")
    elevation = float(elevation)
    warmer = col4.selectbox("λ³΄μ¨μ¬", ['μμ', 'λ§μ'])
    nwarmer = 1 if warmer == 'λ§μ' else 0
    shade = col5.selectbox("μ/μμ§", ['μμ§','μμ§'])
    nshade = 1 if shade == 'μμ§' else 0

    col1, col2, col3, col4 = st.columns(4)
    temperature = col1.slider("κΈ°μ¨", -50.0, 50.0, -7.1)
    humidity =  col2.slider("μ΅λ", 0.0, 100.0, 71.1)
    wind_speed = col3.slider("νμ", 0.0, 50.0, 1.9)
    rainfall = col4.slider("κ°μλ", 0.0, 100.0, 0.0)

    col1, col2 = st.columns([8,1])
    submit = col2.form_submit_button('π λνμμΈ‘')
    
    if submit:
        non_scale = np.array([latitude, longitude, elevation, temperature, rainfall, wind_speed, humidity]).reshape(1, -1)
        with open('scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)

        x = scaler.transform(non_scale)
        x = np.append(x, nwarmer)
        x = np.append(x, nshade).reshape(1, -1)
        full_data = pd.DataFrame(x, columns=["μλ", "κ²½λ", "κ³ λ", "κΈ°μ¨", "κ°μλ", "νμ", "μ΅λ", "λ³΄μ¨μ¬μ¬λΆ", "μμ§/μμ§"])
        x_data = full_data[["κ³ λ", "μλ", "κ²½λ", "λ³΄μ¨μ¬μ¬λΆ"]]
        with open('kmeans_model.pkl', 'rb') as f:
             Kmeans_model = pickle.load(f)

        ncluster = Kmeans_model.predict(x_data)[0] # κ°μ΄ λ°°μ΄
        # print(ncluster)
        # k κ°μ λ°λ₯Έ λͺ¨λΈμ μ°Ύμμ μ¨λ μμΈ‘
        models = ["RFRegress_Model_Zip0.pkl", "RFRegress_Model_Zip1.pkl", "RFRegress_Model_Zip2.pkl", "RFRegress_Model_Zip3.pkl"]
        #models = ["RFRegress_Model0.pkl", "RFRegress_Model1.pkl", "RFRegress_Model2.pkl", "RFRegress_Model3.pkl"]
        select_model = models[ncluster]
        with gzip.open(select_model, 'rb') as m:
            rf_model = pickle.load(m)
        
        #x = np.array([latitude, longitude, elevation, temperature, rainfall, wind_speed, humidity]).reshape(1, -1)
        # x = np.array([34.3375, 126.8486, 34, 2.4, 0, 4.9, 64]).reshape(1, -1) 
        # loc = pd.DataFrame(x[:, 0:2], columns=["lat", "lon"]) # μ§λλ₯Ό μν κ²(μ κ·ν μ΄μ κ°)
        x_data = full_data[["μλ", "κ²½λ", "κ³ λ", "κΈ°μ¨", "κ°μλ", "νμ", "μ΅λ", "μμ§/μμ§"]]
        
        # x_data = pd.DataFrame(x, columns=["μλ", "κ²½λ", "κ³ λ", "κΈ°μ¨", "κ°μλ", "νμ", "μ΅λ", "μμ§/μμ§"])
        y_pred = rf_model.predict(x_data)[0]

        loc_map = folium.Map(location=[latitude, longitude], 
                            zoom_start = 12)
        tooltip_prn = "μμΈ‘μ¨λλ {0:0.3f}μλλ€".format(y_pred)
        folium.Marker([latitude, longitude], icon=folium.Icon(color='red'), 
                        popup=tooltip_prn, tooltip=tooltip_prn).add_to(loc_map)
        st_folium(loc_map, width=1400)
        st.balloons()
        col1.write("#### β¨ μμΈ‘μ¨λλ {0:0.3f} μλλ€ β¨".format(y_pred))