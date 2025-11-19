from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
/* พื้นหลังแบบไล่สี */
body {
    background: linear-gradient(135deg, #d9a7c7, #fffcdc);
}

/* Card style สำหรับคอลัมน์แต่ละอัน */
.card {
    background: rgba(255,255,255,0.7);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

/* Hover effect */
.card:hover {
    transform: scale(1.06);
    box-shadow: 0 14px 28px rgba(0,0,0,0.25);
}

/* Title glow effect */
.title-glow {
    font-size: 42px !important;
    color: #ffffff;
    text-shadow: 0 0 10px #ff6ec4, 0 0 20px #ff6ec4, 0 0 30px #ff6ec4;
    text-align: center;
}

/* Header style */
.sub-header {
    font-size: 22px;
    color: #6d097b;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-glow'>คนที่หล่อขนาดนี้เป็นของคุณละนะ💖</h1>", unsafe_allow_html=True)
st.header('664230022 นายพชรพล เนตรสุวรรณ')
st.image("./img/pro.jpg", width=350)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='sub-header'>Versicolor</p>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<p class='sub-header'>Virginica</p>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<p class='sub-header'>Setosa</p>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""

st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    #st.write(dt.head(10))
    st.bar_chart(dx2)
    st.button("ไม่แสดงข้อมูล")
else:
        st.write("ไม่แสดงข้อมูล")

html_8 = """
    <div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
    <center><h5>ทำนายข้อมูล</h5></center>
    </div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
        #st.write("ทำนาย")
    dt = pd.read_csv("./data/iris.csv") 
    X = dt.drop('variety', axis=1)
    y = dt.variety   
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)   
    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    st.write(Knn_model.predict(x_input))
        
    out=Knn_model.predict(x_input)

    if out[0] == 'Setosa':
        st.image("./pic/iris1.jpg")
    elif out[0] == 'Versicolor':       
        st.image("./pic/iris2.jpg")
    else:
        st.image("./pic/iris3.jpg")
else:
        st.write("ไม่ทำนาย") 