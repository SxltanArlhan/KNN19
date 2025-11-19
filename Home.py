from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: "Prompt", sans-serif;
}

/* พื้นหลัง Gradient */
body {
    background: linear-gradient(135deg, #ffe6e6, #e0c3fc, #8ec5fc);
}

/* กล่อง Card */
.card {
    background: rgba(255,255,255,0.8);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 16px 30px rgba(0,0,0,0.25);
}

/* Title Glow */
.title-glow {
    font-size: 45px !important;
    font-weight: 700;
    color: white;
    text-shadow: 0 0 10px #ff61d8, 0 0 20px #ff61d8, 0 0 30px #ff61d8;
    text-align: center;
    margin-top: -30px;
}

/* Sub-header under images */
.sub-header {
    font-size: 22px;
    color: #6d097b;
    font-weight: bold;
    text-align: center;
}

/* Centering image manually */
.center-img img {
    display: block;
    justify-content: center;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE + PROFILE CENTERED
# ---------------------------------------------------------
st.set_page_config( layout="centered")
st.markdown("<h1 class='title-glow'>คนที่หล่อขนาดนี้เป็นของคุณละนะ💖</h1>", unsafe_allow_html=True)

st.image("./img/pro.jpg")

st.markdown("<h3 style='text-align:center; color:#4a0072;'>664230022 นายพชรพล เนตรสุวรรณ</h3>", unsafe_allow_html=True)
st.markdown("---")


# ---------------------------------------------------------
# IRIS IMAGES
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='sub-header'>Versicolor</p>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)

with col2:
    st.markdown("<p class='sub-header'>Virginica</p>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)

with col3:
    st.markdown("<p class='sub-header'>Setosa</p>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)

st.markdown("---")


# ---------------------------------------------------------
# DATA STATS CARD
# ---------------------------------------------------------
st.markdown("""
<div style="background-color:#FF7C8A;padding:15px;border-radius:15px;">
<center><h4 style="color:white;">📊 สถิติข้อมูลดอกไม้</h4></center>
</div>
""", unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# สรุปข้อมูล
dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["petal.length", "petal.width", "sepal.length", "sepal.width"])


# ปุ่มแสดงกราฟ
if st.button("แสดงการจินตทัศน์ข้อมูล (Bar Chart)"):
    st.bar_chart(dx2)
else:
    st.info("คลิกปุ่มเพื่อแสดงข้อมูล")


st.markdown("---")


# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------
st.markdown("""
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;">
<center><h4 style="color:black;">🔮 ทำนายข้อมูล KNN</h4></center>
</div>
""", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    pt_len = st.slider("petal.length", 1.0, 7.0, 2.0)
    pt_wd = st.slider("petal.width", 0.1, 3.0, 0.5)

with colB:
    sp_len = st.number_input("sepal.length", 4.0, 8.0, 5.0)
    sp_wd = st.number_input("sepal.width", 2.0, 5.0, 3.0)

if st.button("ทำนายผล"):
    dt = pd.read_csv("./data/iris.csv")
    X = dt.drop('variety', axis=1)
    y = dt.variety

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    out = knn.predict(x_input)

    st.success(f"🌸 ผลลัพธ์ที่ทำนายได้: **{out[0]}**")

    if out[0] == 'Setosa':
        st.image("./img/iris1.jpg")
    elif out[0] == 'Versicolor':
        st.image("./img/iris2.jpg")
    else:
        st.image("./img/iris3.jpg")
else:
    st.info("กรอกข้อมูลและกดปุ่มเพื่อทำนาย")
