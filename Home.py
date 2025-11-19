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

# ----------- TITLE -----------
st.markdown("<h1 class='title-glow'>คนที่หล่อขนาดนี้เป็นของคุณละนะ💖</h1>", unsafe_allow_html=True)
st.header('664230022 นายพชรพล เนตรสุวรรณ')
st.image("./img/pro.jpg", width=350)

# ------------- COLUMNS --------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'<p class='sub-header'>Versicolor</p>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Virginica</p>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Setosa</p>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
