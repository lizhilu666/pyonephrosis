import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册中文字体（必须有 simsunb.ttf）
pdfmetrics.registerFont(TTFont('SimSun', 'simsunb.ttf'))

# 页面设置
st.set_page_config(page_title="肾积脓风险评估系统", layout="centered")

st.title("🏥 肾积脓风险评估与临床决策系统")
st.subheader("Pyonephrosis-CDSS")

# ===== 加载模型 =====
model = joblib.load("model.pkl")

# ===== 输入 =====
st.header("📋 患者指标录入")

age = st.number_input("年龄", 0, 120, 50)
pct = st.number_input("PCT (ng/mL)", 0.0, 100.0, 1.0)
crp = st.number_input("CRP (mg/L)", 0.0, 300.0, 10.0)
wbc = st.number_input("白细胞 (×10⁹/L)", 0.0, 50.0, 10.0)
neut = st.number_input("中性粒细胞 (×10⁹/L)", 0.0, 50.0, 8.0)
lymph = st.number_input("淋巴细胞 (×10⁹/L)", 0.0, 50.0, 1.0)

# ===== NLR =====
nlr = neut / lymph if lymph != 0 else 0
st.markdown(f"👉 **NLR（中性粒/淋巴） = {nlr:.2f}**")

# ===== 计算 =====
if st.button("🔍 计算风险"):

    input_dict = {
        "年龄": age,
        "PCT": pct,
        "CRP": crp,
        "白细胞": wbc,
        "中性粒": neut,
        "淋巴细胞": lymph,
        "NLR": nlr
    }

    df = pd.DataFrame([input_dict])

    prob = model.predict_proba(df)[0][1]

    st.subheader("📊 风险评估结果")

    st.metric("脓毒血症风险概率", f"{prob*100:.1f}%")

    if prob < 0.3:
        st.success("低风险 🟢")
    elif prob < 0.6:
        st.warning("中风险 🟡")
    else:
        st.error("高风险 🔴")

    # ===== 干预模拟 =====
    st.subheader("🧠 干预模拟（调整指标 → 风险变化）")

    new_pct = st.slider("调整PCT", 0.0, 100.0, pct)
    new_crp = st.slider("调整CRP", 0.0, 300.0, crp)

    new_input = df.copy()
    new_input["PCT"] = new_pct
    new_input["CRP"] = new_crp

    new_prob = model.predict_proba(new_input)[0][1]

    st.info(f"👉 干预后风险：{new_prob*100:.1f}%")
    st.write(f"风险变化：{(new_prob - prob)*100:.1f}%")

    # ===== SHAP解释 =====
    st.subheader("📈 SHAP解释（影响因素）")

    explainer = shap.Explainer(model, df)
    shap_values = explainer(df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # ===== PDF报告 =====
    st.subheader("📄 下载报告")

    def create_pdf():
        c = canvas.Canvas("report.pdf")

        c.setFont("SimSun", 12)

        c.drawString(50, 800, "肾积脓风险评估报告")
        c.drawString(50, 760, f"风险概率: {prob*100:.1f}%")
        c.drawString(50, 730, f"PCT: {pct}")
        c.drawString(50, 710, f"CRP: {crp}")
        c.drawString(50, 690, f"NLR: {nlr:.2f}")

        c.save()

    if st.button("📥 生成PDF报告"):
        create_pdf()
        with open("report.pdf", "rb") as f:
            st.download_button("下载PDF", f, file_name="report.pdf")
