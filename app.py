import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ===== 中文字体 =====
pdfmetrics.registerFont(TTFont('SimSun', 'simsunb.ttf'))

# ===== 页面 =====
st.set_page_config(page_title="肾积脓风险评估系统", layout="centered")
st.title("🏥 肾积脓风险评估与临床决策系统")
st.subheader("Pyonephrosis-CDSS")

# ===== 加载模型和列 =====
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

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

    # ===== 自动补齐变量（核心修复）=====
    df = pd.DataFrame([input_dict])

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # ===== 预测 =====
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
    st.subheader("🧠 干预模拟")

    new_pct = st.slider("调整PCT", 0.0, 100.0, pct)
    new_crp = st.slider("调整CRP", 0.0, 300.0, crp)

    df_new = df.copy()
    if "PCT" in df_new.columns:
        df_new["PCT"] = new_pct
    if "CRP" in df_new.columns:
        df_new["CRP"] = new_crp

    new_prob = model.predict_proba(df_new)[0][1]

    st.info(f"👉 干预后风险：{new_prob*100:.1f}%")
    st.write(f"风险变化：{(new_prob - prob)*100:.1f}%")

    # ===== SHAP =====
    st.subheader("📈 SHAP解释")

    try:
        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except:
        st.warning("SHAP暂时无法显示（不影响使用）")

    # ===== PDF =====
    st.subheader("📄 报告下载")

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
