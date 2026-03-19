import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64

# ===================== 页面设置 =====================
st.set_page_config(
    page_title="肾积脓风险评估系统（Pyonephrosis-CDSS）",
    layout="centered"
)

# ===================== 中文字体修复 =====================
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Microsoft YaHei", "SimSun", sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ===================== 标题 =====================
st.title("🏥 肾积脓风险评估与临床决策系统")
st.markdown("### Pyonephrosis-CDSS")

st.markdown("---")

# ===================== 输入区域 =====================
st.subheader("📋 患者指标录入")

age = st.number_input("年龄", 0, 100, 50)
pct = st.number_input("PCT (ng/mL)", 0.0, 100.0, 1.0)
crp = st.number_input("CRP (mg/L)", 0.0, 300.0, 10.0)
wbc = st.number_input("白细胞 (×10⁹/L)", 0.0, 50.0, 10.0)
neut = st.number_input("中性粒细胞 (×10⁹/L)", 0.0, 50.0, 8.0)
lymph = st.number_input("淋巴细胞 (×10⁹/L)", 0.0, 10.0, 1.0)

nlr = neut / lymph if lymph != 0 else 0

st.markdown(f"👉 **NLR（中性粒/淋巴）= {nlr:.2f}**")

# ===================== 模型加载 =====================
model = joblib.load("model.pkl")

# ===================== 预测 =====================
if st.button("🔍 计算风险"):
    
    input_data = np.array([[age, pct, crp, wbc, neut, lymph, nlr]])
    
    prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    
    st.subheader("📊 预测结果")
    
    st.metric("发生肾积脓风险", f"{prob*100:.2f}%")
    
    # ===================== 风险分级 =====================
    if prob < 0.3:
        st.success("🟢 低风险")
    elif prob < 0.7:
        st.warning("🟡 中风险")
    else:
        st.error("🔴 高风险（建议紧急干预）")

    # ===================== 干预建议 =====================
    st.subheader("🧠 临床建议")
    
    if pct > 2:
        st.write("⚠️ PCT升高 → 强烈提示感染，建议抗感染治疗")
    
    if nlr > 5:
        st.write("⚠️ NLR升高 → 炎症反应明显")
    
    if prob > 0.7:
        st.write("🚨 建议：立即引流 + 抗感染")

    # ===================== 可视化 =====================
    st.subheader("📈 风险可视化")
    
    fig, ax = plt.subplots()
    ax.bar(["风险"], [prob])
    ax.set_ylim(0,1)
    st.pyplot(fig)

    # ===================== PDF报告 =====================
    st.subheader("📄 下载报告")

    report = f"""
    肾积脓风险评估报告

    年龄: {age}
    PCT: {pct}
    CRP: {crp}
    WBC: {wbc}
    NLR: {nlr:.2f}

    风险概率: {prob*100:.2f}%
    """

    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📥 下载报告</a>'
    st.markdown(href, unsafe_allow_html=True)
