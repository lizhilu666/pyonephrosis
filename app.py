# =========================
# 导入
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from io import BytesIO
import os

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================
# 页面
# =========================
st.set_page_config(layout="wide")
st.title("🏥 肾积脓智能风险评估系统（最终医院上线版）")

# =========================
# 加载模型
# =========================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# 字体自动适配（关键终极修复）
# =========================
def load_font():
    try:
        # 优先当前目录
        font_path = os.path.join(os.getcwd(), "simsun.ttf")
        pdfmetrics.registerFont(TTFont('SimSun', font_path))
        return "SimSun"
    except:
        try:
            # Windows系统字体
            font_path = "C:/Windows/Fonts/simsun.ttc"
            pdfmetrics.registerFont(TTFont('SimSun', font_path))
            return "SimSun"
        except:
            return "Helvetica"  # fallback

font_name = load_font()

# =========================
# 中文映射
# =========================
feature_map = {
    "age": "年龄",
    "fever": "发热",
    "Calculus.diameter": "结石直径",
    "Degree.of.hydronephrosis": "肾积水分级",
    "Leukocytes": "白细胞",
    "PLT": "血小板",
    "HGB": "血红蛋白",
    "NLR": "中性粒/淋巴比值",
    "LMR": "淋巴/单核比值",
    "Cr": "肌酐",
    "TT": "凝血时间TT",
    "PT": "凝血时间PT",
    "APTT": "APTT",
    "HR": "心率"
}

# =========================
# 布局
# =========================
col1, col2 = st.columns([1, 1.5])

# =========================
# 输入区
# =========================
with col1:
    st.subheader("🧾 患者信息录入")

    age = st.number_input("年龄", 0, 100, 50)
    fever = st.selectbox("是否发热", ["否", "是"])
    diameter = st.number_input("结石直径(mm)", 0.0, 50.0, 10.0)
    hydro = st.selectbox("肾积水分级", ["0", "I", "II", "III"])
    leuk = st.number_input("白细胞", 0.0, 50.0, 10.0)
    plt_val = st.number_input("血小板", 0.0, 500.0, 200.0)
    hgb = st.number_input("血红蛋白", 0.0, 200.0, 120.0)
    nlr = st.number_input("中性粒/淋巴比值 NLR", 0.0, 20.0, 3.0)
    lmr = st.number_input("淋巴/单核比值 LMR", 0.0, 20.0, 4.0)
    cr = st.number_input("肌酐 Cr", 0.0, 1000.0, 80.0)
    tt = st.number_input("TT", 0.0, 50.0, 15.0)
    pt = st.number_input("PT", 0.0, 50.0, 12.0)
    aptt = st.number_input("APTT", 0.0, 100.0, 30.0)
    hr = st.number_input("心率", 0.0, 200.0, 80.0)

# =========================
# 数据构建（完全稳定）
# =========================
input_dict = {
    "age": age,
    "fever": 1 if fever == "是" else 0,
    "Calculus.diameter": diameter,
    "Degree.of.hydronephrosis": ["0","I","II","III"].index(hydro),
    "Leukocytes": leuk,
    "PLT": plt_val,
    "HGB": hgb,
    "NLR": nlr,
    "LMR": lmr,
    "Cr": cr,
    "TT": tt,
    "PT": pt,
    "APTT": aptt,
    "HR": hr
}

data = pd.DataFrame([input_dict])
data = data.reindex(columns=columns)
data = data.fillna(0)

# =========================
# 结果区
# =========================
with col2:

    if st.button("🧠 计算风险"):

        # ===== 预测 =====
        risk = model.predict_proba(data)[0][1]

        st.subheader("📊 风险评估结果")
        st.metric("风险概率", f"{risk:.2%}")

        if risk < 0.3:
            level = "低风险"
        elif risk < 0.6:
            level = "中风险"
        else:
            level = "高风险"

        st.write("风险等级：", level)

        # ===== SHAP =====
        st.subheader("🔍 风险来源分析（SHAP）")

        try:
            explainer = shap.LinearExplainer(model, data)
            shap_values = explainer(data)

            shap_df = pd.DataFrame({
                "变量": data.columns,
                "贡献值": shap_values.values[0]
            })

            shap_df["中文"] = shap_df["变量"].map(feature_map)
            shap_df = shap_df.sort_values("贡献值", key=abs, ascending=False)

            st.dataframe(shap_df[["中文","贡献值"]])

        except:
            st.warning("SHAP分析暂不可用")

        # ===== 干预模拟 =====
        st.subheader("🧪 干预模拟（降低风险）")

        def simulate(col, new_val):
            temp = data.copy()
            temp[col] = new_val
            new_risk = model.predict_proba(temp)[0][1]
            return new_risk, risk - new_risk

        r1, d1 = simulate("Leukocytes", leuk*0.7)
        r2, d2 = simulate("NLR", nlr*0.7)
        r3, d3 = simulate("Cr", cr*0.8)

        st.write(f"降低白细胞 → {r1:.2%}（↓{d1:.2%}）")
        st.write(f"降低NLR → {r2:.2%}（↓{d2:.2%}）")
        st.write(f"降低肌酐 → {r3:.2%}（↓{d3:.2%}）")

        # ===== PDF =====
        def generate_pdf():

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)

            styles = getSampleStyleSheet()
            styles["Normal"].fontName = font_name
            styles["Title"].fontName = font_name

            story = []

            story.append(Paragraph("肾积脓风险评估报告（HIS版）", styles["Title"]))
            story.append(Spacer(1, 20))

            table_data = [["指标","数值"]]

            for k,v in input_dict.items():
                table_data.append([feature_map[k], str(v)])

            table_data.append(["风险概率", f"{risk:.2%}"])
            table_data.append(["风险等级", level])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ("GRID",(0,0),(-1,-1),1,colors.black),
                ("BACKGROUND",(0,0),(-1,0),colors.grey),
                ("FONTNAME",(0,0),(-1,-1),font_name)
            ]))

            story.append(table)
            story.append(Spacer(1, 20))

            story.append(Paragraph(
                "临床建议：建议重点控制炎症指标（白细胞、NLR）及肾功能（肌酐）。",
                styles["Normal"]
            ))

            doc.build(story)
            buffer.seek(0)

            return buffer

        pdf = generate_pdf()

        st.download_button(
            "📄 下载PDF报告",
            pdf,
            file_name="肾积脓风险评估报告.pdf"
        )