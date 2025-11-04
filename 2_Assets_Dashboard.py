# -- coding: utf-8 --
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Asset Dashboard | لوحة تحكم الأصول")
st.caption("نظرة عامة على الأسطول وتصنيف المخاطر للمدراء")

# بيانات وهمية لمحاكاة أسطول المضخات
data = {
    'Asset ID': ["مضخة الرياض-A", "مضخة الشرقية-B", "مضخة جدة-C", "مضخة القصيم-D", "مضخة المنطقة الحرة-E"],
    'ARIS Index': [15, 68, 42, 95, 29],
    'Vibration (mm/s)': [4.5, 15.2, 7.8, 22.0, 5.5],
    'Corrosion Score': [0.2, 0.7, 0.4, 0.9, 0.3],
    'Last Action': ['مراقبة دورية', 'إيقاف مخطط', 'فحص UT', 'إيقاف فوري', 'موازنة'],
    'Status': ['آمن', 'مرتفع', 'متوسط', 'فشل وشيك', 'منخفض']
}

df = pd.DataFrame(data)

# ==========================================================
# وظيفة التلوين
# ==========================================================
def color_risk(val):
    if val >= 80:
        color = 'background-color: #ff4b4b' # Red
    elif val >= 50:
        color = 'background-color: #ff9a00' # Orange
    elif val >= 35:
        color = 'background-color: #fff400' # Yellow
    else:
        color = 'background-color: #00ba7c' # Green
    return color

# ==========================================================
# عرض المقاييس الرئيسية في الأعمدة (حركات بركات)
# ==========================================================
st.markdown("### مقاييس الأسطول الرئيسية")
col1, col2, col3, col4 = st.columns(4)

total_assets = len(df)
high_risk = df[df['ARIS Index'] >= 50]
low_risk_count = df[df['ARIS Index'] < 35].shape[0]

col1.metric("إجمالي الأصول", total_assets)
col2.metric("أصول بخطر مرتفع (> 50%)", high_risk.shape[0], delta=f"-{high_risk.shape[0] / total_assets * 100:.1f}%")
col3.metric("أصول بخطر منخفض", low_risk_count)
col4.metric("أقصى ARIS مسجل", df['ARIS Index'].max(), delta_color="inverse")

# ==========================================================
# عرض الجدول مع التنسيق
# ==========================================================

st.markdown("### تصنيف الأصول حسب الخطر (Risk Prioritization)")
st.info("💡 *نقطة قيمة:* تظهر هنا الأصول التي يجب أن يبدأ بها فريق الصيانة (Prioritization).")
st.dataframe(
    df.style.applymap(color_risk, subset=['ARIS Index']), 
    use_container_width=True,
    column_config={
        "ARIS Index": st.column_config.ProgressColumn(
            "ARIS Index",
            help="مؤشر الخطر المحسوب بالذكاء الاصطناعي",
            format="%f",
            min_value=0,
            max_value=100,
        ),
    }
)

# ==========================================================
# توزيع المخاطر (Visualization)
# ==========================================================
st.markdown("### توزيع الأصول حسب مستوى الخطر")
risk_counts = df['Status'].value_counts().reset_index()
risk_counts.columns = ['Risk Level', 'Count']

risk_order = ['آمن', 'منخفض', 'متوسط', 'مرتفع', 'فشل وشيك']
risk_counts['Risk Level'] = pd.Categorical(risk_counts['Risk Level'], categories=risk_order, ordered=True)
risk_counts = risk_counts.sort_values('Risk Level')

fig_bar = px.bar(
    risk_counts, 
    x='Risk Level', 
    y='Count', 
    title='Assets Distribution by Risk Level',
    color='Risk Level',
    color_discrete_map={
        'آمن': '#00ba7c',
        'منخفض': '#fff400',
        'متوسط': '#ff9a00',
        'مرتفع': '#ff9a00',
        'فشل وشيك': '#ff4b4b'
    }
)
st.plotly_chart(fig_bar, use_container_width=True)