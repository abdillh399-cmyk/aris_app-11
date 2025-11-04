# -- coding: utf-8 --
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Customer Journey & History | رحلة العميل وتاريخ الأصل")
st.caption("تحليل التنبؤ مقابل الواقع - يوضح سهولة العمل وقوة النموذج")

# بيانات وهمية لتاريخ المضخة (Vibration and ARIS Trend)
dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=15, freq='W'))
historical_data = {
    'Date': dates,
    'ARIS Index': [10, 18, 25, 33, 40, 52, 65, 78, 90, 85, 70, 50, 40, 35, 30],
    'Vibration (mm/s)': [2.5, 3.0, 3.5, 4.0, 4.5, 5.5, 6.8, 7.5, 8.2, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5],
    'Historical Corrosion Score': [0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4]
}
history_df = pd.DataFrame(historical_data)

st.markdown("### 📊 مسار مؤشر ARIS التاريخي (آخر 15 أسبوعاً)")
st.info("💡 *نقطة قيمة:* يظهر هذا المخطط كيف نجح ARIS في التنبؤ قبل أن يصل الخطر إلى الحد النهائي (80%) مما يمنع الفشل الكارثي.")

fig = px.line(history_df, x='Date', y='ARIS Index', title='ARIS Index Trend Over Time', line_shape='spline')
fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="الحد النهائي (80%)", annotation_position="top left")
# محاكاة لتاريخ اتخاذ قرار الإيقاف (عند الأسبوع 8 تقريباً)
fig.add_vline(x=history_df['Date'].iloc[8], line_dash="dash", line_color="blue", annotation_text="❌ تم الإيقاف المُخطط له في هذا التاريخ", annotation_position="bottom right")

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# تحليل جودة المعدن (التآكل الداخلي)
# ==========================================================
st.markdown("---")
st.markdown("### 🔬 تحليل العوامل المخفية: التآكل مقابل الاهتزاز")
st.warning("*النقد المُعالج:* مشكلة أن الاهتزاز قد يكون منخفضاً لكن التآكل الداخلي يسبب الخطر الحقيقي.")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### (1) الاهتزاز اللحظي (Vibration)")
    fig_vib = px.line(history_df, x='Date', y='Vibration (mm/s)', title='Vibration Trend', line_shape='spline')
    fig_vib.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="حد الاهتزاز المتوسط")
    st.plotly_chart(fig_vib, use_container_width=True)

with col_b:
    st.markdown("#### (2) الخطر التاريخي (Corrosion Score)")
    fig_corr = px.area(history_df, x='Date', y='Historical Corrosion Score', title='Corrosion Trend (ILI/OSI)', line_shape='spline')
    fig_corr.update_traces(fill='tozeroy', line_color='orange')
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
نظام ARIS يتغلب على هذه المشكلة عبر:
1.  *الدمج الذكي:* ARIS هو مؤشر واحد يجمع بين الاهتزاز (العامل السريع) والتآكل (العامل البطيء المخفي)، مما يمنع المفاجآت.
2.  *الـ XAI الموجه:* التفسير الفوري يحدد *السبب الجذري* (الذي قد يكون التآكل حتى لو كان الاهتزاز مقبولاً)، ويوجه فريق الصيانة إلى الفحص الداخلي فوراً.
""")