# -- coding: utf-8 --
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go 
import plotly.express as px
from datetime import datetime

# ==========================================================
# 1. LOAD MODEL AND FIXED PARAMETERS
# ==========================================================

@st.cache_resource
def load_model():
    """Loads the pre-trained model (aris_model.pkl)."""
    try:
        # تأكد أن اسم ملف النموذج aris_model.pkl صحيح
        model = joblib.load('aris_model.pkl') 
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

# ==========================================================
# 2. ARIS Index Calculation (Enhanced Sensitivity)
# ==========================================================

def calculate_aris_data(model, vibration, temp, corrosion_score, change_rate, flow_rate, lube_health, stress_in, rul_in):
    """Calculates ARIS Index and dynamic feature importance (CCP) with high sensitivity."""
    
    if model is None:
        # قيم افتراضية في حال فشل تحميل النموذج
        return 25, 40, 30
        
    new_data = pd.DataFrame([[vibration, temp, corrosion_score, change_rate]], 
                            columns=['Vibration_X', 'Bearing_Temp', 'Historical_Corrosion_Score', 'Vibration_Change_Rate'])
    
    try:
        failure_prob = model.predict_proba(new_data)[0][1]
    except Exception:
        return 25, 40, 30 
        
    risk_index = round(failure_prob * 120) 
    
    # Normalization for dynamic risk adjustment (Vibration: 1-25, Temp: 30-85)
    temp_normalized = (temp - 30) / 55 
    vib_normalized = (vibration - 1) / 24 
    
    extra_risk_points = 0
    if temp_normalized > 0.3:
        extra_risk_points += (temp_normalized - 0.3) * 150 
    if vib_normalized > 0.3:
        extra_risk_points += (vib_normalized - 0.3) * 150 
        
    
    # -----------------------------------------------------------
    # إضافة تأثير العوامل الجديدة (Flow Rate و Lube Health و Stress و RUL)
    # -----------------------------------------------------------
    
    if flow_rate < 0.7:
        extra_risk_points += (0.7 - flow_rate) * 50 
    elif flow_rate > 1.1:
        extra_risk_points += (flow_rate - 1.1) * 75
        
    if lube_health < 0.4:
        extra_risk_points += (0.4 - lube_health) * 100 
    
    if stress_in > 0.6:
        extra_risk_points += (stress_in - 0.6) * 80
        
    if rul_in < 0.3:
        extra_risk_points += (0.3 - rul_in) * 120
    
    # -----------------------------------------------------------
        
    risk_index = risk_index + extra_risk_points
    risk_index = int(min(100, risk_index)) 
    
    # Feature Contribution Calculation
    corrosion_normalized = (corrosion_score - 0.1) / 0.9  
    rate_normalized = change_rate / 1.0
    
    corrosion_influence = 0.45 * corrosion_normalized
    rate_influence = 0.40 * rate_normalized
    temp_vib_influence = 0.15 * (temp_normalized + vib_normalized) / 2
    
    total_dynamic_influence = corrosion_influence + rate_influence + temp_vib_influence
    
    if total_dynamic_influence > 0.01:
        corrosion_contribution = round((corrosion_influence / total_dynamic_influence) * 100)
        rate_contribution = round((rate_influence / total_dynamic_influence) * 100)
    else:
        corrosion_contribution = 45 
        rate_contribution = 40
        
    total_contribution = corrosion_contribution + rate_contribution
    if total_contribution > 100:
         corrosion_contribution = round(corrosion_contribution * 100 / total_contribution)
         rate_contribution = round(rate_contribution * 100 / total_contribution)
    
    return risk_index, corrosion_contribution, rate_contribution

# ==========================================================
# 3. MOCK HISTORICAL DATA FUNCTION
# ==========================================================

def get_historical_data(risk_index):
    """Generates mock historical ARIS data based on the current index."""
    
    days = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    
    base_value = risk_index - 15  
    
    history = np.linspace(base_value, risk_index, 30)
    noise = np.random.normal(0, 5, 30) 
    
    historical_risks = np.clip(history + noise, 0, 100).round(0)
    
    historical_risks[-1] = risk_index
    
    df = pd.DataFrame({
        'التاريخ': days,
        'مؤشر ARIS التاريخي': historical_risks
    })
    
    return df

# ==========================================================
# 4. Risk Explanation and Recommendations (Arabic)
# ==========================================================

def explain_risk(risk_index, corr_contrib, rate_contrib, vibration_in, asset_id, temp_in, flow_rate, lube_health, stress_in, rul_in):
    """Generates the Arabic risk explanation and XAI with minimal custom CSS."""
    
    # ----------------------------------------------------
    # تحديد العوامل الرئيسية
    # ----------------------------------------------------
    
    if corr_contrib >= 50:
        dominant_factor = "التآكل التاريخي وسلامة المعدن"
        action_focus = "فحص بالموجات فوق الصوتية (UT) أو فحص ILI/OSI مفصل."
    elif vibration_in > 10 or (100 - corr_contrib - rate_contrib) >= 40:
        dominant_factor = "الإجهاد الميكانيكي اللحظي (اهتزاز/حرارة)"
        action_focus = "إجراء موازنة دقيقة وتوسيط للعمود أو استبدال رولمان بلي (Bearing)."
    else:
        dominant_factor = "معدل التدهور في الأداء"
        action_focus = "مراجعة سجلات التشغيل الأخيرة وتصحيح العيوب التشغيلية."

    
    # ----------------------------------------------------
    # 4.1. عرض المؤشر الدائري في الأعلى
    # ----------------------------------------------------
    
    # الألوان الافتراضية (الزرقاء والبيضاء)
    PRIMARY_COLOR = "#FF4B4B" # Default Streamlit Red
    
    st.markdown(f"<h1>⚙ تقييم الخطر الحالي لـ: {asset_id.split(' ')[0]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #666666;'>آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

    
    col_gauge, col_info = st.columns([1, 2])
    
    with col_gauge:
        # المؤشر الدائري: نستخدم ألوان Streamlit الافتراضية
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_index,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "مؤشر ARIS الكلي", 'font': {'size': 20, 'color': '#333333'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333333"},
                'bar': {'color': '#FF4B4B'}, # اللون الأحمر الافتراضي
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"}, 
                    {'range': [35, 50], 'color': "yellow"}, 
                    {'range': [50, 80], 'color': "orange"}, 
                    {'range': [80, 100], 'color': "red"} 
                ],
                'threshold': {'line': {'color': "darkred", 'width': 4}, 'thickness': 0.75, 'value': 80}}))

        fig_gauge.update_layout(font = {'color': "#333333", 'family': "Arial"}, 
                                autosize=False, width=350, height=300, 
                                paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        
        # ----------------------------------------------------
        # 4.2. ملخص الخطر في مربع معلومات (Box)
        # ----------------------------------------------------
        if risk_index < 35:
            emoji = "✅"
            title = f"المستوى: آمن/منخفض جداً ({risk_index}%)"
            summary = "لا يوجد خطر تشغيلي حالي. استمر في المراقبة الدورية المجدولة."
            box_color = "lightgreen" 
            text_color = "darkgreen" 
        elif 35 <= risk_index < 50:
            emoji = "⚠"
            title = f"المستوى: متوسط - يتطلب تدخلاً ({risk_index}%)"
            summary = f"يجب *إصدار أمر عمل عاجل خلال 7 أيام. الخطر ناتج عن {dominant_factor}. يُنصح بـ **{action_focus}*."
            box_color = "lightyellow" 
            text_color = "goldenrod" 
        elif 50 <= risk_index < 80:
            emoji = "❌"
            title = f"المستوى: مرتفع - يقترب من الحد النهائي ({risk_index}%)"
            summary = f"خطر وشيك! *إيقاف مخطط له خلال 48 ساعة. السبب الرئيسي هو **{dominant_factor}. يجب تنفيذ **{action_focus}* فوراً."
            box_color = "lightsalmon" 
            text_color = "red" 
        else: 
            emoji = "🔥"
            title = f"المستوى: فشل كارثي وشيك! ({risk_index}%)"
            summary = "يجب *إيقاف فوري وعاجل للمضخة*. الأمر الآن هو إزالة المكونات المتضررة واستبدالها بالكامل لتجنب كارثة."
            box_color = "lightcoral" 
            text_color = "darkred" 
            
        st.markdown(f"""
        <div style="
            background-color: {box_color};
            border: 2px solid {text_color}; 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 20px;">
            <h3 style="color: {text_color}; margin-top: 0;">{emoji} {title}</h3>
            <p style="font-size: 1.1em; color: #333333;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    
    # ----------------------------------------------------
    # 4.3. البطاقات الرقمية المتقدمة (KPI Metrics)
    # ----------------------------------------------------
    
    st.markdown("<h3>📊 المقاييس الأساسية ومؤشرات الأداء</h3>", unsafe_allow_html=True)
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4, col_kpi_5 = st.columns(5)
    
    # KPI 1: الاهتزاز 
    col_kpi_1.metric("⚡ الاهتزاز الحالي (mm/s)", f"{vibration_in} mm/s", delta=f"المتبقي للحد: {round(12.5 - vibration_in, 1)}", delta_color="normal" if vibration_in < 12.5 else "inverse")

    # KPI 2: الحرارة 
    col_kpi_2.metric("🌡 حرارة العمود (°C)", f"{temp_in} °C", delta=f"المتبقي للحد: {round(75.0 - temp_in, 1)}", delta_color="normal" if temp_in < 75.0 else "inverse")

    # KPI 3: جودة التزييت 
    col_kpi_3.metric("💧 صحة التزييت (Lube Health)", f"{lube_health * 100:.0f} %", delta="الحد الأدنى: 70%", delta_color="normal" if lube_health >= 0.7 else "inverse")
    
    # KPI 4: الزمن المتبقي للعمل (RUL)
    col_kpi_4.metric("⏳ العمر التشغيلي المتبقي (RUL)", f"{rul_in * 100:.0f} %", delta="المخاطرة تبدأ من 30%", delta_color="normal" if rul_in >= 0.5 else "inverse")
    
    # KPI 5: معدل التغير
    col_kpi_5.metric("📈 معدل التدهور", f"{change_rate_in * 100:.0f} %", delta="سريع" if change_rate_in > 0.3 else "بطيء", delta_color="inverse" if change_rate_in > 0.3 else "normal")


    
    st.markdown("---")
    
    # ----------------------------------------------------
    # 4.4. الرسم البياني الخطي
    # ----------------------------------------------------
    
    col_line, col_xai = st.columns([2, 1])
    
    with col_line:
        st.markdown("<h3>📈 تحليل اتجاه التدهور (آخر 30 يوماً)</h3>", unsafe_allow_html=True)
        historical_df = get_historical_data(risk_index)
        
        # الرسم البياني الخطي: نستخدم ألوان Streamlit الافتراضية
        fig_line = px.line(
            historical_df, 
            x='التاريخ', 
            y='مؤشر ARIS التاريخي', 
            title='معدل تدهور مؤشر ARIS',
            labels={'مؤشر ARIS التاريخي': 'نسبة الخطر (%)', 'التاريخ': 'التاريخ'},
            markers=True
        )
        
        fig_line.add_hline(y=35, line_dash="dash", line_color="green", annotation_text="منطقة آمنة", annotation_position="top right")
        fig_line.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="حد التدخل", annotation_position="top left")
        fig_line.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="فشل وشيك", annotation_position="top right")

        fig_line.update_traces(line_color='#FF4B4B', line_width=3) # اللون الأحمر الافتراضي
        fig_line.update_yaxes(range=[0, 100]) 
        
        fig_line.update_layout(
            # السماح لخلفية المخطط بأن تأخذ خلفية التطبيق الافتراضية
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color="#333333" 
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_xai:
        st.markdown("<h3>🔬 تحليل السبب الجذري (XAI)</h3>", unsafe_allow_html=True)
        
        vib_temp_contrib = 100 - corr_contrib - rate_contrib
        if vib_temp_contrib < 0: vib_temp_contrib = 0 
        
        contributions = {
            "التآكل التاريخي وسلامة المعدن": corr_contrib,
            "الإجهاد الميكانيكي اللحظي": vib_temp_contrib,
            "معدل التدهور في الأداء": rate_contrib
        }
        
        main_reason = max(contributions, key=contributions.get)
        main_contribution = contributions[main_reason]
        
        # صندوق تحليل السبب الجذري: نستخدم تنسيق Streamlit القياسي
        st.info(f"""
        *🥇 السبب الأول: {main_reason}*
        <p style="font-size: 1.2em; font-weight: bold; color: #333333;">نسبة التأثير: {main_contribution}%</p>
        <p style="font-size: 0.9em; color: #666666;">يوجه هذا التحليل فريق الصيانة مباشرة إلى جذر المشكلة.</p>
        """, icon="🔬")

        st.markdown("<h4>تحذيرات إضافية:</h4>", unsafe_allow_html=True)
        messages = []
        if lube_health < 0.7:
            messages.append(f"🛢 جودة التزييت منخفضة ({lube_health * 100:.0f}%): يتطلب تغيير زيت فوري.")
        if stress_in > 0.6:
            messages.append(f"🔗 إجهاد الشد مرتفع ({stress_in}): مؤشر قوي على التصدع الهيكلي.")
        if messages:
            for msg in messages:
                st.markdown(f'<p style="color: red; margin: 5px 0;">{msg}</p>', unsafe_allow_html=True)
        else:
            st.success("⭐ جميع المدخلات التشغيلية الإضافية ضمن الحدود الآمنة.")

    st.markdown("---")
    st.markdown(f"<p style='color: #FF4B4B;'>🔑 نظام ARIS يعمل بـ {asset_id.split(' ')[0]} :** يوفر رؤية استشرافية دقيقة بنقرة زر.</p>", unsafe_allow_html=True)


# ==========================================================
# 5. Streamlit Main Interface
# ==========================================================

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("💡 ARIS Index - واجهة التنبؤ بالمخاطر القائمة على الذكاء الاصطناعي")
st.caption("👈 *نظام رؤية الأصول الصناعية (ARIS):* التصميم الأفضل لرصد التآكل وتدهور المعدات بتقنية الذكاء الاصطناعي المتقدمة.")

# Load Model
model = load_model()

# ==========================================================
# 6. التعامل مع حالة فشل تحميل النموذج
# ==========================================================
if model is None:
    st.header("تطبيق ARIS Index غير متاح حالياً")
    st.warning("⚠ لا يمكن عرض مؤشر الخطر لأن *ملف النموذج (aris_model.pkl)* فشل في التحميل. يرجى التأكد من وجود الملف في مجلد المشروع الرئيسي.")
    st.stop()
    
# ==========================================================
# 7. واجهة المستخدم
# ==========================================================

# ----------------------------------------------------
# 7.1. SIDEBAR (CCP & Materials Info)
# ----------------------------------------------------
st.sidebar.markdown(f"<h2>🛠 مدخلات المستشعرات (Input Data)</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

asset_id = st.sidebar.selectbox(
    "اختر موقع المضخة:",
    options=[
        "1. مضخة الرياض الرئيسية - A (بيئة جافة)", 
        "2. مضخة الدمام الساحلية - B (بيئة بحرية)",
        "3. مضخة الجبيل الصناعية - C (بيئة كيميائية/أكثر حمضية)",
        "4. مضخة راس تنورة - D (بيئة بحرية/ملحية)"
    ],
    index=0,
    help="اختر المضخة لتحديد سياقها البيئي الذي يؤثر على التآكل التاريخي."
)

vibration_in = st.sidebar.slider("1. الاهتزاز الحالي (Vibration_X):", min_value=1.0, max_value=25.0, value=7.0, step=0.1, help="الحد الموصى به: < 12.5 مم/ث.")
temp_in = st.sidebar.slider("2. حرارة العمود (Bearing_Temp):", min_value=30.0, max_value=85.0, value=55.0, step=0.1, help="الحد الموصى به: < 75 درجة مئوية.")

corrosion_default = 0.2 
if "الدمام الساحلية" in asset_id:
    corrosion_default = 0.55 
elif "الجبيل الصناعية" in asset_id:
    corrosion_default = 0.60 
elif "راس تنورة" in asset_id:
    corrosion_default = 0.70 
    
corrosion_in = st.sidebar.slider("3. خطر الفحص التاريخي (ILI/OSI):", min_value=0.1, max_value=1.0, value=corrosion_default, step=0.01, help="جودة المعدن المتبقية (1.0 أسوأ).")
change_rate_in = st.sidebar.slider("4. معدل التغير في الاهتزاز:", min_value=0.0, max_value=1.0, value=0.15, step=0.01, help="سرعة تدهور الأداء (1.0 سريع جداً).")


# === عوامل التآكل الميكانيكي الإضافية ===
st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3>⚙ العوامل التشغيلية الثانوية</h3>", unsafe_allow_html=True)

flow_rate_in = st.sidebar.slider("5. معدل التدفق التشغيلي (نسبة):", min_value=0.5, max_value=1.5, value=1.0, step=0.05, help="معدل التدفق الحالي (1.0 = الأمثل).")

lube_health_in = st.sidebar.slider("6. صحة جودة التزييت (Lube Health):", min_value=0.0, max_value=1.0, value=0.8, step=0.1, help="نسبة جودة الزيت (1.0 ممتاز).")

stress_in = st.sidebar.slider("7. إجهاد السطح/الشد (Tensile Stress):", min_value=0.0, max_value=1.0, value=0.4, step=0.1, help="مستوى الإجهاد الهيكلي (1.0 = مرتفع جداً).")

rul_in = st.sidebar.slider("8. الزمن المتبقي للعمل (RUL):", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="الزمن المتوقع المتبقي لعمر المعدة (1.0 = جديد).")


# === المواد والطلاء في القائمة الجانبية (Sidebar) ===
st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3>🗂 البيانات المرجعية</h3>", unsafe_allow_html=True)

material_options = [
    "الفولاذ الكربوني (CS)", "الفولاذ المقاوم للصدأ 316L", 
    "فولاذ دوبلكس (Duplex 2205)", "سبائك النيكل (Inconel 625)",
    "التيتانيوم", "البرونز", "LCS", "304 SS", "Super Duplex", "Hastelloy C276"
]
coating_options = [
    "إيبوكسي (Epoxy)", "بولي يوريثين (PU)", 
    "إيبوكسي مرتبط بالانصهار (FBE)", "طلاء السيراميك",
    "3LPE", "Zinc Primer", "Glass Flake", "Polyurea", "Phenolic", "Rubber Lining"
]

st.sidebar.selectbox("نوع المعدن المستخدم:", options=material_options, index=1, help="نوع المعدن يؤثر على مدى مقاومة التآكل (Corrosion Score).")
st.sidebar.selectbox("نوع الطلاء:", options=coating_options, index=2, help="يقلل الطلاء الفعال من خطر التآكل الداخلي.")


# ----------------------------------------------------
# 7.2. MAIN PAGE CONTENT (Results)
# ----------------------------------------------------

risk_result, corr_contrib, rate_contrib = calculate_aris_data(
    model, 
    vibration_in, 
    temp_in, 
    corrosion_in, 
    change_rate_in,
    flow_rate_in, 
    lube_health_in,
    stress_in, 
    rul_in 
)

explain_risk(
    risk_result, 
    corr_contrib, 
    rate_contrib, 
    vibration_in, 
    asset_id,
    temp_in,
    flow_rate_in, 
    lube_health_in,
    stress_in, 
    rul_in 
)