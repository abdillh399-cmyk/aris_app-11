# -- coding: utf-8 --
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random

st.set_page_config(layout="wide")
st.title("🇸🇦 ARIS Digital Twin: شبكة النقل العملاقة (أكثر من 1000 نقطة تحليل)")
st.caption("يعرض هذا النظام محاكاة لشبكة أنابيب النقل الممتدة، ملوّنة بالكامل حسب أعلى مستوى خطر تآكل مسجل.")

st.markdown("---")

# ----------------------------------------------------
# 1. تعريف إحداثيات المدن والمراكز الرئيسية
# ----------------------------------------------------
# هذه قيم وهمية، لكنها تعكس التوزيع الجغرافي الواقعي (تضخيم المسافات)
LOCATIONS = {
    "الدمام": (5000, 2600, 1.0),   
    "راس تنورة": (5150, 2750, 1.2), 
    "الجبيل": (4900, 2800, 1.1),    
    "الرياض": (4670, 2470, 0.8),    
    "القصيم": (4390, 2630, 0.9),    
    "جدة": (3920, 2150, 1.0)      
}

# ----------------------------------------------------
# 2. توليد خطوط الأنابيب الرئيسية (ربط المدن ببعضها)
# ----------------------------------------------------
def generate_main_pipelines(locations):
    segments = []
    
    main_path_1 = ["الدمام", "الرياض", "القصيم"]
    main_path_2 = ["الرياض", "جدة"] 
    coastal_paths = [("الدمام", "راس تنورة"), ("الدمام", "الجبيل")] 
    
    all_paths = [main_path_1, main_path_2]
    
    # زيادة عدد نقاط البيانات في كل مقطع إلى 150 نقطة (لضمان تجاوز 1000 نقطة إجمالاً)
    POINTS_PER_SEGMENT = 150 
    
    for path in all_paths:
        for i in range(len(path) - 1):
            start_loc = locations[path[i]]
            end_loc = locations[path[i+1]]
            
            num_points = POINTS_PER_SEGMENT
            t = np.linspace(0, 1, num_points)
            
            x = start_loc[0] + t * (end_loc[0] - start_loc[0]) + np.sin(t * 8) * 50  
            y = start_loc[1] + t * (end_loc[1] - start_loc[1]) + np.cos(t * 8) * 50 
            z = -start_loc[2] + t * (-end_loc[2] - (-start_loc[2])) + np.sin(t * 5) * 0.5 
            
            is_coastal = path[i] in ["الدمام", "راس تنورة"] 
            corrosion_base = 0.35 if is_coastal else 0.15
            corrosion_score = np.clip(corrosion_base + (t * 0.6) + np.random.normal(0, 0.05, num_points), 0.1, 1.0)
            
            segments.append({'X': x, 'Y': y, 'Z': z, 'Corrosion': corrosion_score, 'Path': f'{path[i]}-{path[i+1]}'})

    # إضافة التفرعات الساحلية
    for start, end in coastal_paths:
        start_loc = locations[start]
        end_loc = locations[end]
        
        num_points = 50 # 50 نقطة للتفرعات
        t = np.linspace(0, 1, num_points)
        x = start_loc[0] + t * (end_loc[0] - start_loc[0])
        y = start_loc[1] + t * (end_loc[1] - start_loc[1])
        z = -start_loc[2] + t * (-end_loc[2] - (-start_loc[2]))
        corrosion_score = np.clip(0.6 + np.random.normal(0, 0.05, num_points), 0.1, 1.0) 
        
        segments.append({'X': x, 'Y': y, 'Z': z, 'Corrosion': corrosion_score, 'Path': f'{start}-{end}'})
    
    return segments


# ----------------------------------------------------
# 3. الرسم البياني 3D (الخريطة والتدفق)
# ----------------------------------------------------

st.markdown("### 🚨 الرؤية الحمراء: شبكة الأنابيب الحيوية وملف المخاطر (EM-Locator)")
st.warning("⚠ يظهر هنا نظام النقل الرئيسي ملوّناً بالكامل حسب أعلى مستوى تآكل مسجل في كل مقطع.")

pipe_segments = generate_main_pipelines(LOCATIONS) 
traces = []

# 1. رسم مسارات الأنابيب الملونة بالكامل حسب الخطر
for segment in pipe_segments:
    max_corr = segment['Corrosion'].max()
    
    if max_corr > 0.75:
        line_color = 'red' # حرج جداً
    elif max_corr > 0.6:
        line_color = 'orange' # مرتفع
    else:
        line_color = 'green' # آمن/منخفض
    
    traces.append(
        go.Scatter3d(
            x=segment['X'], y=segment['Y'], z=segment['Z'],
            mode='lines',
            name=segment['Path'],
            line=dict(color=line_color, width=8), # خطوط سميكة وواضحة (8 وحدات)
            hoverinfo='text',
            text=[f"الخطر: {c:.2f} | المسار: {segment['Path']}" for c in segment['Corrosion']],
            showlegend=False
        )
    )

# 2. رسم مواقع المدن كنقاط مرجعية (Hubs)
city_x = [v[0] for v in LOCATIONS.values()]
city_y = [v[1] for v in LOCATIONS.values()]
city_z = [-v[2] for v in LOCATIONS.values()]
city_names = list(LOCATIONS.keys())

traces.append(
    go.Scatter3d(
        x=city_x, y=city_y, z=city_z,
        mode='markers+text',
        name='مراكز التجميع/المدن',
        text=city_names,
        textposition="top center",
        marker=dict(size=12, color='#FFFFFF', symbol='circle', line=dict(width=3, color='black')),
        hoverinfo='text'
    )
)

# 3. إضافة إطار (Boundary) للشبكة لمحاكاة حدود الخريطة
frame_x = [3800, 5200, 5200, 3800, 3800] 
frame_y = [2000, 2000, 2900, 2900, 2000]
frame_z = [0.5, 0.5, 0.5, 0.5, 0.5] 

traces.append(
    go.Scatter3d(
        x=frame_x, y=frame_y, z=frame_z,
        mode='lines',
        name='إطار الخريطة',
        line=dict(color='lightgray', width=2),
        showlegend=False
    )
)


fig = go.Figure(data=traces)

# 4. تخصيص الخريطة (لجعلها تبدو مسطحة كخريطة للمملكة)
fig.update_layout(
    scene = dict(
        xaxis_title='الموقع الأفقي',
        yaxis_title='الموقع الرأسي',
        zaxis_title='العمق (- متر)',
        # التسطيح (Flatter View)
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=1, z=0.03), 
        camera=dict(
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=0), 
            eye=dict(x=0.5, y=0.5, z=2) # زاوية عرض علوية
        )
    ),
    height=800,
    title='رؤية ARIS لنظام النقل الرئيسي - ملون حسب مخاطر التآكل (التسطيح الهندسي)'
)

# عرض الرسم البياني
st.plotly_chart(fig, use_container_width=True)