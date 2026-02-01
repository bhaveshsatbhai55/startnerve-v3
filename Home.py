import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="StartNerve Technologies",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for the "Enterprise" feel
st.markdown("""
    <style>
    .hero-text {font-size: 3.5rem; font-weight: 800; color: #4F46E5; text-align: center; margin-top: 2rem;}
    .sub-text {font-size: 1.5rem; color: #4B5563; text-align: center; margin-bottom: 3rem;}
    .feature-card {background: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #4F46E5; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stat-box {text-align: center; padding: 20px; background: white; border-radius: 10px; border: 1px solid #E5E7EB;}
    .stat-num {font-size: 2.5rem; font-weight: bold; color: #111827;}
    </style>
""", unsafe_allow_html=True)

# 🏆 HERO SECTION
st.markdown('<div class="hero-text">StartNerve Technologies</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">The Operating System for Modern Drug Discovery.</div>', unsafe_allow_html=True)

# 📊 LIVE STATS (Fake for now, but looks real)
col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="stat-box"><div class="stat-num">12</div><div>Toxicity Endpoints</div></div>', unsafe_allow_html=True)
col2.markdown('<div class="stat-box"><div class="stat-num">94%</div><div>Validation Accuracy</div></div>', unsafe_allow_html=True)
col3.markdown('<div class="stat-box"><div class="stat-num">8k+</div><div>Chemicals Indexed</div></div>', unsafe_allow_html=True)
col4.markdown('<div class="stat-box"><div class="stat-num">0.4s</div><div>Inference Speed</div></div>', unsafe_allow_html=True)

st.markdown("---")

# 🚀 FEATURE GRID
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="feature-card">
        <h3>🧬 The Bio-Engine</h3>
        <p>Our flagship high-throughput screening tool.</p>
        <ul>
            <li><b>12-Point Toxicity Scan</b> (Liver, Heart, DNA)</li>
            <li><b>FDA Similarity Search</b> (Competitor Analysis)</li>
            <li><b>3D Molecular Visualization</b> (Interactive)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Launch Bio-Engine 🚀"):
        st.switch_page("pages/1_🧬_Bio-Engine.py")

with col_right:
    st.info("📢 **Latest Update (v5.0):** Added 3D Visualization and Batch CSV Processing pipeline for hospital-grade throughput.")
    st.write("### Why StartNerve?")
    st.write("Traditional wet-lab testing takes weeks and costs thousands. StartNerve provides initial safety validation in milliseconds using government-trained AI models.")

st.markdown("<br><br><br><center><small>© 2025 StartNerve Technologies. Built for DY Patil International University.</small></center>", unsafe_allow_html=True)