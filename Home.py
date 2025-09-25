# Home.py
import streamlit as st
import os

# Configure page
st.set_page_config(
    page_title="Baseball Analytics Portfolio", 
    layout="wide",
    page_icon="⚾",
    initial_sidebar_state="collapsed"
)

def render_navbar():
    st.markdown("""
    <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styling */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Modern navbar */
        .navbar {
            position: sticky;
            top: 0;
            z-index: 999;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1rem 2rem;
            border-radius: 0 0 1rem 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .navbar-content {
            display: flex;
            justify-content: center;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .navbar-brand {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            text-decoration: none;
        }
        
        /* Card styling */
        .portfolio-card {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(226, 232, 240, 0.8);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .portfolio-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        .card-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .card-description {
            color: #6b7280;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
            flex-grow: 1;
        }
        
        /* Button styling */
        .portfolio-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
            color: white;
            text-decoration: none;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            border: none;
            cursor: pointer;
        }
        
        .portfolio-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* Main title styling */
        .main-title {
            text-align: center;
            color: #1f2937;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .main-subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
    </style>
    
    <div class="navbar">
        <div class="navbar-content">
            <div class="navbar-brand">⚾ Baseball Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_navbar()

# Navigation buttons
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.button("Home", key="nav_home", use_container_width=True, type="primary"):
        pass  # Already on home page

with nav_col2:
    if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
        st.switch_page("pages/demo_dashboard.py")

with nav_col3:
    if st.button("Pitch Prediction", key="nav_pitch", use_container_width=True):
        st.switch_page("pages/pitch_prediction.py")

with nav_col4:
    if st.button("Deception Index", key="nav_deception", use_container_width=True):
        st.switch_page("pages/deception_index.py")

st.markdown("---")

st.markdown('<h1 class="main-title">James Hutchins</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Baseball Analytics Portfolio</p>', unsafe_allow_html=True)

# Portfolio cards
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="portfolio-card">
        <div class="card-icon">📈</div>
        <h3 class="card-title">Batter Analytics Dashboard</h3>
        <p class="card-description">
            Interactive dashboard analyzing batter performance metrics, hot/cold streaks, 
            and advanced statistics with real-time data visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Dashboard", key="dashboard", type="primary", use_container_width=True):
        st.switch_page("pages/demo_dashboard.py")

with col2:
    st.markdown("""
    <div class="portfolio-card">
        <div class="card-icon">⚾</div>
        <h3 class="card-title">Markov Chain Pitch Prediction</h3>
        <p class="card-description">
            Advanced machine learning model using Markov chains to predict pitch sequences 
            and analyze pitcher behavior patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Analysis", key="pitch_pred", type="primary", use_container_width=True):
        st.switch_page("pages/pitch_prediction.py")

with col3:
    st.markdown("""
    <div class="portfolio-card">
        <div class="card-icon">🤖</div>
        <h3 class="card-title">Pitcher Deception Index</h3>
        <p class="card-description">
            Novel metric quantifying pitcher deception through movement patterns, and
            release points.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Research", key="deception", type="primary", use_container_width=True):
        st.switch_page("pages/deception_index.py")

# Add some spacing and additional info
st.markdown("<br><br>", unsafe_allow_html=True)

# Contact section
st.markdown("---")
st.markdown("### 📫 Get in Touch")

contact_col1, contact_col2, contact_col3 = st.columns(3)

with contact_col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📧</div>
        <a href="mailto:james.hutchins2@outlook.com" style="text-decoration: none; color: #3b82f6; font-weight: 500;">
            james.hutchins2@outlook.com
        </a>
    </div>
    """, unsafe_allow_html=True)

with contact_col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💼</div>
        <a href="https://www.linkedin.com/in/james-hutchins-5446aa175/" target="_blank" style="text-decoration: none; color: #3b82f6; font-weight: 500;">
            LinkedIn Profile
        </a>
    </div>
    """, unsafe_allow_html=True)

with contact_col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐙</div>
        <a href="https://github.com/jameshutchins2" target="_blank" style="text-decoration: none; color: #3b82f6; font-weight: 500;">
            GitHub Profile
        </a>
    </div>
    """, unsafe_allow_html=True)

# Footer section
st.markdown("""
<div style="background: #f8fafc; padding: 2rem; border-radius: 1rem; margin-top: 3rem; text-align: center;">
    <h4 style="color: #374151; margin-bottom: 1rem;">About This Portfolio</h4>
    <p style="color: #6b7280; line-height: 1.6; margin-bottom: 1.5rem;">
        This portfolio showcases advanced baseball analytics projects combining statistical analysis, 
        machine learning, and data visualization. Each project demonstrates different aspects of 
        modern sports analytics methodology.
    </p>
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Python</span>
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Streamlit</span>
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Plotly</span>
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Pandas</span>
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Machine Learning</span>
        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">Baseball Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

