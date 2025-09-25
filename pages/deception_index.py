# Deception Index Analysis Page
import streamlit as st
import os

# Configure page
st.set_page_config(
    page_title="Pitcher Deception Index", 
    layout="wide",
    page_icon="🤖"
)

def render_navbar():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
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
            justify-content: space-between;
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
        
        .navbar-links {
            display: flex;
            gap: 2rem;
        }
        
        .navbar-links a {
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }
        
        .navbar-links a:hover {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            transform: translateY(-1px);
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
    if st.button("Home", key="nav_home", use_container_width=True):
        st.switch_page("Home.py")

with nav_col2:
    if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
        st.switch_page("pages/demo_dashboard.py")

with nav_col3:
    if st.button("Pitch Prediction", key="nav_pitch", use_container_width=True):
        st.switch_page("pages/pitch_prediction.py")

with nav_col4:
    if st.button("Deception Index", key="nav_deception", use_container_width=True, type="primary"):
        pass  # Already on deception index page

st.markdown("---")

st.title("Pitcher Deception Index")

# Check if HTML file exists
html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "notebooks", "deceptionIndex.html")

if os.path.exists(html_path):
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; margin-bottom: 2rem;">
        <h4 style="color: #1e40af; margin-bottom: 0.5rem;">🔬 Research Overview</h4>
        <p style="color: #1f2937; margin-bottom: 0;">
            The Pitcher Deception Index is a novel metric that quantifies how effectively a pitcher disguises their intentions. 
            This analysis combines release point consistency, movement patterns, and batter reaction times to create a comprehensive deception score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # View full analysis section
    if st.button("📱 View Full Research", type="primary", use_container_width=True):
        with st.spinner("Loading research..."):
            # Read and display the HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=800, scrolling=True)

else:
    st.error("Research file not found. Please check the file path.")
    st.info(f"Looking for: {html_path}")
