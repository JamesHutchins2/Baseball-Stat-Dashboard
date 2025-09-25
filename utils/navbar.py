import streamlit as st

def render_navbar(current_page="Home"):
    """Render the navigation bar with proper Streamlit navigation"""
    
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
        
        .navbar-nav {
            display: flex;
            gap: 1rem;
        }
        
        .nav-btn {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .nav-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        .nav-btn.active {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
    
    <div class="navbar">
        <div class="navbar-content">
            <div class="navbar-brand">⚾ Baseball Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons using Streamlit columns
    nav_col1, nav_col2, nav_col3, nav_col4, nav_spacer = st.columns([1, 1, 1, 1, 6])
    
    with nav_col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True, 
                    type="primary" if current_page == "Home" else "secondary"):
            st.switch_page("Home.py")
    
    with nav_col2:
        if st.button("📈 Dashboard", key="nav_dashboard", use_container_width=True,
                    type="primary" if current_page == "Dashboard" else "secondary"):
            st.switch_page("pages/demo_dashboard.py")
    
    with nav_col3:
        if st.button("⚾ Pitch Prediction", key="nav_pitch", use_container_width=True,
                    type="primary" if current_page == "Pitch Prediction" else "secondary"):
            st.switch_page("pages/pitch_prediction.py")
    
    with nav_col4:
        if st.button("🤖 Deception Index", key="nav_deception", use_container_width=True,
                    type="primary" if current_page == "Deception Index" else "secondary"):
            st.switch_page("pages/deception_index.py")
    
    st.markdown("---")
