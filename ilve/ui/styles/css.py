"""
CSS styling module for ILVE framework - FIXED for full width.
"""

import streamlit as st

def apply_custom_css():
    """Apply the complete custom CSS styling for the ILVE application."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def get_custom_css() -> str:
    """Return the custom CSS as a string."""
    return """
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #f8f9fa;
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* FIXED: Main content area - removed side margins for full width */
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(8px);
        /* REMOVED: margin: 1.5rem; - this was constraining width */
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        /* Use full available width */
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Ensure tabs and their content use full width */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 15px;
        padding: 0.6rem;
        width: 100% !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        width: 100% !important;
        padding: 0 !important;
    }
    
    /* Fix column layouts to use full width */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
    
    /* Ensure vertical blocks use full width */
    div[data-testid="stVerticalBlock"] {
        width: 100% !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.8rem;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #333;
        margin: 2.5rem 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 4px solid #4CAF50;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    /* Cards and containers */
    .info-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
        padding: 1.8rem;
        border-radius: 18px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        margin: 1.2rem 0;
        box-shadow: 0 5px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
        width: 100%; /* Ensure full width */
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }
    
    .metric-card {
        background: white;
        padding: 1.6rem;
        border-radius: 18px;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        width: 100%; /* Ensure full width */
    }
    
    .metric-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.6rem 0;
    }
    
    .metric-label {
        color: #666;
        font-weight: 600;
        font-size: 1.0rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    /* Interactive elements */
    .control-panel {
        background: white;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.03);
        margin-bottom: 1.2rem;
        width: 100%; /* Ensure full width */
    }
    
    .image-display {
        background: white;
        border-radius: 18px;
        padding: 1.8rem;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid rgba(0, 0, 0, 0.03);
        width: 100%; /* Ensure full width */
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.85rem 1.7rem;
        font-weight: 700;
        font-size: 1.0rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 18px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(8px);
        border-radius: 0 20px 20px 0;
        box-shadow: 5px 0 20px rgba(0,0,0,0.05);
    }
    
    /* Tab styling - enhanced for full width */
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 1.8rem;
        background: transparent;
        border-radius: 10px;
        color: #666;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        color: white !important;
        box-shadow: 0 5px 18px rgba(76, 175, 80, 0.3);
    }
    
    /* Progress indicators */
    .progress-container {
        background: rgba(76, 175, 80, 0.1);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        width: 100%; /* Ensure full width */
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
        margin: 0.6rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FFCA28 0%, #FFB300 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
        margin: 0.6rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #EF5350 0%, #D32F2F 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
        margin: 0.6rem 0;
    }
    
    /* Loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.8rem;
        }
        
        .metric-card {
            margin-bottom: 1.2rem;
        }
        
        .main .block-container {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            border-radius: 10px;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.03);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
        border-radius: 5px;
    }
    
    /* Tooltips and help text */
    .help-text {
        background: rgba(76, 175, 80, 0.1);
        padding: 0.9rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.6rem 0;
        font-size: 0.95rem;
        color: #444;
        width: 100%; /* Ensure full width */
    }
    
    /* Feature highlights */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(102, 187, 106, 0.1) 0%, rgba(67, 160, 71, 0.1) 100%);
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        width: 100%; /* Ensure full width */
    }
    
    .feature-highlight h4 {
        color: #388E3C;
        margin-bottom: 0.6rem;
        font-weight: 700;
    }
    
    .learning-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        width: 100%; /* Ensure full width */
    }
    
    .learning-card h5 {
        color: #2196F3;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
    }
    
    .learning-card p {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #333;
    }
    
    .learning-card strong {
        color: #4CAF50;
    }
    
    /* PLOTLY CHARTS - Ensure they use full width */
    .js-plotly-plot .plotly .main-svg {
        width: 100% !important;
    }
    
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* STREAMLIT COLUMNS - Force full width usage */
    .element-container {
        width: 100% !important;
    }
    
    /* FORMS - Ensure they use full width */
    .stForm {
        width: 100% !important;
    }
    
    /* EXPANDERS - Ensure they use full width */
    .streamlit-expanderHeader {
        width: 100% !important;
    }
    
    .streamlit-expanderContent {
        width: 100% !important;
    }
</style>
"""

def get_component_specific_css(component_name: str) -> str:
    """Get CSS specific to a particular component."""
    component_styles = {
        "model_selector": """
        <style>
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
            width: 100%; /* Full width */
        }
        
        .model-card-selected {
            border: 2px solid #4CAF50 !important;
            background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%) !important;
        }
        </style>
        """,
        
        "interactive_explorer": """
        <style>
        .control-sliders {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            width: 100%; /* Full width */
        }
        
        .image-output {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            width: 100%; /* Full width */
        }
        </style>
        """,
        
        "analysis_results": """
        <style>
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
            width: 100%; /* Full width */
        }
        
        .analysis-item {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
            width: 100%; /* Full width */
        }
        
        .analysis-item:hover {
            transform: translateY(-2px);
        }
        </style>
        """
    }
    
    return component_styles.get(component_name, "")

def apply_component_css(component_name: str):
    """Apply CSS specific to a component."""
    css = get_component_specific_css(component_name)
    if css:
        st.markdown(css, unsafe_allow_html=True)