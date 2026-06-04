import os
import io
import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv

# Import modular utils
from utils.ui_constants import UI_STRINGS
from utils.model_utils import load_cnn_model, get_cached_prediction, preprocess_pil
from utils.data_utils import load_dataset, get_breed_info, get_breed_context_string
from utils.gemini_utils import configure_gemini, generate_breed_info, generate_chat_response, generate_manual_chat_response
from utils.translation_utils import translate_to_english, translate_from_english, translate_text
from utils.tts_utils import play_tts, generate_audio_bytes

# Load environment (optional, mostly for other vars if any)
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="🐄 PashuDhan AI", layout="wide")
# Initialize theme state if not present
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "theme_toggle" not in st.session_state:
    st.session_state.theme_toggle = st.session_state.dark_mode

def on_theme_change():
    st.session_state.dark_mode = st.session_state.theme_toggle

# Theme Variables based on Dark Mode
if st.session_state.dark_mode:
    bg_color = "#0B0F19"
    sidebar_bg = "#111827"
    text_color = "#E2E8F0"
    text_muted = "#94A3B8"
    card_bg = "#161B2B"
    border_color = "#1E293B"
    shadow = "0 8px 30px rgba(0, 0, 0, 0.4)"
    primary_color = "#FF7336"
    primary_hover = "#EA580C"
    opt_icon_color = "#94A3B8"
    opt_hover_color = "#1E293B"
    input_bg = "#111827"
    input_border = "#1E293B"
else:
    bg_color = "#F8FAFC"
    sidebar_bg = "#FFFFFF"
    text_color = "#1E293B"
    text_muted = "#64748B"
    card_bg = "#FFFFFF"
    border_color = "#E2E8F0"
    shadow = "0 8px 30px rgba(0, 0, 0, 0.05)"
    primary_color = "#DD5716"
    primary_hover = "#C2410C"
    opt_icon_color = "#475569"
    opt_hover_color = "#F1F5F9"
    input_bg = "#FFFFFF"
    input_border = "#E2E8F0"

# Render sidebar brand title at the very top
st.sidebar.markdown(f"""
<div style="
    font-size: 1.35rem; 
    font-weight: 800; 
    margin-top: -10px;
    margin-bottom: 12px; 
    color: {text_color};
    display: flex;
    align-items: center;
    gap: 8px;
">
    🐄 PashuDhan AI
</div>
""", unsafe_allow_html=True)

# Sidebar theme toggle right under the header
theme_label = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
st.sidebar.toggle(theme_label, key="theme_toggle", on_change=on_theme_change)

# Inject Global Styling & Typography overrides
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

/* Main container spacing & sizing */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px !important;
}}

/* Typography Scaling */
h1 {{
    font-size: 1.45rem !important;
    margin-bottom: 0.5rem !important;
    color: {text_color} !important;
}}
h2 {{
    font-size: 1.15rem !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
    color: {text_color} !important;
}}
h3 {{
    font-size: 0.95rem !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.3rem !important;
    color: {text_color} !important;
}}
p, span, label, li {{
    font-size: 0.85rem !important;
    line-height: 1.45 !important;
    color: {text_color} !important;
}}
small {{
    color: {text_muted} !important;
    font-size: 0.75rem !important;
}}

/* Sidebar Specific Scaling */
section[data-testid="stSidebar"] h1 {{
    font-size: 1.1rem !important;
}}
section[data-testid="stSidebar"] h2 {{
    font-size: 0.95rem !important;
}}
section[data-testid="stSidebar"] h3 {{
    font-size: 0.88rem !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li {{
    font-size: 0.8rem !important;
}}

/* Reduce vertical gaps between sidebar widgets */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
    padding-bottom: 0.35rem !important;
    padding-top: 0px !important;
}}

/* App Container and Main Page views */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
}}

/* Sidebar Container - Restrict width styling only when it is expanded */
section[data-testid="stSidebar"][aria-expanded="true"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {border_color} !important;
    min-width: 340px !important;
    max-width: 340px !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {border_color} !important;
}}

/* Replace collapse button material text with unicode arrow symbol to prevent "keyboard" text leak */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="collapsedControl"] button span {{
    display: none !important;
}}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    background-color: transparent !important;
    border: none !important;
}}
[data-testid="stSidebarCollapseButton"] button::before,
[data-testid="collapsedControl"] button::before {{
    content: "▶" !important;
    display: inline-block !important;
    font-size: 1.1rem !important;
    color: {text_color} !important;
    font-weight: bold !important;
    visibility: visible !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button::before {{
    content: "◀" !important;
    display: inline-block !important;
    font-size: 1.1rem !important;
    color: {text_color} !important;
    font-weight: bold !important;
    visibility: visible !important;
}}

/* Ensure dropdown selectbox and input fields have clear borders and correct padding */
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] div[data-baseweb="input"] {{
    border: 1px solid {input_border} !important;
    border-radius: 8px !important;
    background-color: {input_bg} !important;
    height: 38px !important;
}}

/* Force input background and text color directly */
div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
div[data-testid="stTextInput"] input {{
    background-color: {input_bg} !important;
    color: {text_color} !important;
}}
div[data-testid="stTextInput"] button,
div[data-testid="stTextInput"] button[title="Show password text"],
div[data-testid="stTextInput"] button[title="Hide password text"] {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    color: {text_color} !important;
}}
div[data-testid="stTextInput"] button:hover,
div[data-testid="stTextInput"] button[title="Show password text"]:hover,
div[data-testid="stTextInput"] button[title="Hide password text"]:hover {{
    background-color: rgba(255, 255, 255, 0.1) !important;
}}
div[data-testid="stTextInput"] button svg,
div[data-testid="stTextInput"] button[title="Show password text"] svg,
div[data-testid="stTextInput"] button[title="Hide password text"] svg {{
    fill: {text_color} !important;
    color: {text_color} !important;
}}

/* Ensure the dropdown selection text has high contrast */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] p,
div[data-baseweb="select"] input {{
    color: {text_color} !important;
    font-size: 0.85rem !important;
}}

/* Dropdown list popover colors */
div[data-baseweb="menu"],
div[role="listbox"],
ul[data-testid="stSelectboxVirtualDropdown"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
}}
div[data-baseweb="menu"] li,
div[role="listbox"] li,
ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"] {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
}}
div[data-baseweb="menu"] li:hover,
div[role="listbox"] li:hover,
ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover {{
    background-color: {primary_color} !important;
    color: white !important;
}}

/* Override focus and hover borders of input elements */
div[data-baseweb="select"]:hover > div,
div[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
div[data-baseweb="select"]:focus-within > div,
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
    border-color: {primary_color} !important;
}}

/* Option Menu styling overrides */
.container-fluid,
.container,
.row,
.navbar,
.nav,
.nav-item,
.nav-link,
[role="navigation"],
div[role="navigation"] div,
div[role="navigation"] ul,
div[role="navigation"] li,
div[role="navigation"] a {{
    background-color: transparent !important;
    border: none !important;
}}
iframe[title="streamlit_option_menu.option_menu"] {{
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
}}
.nav-link {{
    color: {text_color} !important;
    font-size: 0.85rem !important;
    padding: 6px 10px !important;
    margin-bottom: 4px !important;
    border-radius: 6px !important;
}}
.nav-link.active,
.active,
div[role="navigation"] a.active,
div[role="navigation"] li.active {{
    background-color: {primary_color} !important;
    color: white !important;
}}

/* Custom primary buttons styling */
div.stButton > button:first-child {{
    background-color: {primary_color} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    transition: all 0.2s ease-in-out !important;
}}
div.stButton > button:first-child:hover {{
    background-color: {primary_hover} !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    transform: translateY(-1px) !important;
    color: white !important;
}}
div.stButton > button:first-child:active {{
    transform: translateY(1px) !important;
}}

/* Chat Input Styling */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stChatInput"] {{
    background-color: {bg_color} !important;
    background: {bg_color} !important;
    border-top: none !important;
    box-shadow: none !important;
}}
.stChatInputContainer {{
    border-color: {primary_color} !important;
    background-color: {input_bg} !important;
}}

/* File uploader area */
div[data-testid="stFileUploader"] {{
    background-color: {card_bg} !important;
    border: 1.5px dashed {border_color} !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
}}
div[data-testid="stFileUploader"] div,
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] button {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
}}

/* Streamlit Notifications / Alert Box Overrides (st.info, st.error, st.warning) */
div[data-testid="stNotification"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
    box-shadow: {shadow} !important;
}}
div[data-testid="stNotification"] p {{
    font-size: 0.88rem !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}}

/* Chat bubbles customization */
div[data-testid="chatAvatarIcon-user"] {{
    background-color: {primary_color} !important;
}}
div[data-testid="stChatMessage"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    padding: 0.75rem 1rem !important;
    box-shadow: {shadow} !important;
}}
</style>
""", unsafe_allow_html=True)

# Always enable translation if available
try:
    from deep_translator import GoogleTranslator
    HAVE_TRANSLATOR = True
except ImportError:
    HAVE_TRANSLATOR = False
enable_translator = HAVE_TRANSLATOR

# Check if TTS is available
try:
    from gtts import gTTS
    HAVE_GTTS = True
except ImportError:
    HAVE_GTTS = False

# Sidebar Language Selection

lang_choice = st.sidebar.selectbox("Interface Language / भाषा / ਭਾਸ਼ਾ", list(UI_STRINGS.keys()))
S = UI_STRINGS[lang_choice]
TARGET_LANG = lang_choice

# API Key Input
user_api_key = st.sidebar.text_input(translate_text(S.get("api_key_label", "Gemini API Key"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), type="password", help=translate_text(S.get("api_key_help", "Enter your Google Gemini API key here."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key="gemini_api_key")
if user_api_key:
    configure_gemini(user_api_key)
else:
    warning_text = translate_text(S.get("api_key_warning", "Please enter your Google Gemini API key to use chat features."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
    st.sidebar.markdown(f"""
    <div style="
        padding: 10px 14px;
        border-radius: 8px;
        background-color: {card_bg};
        border: 1px solid {border_color};
        color: #EAB308;
        font-size: 0.82rem;
        margin-bottom: 12px;
        line-height: 1.4;
    ">
        {warning_text}
    </div>
    """, unsafe_allow_html=True)


# Initialize last_lang if not present
if "last_lang" not in st.session_state:
    st.session_state.last_lang = TARGET_LANG

# Handle Language Change
if st.session_state.last_lang != TARGET_LANG:
    # Clear all history and cached info on language change
    st.session_state.chat_history = []
    st.session_state.manual_chat_history = []
    st.session_state.current_breed_info = None
    # Resetting predicted breed will trigger re-generation of info if image persists
    st.session_state.current_predicted_breed = None 
    st.session_state.audio_cache = {}
    
    # Update last_lang
    st.session_state.last_lang = TARGET_LANG
    st.rerun()

st.title(S["title"])

# ---------------- SIDEBAR NAVIGATION & TOGGLES ----------------
# st.sidebar.markdown("---")
# st.sidebar.header(S.get("nav_title", "Navigation"))

# nav_choice = st.sidebar.radio("Navigation", [S["tab_predict"], S["tab_ask"]], label_visibility="collapsed")

from streamlit_option_menu import option_menu

with st.sidebar:
    # Initialize active_tab_index if not present
    if "active_tab_index" not in st.session_state:
        st.session_state.active_tab_index = 0

    nav_options = [S["tab_predict"], S["tab_ask"]]

    # Clean markdown header instead of option menu default title frame
    st.markdown(f"""
    <div style="
        font-size: 0.95rem; 
        font-weight: 700; 
        margin-top: 15px; 
        margin-bottom: 8px; 
        color: {text_color};
        opacity: 0.8;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        {S.get("nav_title", "Navigation")}
    </div>
    """, unsafe_allow_html=True)

    nav_choice = option_menu(
        menu_title=None,
        options=nav_options,
        icons=[S.get("icon_predict", "camera-fill"), S.get("icon_ask", "chat-dots-fill")],
        default_index=st.session_state.active_tab_index,
        styles={
            "container": {"padding": "0!important", "background-color": sidebar_bg, "border": "none!important", "box-shadow": "none!important"},
            "nav": {"background-color": sidebar_bg, "border": "none!important", "box-shadow": "none!important"},
            "nav-item": {"background-color": sidebar_bg, "border": "none!important", "box-shadow": "none!important"},
            "icon": {"color": opt_icon_color, "font-size": "15px"},
            "nav-link": {
                "font-size": "0.82rem",
                "text-align": "left",
                "margin": "0px 0px 4px 0px",
                "padding": "6px 10px",
                "--hover-color": opt_hover_color,
                "color": text_color,
                "border-radius": "6px",
                "background-color": sidebar_bg
            },
            "icon-selected": {
                "color": "#FFFFFF"
            },
            "nav-link-selected": {
                "background-color": primary_color,
                "color": "#FFFFFF",                 
            },
        }
    )

    # Update active_tab_index based on selection
    if nav_choice in nav_options:
        st.session_state.active_tab_index = nav_options.index(nav_choice)



# ---------------- SESSION STATE INIT ----------------
# 1. History for Predict Tab
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_predicted_breed" not in st.session_state:
    st.session_state.current_predicted_breed = None
if "current_breed_context" not in st.session_state:
    st.session_state.current_breed_context = ""
if "current_breed_info" not in st.session_state:
    st.session_state.current_breed_info = None
if "persisted_image" not in st.session_state:
    st.session_state.persisted_image = None

# 2. History for Ask/Manual Tab
if "manual_chat_history" not in st.session_state:
    st.session_state.manual_chat_history = []
if "manual_current_breed" not in st.session_state:
    st.session_state.manual_current_breed = None

# 3. Audio Cache
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

# Load resources
model = load_cnn_model()
dataset = load_dataset()

if model is None:
    st.warning(S["no_model"])
if dataset is None:
    st.warning(S["no_dataset"])

# Helper to play and cache audio
def play_and_cache_audio(text, key):
    if HAVE_GTTS:
        spinner_text = translate_text(S.get("generating_audio", "Generating audio..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
        with st.spinner(spinner_text):
            audio_bytes = generate_audio_bytes(text, TARGET_LANG)
            
        if audio_bytes:
            st.session_state.audio_cache[key] = audio_bytes
            st.rerun()
        else:
            st.error(S.get('tts_failed','Audio failed:'))

# Render breed info helper (Modified to accept text)
def render_breed_info_ui(info_text: str):
    header = translate_text(S.get("breed_information","Breed Information"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
    st.markdown(f"""
    <div style="
        padding: 24px;
        border-radius: 16px;
        background-color: {card_bg};
        border: 1px solid {border_color};
        color: {text_color};
        line-height: 1.7;
        margin-bottom: 20px;
        box-shadow: {shadow};
        transition: transform 0.2s ease-in-out;
    ">
        <h3 style="margin-top:0; color: {primary_color}; display: flex; align-items: center; gap: 10px; font-weight: 700;">
            🐄 {header}
        </h3>
        <div style="white-space: pre-wrap; font-size: 1.05rem; opacity: 0.95;">{info_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # On-demand Listen Button (Cached)
    key = "breed_info_audio"
    if HAVE_GTTS:
        if key in st.session_state.audio_cache:
            st.audio(st.session_state.audio_cache[key], format="audio/mp3")
        else:
            if st.button(translate_text(S.get("listen", "Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
                play_and_cache_audio(info_text, key)


# ==============================================================================
# TAB 1: PREDICT BREED & CONTINUOUS CHAT
# ==============================================================================
if nav_choice == S["tab_predict"]:
    
    uploaded_image = None

    # Check if we have a persisted image
    if st.session_state.persisted_image is not None:
        st.image(st.session_state.persisted_image, width=350)
        uploaded_image = st.session_state.persisted_image
        
        if st.button(translate_text("New Prediction", TARGET_LANG, enable_translator, HAVE_TRANSLATOR)):
            st.session_state.persisted_image = None
            st.session_state.current_predicted_breed = None
            st.session_state.chat_history = []
            st.session_state.current_breed_info = None
            # Clear audio cache related to this tab? Maybe just breed info
            if "breed_info_audio" in st.session_state.audio_cache:
                del st.session_state.audio_cache["breed_info_audio"]
            # Also clear chat audio keys?
            keys_to_remove = [k for k in st.session_state.audio_cache.keys() if k.startswith("chat_")]
            for k in keys_to_remove:
                del st.session_state.audio_cache[k]
            
            st.rerun()

    else:
        # Informational message (Red Warning)
        st.error(translate_text(S.get("initial_setup_instructions", "**Important Setup Instructions**\n"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
        info_msg = (
            "- **Please use your Gemini API key.**\n"
            "- **Please select the language of Pashudhan AI for before starting conversation.**"
        )
        st.info(translate_text(info_msg, TARGET_LANG, enable_translator, HAVE_TRANSLATOR))

        st.subheader(translate_text(S.get("upload_or_capture", S.get("upload","Upload or Capture Image")), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))

        # from streamlit_option_menu import option_menu
        # Clean header for input type selection
        st.markdown(f"""
        <div style="
            font-size: 0.92rem; 
            font-weight: 700; 
            margin-top: 12px; 
            margin-bottom: 8px; 
            color: {text_color};
            opacity: 0.8;
        ">
            {translate_text(S.get("select_input","Select input type"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)}
        </div>
        """, unsafe_allow_html=True)

        img_choice = option_menu(
            menu_title=None,
            options=[
                translate_text(S["upload"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR),
                translate_text(S["url"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR),
                translate_text(S["webcam"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
            ],
            icons=["upload", "link", "camera"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": bg_color, "border": "none!important", "box-shadow": "none!important"},
                "nav": {"background-color": bg_color, "border": "none!important", "box-shadow": "none!important"},
                "nav-item": {"background-color": bg_color, "border": "none!important", "box-shadow": "none!important"},
                "icon": {"color": opt_icon_color, "font-size": "15px"},
                "nav-link": {
                    "font-size": "0.85rem",
                    "text-align": "center",
                    "margin": "0px 4px 0px 4px",
                    "--hover-color": opt_hover_color,
                    "border-radius": "6px",
                    "color": text_color,
                    "background-color": bg_color
                },
                "nav-link-selected": {
                    "background-color": primary_color,
                    "color": "white",
                },
            }
        )

        if img_choice == translate_text(S["upload"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR):
            file = st.file_uploader(translate_text(S.get("choose_image","Choose image"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), type=["jpg","jpeg","png"])
            if file:
                uploaded_image = Image.open(file)
                st.session_state.persisted_image = uploaded_image
                st.rerun() # Rerun to switch to "persisted" view
        elif img_choice == translate_text(S["url"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR):
            url = st.text_input(translate_text(S.get("enter_url","Enter image URL"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
            if url:
                try:
                    resp = requests.get(url, timeout=10)
                    uploaded_image = Image.open(io.BytesIO(resp.content))
                    st.session_state.persisted_image = uploaded_image
                    st.rerun()
                except Exception:
                    st.error(translate_text(S.get("invalid_url","Invalid URL"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
        elif img_choice == translate_text(S["webcam"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR):
            cam = st.camera_input(translate_text(S.get("take_photo","Take photo"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
            if cam:
                uploaded_image = Image.open(cam)
                st.session_state.persisted_image = uploaded_image
                st.rerun()

    # Logic to run prediction AND manage chat session reset
    if uploaded_image is not None and model is not None:
        try:
            # Run prediction (Cached)
            # 1. Preprocess to get array (fast, no need to cache)
            input_array = preprocess_pil(uploaded_image)
            # 2. Run inference (slow, cached based on array content)
            prediction_result = get_cached_prediction(model, input_array)
            predicted_breed_label = prediction_result.get("label")
            
            # Display Prediction Info
            st.info(translate_text(f"{S['predicted']}: {predicted_breed_label}", TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
            st.write(translate_text(f"{S['confidence']}: {prediction_result.get('confidence')*100:.2f}%", TARGET_LANG, enable_translator, HAVE_TRANSLATOR))

            # CHECK: Is this a NEW breed detection? If so, reset chat history
            if st.session_state.current_predicted_breed != predicted_breed_label:
                st.session_state.current_predicted_breed = predicted_breed_label
                st.session_state.chat_history = [] # Clear history
                st.session_state.current_breed_info = None # Clear info
                # Clear audio cache for new breed
                if "breed_info_audio" in st.session_state.audio_cache:
                    del st.session_state.audio_cache["breed_info_audio"]
                
                # Fetch info for context
                context_str = get_breed_context_string(dataset, predicted_breed_label)
                st.session_state.current_breed_context = context_str

                # --- GENERATE GEMINI INFO ---
                if user_api_key:
                    with st.spinner(translate_text(S.get("generating_info", "Generating breed information..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)):
                        info_text_raw = generate_breed_info(user_api_key, predicted_breed_label, context_str)
                    
                    if info_text_raw:
                        info_text_display = translate_from_english(info_text_raw, TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
                        st.session_state.current_breed_info = info_text_display
                        # Auto-play removed
                else:
                    # Fallback if no API key (optional, maybe just show warning)
                    pass

            # Show Breed Details (Gemini Info or Fallback)
            if st.session_state.current_breed_info:
                render_breed_info_ui(st.session_state.current_breed_info)
            elif not user_api_key:
                 st.warning(translate_text(S.get("api_key_warning", "Please enter Gemini API Key to see detailed AI-generated breed information."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))
                 # Fallback table
                 info_row = get_breed_info(dataset, predicted_breed_label)
                 if info_row:
                     fallback_text = "\n".join([f"{k.replace('_',' ').title()}: {v}" for k, v in info_row.items() if str(v).strip() not in ["nan", "None", ""]])
                     # Translate fallback text
                     fallback_text = translate_text(fallback_text, TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
                     render_breed_info_ui(fallback_text)

        except Exception as e:
            st.error(f"{translate_text(S.get('error_prediction', 'Error during prediction:'), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)} {translate_text(str(e), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)}")

    # ---------------- CONTINUOUS CHAT FOR PREDICTED BREED ----------------
    if st.session_state.current_predicted_breed:
        st.markdown("---")
        st.subheader(translate_text(S.get("chat_history", "Chat about this Breed"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))

        # 1. Display Chat History
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and HAVE_GTTS:
                    key = f"chat_{i}"
                    if key in st.session_state.audio_cache:
                        st.audio(st.session_state.audio_cache[key], format="audio/mp3")
                    else:
                        if st.button(translate_text(S.get("listen", "Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
                            play_and_cache_audio(message["content"], key)

        # 2. Chat Input
        placeholder_text = translate_text(S.get("follow_up_placeholder", "Ask follow up..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
        if user_input := st.chat_input(placeholder_text):
            
            # A. Show User Message
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # B. Generate Response
            if user_api_key:
                dataset_context = st.session_state.current_breed_context
                breed_name = st.session_state.current_predicted_breed
                
                q_en = translate_to_english(user_input, enable_translator, HAVE_TRANSLATOR)
                
                with st.spinner(translate_text(S.get("thinking", "PashuDhan AI is thinking..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)):
                    answer_raw = generate_chat_response(user_api_key, breed_name, dataset_context, q_en)
                
                if answer_raw:
                    answer_display = translate_from_english(answer_raw, TARGET_LANG, enable_translator, HAVE_TRANSLATOR)

                    # C. Show Assistant Message
                    with st.chat_message("assistant"):
                        st.markdown(answer_display)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_display})
                    st.rerun() # Rerun to show the new message in the loop with the button
            else:
                st.error(translate_text(S.get("api_key_error", "Please insert your Google Gemini API key in the sidebar to use the chat features."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))


# ==============================================================================
# TAB 2: ASK PASHU AI (ANY BREED - CONTINUOUS CHAT)
# ==============================================================================
elif nav_choice == S["tab_ask"]:
    st.subheader(translate_text(S.get("ask_question", "Ask a Question about the Cattle"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))

    if dataset is not None:
        # 1. Select Breed
        breed_names = dataset["breed"].dropna().unique().tolist()
        breed_names.sort()
        
        # Explicit persistence for Tab 2 selection
        if "manual_selected_index" not in st.session_state:
            st.session_state.manual_selected_index = 0
            
        def on_breed_change():
            st.session_state.manual_chat_history = [] # Reset history on change
            # Clear manual chat audio cache
            keys_to_remove = [k for k in st.session_state.audio_cache.keys() if k.startswith("manual_chat_")]
            for k in keys_to_remove:
                del st.session_state.audio_cache[k]

        selected_breed = st.selectbox(
            translate_text(S.get("select_breed", "Select Breed"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), 
            breed_names, 
            index=breed_names.index(st.session_state.manual_current_breed) if st.session_state.manual_current_breed in breed_names else 0,
            key="manual_select_box"
        )
        
        # Update state if changed (Streamlit handles the widget state, but we sync our var)
        if st.session_state.manual_current_breed != selected_breed:
             st.session_state.manual_current_breed = selected_breed
             st.session_state.manual_chat_history = []
             # Clear audio cache
             keys_to_remove = [k for k in st.session_state.audio_cache.keys() if k.startswith("manual_chat_")]
             for k in keys_to_remove:
                del st.session_state.audio_cache[k]
             st.rerun()

        # Calculate Context for this breed
        manual_breed_context = f"Breed: {selected_breed}\n"
        context_text = get_breed_context_string(dataset, selected_breed)
        if context_text:
            manual_breed_context += context_text
        else:
            manual_breed_context = f"{translate_text(S.get('no_dataset_info', 'No dataset info available for'), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)} {selected_breed}."
        
        # 2. Display Chat History
        for i, message in enumerate(st.session_state.manual_chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and HAVE_GTTS:
                    key = f"manual_chat_{i}"
                    if key in st.session_state.audio_cache:
                        st.audio(st.session_state.audio_cache[key], format="audio/mp3")
                    else:
                        if st.button(translate_text(S.get("listen", "Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
                            play_and_cache_audio(message["content"], key)

        # 3. Chat Input (Continuous)
        ph_text = translate_text(S.get("manual_chat_placeholder", "Ask anything about this breed..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
        if manual_input := st.chat_input(ph_text):
            
            # A. Display User Message
            st.chat_message("user").markdown(manual_input)
            st.session_state.manual_chat_history.append({"role": "user", "content": manual_input})

            # B. Generate Response
            if user_api_key:
                q_en = translate_to_english(manual_input, enable_translator, HAVE_TRANSLATOR)
                
                with st.spinner(translate_text(S.get("thinking", "PashuDhan AI is thinking..."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR)):
                    answer_raw = generate_manual_chat_response(user_api_key, selected_breed, manual_breed_context, q_en)
                
                if answer_raw:
                    answer_display = translate_from_english(answer_raw, TARGET_LANG, enable_translator, HAVE_TRANSLATOR)

                    # C. Display Assistant Message
                    with st.chat_message("assistant"):
                        st.markdown(answer_display)
                    
                    st.session_state.manual_chat_history.append({"role": "assistant", "content": answer_display})
                    st.rerun()
            else:
                st.error(translate_text(S.get("api_key_error", "Please insert your Google Gemini API key in the sidebar to use the chat features."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))