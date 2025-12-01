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
st.markdown("""
<style>
/* Global Button Style */
div.stButton > button:first-child {
    background-color: #DD5716;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #b94510;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
div.stButton > button:first-child:active {
    background-color: #96380d;
    color: white;
}

/* Chat Input Style */
.stChatInputContainer {
    border-color: #DD5716 !important;
}
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
    st.sidebar.warning(translate_text(S.get("api_key_warning", "⚠️ Please enter your Google Gemini API key to use chat features."), TARGET_LANG, enable_translator, HAVE_TRANSLATOR))


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

    nav_choice = option_menu(
        menu_title=S.get("nav_title", "Navigation"),
        options=nav_options,
        icons=[S.get("icon_predict", "camera-fill"), S.get("icon_ask", "chat-dots-fill")],
        menu_icon="cast",
        default_index=st.session_state.active_tab_index,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4A4A4A", "font-size": "20px"},   # Normal icon color
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px 0px 10px 0px",
                "--hover-color": "#D1D1D1"
            },
            "icon-selected": {
                "color": "#FFFFFF"
            },
            "nav-link-selected": {
                "background-color": "#DD5716",
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
        padding: 20px;
        border-radius: 12px;
        background-color: #FAFAFA;
        border: 2px solid #DD5716;
        color: #1e1e1e;
        line-height: 1.6;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    ">
        <h3 style="margin-top:0; color: #DD5716; display: flex; align-items: center; gap: 10px;">
            {header}
        </h3>
        <div style="white-space: pre-wrap; font-size: 1.05rem;">{info_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # On-demand Listen Button (Cached)
    key = "breed_info_audio"
    if HAVE_GTTS:
        if key in st.session_state.audio_cache:
            st.audio(st.session_state.audio_cache[key], format="audio/mp3")
        else:
            if st.button(translate_text(S.get("listen", "🔊 Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
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
        
        if st.button("🔄 " + translate_text("New Prediction", TARGET_LANG, enable_translator, HAVE_TRANSLATOR)):
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
        img_choice = option_menu(
            menu_title=translate_text(S.get("select_input","Select input type"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR),
            options=[
                translate_text(S["upload"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR),
                translate_text(S["url"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR),
                translate_text(S["webcam"], TARGET_LANG, enable_translator, HAVE_TRANSLATOR)
            ],
            icons=["upload", "link", "camera"],
            menu_icon="image",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#4A4A4A", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#e0e0e0",
                    "border-radius": "8px"
                },
                "nav-link-selected": {
                    "background-color": "#DD5716",
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
                        if st.button(translate_text(S.get("listen", "🔊 Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
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
                        if st.button(translate_text(S.get("listen", "🔊 Listen"), TARGET_LANG, enable_translator, HAVE_TRANSLATOR), key=f"btn_{key}"):
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