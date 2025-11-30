import os
import io
import tempfile
import re
import time
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests

# ML imports
import tensorflow as tf

# ---------- Load .env ----------
from dotenv import load_dotenv
load_dotenv()     # IMPORTANT: loads GEMINI_API_KEY + OPENAI_API_KEY from .env

# Gemini (required)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    genai = None

# Optional EfficientNet preprocess
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    def preprocess_input(x):
        return x.astype("float32") / 255.0

# Optional speech recognition
try:
    import speech_recognition as sr
    HAVE_SR = True
except Exception:
    HAVE_SR = False

# Optional TTS
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# Optional translator
try:
    from deep_translator import GoogleTranslator
    HAVE_TRANSLATOR = True
except Exception:
    HAVE_TRANSLATOR = False

# Streamlit setup
st.set_page_config(page_title="PashuDhan AI", layout="wide")
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #28a745;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# UI strings (multilingual)
UI_STRINGS = {
    "English": {
        "title": "PashuDhan AI",
        "nav_title": "Navigation",
        "tab_predict": "Predict Breed & Chat",
        "tab_ask": "Ask PashuAI (General)",
        "enable_audio": "Enable Audio",
        "enable_translation": "Enable Translation",
        "upload": "Upload Image",
        "url": "Image URL",
        "webcam": "Use Webcam",
        "generate": "Generate Response",
        "predicted": "Predicted Breed",
        "confidence": "Confidence",
        "no_model": "Model not found. Place my_model.h5 in folder.",
        "no_dataset": "dataset.csv missing.",
        "ask_question": "Ask a Question about the Cattle",
        "select_input": "Select input type",
        "take_photo": "Take photo",
        "choose_image": "Choose image",
        "enter_url": "Enter image URL",
        "response": "Response",
        "invalid_url": "Invalid URL",
        "no_dataset_info": "No dataset info found for this breed.",
        "enter_question": "Enter question.",
        "breed_information": "Breed Information",
        "tts_failed": "Audio failed:",
        "chat_history": "Chat with AI about this Breed",
        "follow_up_placeholder": "Ask a follow-up question about this animal...",
        "separate_section": "Independent Query Section",
        "manual_chat_placeholder": "Ask anything about this breed..."
    },
    "Hindi": {
        "title": "पशुधन AI",
        "nav_title": "नेविगेशन",
        "tab_predict": "नस्ल पहचान और चैट",
        "tab_ask": "पशुधन AI से पूछें (सामान्य)",
        "enable_audio": "ऑडियो सक्षम करें",
        "enable_translation": "अनुवाद सक्षम करें",
        "upload": "छवि अपलोड करें",
        "url": "छवि URL",
        "webcam": "वेबकैम का उपयोग करें",
        "generate": "जवाब प्राप्त करें",
        "predicted": "अनुमानित नस्ल",
        "confidence": "विश्वास स्तर",
        "no_model": "मॉडल नहीं मिला। कृपया my_model.h5 फ़ोल्डर में रखें।",
        "no_dataset": "dataset.csv नहीं मिला।",
        "ask_question": "पशु के बारे में प्रश्न पूछें",
        "select_input": "इनपुट प्रकार चुनें",
        "take_photo": "फोटो लें",
        "choose_image": "छवि चुनें",
        "enter_url": "छवि URL दर्ज करें",
        "response": "जवाब",
        "invalid_url": "अमान्य URL",
        "no_dataset_info": "इस नस्ल के लिए dataset जानकारी नहीं मिली।",
        "enter_question": "कृपया प्रश्न दर्ज करें।",
        "breed_information": "नस्ल जानकारी",
        "tts_failed": "ऑडियो त्रुटि:",
        "chat_history": "इस नस्ल के बारे में AI से चैट करें",
        "follow_up_placeholder": "इस जानवर के बारे में और सवाल पूछें...",
        "separate_section": "स्वतंत्र प्रश्न अनुभाग",
        "manual_chat_placeholder": "इस नस्ल के बारे में कुछ भी पूछें..."
    },
    "Marathi": {
        "title": "पशुधन AI",
        "nav_title": "नेव्हिगेशन",
        "tab_predict": "जात ओळख आणि चैट",
        "tab_ask": "पशुधन AI ला विचारा (सामान्य)",
        "enable_audio": "ऑडिओ सक्षम करा",
        "enable_translation": "भाषांतर सक्षम करा",
        "upload": "प्रतिमा अपलोड करा",
        "url": "प्रतिमा URL",
        "webcam": "वेबकॅम वापरा",
        "generate": "उत्तर तयार करा",
        "predicted": "भाकीत केलेली जात",
        "confidence": "विश्वास",
        "no_model": "मॉडेल सापडले नाही. कृपया my_model.h5 ठेवा.",
        "no_dataset": "dataset.csv सापडला नाही.",
        "ask_question": "गुरांबद्दल प्रश्न विचारा",
        "select_input": "इनपुट प्रकार निवडा",
        "take_photo": "फोटो घ्या",
        "choose_image": "प्रतिमा निवडा",
        "enter_url": "प्रतिमा URL भरा",
        "response": "उत्तर",
        "invalid_url": "अवैध URL",
        "no_dataset_info": "या जातीबद्दल dataset माहिती सापडली नाही.",
        "enter_question": "कृपया प्रश्न भरा.",
        "breed_information": "जात माहिती",
        "tts_failed": "ऑडिओ त्रुटी:",
        "chat_history": "AI सोबत चर्चा करा",
        "follow_up_placeholder": "पुढील प्रश्न विचारा...",
        "separate_section": "स्वतंत्र प्रश्न विभाग",
        "manual_chat_placeholder": "या जातीबद्दल काहीही विचारा..."
    },
    "Punjabi": {
        "title": "ਪਸ਼ੁਧਨ AI",
        "nav_title": "ਨੇਵੀਗੇਸ਼ਨ",
        "tab_predict": "ਨਸਲ ਪਹਚਾਣ ਅਤੇ ਚੈਟ",
        "tab_ask": "ਪਸ਼ੂਧਨ AI ਤੋਂ ਪੁੱਛੋ (ਆਮ)",
        "enable_audio": "ਆਡੀਓ ਸਮਰੱਥ ਕਰੋ",
        "enable_translation": "ਅਨੁਵਾਦ ਸਮਰੱਥ ਕਰੋ",
        "upload": "ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",
        "url": "ਤਸਵੀਰ URL",
        "webcam": "ਵੇਬਕੈਮ ਨਾਲ ਤਸਵੀਰ ਲਓ",
        "generate": "ਜਵਾਬ ਤਿਆਰ ਕਰੋ",
        "predicted": "ਅਨੁਮਾਨਿਤ ਨਸਲ",
        "confidence": "ਭਰੋਸਾ",
        "no_model": "ਮਾਡਲ ਨਹੀਂ ਮਿਲਿਆ। ਕਿਰਪਾ ਕਰਕੇ my_model.h5 ਫੋਲਡਰ ਵਿਚ ਰੱਖੋ।",
        "no_dataset": "dataset.csv ਨਹੀਂ ਮਿਲਿਆ।",
        "ask_question": "ਪਸ਼ੂ ਬਾਰੇ ਪ੍ਰਸ਼ਨ ਪੁੱਛੋ",
        "select_input": "ਇਨਪੁਟ ਕਿਸਮ ਚੁਣੋ",
        "take_photo": "ਫੋਟੋ ਖਿੱਚੋ",
        "choose_image": "ਤਸਵੀਰ ਚੁਣੋ",
        "enter_url": "ਤਸਵੀਰ ਦਾ URL ਦਾਖ਼ਲ ਕਰੋ",
        "response": "ਜਵਾਬ",
        "invalid_url": "ਅਵੈਧ URL",
        "no_dataset_info": "ਇਸ ਨਸਲ ਲਈ ਡੇਟਾਸੇਟ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।",
        "enter_question": "ਕਿਰਪਾ ਕਰਕੇ ਪ੍ਰਸ਼ਨ ਦਿਓ।",
        "breed_information": "ਨਸਲ ਜਾਣਕਾਰੀ",
        "tts_failed": "ਆਡੀਓ ਤਰੁੱਟੀ:",
        "chat_history": "AI ਨਾਲ ਗੱਲਬਾਤ ਕਰੋ",
        "follow_up_placeholder": "ਹੋਰ ਸਵਾਲ ਪੁੱਛੋ...",
        "separate_section": "ਵੱਖਰਾ ਸਵਾਲ ਸੈਕਸ਼ਨ",
        "manual_chat_placeholder": "ਇਸ ਨਸਲ ਬਾਰੇ ਕੁਝ ਵੀ ਪੁੱਛੋ..."
    },
}

# sidebar language selection
lang_choice = st.sidebar.selectbox("Interface Language / भाषा / ਭਾਸ਼ਾ", list(UI_STRINGS.keys()))
S = UI_STRINGS[lang_choice]
TARGET_LANG = lang_choice

st.title(S["title"])

# ---------------- SIDEBAR NAVIGATION & TOGGLES ----------------
st.sidebar.markdown("---")
st.sidebar.header(S.get("nav_title", "Navigation"))
nav_choice = st.sidebar.radio("", [S["tab_predict"], S["tab_ask"]])

st.sidebar.markdown("---")
enable_tts = st.sidebar.checkbox(S["enable_audio"], True)
enable_translator = st.sidebar.checkbox(S["enable_translation"], HAVE_TRANSLATOR)

# ---------------- SESSION STATE INIT ----------------
# 1. History for Predict Tab
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_predicted_breed" not in st.session_state:
    st.session_state.current_predicted_breed = None
if "current_breed_context" not in st.session_state:
    st.session_state.current_breed_context = ""

# 2. History for Ask/Manual Tab
if "manual_chat_history" not in st.session_state:
    st.session_state.manual_chat_history = []
if "manual_current_breed" not in st.session_state:
    st.session_state.manual_current_breed = None

# Load model + dataset
MODEL_PATH = "my_model.h5"
DATASET_PATH = "dataset.csv"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None

@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return None

model = load_cnn_model()
dataset = load_dataset()

if model is None:
    st.warning(S["no_model"])
if dataset is None:
    st.warning(S["no_dataset"])

# Breed labels
BREEDS = [
'Alambadi','Amritmahal','Ayrshire','Banni','Bargur','Bhadawari','Brown_Swiss',
'Dangi','Deoni','Gir','Guernsey','Hallikar','Hariana','Holstein_Friesian',
'Jaffrabadi','Jersey','Kangayam','Kankrej','Kasargod','Kenkatha','Kherigarh',
'Khillari','Krishna_Valley','Malnad_gidda','Mehsana','Murrah','Nagori','Nagpuri',
'Nili_Ravi','Nimari','Ongole','Pulikulam','Rathi','Red_Sindhi','Sahiwal',
'Tharparkar','Toda','Umblachery','Vechur'
]

# Preprocess + predict
def preprocess_pil(img: Image.Image, size=224):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_breed(img: Image.Image):
    x = preprocess_pil(img)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return {"label": BREEDS[idx], "confidence": float(probs[idx])}

# Translation helpers
LANG_CODE_MAP = {"English":"en","Hindi":"hi","Marathi":"mr","Punjabi":"pa"}

def translate_to_english(text: str) -> str:
    if not enable_translator or not HAVE_TRANSLATOR:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "English" or not enable_translator or not HAVE_TRANSLATOR:
        return text
    try:
        code = LANG_CODE_MAP.get(target_lang, "en")
        return GoogleTranslator(source="en", target=code).translate(text)
    except Exception:
        return text

def translate_text(text: str, target_lang: str) -> str:
    if not enable_translator or not HAVE_TRANSLATOR:
        return text
    try:
        code = LANG_CODE_MAP.get(target_lang, "en")
        return GoogleTranslator(source="auto", target=code).translate(text)
    except Exception:
        return text

# TTS cleaning & playback
def clean_for_tts(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`~#>\-•→←↑↓✔✓➤★☆▶►]", " ", text)
    text = re.sub(r":[a-zA-Z_]+:", " ", text)
    text = re.sub(r"[^\w\s\.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def play_tts(text: str, lang_key: str = TARGET_LANG):
    if not HAVE_GTTS:
        return
    try:
        clean_text = clean_for_tts(text)
        lang_code = LANG_CODE_MAP.get(lang_key, "en")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                gTTS(clean_text, lang=lang_code).save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                gTTS(clean_text, lang="en").save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")
    except Exception as e:
        st.error(f"{S.get('tts_failed','Audio failed:')} {e}")

# Render breed info
def render_breed_info(info_row: dict):
    header = translate_text(S.get("breed_information","Breed Information"), TARGET_LANG)
    st.markdown(f"""
    <div style="
        padding: 18px;
        border-radius: 12px;
        background-color: #1e1e1e;
        border: 1px solid #444;
        color: white;
        line-height: 1.6;
    ">
        <h3 style="margin-top:0;">🐄 {header}</h3>
    """, unsafe_allow_html=True)

    text_lines = []
    for k, v in info_row.items():
        if str(v).strip() not in ["nan","None",""]:
            clean_k = translate_text(k.replace("_"," ").title(), TARGET_LANG)
            clean_v = translate_text(str(v), TARGET_LANG)
            st.markdown(f"<p><strong>{clean_k}:</strong> {clean_v}</p>", unsafe_allow_html=True)
            text_lines.append(f"{clean_k}: {clean_v}")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: PREDICT BREED & CONTINUOUS CHAT
# ==============================================================================
if nav_choice == S["tab_predict"]:
    st.subheader(translate_text(S.get("upload_or_capture", S.get("upload","Upload or Capture Image")), TARGET_LANG))

    img_choice = st.radio(translate_text(S.get("select_input","Select input type"), TARGET_LANG),
                          [translate_text(S["upload"], TARGET_LANG),
                           translate_text(S["url"], TARGET_LANG),
                           translate_text(S["webcam"], TARGET_LANG)])

    uploaded_image = None

    if img_choice == translate_text(S["upload"], TARGET_LANG):
        file = st.file_uploader(translate_text(S.get("choose_image","Choose image"), TARGET_LANG), type=["jpg","jpeg","png"])
        if file:
            uploaded_image = Image.open(file)
            st.image(uploaded_image, width=350)
    elif img_choice == translate_text(S["url"], TARGET_LANG):
        url = st.text_input(translate_text(S.get("enter_url","Enter image URL"), TARGET_LANG))
        if url:
            try:
                resp = requests.get(url, timeout=10)
                uploaded_image = Image.open(io.BytesIO(resp.content))
                st.image(uploaded_image, width=350)
            except Exception:
                st.error(translate_text(S.get("invalid_url","Invalid URL"), TARGET_LANG))
    elif img_choice == translate_text(S["webcam"], TARGET_LANG):
        cam = st.camera_input(translate_text(S.get("take_photo","Take photo"), TARGET_LANG))
        if cam:
            uploaded_image = Image.open(cam)
            st.image(uploaded_image, width=350)

    # Logic to run prediction AND manage chat session reset
    if uploaded_image is not None and model is not None:
        try:
            # Run prediction
            prediction_result = predict_breed(uploaded_image)
            predicted_breed_label = prediction_result.get("label")
            
            # Display Prediction Info
            st.info(translate_text(f"{S['predicted']}: {predicted_breed_label}", TARGET_LANG))
            st.write(translate_text(f"{S['confidence']}: {prediction_result.get('confidence')*100:.2f}%", TARGET_LANG))

            # CHECK: Is this a NEW breed detection? If so, reset chat history
            if st.session_state.current_predicted_breed != predicted_breed_label:
                st.session_state.current_predicted_breed = predicted_breed_label
                st.session_state.chat_history = [] # Clear history
                
                # Fetch info for context
                context_str = ""
                if dataset is not None:
                    breed_row = dataset[dataset["breed"].astype(str).str.lower() == predicted_breed_label.lower()]
                    if not breed_row.empty:
                        info_row = breed_row.iloc[0].to_dict()
                        context_str = "\n".join([f"{k}: {v}" for k, v in info_row.items() if str(v).strip() not in ["nan", "None", ""]])
                
                st.session_state.current_breed_context = context_str

            # Show Breed Details
            if dataset is not None:
                 breed_row = dataset[dataset["breed"].astype(str).str.lower() == predicted_breed_label.lower()]
                 if not breed_row.empty:
                     info_row = breed_row.iloc[0].to_dict()
                     render_breed_info(info_row)
                 else:
                     st.warning(translate_text(S["no_dataset_info"], TARGET_LANG))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # ---------------- CONTINUOUS CHAT FOR PREDICTED BREED ----------------
    if st.session_state.current_predicted_breed:
        st.markdown("---")
        st.subheader(translate_text(S.get("chat_history", "Chat about this Breed"), TARGET_LANG))

        # 1. Display Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Chat Input
        placeholder_text = translate_text(S.get("follow_up_placeholder", "Ask follow up..."), TARGET_LANG)
        if user_input := st.chat_input(placeholder_text):
            
            # A. Show User Message
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # B. Generate Response
            try:
                if genai:
                    dataset_context = st.session_state.current_breed_context
                    breed_name = st.session_state.current_predicted_breed
                    
                    q_en = translate_to_english(user_input)
                    
                    final_prompt = f"""
                    You are a veterinary expert. 
                    The user has uploaded an image of a cattle breed identified as: {breed_name}.
                    
                    Here is the official dataset information about this breed:
                    {dataset_context}

                    User Question: {q_en}

                    Answer the question based on the dataset info and your general knowledge. 
                    Keep the answer helpful and concise.
                    """

                    # Use valid model
                    chat_model = genai.GenerativeModel("gemini-2.5-pro")
                    response = chat_model.generate_content(final_prompt)
                    answer_raw = response.text.strip()

                    answer_display = translate_from_english(answer_raw, TARGET_LANG)

                    # C. Show Assistant Message
                    with st.chat_message("assistant"):
                        st.markdown(answer_display)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_display})

                    if enable_tts:
                        play_tts(answer_display, TARGET_LANG)
                else:
                    st.error("Gemini API not configured.")
            except Exception as e:
                st.error(f"Error generating chat response: {e}")


# ==============================================================================
# TAB 2: ASK PASHU AI (ANY BREED - CONTINUOUS CHAT)
# ==============================================================================
elif nav_choice == S["tab_ask"]:
    st.subheader(translate_text(S.get("ask_question", "Ask a Question about the Cattle"), TARGET_LANG))

    if dataset is not None:
        # 1. Select Breed
        breed_names = dataset["breed"].dropna().unique().tolist()
        breed_names.sort()
        selected_breed = st.selectbox(translate_text("Select Breed", TARGET_LANG), breed_names, key="manual_select")

        # Handle Context Switching (Clear history if breed changes)
        if st.session_state.manual_current_breed != selected_breed:
            st.session_state.manual_current_breed = selected_breed
            st.session_state.manual_chat_history = [] # Reset history for new breed

        # Calculate Context for this breed
        manual_breed_context = f"Breed: {selected_breed}\n"
        breed_row = dataset[dataset["breed"].astype(str).str.lower() == selected_breed.lower()]
        if not breed_row.empty:
            info_row = breed_row.iloc[0].to_dict()
            context_text = "\n".join([f"{k.replace('_',' ').title()}: {v}" for k, v in info_row.items() if str(v).strip() not in ["nan", "None", ""]])
            manual_breed_context += context_text
        else:
            manual_breed_context = f"No dataset info available for {selected_breed}."
        
        # 2. Display Chat History
        for message in st.session_state.manual_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Chat Input (Continuous)
        ph_text = translate_text(S.get("manual_chat_placeholder", "Ask anything about this breed..."), TARGET_LANG)
        if manual_input := st.chat_input(ph_text):
            
            # A. Display User Message
            st.chat_message("user").markdown(manual_input)
            st.session_state.manual_chat_history.append({"role": "user", "content": manual_input})

            # B. Generate Response
            try:
                if genai:
                    q_en = translate_to_english(manual_input)
                    
                    # Build Prompt with specific context
                    prompt = f"""
                    You are a veterinary AI expert. 
                    User is asking about the cattle breed: {selected_breed}.
                    
                    Official Dataset Info:
                    {manual_breed_context}

                    User Question: {q_en}
                    
                    Answer based on the dataset and general veterinary knowledge.
                    """

                    chat_model = genai.GenerativeModel("gemini-2.5-pro")
                    response = chat_model.generate_content(prompt)
                    answer_raw = response.text.strip()
                    
                    answer_display = translate_from_english(answer_raw, TARGET_LANG)

                    # C. Display Assistant Message
                    with st.chat_message("assistant"):
                        st.markdown(answer_display)
                    
                    st.session_state.manual_chat_history.append({"role": "assistant", "content": answer_display})

                    if enable_tts:
                        play_tts(answer_display, TARGET_LANG)
                else:
                    st.error("Gemini API not configured.")
            except Exception as e:
                st.error(f"Error generating response: {e}")