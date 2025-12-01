import re
import tempfile
import streamlit as st
from gtts import gTTS
from utils.translation_utils import LANG_CODE_MAP

# TTS cleaning & playback
def clean_for_tts(text: str) -> str:
    if not text:
        return text
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove markdown/formatting symbols but keep punctuation and letters (including unicode)
    text = re.sub(r"[*_`~#>\-•→←↑↓✔✓➤★☆▶►]", " ", text)
    # Remove specific patterns like :emoji_name:
    text = re.sub(r":[a-zA-Z_]+:", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_audio_bytes(text: str, lang_key: str) -> bytes:
    try:
        clean_text = clean_for_tts(text)
        lang_code = LANG_CODE_MAP.get(lang_key, "en")
        
        # Try target language
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                gTTS(clean_text, lang=lang_code).save(tmp.name)
                with open(tmp.name, "rb") as f:
                    return f.read()
        except Exception:
            # Fallback to English
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                gTTS(clean_text, lang="en").save(tmp.name)
                with open(tmp.name, "rb") as f:
                    return f.read()
    except Exception as e:
        return None

def play_tts(text: str, lang_key: str, have_gtts: bool, error_msg: str):
    if not have_gtts:
        return
    
    audio_bytes = generate_audio_bytes(text, lang_key)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.error(f"{error_msg}")
