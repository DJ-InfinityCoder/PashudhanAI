from deep_translator import GoogleTranslator

# Translation helpers
LANG_CODE_MAP = {"English":"en","Hindi":"hi","Marathi":"mr","Punjabi":"pa"}

def translate_to_english(text: str, enable_translator: bool, have_translator: bool) -> str:
    if not enable_translator or not have_translator:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def translate_from_english(text: str, target_lang: str, enable_translator: bool, have_translator: bool) -> str:
    if target_lang == "English" or not enable_translator or not have_translator:
        return text
    try:
        code = LANG_CODE_MAP.get(target_lang, "en")
        return GoogleTranslator(source="en", target=code).translate(text)
    except Exception:
        return text

def translate_text(text: str, target_lang: str, enable_translator: bool, have_translator: bool) -> str:
    if not enable_translator or not have_translator:
        return text
    try:
        code = LANG_CODE_MAP.get(target_lang, "en")
        return GoogleTranslator(source="auto", target=code).translate(text)
    except Exception:
        return text
