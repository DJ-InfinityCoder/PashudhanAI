import streamlit as st
try:
    from google import genai
except Exception:
    genai = None

def configure_gemini(api_key):
    if api_key:
        try:
            # Instantiate client to test configuration correctness
            client = genai.Client(api_key=api_key)
            return True
        except Exception as e:
            st.sidebar.error(f"Error configuring API key: {e}")
            return False
    return False

def generate_breed_info(api_key, breed_label, context_str):
    if not genai or not api_key:
        return None
    
    try:
        prompt = f"""
        You are a veterinary expert.
        The user has identified a cattle breed: {breed_label}.
        
        Here is the official dataset information:
        {context_str}
        
        Please generate a formal, structured description of this breed based on the provided context.
        The description should be comprehensive yet concise, suitable for an educational display.
        Structure it with clear sections or paragraphs.
        Do not use markdown tables.
        """
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating breed info: {e}")
        return None

def generate_breed_summary(api_key, breed_label, context_str):
    if not genai or not api_key:
        return None
    
    try:
        summary_prompt = f"""
        You are a veterinary expert.
        The user has identified a cattle breed: {breed_label}.
        
        Here is the official dataset information:
        {context_str}
        
        Please generate a structured, speakable summary of this breed.
        Include:
        - Key Characteristics
        - Origin
        - Utility (Milk/Draught)
        - Interesting Fact
        
        Keep it concise and suitable for audio playback.
        """
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=summary_prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def generate_chat_response(api_key, breed_name, context_str, user_question):
    if not genai or not api_key:
        return None
        
    try:
        final_prompt = f"""
        You are a veterinary expert. 
        The user has uploaded an image of a cattle breed identified as: {breed_name}.
        
        Here is the official dataset information about this breed:
        {context_str}

        User Question: {user_question}

        Answer the question based on the dataset info and your general knowledge. 
        Keep the answer helpful and concise.
        """

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=final_prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating chat response: {e}")
        return None

def generate_manual_chat_response(api_key, selected_breed, manual_breed_context, user_question):
    if not genai or not api_key:
        return None
        
    try:
        prompt = f"""
        You are a veterinary AI expert. 
        User is asking about the cattle breed: {selected_breed}.
        
        Official Dataset Info:
        {manual_breed_context}

        User Question: {user_question}
        
        Answer based on the dataset and general veterinary knowledge.
        """

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None
