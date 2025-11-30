# 🐄 Pashudhan AI

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Smart Breed & Body Trait Detection for Indigenous Cattle**

Pashudhan AI is a comprehensive solution designed to revolutionize livestock management in India. By combining **Computer Vision** and **Generative AI**, it accurately identifies indigenous cattle breeds and provides detailed, actionable insights about their traits, health, and productivity in both English and Hindi.

---

## 📋 Table of Contents
- [About](#-about)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Scope](#-future-scope)
- [References](#-references)

---

## 🧐 About

India is home to a vast diversity of indigenous cattle breeds, each with unique characteristics. However, accurate identification often requires expert knowledge, which isn't always accessible to farmers.

**Pashudhan AI** bridges this gap by:
1.  **Identifying** the breed from an image using a fine-tuned **EfficientNetB0** CNN.
2.  **Retrieving** verified breed traits from a curated dataset.
3.  **Generating** a farmer-friendly description using **Google's Gemini LLM**.

This tool empowers farmers, veterinarians, and researchers with instant, accurate information to improve breeding, healthcare, and productivity.

---

## 🚀 Key Features

*   **📸 Automated Breed Classification**: Identifies 39+ Indian cattle breeds with **~91% accuracy**.
*   **🧠 Generative Insights**: Uses Google Gemini to explain breed traits, origin, and utility in simple language.
*   **🗣️ Multilingual Support**: Provides output in **English** and **Hindi** for wider accessibility.
*   **📊 Structured Knowledge Base**: Backed by a custom "Hamara Dataset" of breed-specific physical and biological traits.
*   **💻 User-Friendly Interface**: Built with **Streamlit** for easy image upload, webcam capture, and interactive Q&A.
*   **💬 AI Chatbot**: Ask open-ended questions about cattle management and get AI-driven answers grounded in verified data.

---

## 🛠 Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Deep Learning**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/) (EfficientNetB0)
*   **Generative AI**: [Google Gemini API](https://ai.google.dev/)
*   **Computer Vision**: [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 📂 Dataset

The model is trained on a robust combination of data:
1.  **Image Dataset**: Based on the *Indian Bovine Breeds* dataset, augmented (rotation, zoom, brightness) to balance class distribution across 39 breeds.
2.  **Trait Dataset**: A custom-curated dataset containing detailed attributes for each breed:
    *   Origin & Region
    *   Physical Characteristics (Horn, Color, Size)
    *   Milk Yield & Productivity
    *   Behavioral Traits

---

## 📈 Model Performance

We fine-tuned an **EfficientNetB0** architecture, achieving state-of-the-art results:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **91.28%** |
| **Macro F1-Score** | **0.91** |
| **Validation Loss** | **0.84** |

*   **High Precision**: Breeds like *Kangayam*, *Dangi*, and *Guernsey* achieved near 100% classification accuracy.
*   **Robustness**: The model generalizes well, with validation accuracy consistently tracking training progress.

---

## 🏁 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/DJ-InfinityCoder/PashudhanAI.git
    cd PashudhanAI
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    ```

---

## 🎈 Usage

Run the Streamlit application:
```bash
streamlit run main.py
```
1.  **Upload** an image or use the **Camera**.
2.  View the **Predicted Breed** and confidence score.
3.  Read the **AI-Generated Report** in English or Hindi.
4.  Use the **Chat** feature to ask specific questions about the breed.

---

## 🔮 Future Scope

*   **Offline Inference**: Deploying quantized models on mobile devices for use without internet.
*   **Health Detection**: Extending the model to detect common skin diseases (e.g., Lumpy Skin Disease).
*   **Expanded Dataset**: Including more rare breeds and varied environmental conditions.

---

## 📚 References

*   *Jogi et al. (2024)*. "Cattle Breed Classification Techniques".
*   [Indian Bovine Breeds Dataset](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds)
*   [Google Gemini API Documentation](https://ai.google.dev/docs)

---

<p align="center">
  Made with ❤️ for Indian Farmers 🇮🇳
</p>
