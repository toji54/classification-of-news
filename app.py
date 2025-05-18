import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Конфигурация
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
MODEL_PATH = 'model_glove_lstm.h5'  # Файл модели в формате H5
TOKENIZER_PATH = 'tokenizer.pkl'    # Файл токенизатора

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

# Загрузка токенизатора
@st.cache_resource
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки токенизатора: {e}")
        return None

# Интерфейс приложения
def main():
    st.title("📰 News Classifier (GloVe+LSTM)")
    st.write("Классифицирует новостной текст на английском языке по 4 категориям: World, Sports, Business, Sci/Tech")
    
    # Загрузка компонентов
    model = load_model()
    tokenizer = load_tokenizer()
    
    if model is None or tokenizer is None:
        st.stop()
    
    # Ввод текста
    text = st.text_area("Введите текст новости (только на английском языке):", height=150,
                       placeholder="Example: Tesla announced new battery technology...")
    
    if st.button("Классифицировать"):
        if not text.strip():
            st.warning("Please enter some text")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Токенизация и предсказание
                    seq = tokenizer.texts_to_sequences([text])
                    padded = pad_sequences(seq, maxlen=100)
                    pred = model.predict(padded, verbose=0)
                    
                    # Отображение результатов
                    category = CLASS_NAMES[np.argmax(pred)]
                    confidence = np.max(pred)
                    
                    st.success(f"Category: {category}")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Детализация
                    with st.expander("Details"):
                        for name, prob in zip(CLASS_NAMES, pred[0]):
                            st.write(f"{name}: {prob:.4f}")
                            
                except Exception as e:
                    st.error(f"Classification error: {e}")

if __name__ == "__main__":
    main()