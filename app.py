import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import zipfile
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Конфигурация
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
ZIP_PATH = 'model_glove_lstm.zip'  # ZIP-архив с моделью
TOKENIZER_PATH = 'tokenizer.pkl'   # Отдельный файл токенизатора

# Проверка и распаковка модели
def prepare_model():
    # Если папка модели уже существует - пропускаем распаковку
    if os.path.exists('model_glove_savedmodel'):
        return True
        
    if not os.path.exists(ZIP_PATH):
        st.error(f"ZIP-архив с моделью '{ZIP_PATH}' не найден!")
        return False
        
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
        return True
    except Exception as e:
        st.error(f"Ошибка распаковки модели: {e}")
        return False

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('model_glove_savedmodel')
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
    st.write("Classifies English news text into 4 categories")
    
    # Подготовка модели
    if not prepare_model():
        st.stop()
    
    # Загрузка компонентов
    model = load_model()
    tokenizer = load_tokenizer()
    
    if model is None or tokenizer is None:
        st.stop()
    
    # Ввод текста
    text = st.text_area("Enter news text (English only):", height=150,
                       placeholder="Example: Tesla announced new battery technology...")
    
    if st.button("Classify"):
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