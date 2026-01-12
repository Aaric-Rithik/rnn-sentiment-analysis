import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os

# Set page config
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-size: 1.2em;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-size: 1.2em;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #f39c12;
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class='main-header'>🎬 Movie Review Sentiment Analysis</div>
<div class='sub-header'>AI-Powered Sentiment Detection Using Deep Learning</div>
""", unsafe_allow_html=True)

st.divider()

# Load model and word index
@st.cache_resource
def load_resources():
    """Load model and word index"""
    try:
        model = load_model('simple_rnn_imdb.h5')
        word_index = imdb.get_word_index()
        return model, word_index
    except FileNotFoundError:
        return None, None

model, word_index = load_resources()

if model is None or word_index is None:
    st.error("""
    ❌ **Model file not found!**
    
    Please ensure `simple_rnn_imdb.h5` exists in the project directory.
    
    **Setup Instructions:**
    1. Train the model using SimpleRNN.ipynb
    2. Save as: `simple_rnn_imdb.h5`
    3. Place in project root folder
    """)
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# Reverse word index
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    """Decode review from integers to words"""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """Convert user text to model input format"""
    # Convert to lowercase
    words = text.lower().split()
    
    # Convert words to integers
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Pad to 500 words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return padded_review

# User input section
st.header("📝 Enter Your Movie Review")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "Write a movie review:",
        placeholder="Example: This movie was absolutely amazing! I loved every minute of it.",
        height=150,
        label_visibility="collapsed"
    )
with col2:
    st.write("")
    st.write("")
    word_count = len(user_input.split())
    st.metric("Words", word_count)

# Prediction section
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, key="analyze")

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True, key="clear")

with col3:
    example_btn = st.button("📋 Example", use_container_width=True, key="example")

# Handle button clicks
if analyze_btn:
    if user_input.strip():
        with st.spinner("🤔 Analyzing sentiment..."):
            # Preprocess
            preprocessed_input = preprocess_text(user_input)
            
            # Make prediction
            prediction = model.predict(preprocessed_input, verbose=0)
            sentiment_score = prediction[0][0]
            
            # Display results
            st.divider()
            st.success("✅ Analysis Complete!")
            
            # Sentiment result
            if sentiment_score > 0.52:
                sentiment = "😊 POSITIVE"
                color = "green"
                emoji = "👍"
            elif sentiment_score < 0.48:
                sentiment = "😢 NEGATIVE"
                color = "red"
                emoji = "👎"
            else:
                sentiment = "😐 NEUTRAL"
                color = "orange"
                emoji = "🤷"
            
            # Display prediction with better formatting
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {emoji} Sentiment")
                st.markdown(f"<div class='sentiment-{color.lower()}'>{sentiment}</div>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Confidence")
                st.markdown(f"### {sentiment_score:.1%}", help="Model's confidence in this prediction")
            
            # Progress bar
            st.progress(float(sentiment_score))
            
            # Display confidence breakdown
            st.subheader("📈 Detailed Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Positive Score", f"{sentiment_score:.2%}")
            with col2:
                st.metric("Uncertainty", f"{abs(sentiment_score - 0.5):.2%}")
            with col3:
                st.metric("Negative Score", f"{1-sentiment_score:.2%}")
            
            # Display review statistics
            st.subheader("📊 Review Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            words = user_input.split()
            sentences = user_input.split('.')
            
            with col1:
                st.metric("Words", len(words))
            with col2:
                st.metric("Sentences", len([s for s in sentences if s.strip()]))
            with col3:
                st.metric("Characters", len(user_input))
            with col4:
                avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
                st.metric("Avg Word Length", f"{avg_word_len:.1f}")
            
            # Show interpretation
            st.divider()
            st.subheader("💡 Interpretation")
            
            if sentiment_score > 0.70:
                interpretation = "🌟 **Excellent sentiment!** This review is very positive and expresses strong satisfaction."
            elif sentiment_score > 0.60:
                interpretation = "😊 **Good sentiment.** The reviewer clearly enjoyed the movie."
            elif sentiment_score > 0.52:
                interpretation = "👍 **Positive sentiment.** The reviewer liked the movie overall."
            elif sentiment_score > 0.48:
                interpretation = "😐 **Mixed sentiment.** The reviewer has both positive and negative points."
            elif sentiment_score > 0.30:
                interpretation = "👎 **Negative sentiment.** The reviewer had more complaints than praise."
            else:
                interpretation = "😞 **Very negative sentiment.** This review expresses strong dissatisfaction."
            
            st.info(interpretation)
            
    else:
        st.warning("⚠️ Please enter a review to analyze!")

elif clear_btn:
    st.rerun()

elif example_btn:
    st.session_state.example_shown = True

# Show example if requested
if "example_shown" in st.session_state and st.session_state.example_shown:
    st.divider()
    st.subheader("📋 Example Reviews")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("### 😊 Positive Example")
        positive_review = """
This movie was absolutely fantastic! The acting was superb, the plot was engaging and well-written, 
and the cinematography was stunning. Every scene was carefully crafted, and the director did an 
amazing job bringing the story to life. I was on the edge of my seat throughout the entire film. 
Highly recommended for anyone looking for quality entertainment!
        """
        if st.button("Try This", key="pos_example"):
            st.session_state.user_input = positive_review
            st.rerun()
        st.write(positive_review)
    
    with example_col2:
        st.markdown("### 😢 Negative Example")
        negative_review = """
Terrible movie. Complete waste of my time and money. The acting was awful, the plot made no sense, 
and the direction was poor. I couldn't wait for it to end. The dialogue was cringey, and the special 
effects were subpar. I have no idea how this film ever got made. Save yourself the trouble and 
watch something else instead.
        """
        if st.button("Try This", key="neg_example"):
            st.session_state.user_input = negative_review
            st.rerun()
        st.write(negative_review)

# Sidebar
st.sidebar.header("ℹ️ About This Model")
st.sidebar.info(
    """
    **Project:** RNN Sentiment Analysis
    
    **Model Architecture:** SimpleRNN
    
    **Dataset:** IMDB Reviews (25,000 training samples)
    
    **Accuracy:** ~88%
    """
)

st.sidebar.header("🏗️ Model Details")
st.sidebar.write(
    """
    **Architecture:**
    - Input Layer: 500 words max
    - Embedding Layer: 128 dimensions
    - SimpleRNN Layer: 128 units (ReLU)
    - Output Layer: 1 unit (Sigmoid)
    
    **Training:**
    - Optimizer: Adam
    - Loss: Binary Crossentropy
    - Batch Size: 32
    - Early Stopping: Enabled
    - Epochs: 10 (max)
    """
)

st.sidebar.header("📚 How It Works")
st.sidebar.markdown(
    """
    1. **Input**: Your review text
    2. **Preprocessing**: Convert to numbers & pad
    3. **Embedding**: Map words to 128-dim vectors
    4. **RNN Processing**: Read sequence, remember context
    5. **Output**: Probability (0-1)
    
    0.0 = Negative | 0.5 = Neutral | 1.0 = Positive
    """
)

st.sidebar.header("🔗 Links")
st.sidebar.markdown(
    """
    - [Project on GitHub](#)
    - [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
    - [TensorFlow Docs](https://tensorflow.org/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    """
)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
    <p>🤖 Made with ❤️ using TensorFlow, Keras, and Streamlit</p>
    <p><small>Sentiment Analysis Model | Deep Learning | NLP</small></p>
    <p><small>© 2026 | All Rights Reserved</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
