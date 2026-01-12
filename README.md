# rnn-sentiment-analysis
Production-ready sentiment analysis using SimpleRNN neural network.  Classifies IMDB movie reviews as positive or negative with real-time  predictions. Deployed on Streamlit Cloud.

CONTENT
# RNN Sentiment Analysis - IMDB Movie Reviews

A sentiment analysis web application using SimpleRNN neural network 
trained on IMDB movie reviews. Classifies text as positive or negative.

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup Steps

1. Clone repository
```bash
git clone https://github.com/yourusername/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis
2.Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

3.Install dependencies
4.pip install -r requirements.txt
5.Run application
streamlit run app.py

FOLDER STRUCTURE:

rnn-sentiment-analysis/
├── app.py
├── main.py
├── simple_rnn_imdb.h5
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── config.toml


Application opens at: http://localhost:8501

**Model Details**
Architecture: SimpleRNN with 128 units
Embedding: 128 dimensions
Input: 500-word sequences
Accuracy: 86% on IMDB test set
Output: Positive (0.5-1.0) or Negative (0-0.5)
How to Use
Enter a movie review in the text box
Click "Classify" button
View sentiment prediction and confidence score
Deployment
Deployed on Streamlit Cloud:
Live Demo

Technologies
Python
TensorFlow / Keras
Streamlit
NumPy
Scikit-learn
Model Performance

Training Accuracy: 86%
Dataset: 25,000+ IMDB reviews
Inference Time: 200-500ms



## **INSTALLATION STEPS (For Users)**

```bash
git clone https://github.com/yourusername/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis
python -m venv venv
venv\Scripts\activate
pip install -r [requirements.txt](http://_vscodecontentref_/1)
streamlit run [app.py](http://_vscodecontentref_/2)
