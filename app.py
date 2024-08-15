# app.py
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('sentiment_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('text_analysis_front_end.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['review']
        text_sequence = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_sequence, maxlen=200)
        prediction = model.predict(text_padded)
        sentiment = "Positive" if prediction[0][1] > 0.5 else "Negative"
        return render_template('text_analysis_front_end.html', prediction=sentiment, review=text)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

