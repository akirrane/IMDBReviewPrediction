from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import predict_response
import re

# Creating the flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Give cross-origin resource sharing capabilities

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Post functionality for the back and forth conversation with the chatbot
@app.post("/answer")
def answer():
    text = request.get_json().get("message")
    # Check user input for a valid alpha request
    if re.search(r'\d', text):
        botAnswer = 'A valid review does not contain numbers. Please try again.'
    else:
        botAnswer = predict_response(text)
    if botAnswer == 'pos':
        botAnswer = 'The review is positive.'
    if botAnswer == 'neg':
        botAnswer = 'The review is negative.'
    ansDict = {"answer": botAnswer}
    return jsonify(ansDict)  # Return the answer in JSON

if __name__ == "__main__":
    app.run(debug=True)
