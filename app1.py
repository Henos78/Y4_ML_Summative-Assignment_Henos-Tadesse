
#Use this one to deploy the flask app
from flask import Flask, render_template, request, jsonify
from chatbot import load_dataset, tokenize_and_encode, generate_answer, model

app = Flask(__name__)

df = load_dataset('intents1.json')
tokenizer, X, lbl_enc, y = tokenize_and_encode(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    model_response = generate_answer(user_input, model, tokenizer, X, lbl_enc, df)
    return jsonify({'user_input': user_input, 'model_response': model_response})

if __name__ == '__main__':
    app.run(debug=True)
