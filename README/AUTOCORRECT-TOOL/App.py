# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from utils.autocorrect import Autocorrect

app = Flask(__name__)

# Initialize autocorrect system
autocorrect = None

@app.before_first_request
def initialize():
    global autocorrect
    autocorrect = Autocorrect('word_frequencies.csv')
    autocorrect.generate_chart()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text or not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        corrected_text = autocorrect.correct_sentence(text)
        return jsonify({'corrected_text': corrected_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
