from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords

# __name__ is equal to app.py
app = Flask(__name__)

# load model from model.pck
model = joblib.load('SGD_model.pkl')



@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
	REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
	BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
	STOPWORDS = set(stopwords.words('english'))

	text =  [request.form['JobTitle']]
	df = pd.DataFrame({'T':text})

	def clean_text(text):
	    text = text.lower() # lowercase text
	    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
	    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
	    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
	    return text

	df['T'] = df['T'].apply(clean_text)

	industry = model.predict(df['T'])[0]
	
	return render_template("index.html", industry=industry)	







if __name__ == "__main__":
    app.run()
