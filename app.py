from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # allow CORS for all routes
with open('svm_pickle_model.pkl', 'rb') as f:
    svm_dict = pickle.load(f)

svm_model = svm_dict["model"]
feature_names = svm_dict["feature_names"]
# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()




@app.route('/api/submit', methods=['POST'])
def submit_data():

    input_array= request.get_json()
    input_array = np.array(input_array)
    print("array is",input_array)
    #print("input array",input_array)

    #Perform sentiment analysis and store the results in a list
    #sentiment_scores = [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    sentiment_scores=[]
    for text in input_array:
        score = sid.polarity_scores(str(text))
        if score['compound'] > 0:
            sentiment_scores.append(1)
        elif score['compound'] == 0:
            sentiment_scores.append(0)
        else:
            sentiment_scores.append(-1)
    sentiment_scores=np.array(sentiment_scores)
    reshaped_array = sentiment_scores.reshape(1, -1)
    
    predictions = svm_model.predict(reshaped_array)
    disorder = predictions[0]

    # # data = request.get_json()
    # # # process the data here
    # data = {'name': 'John', 'age': 30, 'city': 'New York'}
    # eturn the processed data as JSON
    #print("disorder",disorder)
    return jsonify({'You are suffering from': disorder})
if __name__ == '__main__':
    app.run(debug=True)
