import pickle
with open('svm_pickle_model.pkl', 'rb') as f:
    svm_dict = pickle.load(f)

# Extract the SVM model and feature names from the dictionary
svm_model = svm_dict["model"]
feature_names = svm_dict["feature_names"]
# Predict the disorder type using the SVM model
sentiment_scores=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
predictions = svm_model.predict([sentiment_scores])
disorder = predictions[0]
print("disorder",disorder)