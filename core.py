import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import classification_report
from sklearn import  svm
import string


header = ["utterance", "label"]
data = pd.read_csv('LabelledData_Modified.txt',sep=',',header=None, names=header)
train_data = data[:1300]
test_data = data[:1482]

def remove_non_ascii(textVal):
    return ''.join([i if ord(i) < 128 else ' ' for i in textVal])

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
train_vectors = vectorizer.fit_transform(train_data.utterance)
test_vectors = vectorizer.transform(test_data.utterance)

# from sklearn.ensemble import RandomForestClassifier
# classifier_randomforest = RandomForestClassifier(n_estimators=20)
# t0 = time.time()
# classifier_randomforest.fit(train_vectors,train_data.label)
# t1 = time.time()
# prediction_randomforest = classifier_randomforest.predict(test_vectors)
# t2 = time.time()
# time_random_forest_train = t1-t0
# time_random_forest_predict = t2-t1
#
# print("Results for RandomForestClassifier()")
# print("Training time: %fs; Prediction time: %fs" % (time_random_forest_train, time_random_forest_predict))
# print(classification_report(test_data.label, prediction_randomforest))


classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_data.label)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1


print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_data.label, prediction_linear))

# #==========================================
# # Model persisitance with pickle
# #==========================================
#
# import pickle
# with open ('labelpredicter.pkl','wb') as f:
#     pickle.dump((vectorizer,prediction_linear), f)

# new_data = pd.read_csv("result.csv", error_bad_lines=False)


#new_data = pd.read_csv("result.csv", error_bad_lines=False)
#
#test_data_utterance = new_data["Utterance"]
#test_data_label = new_data["Actual"]
#
#vectored =  vectorizer.transform(test_data_utterance)
#predicted_label = classifier_linear.predict(vectored)
#new_data['Predicted'] = predicted_label
#
# new_dict = {
#
#     "Predicted" : predicted_label,
#     "Utterance": test_data_utterance,
#     "Actual": test_data_label
#
# }
#
#
#new_dict_df = pd.DataFrame(new_dict)
#new_dict_df.to_csv("results.csv",sep='\t')