import pickle
import pandas as pd

with open ('labelpredicter.pkl','rb')as f:
   vectorizer,clf2 = pickle.load(f)

new_data = pd.read_csv("result.csv", error_bad_lines=False)
vectored =  vectorizer.transform(new_data.utterance)
predicted_label = clf2.predict(vectored)
new_data['Predicted'] = predicted_label

