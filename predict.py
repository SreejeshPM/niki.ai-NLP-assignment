# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 23:55:04 2018

@author: Lenovo
"""

import pickle
import pandas as pd

with open ('labelpredicter.pkl','rb')as f:
   vectorizer,clf2 = pickle.load(f)

new_data = pd.read_csv("final_test.csv",error_bad_lines=False)

test_data_utterance = new_data["Utterance"]

vectored =  vectorizer.transform(new_data.Utterance)
predicted_label = clf2.predict(vectored)

new_dict = {

"Utterance": test_data_utterance,     
"Predicted" : predicted_label
     
 }


new_dict_df = pd.DataFrame(new_dict)
new_dict_df.to_csv("results_new.csv", sep=',')