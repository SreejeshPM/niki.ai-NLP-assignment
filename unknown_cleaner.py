# -*- coding: utf-8 -*-

import pandas as pd

names = ["Utterance","label"]
data = pd.read_csv("LabelledData.txt", sep=',', names= names)

def UnknownUtterances(data):
    return {row["Utterance"]:row["label"] for index, row in data.iterrows() if row["label"] == 'unknown'}

result = UnknownUtterances(data)

KeyWords = ["what","when","who"]

temp_dict =   {}
l1 = []

def UnknownUtterancesNotMatching(result):
    for k,v in result.iteritems():
        first_word = k.split(" ")[0]
        if first_word not in KeyWords:
            new_word = k.split(" ")[0]
            temp_dict.update({k:new_word})
        
#            return({k:new_word})    
        
    return temp_dict
#    return {k:v for k,v in result.items() if first_word not in KeyWords}


UnknownValues= UnknownUtterancesNotMatching(result)
        