# niki.ai-NLP-assignment
This repo contains the assignment solution and code. For the NLP task by niki.ai

Data overview
    The train data is 1483 rows and 2 columns
    I called the header as  “Utterance” and “label”

    In the initail EDA i saw that the amount of “unknown” labels are very high. So i created a program “unknown_cleaner.py”
    which wll identify  only the unknown labeled data and labels them. So 261 data out of 271(unknown data) data got new labels. The newly formed labels are 

    new_labels= [define,give,in,name, on , the, where, which, whom, whose, why, how,]
    old_labels=[what,when, who, ,affirmation] 

    Then combined both and created new training data. “LabelledData_Modified.txt’

Program Overview

    The core functionality is defined in “Core.py”.

    Preprocessing
    Non_ascii remover
    Tokenizing
    Stemming
    Vectorizing
    Estimator Used
    SVC with kernal=’linear’

Model Prediction
    pickled  the model for persistance
    Created “predict.py” for prediction service




NOTE : 

Evaluvation Metric :



