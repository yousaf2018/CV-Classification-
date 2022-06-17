#Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

#Importing dataset 
dataset = pd.read_csv("CvList.csv")
Cv_Count = len(dataset)
print("Total number of cv --> "+ str(Cv_Count))


#Pre-processing the each cv data for model training
Pre_Processed_Cv_Data = []
print("Pre-Processing is started wait :)")
for count in range(0, Cv_Count):
    #Replacing  all characters other than a-z and A-Z with space 
    each_cv = re.sub('[^a-zA-Z]', ' ', dataset['Resume'][count])
    each_cv = each_cv.lower()
    each_cv = each_cv.split()
    ps =  PorterStemmer()
    each_cv = [ps.stem(word) for word in each_cv if not word in set(stopwords.words('english'))]
    each_cv = ' '.join(each_cv)
    Pre_Processed_Cv_Data.append(each_cv)
    print(f"CV {count+1} is preprocessed")
saving_pre_processed_cv = pd.DataFrame(zip(Pre_Processed_Cv_Data, dataset.iloc[:, 0].values), columns = ['Resume', 'Labels'])
saving_pre_processed_cv.to_csv("Pre_Processed_Data.csv")

#Creating bag of words 
cv = CountVectorizer()
X = cv.fit_transform(Pre_Processed_Cv_Data).toarray()
Y = dataset.iloc[:, 0].values
print(len(X[0]))


#Splitting dataset into training and testing
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)


#Training naive base bayes classifier for cv classification 
classifier = GaussianNB()
classifier.fit(train_x, train_y)

#Testing trained model on unseen dataset
y_pred = classifier.predict(test_x)
for count in range(0, len(y_pred)):
    print("Actual -->" + test_y[count] + "  Prediction -->"+ y_pred[count])


#Model evaluations 
Confusion_Matrix = (test_y, y_pred)
accuracy = accuracy_score(test_y, y_pred)
print(accuracy)
