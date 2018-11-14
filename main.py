import pandas as pd
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import string 
from nltk import pos_tag
from nltk .stem import wordnet
from nltk.stem import WordNetLemmatizer
import random
import sys


review_file = sys.argv[1] 

desc_review = open(review_file,"r").read()
desc_output = open("output.txt","a")

data= []    #create tupples with categories and tokenized text
for cat in movie_reviews.categories():
    for review in movie_reviews.fileids(cat):
        data.append((movie_reviews.words(review),cat))


random.shuffle(data)   #because all neg were at the beginning 
data[:10]  


#let's get rid od stopsword 
lemmatizer = WordNetLemmatizer()    # nous donne la forme canonique d'un mot, exemple: playing -> play
stop =stopwords.words('english')   
punc =list(string.punctuation)
unwanted_words= stop+punc



def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN



def text_process(mess):
    data_propre=[]
    for w in mess:
        if w.lower() not in unwanted_words:
            pos = pos_tag([w])[0][1]
            propre= lemmatizer.lemmatize(w,pos=get_pos(pos))
            data_propre.append(propre.lower())
    return data_propre


data = [(text_process(review),category) for review, category in data]

training_set=data[:1800]
test_set =data[1800:]

all_words_in_data=[]
for review in training_set:
    all_words_in_data+=review[0]




all_words_in_data[:10]
frequence = nltk.FreqDist(all_words_in_data)
often= frequence.most_common(300)      #I am taking the 300 which appears the most in all the data 


features  = [i[0] for i in often]
features


def dic_of_features(words):          
    current_features = {}
    words_set=set(words)  # set of all words
    for w in features:
        #assign true/false.
        current_features[w] = w in words_set
        
    return current_features


training_data = [ (dic_of_features(doc),category)   for  doc,category in training_set]
testing_data = [ (dic_of_features(doc),category)   for  doc,category in test_set]

pos_sample = list(filter(lambda x: x[-1] == "pos", testing_data))[0][0]

from nltk import NaiveBayesClassifier
from sklearn.metrics import classification_report
classifier = NaiveBayesClassifier.train(training_data)
prediction_accuracy =nltk.classify.accuracy(classifier,testing_data)


testing_prediction_data = [i[0] for i in testing_data]   #je prend juste mes mots
testing_prediction_true = [True if i[-1] == "pos" else False for i in testing_data]   # et je prend mon pos ou neg 


test_predictions = [True if classifier.classify(i) == "pos" else False for i in testing_prediction_data]

print(classification_report(testing_prediction_true,test_predictions))



def input_word_to_classification(words):
	rep= classifier.classify(dic_of_features(text_process(word_tokenize(words))))
	
	if rep =='pos':

		desc_output.write("1\n")
	else: 
		desc_output.write("0\n")


	desc_output.close()


input_word_to_classification(desc_review)
print("Your answer is in  the file called output.txt")


