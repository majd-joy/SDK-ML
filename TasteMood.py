#laibraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import joblib
import streamlit 

import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report

# read the dataset

data = pd.read_csv(r"C:\Users\NTC\Downloads\archive (6).zip")


# apply the preprocessing 
def clean_review_date(date_str):
    try:
        date_str = date_str.split('•')[0].strip()  # Remove the special character and trim spaces
        if "Sept" in date_str:
            date_str = date_str.replace("Sept", "Sep")
        return pd.to_datetime(date_str, format='%b %Y', errors='coerce')
    except ValueError:
        return pd.NaT
    

data['Cleaned Review Date'] = data['Review Date'].apply(clean_review_date)




def preprocess_text(text):
  stpunc = string.punctuation 
  # text = text.translate(str.maketrans('', '', st))
  data_processed = text.lower()
  data_processed = word_tokenize(data_processed)
  stop_words = set(stopwords.words('english'))
  st = [] # You need to define st
  data_processed = [word for word in data_processed if word not in stop_words and word not in   stpunc ]
  lemmatizer = WordNetLemmatizer()
  data_processed = [lemmatizer.lemmatize(word) for word in data_processed]
  data_processed = ' '.join(data_processed)

  return data_processed


data["Review"] = data["Review"].apply(preprocess_text)


#Let's Label the data as 0 & 1 i.e. POsitive as 1 & Negative as 0
data.loc[:,'Sentiment'] = data.Sentiment.map({'Negative':0, 'Positive':1})
data['Sentiment'] = data['Sentiment'].astype(int)
data.head()


count = CountVectorizer()
text = count.fit_transform(data['Review'])
#Train & test split
x_train, x_test, y_train, y_test = train_test_split(text, data['Sentiment'], test_size=0.30, random_state=100)



# Multinomial Naive Bayes model


multinomial_nb_model = MultinomialNB()
multinomial_nb_model.fit(x_train, y_train)  # Train the model

prediction = multinomial_nb_model.predict(x_test)

print("Multinomial NB")
print("Accuracy score: {}". format(accuracy_score(y_test, prediction)) )
print(classification_report(y_test, prediction))

prediction = multinomial_nb_model.predict(x_train)

print("Multinomial NB")
print("Accuracy score: {}". format(accuracy_score(y_train, prediction)) )
print(classification_report(y_train, prediction))
negative_list = ["loathe",
"abhor",
"dislike",
"despise",
"detest",
"regret",
"disgust",
"displease",
"execrate" ,
"hatred",
"loathing",
"abhorrences",
"dislike",
"enmity",
"detestation",
"odium",
"hostility",
"abomination","hate"]
def pred_user_input(text):
  text_pros = preprocess_text(text)
  text = count.transform([text_pros])
  prediction = multinomial_nb_model.predict(text)
  for i in negative_list :
     if i in text_pros :
        prediction = 0
        return "Negative"
     else :
        prediction = 1 
        return "Positive"
  if prediction == 1:
    return "Positive"
  else:
    return "Negative"

     
  return text
#? save model and predictions in file and return results 
joblib.dump(multinomial_nb_model, 'multinomial_nb_model.pkl')

# pred_user_input("I absolutely loved this product. It was fantastic and I would recommend it to anyone.")

# pred_user_input("I disliked this product. It was terrible and I would not recommend it to anyone.")
joblib.load('multinomial_nb_model.pkl')



#? create the UI 



# if 'locale' not in streamlit.session_state:
#     streamlit.session_state.locale = 10  # Initialize with your desired value


streamlit.title("TasteMood")


user_input = streamlit.text_input("Enter your review:").lower()

result = pred_user_input(preprocess_text(user_input ))
streamlit.button("Submit" )
# result = pred_user_input( user_input )

streamlit.write("Sentiment: ", result)


