import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#lets load the saved vectorizer and naive model
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))




#transform text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
nltk.download('punkt')
nltk.download('stopwords')
ps =PorterStemmer()

def transform_text(text):
      text= text.lower() #Converting to lowercase
      text= nltk.word_tokenize(text) #Tokenize Removing special characters and retaining alphanumeric words
      text=[word for word in text if word.isalnum()]
    #   Removing stopwords and punctuation 
      text= [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
      #Applying stemning
      text = [ps.stem(word) for word in text]

      return " ".join(text)
 
#saving streamlit code 

st.title("Email spam Classifier")
input_sms=st.text_area("Enter message")


if st.button('predict'):
    #preprocess
    tranformed_sms=transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([tranformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #display
    if result ==1:
        st.header("spam")
    else:
        st.header("Not Spam")