
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframecsv = pd.read_csv('news.csv')
text = dataframecsv['text']
label = dataframecsv['label']
text_train, text_test, label_train, label_test = train_test_split(text, label, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_text_train = tfvect.fit_transform(text_train)
    tfid_text_test = tfvect.transform(text_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


print(fake_news_det(input("Enter news:")))

