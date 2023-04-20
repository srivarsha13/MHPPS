import pandas as pd
from pandas import DataFrame
import numpy as np
import re
import string
import time
import tweepy
from tweepy import OAuthHandler
from tweepy import API
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def init_tweepy_param() :
    consumer_key = 'MkdWGIa1eaEaKQxgf9BGEGZku'
    consumer_secret = 'ERR9AEn9EoOawTlv8M9Iy2A9wnZs0U8QYAHof7KlqMubKSah7t'
    access_token = '1615588996855500801-BIQqcoR7ui8mQE3JNfVOzLjStX8YjW'
    access_token_secret = '9rq9z1oYwMtkmnM9pnRs9yyF0wycVRL7jWhEZ0lkStwpI'
    return consumer_key, consumer_secret, access_token, access_token_secret

def init_regex():
    #regular expressions for tokenization
    regexes = [
        
        #punctuation
        r'(?:(\w+)\'s)',
        
        r'(?:\s(\w+)\.+\s)',
        r'(?:\s(\w+),+\s)',
        r'(?:\s(\w+)\?+\s)',
        r'(?:\s(\w+)!+\s)',
        
        r'(?:\'+(\w+)\'+)',
        r'(?:"+(\w+)"+)',
        r'(?:\[+(\w+)\]+)',
        r'(?:{+(\w+)}+)',
        r'(?:\(+(\w+))',
        r'(?:(\w+)\)+)',

        #words containing numbers & special characters & punctuation
        r'(?:(?:(?:[a-zA-Z])*(?:[0-9!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~])+(?:[a-zA-Z])*)+)',
        
        #pure words
        r'([a-zA-Z]+)',

    ]

    #compiling regular expression
    regex = re.compile(r'(?:'+'|'.join(regexes)+')', re.VERBOSE | re.IGNORECASE)
    return regex


def preprocess(documents):
    regex = init_regex()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    #fetching list of stopwords
    punctuation = list(string.punctuation)
    swords = stopwords.words('english') + ['amp'] + ['infp', 'infj', 'intp', 'intj', 'isfp', 'isfj', 'enfp', 'enfj', 'entp', 'entj', 'esfp', 'esfj', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january', 'feburary', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',  'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',  'jan', 'feb', 'mar', 'apr', 'may', 'jun' 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'tommorow', 'today', 'yesterday'] + ['mr', 'mrs']


    processed_documents = []
    for i,document in enumerate(documents):
        print('{0}/{1}'.format(i+1, len(documents)))
        
        #tokenization
        tokens = regex.findall(document)

        #skipping useless tokens
        t_regex = re.compile(r"[^a-zA-Z]")
        document = []
        
        for token in tokens:
            token = np.array(token)
            token = np.unique(token[token != ''])
            
            if len(token) > 0:
                token = token[0].lower()
            else:
                continue
                
            if re.search(t_regex, token) == None and token not in swords:
                token = lemmatizer.lemmatize(token)
                document.append(token)
                
        document = ' '.join(document)

        #skipping
        if len(document) >= 0:
            processed_documents.append(document)

    print()
    return np.array(processed_documents)

def get_extro_pred(username):
    num_tweets=50
    consumer_key, consumer_secret, access_token, access_token_secret = init_tweepy_param()
    # Authorize our Twitter credentials
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    tweets = api.user_timeline(screen_name=username, 
                            # 200 is the maximum allowed count
                            count=50,
                            include_rts = False,
                            # Necessary to keep full_text 
                            # otherwise only the first 140 words are extracted
                            tweet_mode = 'extended'
                            )
    
    #Transform the tweepy tweets into a 2D array that will populate the CSV file
    outtweets = [[tweet.id_str, 
                tweet.created_at, 
                tweet.favorite_count, 
                tweet.retweet_count, 
                tweet.full_text.encode("utf-8").decode("utf-8")] 
                for idx,tweet in enumerate(tweets)]
    df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])
    df.to_csv('%s_tweets.csv' % username, index=False)
    
    rnd_ieModel = joblib.load('C:\\FinalYearProject\\TweetAnalysis\\rf_model.sav')
    filename = username + "_tweets.csv"
    df = pd.read_csv("C:\FinalYearProject\TweetAnalysis\\" + filename)
    
    #Preprocessing tweets
    posts = df['text']
    print()
    print("Preprocessing tweets...")
    print()
    posts = preprocess(posts)
    cv = CountVectorizer(analyzer="word", max_features=150).fit(posts)
    x = cv.transform(posts)
    tf = TfidfTransformer()
    x_tf =  tf.fit_transform(x).toarray()

    #Predict extroversion
    predicted_ie = rnd_ieModel.predict(x_tf)
    rnd_ieModel.predict(x_tf)
    IEpred=[]
    for pred in predicted_ie:
        if pred==0:
            IEpred+=['I']
        elif pred==1:
            IEpred+=['E']
    
    extro_pred = max(set(IEpred), key = IEpred.count)
    pred_acc = IEpred.count(extro_pred)/len(IEpred)
    if(extro_pred=='I'):
        extro_pred="Introvert"
    else:
        extro_pred="Extrovert"

    return extro_pred, pred_acc

""" username = input("Enter username : ")
pred, acc = get_extro_pred(username)
print("Extroversion Prediction : " + pred)
print("Prediction Accuracy : %.2f" % round(acc*100, 2) + " %")
print() """