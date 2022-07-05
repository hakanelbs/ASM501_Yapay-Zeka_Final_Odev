import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns 
import string
import zeyrek
import gensim

from emot.emo_unicode import UNICODE_EMOJI # For emojis
import emot
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import xgboost

from wordcloud import WordCloud

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import glob
sns.set_style("darkgrid")
from collections import defaultdict



# Veri Seti Metin Manipülasyon Metodu 
def data_man(df_tr):
    #lower
    df_tr['Reviews Content'] = df_tr['Reviews Content'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    #################################################################
    #drop number
    df_tr['Reviews Content'] = df_tr['Reviews Content'].str.replace('\d','')
    #################################################################
    #drop url
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    df_tr['Reviews Content']=df_tr['Reviews Content'].apply(lambda x : remove_URL(x))
    #################################################################
    #drop punc
    df_tr['Reviews Content'] = df_tr['Reviews Content'].str.replace(r'[^\w\s]+', '')
    #################################################################
    #convert_emojis
    def convert_emojis(text):
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, " "+"_".join(UNICODE_EMOJI[emot].replace(","," ").replace(":","").split())+" ")
        return text
    # Example
    text1 = "Hilarious 😂. The feeling of making a sale 😎, The feeling of actually fulfilling orders 😒"
    convert_emojis(text1) 
    for i in range(len(df_tr)):
        df_tr["Reviews Content"].loc[i] = convert_emojis(df_tr["Reviews Content"].loc[i])
        
    #################################################################
    #lemmetize 
    analyzer = zeyrek.MorphAnalyzer()
    df_tr['Reviews Content'] = df_tr['Reviews Content'].apply(lambda x: " ".join([analyzer.lemmatize(word)[0][1][0] for word in x.split()])) 
    #################################################################
    #drop stopwords
    sw = pd.read_csv("turkce_stopwords.csv" , header = None)
    df_tr['Reviews Content'] = df_tr['Reviews Content'].apply(lambda x: " ".join(x for x in x.split() if x not in list(sw[0])))
    #################################################################
    df_tr['Reviews Content'] = df_tr['Reviews Content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    #################################################################
                   
    #################################################################
    # drop 2 kelimelime
    df_tr['Reviews Content'] = df_tr['Reviews Content'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    #################################################################
    # drop NaN
    #df_tr = df_tr.dropna(subset=['Reviews Content' ,"Ana Başlık"])
    df_tr = df_tr[df_tr['Reviews Content'] != '']
    #################################################################
    return df_tr


# Word2Vec Yöntemi İle Vektör Uzayına Çevrilmiş Verileri Numpy.Array Formatına Çevirme Metodu
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec




#Hazırlanan Eğitim ve Valitasyon Veri Seti
hu_data = pd.read_excel(r"C:\Users\Hakan\Desktop\youtube_scrapper\nlp\uygulama\hu_mixed_all.xlsx")


# Eğitim ve Valitasyon Veri Setlerinin Tanımlanması
df_train = hu_data[hu_data["type"] == "train"]
df_test = hu_data[hu_data["type"] == "test"]
df_test = df_test[df_test["Ana Başlık"] != 2]
df_test_1 = df_test[df_test["Ana Başlık"] == 1].loc[64991:65139]
df_test_0 = df_test[df_test["Ana Başlık"] == 0]
df_test = pd.concat([df_test_1 , df_test_0])


# Word2Vec Kelime Gömme Modeline Çevirme İşlemleri
tokenized_tweet = df_train['Reviews Content'].apply(lambda x: x.split()) # tokenizing 
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size = 100,# desired no. of features/independent variables
            window=5, # context window size
            min_count=2, # Ignores all words with total frequency lower than 2.                                  
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 32, # no.of cores
            seed = 34
) 

tokenized_tweet_train = df_train['Reviews Content'].apply(lambda x: x.split()) # tokenizing 
tokenized_tweet_test  = df_test['Reviews Content'].apply(lambda x: x.split()) # tokenizing 

tokenized_tweet_train = tokenized_tweet_train.reset_index()
tokenized_tweet_train = tokenized_tweet_train[["Reviews Content"]]
wordvec_arrays = np.zeros((len(tokenized_tweet_train), 100)) 
for i in range(len(tokenized_tweet_train)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet_train["Reviews Content"].loc[i], 100)
train_wordvec_df = pd.DataFrame(wordvec_arrays)

tokenized_tweet_test = tokenized_tweet_test.reset_index()
tokenized_tweet_test = tokenized_tweet_test[["Reviews Content"]]
wordvec_arrays = np.zeros((len(tokenized_tweet_test), 100)) 
for i in range(len(tokenized_tweet_test)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet_test["Reviews Content"].loc[i], 100)
test_wordvec_df = pd.DataFrame(wordvec_arrays)


#Valitasyon Verilerinin Eğitimi ve Test Sonuçları
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(train_wordvec_df,df_train["Ana Başlık"])
y_pred = xgb_model.predict(test_wordvec_df)
accuracy = cross_val_score(xgb_model, 
                                   test_wordvec_df, 
                                   df_test["Ana Başlık"].astype(int).values, 
                                   cv = 10).mean()



print("Accuracy :", accuracy)
print("Doğruluk Matrisi : \n", confusion_matrix(df_test["Ana Başlık"].astype(int).values, y_pred))
print("F1_score :", f1_score(df_test["Ana Başlık"].astype(int).values, y_pred,average='micro'))


#Youtube Yorumlarını Eğitme ve Doğruluk Oranı Hesaplama 
path = r"C:\Users\Hakan\Desktop\youtube_scrapper\nurgül_türk_full_yorum"
filenames = glob.glob(path + "\*.csv")
filenames.sort(key=os.path.getmtime)

positive_rate_list = []
for file in filenames:
   temp_df = (pd.read_csv(file))
   temp_df.drop_duplicates(subset=['author'] , inplace= True)
   temp_df.dropna(inplace=True) 
   temp_df = temp_df[["text"]]
   temp_df = temp_df.loc[1:]
   temp_df.reset_index(inplace=True , drop=True)
   temp_df = temp_df.rename(columns={'text': 'Reviews Content'})
   temp_df = data_man(temp_df)
   temp_df.reset_index(inplace= True , drop=True)
   
   tokenized_tweet_test  = temp_df['Reviews Content'].apply(lambda x: x.split()) # tokenizing 
   tokenized_tweet_test = tokenized_tweet_test.reset_index()
   tokenized_tweet_test = tokenized_tweet_test[["Reviews Content"]]
   wordvec_arrays = np.zeros((len(tokenized_tweet_test), 100)) 
   for i in range(len(tokenized_tweet_test)):
      wordvec_arrays[i,:] = word_vector(tokenized_tweet_test["Reviews Content"].loc[i], 100)
   test_wordvec_df = pd.DataFrame(wordvec_arrays)
   y_pred_temp = xgb_model.predict(test_wordvec_df)
   temp_pred_df = pd.DataFrame(y_pred_temp , columns = ["pred"])
   positive_rate_list.append([file , len(temp_pred_df[temp_pred_df["pred"] == 1]) / len(temp_pred_df)])
   
   
df = pd.DataFrame (positive_rate_list, columns = ['name', 'rate'])
df.to_excel("yagmur_positive_rate.xlsx" , index = False)