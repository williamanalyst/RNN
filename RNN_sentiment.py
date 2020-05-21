# In this model, it had been tested that the readers do not have a preference over the country that news are related to. 
#
# In[]:
#
#import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from sentiment_utils import *
#from keras.models import Model
#from keras.layers import Dense, Input, Dropout, LSTM, Activation
#from keras.layers import Dropout
from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#from keras.initializers import glorot_uniform
#from keras.utils import np_utils
from keras.callbacks import EarlyStopping
#from nltk.corpus import stopwords
np.random.seed(1)
from sklearn.model_selection import train_test_split
#
os.chdir('C:\Python_Project\Work_Automation\Data')
miq_viewlog = pd.read_excel('miq_viewlog.xlsx', sheet_name = '20190101')
miq_content = pd.read_excel('miq_contents.xlsx', sheet_name = 'contents')
#
miq_viewlog = miq_viewlog.dropna(subset = ['Page path level 2'])
miq_content = miq_content.dropna(subset = ['name'])
#
miq_viewlog['page_id'] = miq_viewlog['Page path level 2'].apply(lambda x: int(str(x)[1:]))
#
miq_viewlog_selected = miq_viewlog[['page_id', 'Pageviews', 'Unique Pageviews' ]]
miq_content_selected = miq_content[['id', 'name', 'country', 'view_restrict']]
#
miq_viewlog_combined = pd.merge(left = miq_viewlog_selected, right = miq_content_selected,
                                left_on = 'page_id', right_on = 'id',
                                how = 'left').iloc[:, 0:-2][['id', 'name', 'Pageviews']]
miq_viewlog_combined = miq_viewlog_combined.dropna(subset = ['name'])\
.loc[(miq_viewlog_combined['Pageviews'] > 5) & 
     (miq_viewlog_combined['Pageviews'] < 100)] # remove articles without names # remove articles with too little views
miq_viewlog_combined = miq_viewlog_combined.loc[(~miq_viewlog_combined['name'].str.contains('price report', case = False)) &
        (~miq_viewlog_combined['name'].str.contains('price index', case = False))&
        (~miq_viewlog_combined['name'].str.contains('origin index', case = False)) &
        (~miq_viewlog_combined['name'].str.contains('varietal index', case = False)) &
        (~miq_viewlog_combined['name'].str.contains('purchasing power', case = False)) ] # remove reports added by William
#
miq_viewlog_combined['Pageviews'].describe()
miq_viewlog_combined['Pageviews'].hist( bins = 500)
plt.xlim(0, 100)
#
miq_viewlog_combined['name'] = miq_viewlog_combined['name'].str.replace('\d+', '')
#
miq_viewlog_combined['Classification'] = miq_viewlog_combined['Pageviews'].apply(lambda x: 'Liked' if x > 35
                    else 'Disliked')
#
miq_viewlog_combined = miq_viewlog_combined.drop(miq_viewlog_combined.loc[miq_viewlog_combined['Classification'] ==
      'Disliked'].sample(n = int(len(miq_viewlog_combined.loc[miq_viewlog_combined['Classification'] =='Disliked'])
     - len(miq_viewlog_combined.loc[miq_viewlog_combined['Classification'] =='Liked']))).index)
#
group_count = miq_viewlog_combined.groupby('Classification').agg({'Pageviews': np.count_nonzero})
sns.barplot(x = group_count.index, y = group_count['Pageviews'])

miq_viewlog_combined['class_id'] = miq_viewlog_combined['Classification'].map({'Disliked': 0, 'Liked': 1})
#
train, test = train_test_split(miq_viewlog_combined, test_size=0.2, shuffle=True)
# random_state=42, 
#y = train['class_id']
##
#from sklearn.model_selection import train_test_split
#X_train , X_val , Y_train , Y_val = train_test_split(train['name'],y,test_size = 0.25)

#
###############################################################################
###############################################################################
#
#
# In[1]: N-Grams method 
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.tokenize import TweetTokenizer
#tokenizer = TweetTokenizer()
##
#vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)
#full_text = list(train['name'].values) + list(test['name'].values)
#vectorizer.fit(full_text)
#train_vectorized = vectorizer.transform(train['name'])
#test_vectorized = vectorizer.transform(test['name'])
##
#y = train['class_id']
##
#from sklearn.model_selection import train_test_split
#x_train , x_val, y_train , y_val = train_test_split(train_vectorized,y,test_size = 0.2)
##
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
##from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import VotingClassifier
#from sklearn.multiclass import OneVsRestClassifier
##
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
## Training Logistic Reagression model and an SVM
#lr = LogisticRegression()
#ovr = OneVsRestClassifier(lr)
#ovr.fit(x_train,y_train)
#print(classification_report( ovr.predict(x_val) , y_val))
#print(accuracy_score( ovr.predict(x_val) , y_val ))
##
#svm = LinearSVC()
#svm.fit(x_train,y_train)
#print(classification_report( svm.predict(x_val) , y_val))
#print(accuracy_score( svm.predict(x_val) , y_val ))
##
#estimators = [ ('svm',svm) , ('ovr' , ovr) ]
#clf = VotingClassifier(estimators , voting='hard')
#clf.fit(x_train,y_train)
#print(classification_report( clf.predict(x_val) , y_val))
#print(accuracy_score( clf.predict(x_val) , y_val ))
##
#
#
# In[]:
from keras.utils import to_categorical
target=train.class_id.values
y=to_categorical(target)
y # output/ target
#
max_features = 100
max_words = 120 # 
batch_size = 32
epochs = 100
num_classes=2
#
from sklearn.model_selection import train_test_split
X_train , X_val , Y_train , Y_val = train_test_split(train['name'],y,test_size = 0.25) # split the input text into train and test set
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
#from keras.layers import Dense,GRU,LSTM,Embedding
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
#from keras.layers import SpatialDropout1D,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
#from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
#from keras.callbacks import EarlyStopping # did not use ModelCheckpoint,   Callback, TensorBoard
#
tokenizer = Tokenizer(num_words=max_features) # max_features applied for the 
tokenizer.fit_on_texts(list(X_train))
#
#
X_train_original = X_train.copy()
dict_to_check = tokenizer.word_index # the library for each text
#
X_train = tokenizer.texts_to_sequences(X_train) # 
X_val = tokenizer.texts_to_sequences(X_val)
#
X_test = tokenizer.texts_to_sequences(test['name'])
X_test =pad_sequences(X_test, maxlen=max_words)
#
len(X_test)
#
X_train =pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test =pad_sequences(X_test, maxlen=max_words)
#
###############################################################################
###############################################################################
#
# In[2]: GRU Method
##
#model_GRU=Sequential()
#model_GRU.add(Embedding(90, 80,mask_zero=True))
#model_GRU.add(GRU(32,dropout=0.4,return_sequences=True))
#model_GRU.add(GRU(16,dropout=0.5,return_sequences=False))
#model_GRU.add(Dense(num_classes,activation='softmax'))
#model_GRU.compile(loss='categorical_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
#model_GRU.summary()
##
#history1=model_GRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)
##
#sub = pd.read_csv('D:/Collection_Dataset/movie-review-sentiment-analysis-kernels-only/check_file.csv' , sep = ',')
##
#y_pred1=model_GRU.predict_classes(X_test, verbose=1)
#sub.Sentiment=y_pred1 # 
#os.chdir('D:/Collection_Dataset/movie-review-sentiment-analysis-kernels-only')
#sub.to_csv('sub1_GRU.csv',index=False)
#sub.head()
##
#plt.plot(history1.history['accuracy'])
#plt.plot(history1.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
##
#model2_GRU=Sequential()
#model2_GRU.add(Embedding(max_features,100,mask_zero=True))
#model2_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
#model2_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
#model2_GRU.add(Dense(num_classes,activation='sigmoid'))
#model2_GRU.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.0005),metrics=['accuracy'])
#model2_GRU.summary()
##
#history2=model2_GRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1) # 
##
#y_pred2=model2_GRU.predict_classes(X_test, verbose=1)
#sub.Sentiment=y_pred2
##os.chdir('D:/Collection_Dataset/movie-review-sentiment-analysis-kernels-only')
#sub.to_csv('sub2_GRU.csv',index=False)
#sub.head()
#
# In[]: lSTM Method (Tested)
#
# 
#
model3_LSTM=Sequential()
model3_LSTM.add(Embedding(64, 32, mask_zero=True)) # top_words = 90, embedding_vector_length = 80
model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=True)) # hidden layer 1
model3_LSTM.add(LSTM(16,dropout=0.5,return_sequences=False)) # hidden layer 2
model3_LSTM.add(Dense(2,activation='sigmoid')) # output layer 
model3_LSTM.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.0005),metrics=['accuracy']) #
model3_LSTM.summary()
#
earlystop = EarlyStopping(monitor='val_loss', min_delta= 0, patience=3, verbose= 1, mode='auto')
#
history3=model3_LSTM.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                         epochs= 100, batch_size= 32, verbose= 1, callbacks=[earlystop])
# X_train has a max_feature
# 
# 
# 
y_pred3=model3_LSTM.predict_classes(X_test, verbose=1)
sub = pd.read_csv('D:/Collection_Dataset/movie-review-sentiment-analysis-kernels-only/check_file.csv' , sep = ',')
sub.Sentiment=y_pred3
#
sub.to_csv('sub3_LSTM.csv',index=False)
sub.head()
#
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# 
# In[]:
#
# Check inputs and weights:
#
for layer in model3_LSTM.layers:
    weights = layer.get_weights() # list of numpy arrays
    print(weights)
#
weight0 = model3_LSTM.layers[0].get_weights()[0] # 64 * 32 # top words = 64, embeded vector length = 32
weight1 = model3_LSTM.layers[1].get_weights() # 32 * 128 + 32 * 128 + 128
weight2 = model3_LSTM.layers[2].get_weights() # 32 * 64 + 16 * 64 + 64
weight3 = model3_LSTM.layers[3].get_weights() # 16 * 2 + 2 # 16 key features + 2 bias

#fit_test = tokenizer.fit_on_texts("The earth is an awesome place live")
#sequence_test = tokenizer.fit_on_texts("The earth is an awesome place live")
#
dict_to_check
#dict_to_check['New']
#dict_to_check['harvest']
#
def get_word_id(word):
    if word in dict_to_check:
        ref = dict_to_check[word]
    else:
        ref = 0
    return ref
    print(ref)
#
word = 'harvest'
get_word_id(word)
#
test1 = "New Zealand goes into lockdown, but harvest allowed to continue"
test2 = "UK drinks businesses urge Chancellor to save them by scrapping duty payments for six months"
#
test_input = "Polish wine consumption, by country of origin"
#
input_converted = [get_word_id(word) for word in test_input.split(' ')]
#
base = np.zeros(max_words - len(input_converted)).reshape(1, max_words - len(input_converted))
#
input_converted2 = np.array([get_word_id(word) for word in test_input.split(' ')]).reshape(1, len(input_converted))
input_concated = np.concatenate((base, input_converted2), axis = 1)
#
y_pred_val = model3_LSTM.predict_classes(input_concated, verbose=1)
print(y_pred_val)
