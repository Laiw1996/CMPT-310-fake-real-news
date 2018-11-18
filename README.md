# Neural network learning documentation

### Run the program with:

jupyter notebook


### Requirements:
sklearn <br/>
nltk <br/>
keras <br/>
matplotlib <br/>


### Dataset
1. The dataset contains 1500 real and 1500 fake cases. Real and fake articles are stored in two separate directories called “real” and “fake”. 
2. The type of the data files is all plain text. 


# Major steps

### Pre-processing
1. Data cleaning.
2. Remove numbers
3. Remove punctuations
4. Lemmatization
5. Term extraction (remove less important terms)
6. Vectorization. First build a vocabulary set based on the selected terms, in which each term has an unique index. Then convert each review into a 1-D vector. Each vector element corresponds to its index in the vocabulary set. Note: for those not in the vocabulary, we could use an unknown sign to represent them.
7. After preprocessing (with stop-word removal lemmatization and stemming), it turns out there are 1304658 words in the whole dataset. The maximum new length is 10290, and the shortest is 2. The average new length is 434.9. We will build a vocabulary set on the 1304658 words and only keep the most frequent ones (need a hyper-parameter here), less frequent ones are represented with a UNKNOWN token.


# Nerual network models

### 1. Word2Vec CBOW
#### Layer (type)                 Output Shape              Param #  
=================================================================
input_92 (InputLayer)        (None, 400)               0         
_________________________________________________________________
embedding_92 (Embedding)     (None, 400, 50)           1000000   
_________________________________________________________________
global_average_pooling1d_1 ( (None, 50)                0         
_________________________________________________________________
dense_92 (Dense)             (None, 2)                 102       
=================================================================
Total params: 1,000,102
Trainable params: 1,000,102
Non-trainable params: 0


### 2. CNN - LSTM
#### Layer (type)                 Output Shape              Param # 
=================================================================
input_93 (InputLayer)        (None, 400)               0         
_________________________________________________________________
embedding_93 (Embedding)     (None, 400, 50)           1000000   
_________________________________________________________________
conv1d_173 (Conv1D)          (None, 396, 256)          64256     
_________________________________________________________________
max_pooling1d_173 (MaxPoolin (None, 79, 256)           0         
_________________________________________________________________
dropout_183 (Dropout)        (None, 79, 256)           0         
_________________________________________________________________
conv1d_174 (Conv1D)          (None, 75, 64)            81984     
_________________________________________________________________
max_pooling1d_174 (MaxPoolin (None, 15, 64)            0         
_________________________________________________________________
dropout_184 (Dropout)        (None, 15, 64)            0         
_________________________________________________________________
lstm_92 (LSTM)               (None, 16)                5184      
_________________________________________________________________
dense_93 (Dense)             (None, 2)                 34        
=================================================================
Total params: 1,151,458
Trainable params: 1,151,458
Non-trainable params: 0


# Training
### The whole training process will take about 15 minutes.

### Parameter for training with CNN-LSTM:
input_length = 400
batch_size=64
epochs=6
dropout=0.2

### Results
The average 10-fold cross validation score: 0.903





