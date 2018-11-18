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

# Model
### CBOW Model
![alt text] (https://github.com/Laiw1996/fake-real-news/blob/master/CBOW-architecture-predicts-the-current-word-based-on-the-context.png)

# Training
### Parameter for training with CNN-LSTM:
input_length = 400 <br/>
batch_size=64 <br/>
epochs=6 <br/>
dropout=0.2 <br/>

### Results
The average 10-fold cross validation score: 0.903





