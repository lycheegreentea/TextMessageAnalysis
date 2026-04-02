# Text Message Analysis V1

## What is this?
This is a NLP tool that allows trains a word embedding (Word2Vec) model on your text messages and allows you to run experiments through a locally hosted desktop app. 

## Usage
1. Double click on the exe
2. A word2vec model will be trained on your text messages
3. A flask app will open:
  a. allows you to experiment with analogies, similarity, sentiments, and word paths 
  b. allows you to open a graph of the top 10,000 words in your vocabulary
## Privacy 
 **This is all run in localhost, none of your data leaves your computer** 


## Creation
### Tech Stack
* **Python** : Flask + word2vec training
  * gensim : training word2vec models
  * jieba : Chinese tokenization
  * PyInstaller : Packaging entire script
* **HTML/CSS/JS** : Frontend

### Features
* Includes Chinese character tokenization with jieba. 
* Utilizes  [Tensorflow's embedding projector](https://github.com/tensorflow/embedding-projector-standalone) for 3D graph creation.