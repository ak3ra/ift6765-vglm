import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
import random
# from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


import pickle
# nltk.download('omw-1.4')

##---- Porter Stemmer---#
stemmer_ps = PorterStemmer()


def sentenceStemmerPS(sentence):
    # we need to tokenize the sentence or else stemming will return the entire sentence as is.
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer_ps.stem(word))
        # adding a space so that we can join all the words at the end to form the sentence again.
        stem_sentence.append(" ")
    return "".join(stem_sentence)

####---Lemmatizer----#
lemmatizer = WordNetLemmatizer()
def sentenceLemmatizer(sentence):
    token_words = word_tokenize(sentence)
# we need to tokenize the sentence or else lemmatizing will return the entire sentence as is.
    lemma_sentence = []
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)



### Parameteres ----_#
import argparse
parser = argparse.ArgumentParser(description='Build Image Lookup Table')

parser.add_argument('--stopwords_dir', default="dataset/stopwords_en.txt", help='path of the stopwords-en.txt')
# parser.add_argument('--image_dir', default="data/task1/image_splits/train.txt", help='path of the image_splits of training set of multi30k')
parser.add_argument('--cap2image_file', default="dataset/cap2image.pickle", help='output file for (topic) word to image id lookup table')
parser.add_argument('--captions_dir', default='dataset/captions_train2014.json', help='path to COCO annotation file')
parser.add_argument('--tfidf', default=8, type=int, help='tfidf topics')
parser.add_argument('--num_img', default=5, type=int, help='number of images')


args = parser.parse_args()

stopwords_dir = args.stopwords_dir
captions_dir = args.captions_dir
cap2image_file = args.cap2image_file
tfidf = args.tfidf
num_img = args.num_img

# captions_dir = 'dataset/captions_train2014.json'
caption_dataset = json.load(open(captions_dir,'r'))

captionList = []
imageidList = []
captionSentList= []
stop_words = ['.']
# stopwords_dir='./dataset/stopwords_en.txt'
if stopwords_dir:
    with open((stopwords_dir), "r") as data:
        for word in data:
            stop_words.append(word.strip())

for annotation in caption_dataset['annotations'][:10]:
    caption = annotation['caption'].lower()
    caption_lemmatized = sentenceLemmatizer(caption)
    # caption_without_sw = [word for word in word_tokenize(caption) if not word in stopwords.words('english')]
    caption_without_sw = [word for word in word_tokenize(caption_lemmatized) if not word in stop_words]

    captionList.append(caption_without_sw)
    captionSentList.append(' '.join(caption_without_sw))
    
    imageidList.append(annotation['image_id'])



n = tfidf
words, weight = None, None
if n > 0:
    print("tf-idf processing")
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(captionSentList))
    words = vectorizer.get_feature_names() 
    weight = tfidf.toarray()


cap2ids={}

for idx, cap in enumerate(captionList):
    imageID = imageidList[idx]
    if n > 0:
        w = weight[idx]
        loc = np.argsort(-w)
        top_words = []
        for i in range(n):
            if w[loc[i]] > 0.0:
                top_words.append(words[loc[i]])
        top_cap = []
        for word in cap:
            if word.lower() in top_words:
                top_cap.append(word)

    for word in top_cap:

            if word not in cap2ids:
                cap2ids[word] = [imageID]  # index 0 is used for placeholder
            else:
                if imageID  not in cap2ids[word]:
                    cap2ids[word].append(imageID)

pickle.dump(cap2ids,open(cap2image_file,"wb"))
print("data process finished!")
# print(len(cap2ids))
# print(total_img)