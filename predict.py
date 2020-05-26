import pickle
import numpy as np

from keras.models import load_model
from nltk.corpus import stopwords

from hanModel import AttentionLayer
from utils import cleanString, wordToSeq


MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
MAX_SENTENCE_NUM = 25  # maximum number of sentences in one document
MAX_WORD_NUM = 40  # maximum number of words in each sentence
EMBED_SIZE = 100  # vector size of word embedding
BATCH_SIZE = 50
NUM_EPOCHS = 2

dataset_name = 'imdb_reviews'

# Load model from saved hdf5 file
model = load_model('models/model_' + str(dataset_name) + '_' + str(NUM_EPOCHS) + '_epoch.h5', custom_objects={'AttentionLayer': AttentionLayer})

# Good Review
review = "As a basketball lover that grew up watching Michael as an idol this series called The last dance is just an amazing dream. I was only a teenager during the 90's with barely access to NBA basketball or footage. So, all the episodes are just amazing. Pretty intense, interesting and an amazing way to understand how Michael Jordan become the best NBA basketball Player of all time and at the same time understand how was his last dance with the Chicago Bulls."

# Bad Review
# review = "This series is really nicely shot, with some stunning landscapes. The cinematographer should be commended and for this I give it 1 extra star. I have read and watched many (many!) fantasy stories and while some have obvious inconsistencies and plot holes that are to a greater or lesser degree forgiveable, the whole premise of this series is so flawed as to make suspension of disbelief impossible. Nothing in this world of blind people makes any sense. How can it be that there are 'guide ropes' in the villiage (that are only used once) and to go into a cave (that are used every time they enter or leave) but they are otherwise able to stride around the countryside confidently, climb ladders and even fight without them? Why would they end up living up a mountain? Surely civilisation would gather in more temperate climates where food was easier to cultivate...talking of which, what do they eat? No meat ovbiously (but I would like to see them try to catch a rabbit!) but how do they even farm? How on earth can a person who is blind from birth, born to generations of blind ancestors, learn falconry? Think about it for a moment... It's so mixed-up, sometimes you think yeah, if this was the first generation of blind people that might be OK and other times it's well, maybe if enough time had passed they'd learn that. I could go on, but suffice it to say that there are so, so many problems that I just can't accept that such a civilisation could exist. Added to that there are also many performance and production mistakes (people hand objects to each other directly, pick things up, dress in matching colours etc) that jar you out of the world too."
stopWords = set(stopwords.words('english'))

with open('indices/word_index_' + dataset_name + '.txt', 'rb') as f:
    word_index = pickle.load(f)

review_cleaned = cleanString(review, stopWords)
input_array = wordToSeq(review_cleaned, word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)

res = model.predict(np.expand_dims(input_array, axis=0)).flatten()
cat = np.argmax(res.flatten()) + 1

print(review_cleaned)
print(res)
print(cat)