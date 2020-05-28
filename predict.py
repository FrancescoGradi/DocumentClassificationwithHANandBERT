import pickle
import numpy as np

from tensorflow.keras.models import load_model, Model
from nltk.corpus import stopwords
from nltk import tokenize

from hanModel import AttentionLayer, HanModel
from utils import cleanString, wordToSeq

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


MAX_FEATURES = 200000  # maximum number of unique words that should be included in the tokenized word index
MAX_SENTENCE_NUM = 15  # maximum number of sentences in one document
MAX_WORD_NUM = 25  # maximum number of words in each sentence
EMBED_SIZE = 100  # vector size of word embedding
BATCH_SIZE = 50
NUM_EPOCHS = 10
INIT_LR = 1e-2

dataset_name = 'yelp_2014'

# Load model from saved hdf5 file
model = load_model('models/model_yelp_2014/20200527-220312.h5', custom_objects={'AttentionLayer': AttentionLayer})

# Good Review
#review = "As a basketball lover that grew up watching Michael as an idol this series called The last dance is just an amazing dream. I was only a teenager during the 90's with barely access to NBA basketball or footage. So, all the episodes are just amazing. Pretty intense, interesting and an amazing way to understand how Michael Jordan become the best NBA basketball Player of all time and at the same time understand how was his last dance with the Chicago Bulls."

# Bad Review
#review = "This series is really nicely shot, with some stunning landscapes. The cinematographer should be commended and for this I give it 1 extra star. I have read and watched many (many!) fantasy stories and while some have obvious inconsistencies and plot holes that are to a greater or lesser degree forgiveable, the whole premise of this series is so flawed as to make suspension of disbelief impossible. Nothing in this world of blind people makes any sense. How can it be that there are 'guide ropes' in the villiage (that are only used once) and to go into a cave (that are used every time they enter or leave) but they are otherwise able to stride around the countryside confidently, climb ladders and even fight without them? Why would they end up living up a mountain? Surely civilisation would gather in more temperate climates where food was easier to cultivate...talking of which, what do they eat? No meat ovbiously (but I would like to see them try to catch a rabbit!) but how do they even farm? How on earth can a person who is blind from birth, born to generations of blind ancestors, learn falconry? Think about it for a moment... It's so mixed-up, sometimes you think yeah, if this was the first generation of blind people that might be OK and other times it's well, maybe if enough time had passed they'd learn that. I could go on, but suffice it to say that there are so, so many problems that I just can't accept that such a civilisation could exist. Added to that there are also many performance and production mistakes (people hand objects to each other directly, pick things up, dress in matching colours etc) that jar you out of the world too."

# 4 stars
#review = "The ingredients are always fresh and I like that I can customize everything. The price for the pita\/salad is reasonable, but after the delivery fee of $3, it can be pricey. Definitely a great option if I forget my lunch at home."

# 1 Star
review = "The tables and floor were dirty. I was the only customer on a Saturday nite and  the person working the counter ignored me I had a corned beef sandwich. I took three bites  and threw it in the trash"

stopWords = set(stopwords.words('english'))

with open('indices/word_index_' + dataset_name + '.txt', 'rb') as f:
    word_index = pickle.load(f)

review_cleaned = cleanString(review, stopWords)
input_array = wordToSeq(review_cleaned, word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)

with open('datasets/' + dataset_name + '_cleaned.txt', 'rb') as f:
    data_cleaned = pickle.load(f)


sent_att_weights = Model(model.inputs, model.get_layer('sent_attention').output, name='SentenceAttention')
word_encoder = model.get_layer('sent_linking').layer

# word_encoder = Model(model.get_layer('word_input').inputs, model.get_layer('word_attention').output)
print(sent_att_weights.summary())

output_array = sent_att_weights.predict(np.resize(input_array, (1, MAX_SENTENCE_NUM, MAX_WORD_NUM)))[1]
print(output_array)
print(output_array.flatten().argsort())

# Get n sentences with most attention in document
n_sentences = 2
sent_index = output_array.flatten().argsort()[-n_sentences:]
print(sent_index)
sent_index = np.sort(sent_index)
print(sent_index)
sent_index = sent_index.tolist()

print(sent_index)
print(' ')

# Create summary using n sentences
sent_list = tokenize.sent_tokenize(review)
print(sent_list)

summary = [sent_list[i] for i in sent_index]


def wordAttentionWeights(sequenceSentence, weights):
    """
    The same function as the AttentionLayer class.
    """
    uit = np.dot(sequenceSentence, weights[0]) + weights[1]
    uit = np.tanh(uit)

    ait = np.dot(uit, weights[2])
    ait = np.squeeze(ait)
    ait = np.exp(ait)
    ait /= np.sum(ait)

    return ait

print(' '.join(summary))

# Summary as input for word attention
summary_cleaned = cleanString(' '.join(summary), stopWords)
word_input_array = wordToSeq(summary_cleaned, word_index, MAX_SENTENCE_NUM, MAX_WORD_NUM, MAX_FEATURES)

# Create model from word input to output of dense layer right before the attention layer
hidden_word_encoding_out = Model(inputs=word_encoder.input, outputs=word_encoder.get_layer('word_dense').output)
# Load weights from trained attention layer
word_context = word_encoder.get_layer('word_attention').get_weights()
# Compute output of dense layer
hidden_word_encodings = hidden_word_encoding_out.predict(word_input_array)
# Compute context vector using output of dense layer
ait = wordAttentionWeights(hidden_word_encodings, word_context)

# Get n words with most attention in document
n_words = 5

flattenlist = []
words_unpadded = []
for idx, sent in enumerate(tokenize.sent_tokenize(summary_cleaned)):
    if (idx >= MAX_SENTENCE_NUM):
        break
    attword_list = tokenize.word_tokenize(sent.rstrip('.'))
    ait_short = (1000 * ait[idx][:len(attword_list)]).tolist()
    words_unpadded.extend(ait_short)
    flattenlist.extend(attword_list)

words_unpadded = np.array(words_unpadded)
sorted_wordlist = [flattenlist[i] for i in words_unpadded.argsort()]

mostAtt_words = []
i = 0
for word in reversed(sorted_wordlist):
    if word not in mostAtt_words:
        mostAtt_words.append(word)
        i += 1
    if (i >= n_words):
        break


res = model.predict(np.expand_dims(input_array, axis=0)).flatten()
cat = np.argmax(res.flatten()) + 1

print('Review')
print(review)
print('Category')
print(res)
print(cat)
print('Summary')
print(summary)
print('Important Words')
print(mostAtt_words)