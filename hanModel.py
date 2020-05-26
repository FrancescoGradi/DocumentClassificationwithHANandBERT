from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, GRU, TimeDistributed, Bidirectional, Embedding, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from tensorflow.keras import initializers as initializers


class AttentionLayer(layers.Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    """

    def __init__(self, attention_dim=100, return_coefficients=False, **kwargs):
        # Initializer
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                                 initializer=self.init,
                                 trainable=True,
                                 name='W')
        self.b = self.add_weight(shape=(self.attention_dim,),
                                 initializer=self.init,
                                 trainable=True,
                                 name='b')
        self.u = self.add_weight(shape=(self.attention_dim, 1),
                                 initializer=self.init,
                                 trainable=True,
                                 name='u')

        super(AttentionLayer, self).build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention_dim': self.attention_dim,
        })
        return config

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W), self.b)
        uit = K.tanh(uit)

        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]



'''

# Network with subclassing style doesnt work

class WordEncoder(Model):

    def __init__(self, len_word_index, embedding_matrix, MAX_WORD_NUM, EMBED_SIZE):
        super(WordEncoder, self).__init__()

        self.word_input = Input(shape=(MAX_WORD_NUM,), dtype='int32', name='word_input')
        self.word_sequences = Embedding(len_word_index + 1, EMBED_SIZE, weights=[embedding_matrix],
                                         input_length=MAX_WORD_NUM, trainable=False, name='word_embedding')
        self.word_gru = Bidirectional(GRU(EMBED_SIZE/2, return_sequences=True), name='word_gru')
        self.word_dense = Dense(EMBED_SIZE, activation='relu', name='word_dense')
        self.word_att = AttentionLayer(EMBED_SIZE, return_coefficients=False, name='word_attention')

    def call(self, inputs):
        x = self.word_input(inputs)
        x = self.word_sequences(x)
        x = self.word_gru(x)
        x = self.word_dense(x)
        x = self.word_att(x)

        return x


class HanModel(Model):

    def __init__(self, word_encoder, n_classes, len_word_index, embedding_matrix, MAX_SENTENCE_NUM=40, MAX_WORD_NUM=50, EMBED_SIZE=100):
        super(HanModel, self).__init__()

        self.sent_input = Input(shape=(MAX_SENTENCE_NUM, MAX_WORD_NUM), dtype='int32', name='sent_input')
        self.sent_encoder = TimeDistributed(word_encoder, name='sent_linking')
        self.sent_gru = Bidirectional(GRU(EMBED_SIZE/2, return_sequences=True), name='sent_gru')
        self.sent_dense = Dense(EMBED_SIZE, activation='relu', name='sent_dense')
        self.sent_att = AttentionLayer(EMBED_SIZE, return_coefficients=False, name='sent_attention')
        self.sent_drop = Dropout(0.5, name='sent_dropout')
        self.preds = Dense(n_classes, activation='softmax', name='output')

    def call(self, inputs):
        x = self.sent_input(inputs)
        x = self.sent_encoder(x)
        x = self.sent_gru(x)
        x = self.sent_dense(x)
        x = self.sent_att(x)
        x = self.sent_drop(x)

        return self.preds(x)

'''

def HanModel(n_classes, len_word_index, embedding_matrix, MAX_SENTENCE_NUM=40, MAX_WORD_NUM=50, EMBED_SIZE=100):

    # Word Encoder
    word_input = Input(shape=(MAX_WORD_NUM,), dtype='int32', name='word_input')
    word_sequences = Embedding(len_word_index + 1, EMBED_SIZE, weights=[embedding_matrix], input_length=MAX_WORD_NUM,
                               trainable=False, name='word_embedding')(word_input)
    word_gru = Bidirectional(GRU((int)(EMBED_SIZE / 2), return_sequences=True,
                                 recurrent_regularizer=regularizers.l2(0.001)), name='word_gru')(word_sequences)
    word_dense = Dense(EMBED_SIZE, activation='relu', name='word_dense')(word_gru)
    word_att = AttentionLayer(EMBED_SIZE, return_coefficients=False, name='word_attention')(word_dense)

    word_encoder = Model(inputs=word_input, outputs=word_att)

    # Sentence Attention model
    sent_input = Input(shape=(MAX_SENTENCE_NUM, MAX_WORD_NUM), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(word_encoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(GRU((int)(EMBED_SIZE / 2), return_sequences=True,
                                 recurrent_regularizer=regularizers.l2(0.001)), name='sent_gru')(sent_encoder)
    sent_dense = Dense(EMBED_SIZE, activation='relu', name='sent_dense')(sent_gru)
    sent_att = AttentionLayer(EMBED_SIZE, return_coefficients=False, name='sent_attention')(sent_dense)
    sent_drop = Dropout(rate=0.2, name='sent_dropout')(sent_att)
    preds = Dense(n_classes, activation='softmax', name='output')(sent_drop)

    return Model(sent_input, preds, name='HanModel')