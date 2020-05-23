from nltk import tokenize
from nltk.stem import WordNetLemmatizer


def cleanString(text, stop_words):
    """
        Cleans input string using set rules.
        Cleaning rules:         Every word is lemmatized and lowercased. Stopwords and non alpha-numeric words are
                                removed.
                                Each sentence ends with a period.
        Input:   text       - string(in sentence structure)
                 stop_words   - set of strings which should be removed from text
        Output:  returnString - cleaned input string
    """

    lemmatizer = WordNetLemmatizer()
    cleaned_string = ''
    sentence_token = tokenize.sent_tokenize(text)

    for j in range(len(sentence_token)):
        single_sentence = tokenize.word_tokenize(sentence_token[j])
        sentences_filtered = [(i, lemmatizer.lemmatize(w.lower())) for i, w in enumerate(single_sentence)
                              if w.lower() not in stop_words and w.isalnum()]

        word_list = [x[1] for x in sentences_filtered]
        cleaned_string += ' '.join(word_list) + ' . '

    return cleaned_string