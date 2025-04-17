from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string
import nltk
import torch

class Lowercase:
    def __call__(self, text):
        return text.lower()

class Tokenize:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.tokenize(text)
    
class StopwordsRemoval:
    def __init__(self, stopwords):
        # init list of stopwords 
        self.stopwords = set(stopwords)

    # callable object like function
    def __call__ (self, tokens):
        # this object takes tokens (list)
        stopwords_removed_tokens = [word for word in tokens if word not in self.stopwords and word not in string.punctuation]
        return stopwords_removed_tokens

class POStagging:
    def __call__(self, tokens):
        # result from POStaggain is ('running', 'VBP')
        pos_tagged_tokens = nltk.pos_tag(tokens)
        return pos_tagged_tokens
    
class Lemmatization:
    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer
    
    # lemmatize tuple of word and its tag ('running', 'verb')
    def __call__(self, pos_tagged_tokens):
        lemmatized_tokens = [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(pos)) for word, pos in pos_tagged_tokens]
        return " ".join(lemmatized_tokens)
    
    @staticmethod
    def _get_wordnet_pos(treebank_tag): #independent of instance
        """Convert Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun if unknown
    
class TFIDF:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer  # a sklearn TF-IDF vectorizer

    def __call__(self, preprocess_text):
        # TF-IDF vectorizer expects an iterable of documents
        tfidf_matrix = self.vectorizer.fit_transform(preprocess_text)
        return tfidf_matrix.toarray()
    
class Compose:
    def __init__(self, transforms):
        # list of transformation object
        self.transforms = transforms

    # apply different transformations in order
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
class ToTensor:
    def __call__(self, X, y):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        return X_tensor, y_tensor