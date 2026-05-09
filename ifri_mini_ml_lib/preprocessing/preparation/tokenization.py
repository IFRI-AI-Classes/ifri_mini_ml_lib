import re 


class Tokenizer:

    def __init__(self, lowercase=True, handle_apostrophes=True, handle_contractions=True, remove_punctuation=True ,min_token_length=1):
        self.lowercase = lowercase
        self.handle_apostrophes = handle_apostrophes
        self.handle_contractions = handle_contractions
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length


    def tokenize_corpus(self, corpus):
        return [self.tokenize(doc) for doc in corpus]

    def _remove_punctuation(self,text):
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text) 
        return text


    def _handle_apostrophes(self,text):
        text = re.sub(r"['\u2019]", " ", text)
        return text

    def _handle_contractions(self, text):
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"shouldn't", "should not", text)
        text = re.sub(r"wouldn't", "would not", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"didn't", "did not", text)
        text = re.sub(r"isn't", "is not", text)
        text = re.sub(r"aren't", "are not", text)
        text = re.sub(r"wasn't", "was not", text)
        text = re.sub(r"weren't", "were not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ve", " have", text)

        pronouns = r"(it|he|she|that|what|who|here|there|how)"
        text = re.sub(pronouns + r"'s", r"\1 is", text)

        text = re.sub(r"'s\b", "", text)

        return text

    def lower_case(self,text):
        return text.lower()
    
  
    def tokenize(self, text):
        if self.lowercase:
            text = self.lower_case(text)
        if self.handle_contractions:
            text = self._handle_contractions(text)
        if self.handle_apostrophes:
            text = self._handle_apostrophes(text)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        tokens = text.split()

        if self.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def tokenize_corpus(self, corpus):
        return [self.tokenize(doc) for doc in corpus]




    t = Tokenizer()

    
    print(t.tokenize("i can't have john's money and she won't take it's place"))


    print(t.tokenize("l'élève va à l'école et c'est très bien"))

    corpus = [
        "I can't do this ",
        "John's car is amazing",
        "l'intelligence artificielle c'est fascinant"
    ]
    print(t.tokenize_corpus(corpus))

    t2 = Tokenizer(handle_contractions=False, min_token_length=3)
    print(t2.tokenize("i can't stop thinking about it "))


