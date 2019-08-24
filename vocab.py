class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word_count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        words = sentence.split(" ")
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index.update({word:self.n_words})
            self.word_count.update({word:1})
            self.index2word.update({self.n_words:word})
            self.n_words = self.n_words + 1
        else:
            self.word_count[word] = self.word_count[word] + 1
