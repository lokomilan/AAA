class CountVectorizer():

    def __init__(self):
        self.feature_names = []
        self.feature_count = 0

    def get_feature_names(self):
        return self.feature_names

    def fit_transform(self, corpus):

        for sentence in corpus:
            for word in sentence.split():
                if not word.lower() in self.feature_names:
                    self.feature_names.append(word.lower())
                    self.feature_count += 1

        matrix = []
        word_to_pos = {self.feature_names[i]: i for i in range(
            self.feature_count)}

        for sentence in corpus:
            matrix.append(self.feature_count * [0])
            for word in sentence.split():
                matrix[-1][word_to_pos[word.lower()]] += 1

        return matrix
