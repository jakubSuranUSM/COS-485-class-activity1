from collections import defaultdict


class UnigramLanguageModel:
    def __init__(self):
        self.unigram_counts = {}
        self.probabilities = {}
        self.total_documents = 0

    def train_for_label(self, label, text):
        self.unigram_counts[label] = defaultdict(int)
        for word in text:
            self.unigram_counts[label][word] += 1

        total_words = sum(self.unigram_counts[label].values())
        print(total_words)
        for word in self.unigram_counts[label].keys():
            self.probabilities[word] = self.unigram_counts[label][word] / total_words

    def predict(self, text):
        probability = 1.0
        for word in text:
            probability *= self.probabilities[word]
        return probability


