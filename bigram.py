from collections import defaultdict
import re


class BigramLanguageModel:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.total_documents = 0

    def train(self, labeled_corpus):
        for text, label in labeled_corpus:
            self.total_documents += 1
            words = re.findall(r'\b\w+\b', text.lower())  # Simple word tokenization
            for bigram in zip(words, words[1:]):
                self.bigram_counts[bigram] += 1
                self.unigram_counts[bigram[0]] += 1

    def calculate_bigram_probability(self, bigram):
        if self.unigram_counts[bigram[0]] == 0:
            return 0.0
        return self.bigram_counts[bigram] / self.unigram_counts[bigram[0]]

    def predict_label(self, unlabeled_text):
        words = re.findall(r'\b\w+\b', unlabeled_text.lower())
        bigrams = list(zip(words, words[1:]))
        probabilities = defaultdict(float)

        print(bigrams)
        for label in ['positive', 'negative']:
            probability = 1.0
            for bigram in bigrams:
                probability *= self.calculate_bigram_probability(bigram)
            probabilities[label] = probability * (self.unigram_counts[bigrams[0][0]] / self.total_documents)

        print(probabilities)
        return max(probabilities, key=probabilities.get)

# Example usage
labeled_corpus = [
    ("positive text 1", "positive"),
    ("positive text 2", "positive"),
    ("negative text 1", "negative"),
    ("negative text 2", "negative"),
    # ... more labeled data
]

# Train a bigram model
bigram_model = BigramLanguageModel()
bigram_model.train(labeled_corpus)

# Unlabeled texts
unlabeled_texts = [
    "unlabeled text 1",
    "unlabeled text 2",
    "negative text 1"
]

# Predict labels for unlabeled texts
predicted_labels = []
for text in unlabeled_texts:
    predicted_label = bigram_model.predict_label(text)
    predicted_labels.append((text, predicted_label))

# Display the results
for text, label in predicted_labels:
    print(f"Text: {text}  |  Predicted Label: {label}")


print(bigram_model.total_documents)
print(bigram_model.unigram_counts)
print(bigram_model.bigram_counts)
