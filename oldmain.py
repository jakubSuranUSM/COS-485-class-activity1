from transformers import AutoTokenizer
from unigram import UnigramLanguageModel

f = open("lyrics.txt")
text = f.read()
f.close()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokensLyric1 = tokenizer.tokenize(text)

model = UnigramLanguageModel()
model.train_for_label("DSB", tokensLyric1)
print(model.unigram_counts)
print(model.probabilities)

prob = model.predict(['Just', 'a', 'bert'])
print(prob)

#
# line = "Hidin' in the night"
# tokensLine = tokenizer.tokenize(line)
# print(f"line: {tokensLine}")
#
#
#
#
#
