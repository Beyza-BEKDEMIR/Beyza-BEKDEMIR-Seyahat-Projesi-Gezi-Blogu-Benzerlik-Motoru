import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import csv

nltk.download('punkt')

# --- Zipf Grafiği Çizme Fonksiyonu ---
def plot_zipf(tokens, title):
    word_counts = Counter(tokens)
    sorted_word_counts = word_counts.most_common()

    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = np.array([freq for _, freq in sorted_word_counts])

    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, frequencies, alpha=0.6, color='blue')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Zipf's Law - {title}")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequency)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# --- 1. Ham Veri ---

with open("travel_blogs/all_travel_blogs.txt", "r", encoding="utf-8") as file:
    content = file.read()

tokens_raw = word_tokenize(content) 
plot_zipf(tokens_raw, "Ham Veri")

# --- 2. Lemmatized Veri ---

lemma_tokens = []
with open("travel_blogs/lemmatized_travel_blogs.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # başlık atla
    for row in reader:
        if len(row) > 1:
            tokens = row[1].split(", ")
            lemma_tokens.extend(tokens)

plot_zipf(lemma_tokens, "Lemmatize Edilmiş Veri")

# --- 3. Stemmed Veri ---

stem_tokens = []
with open("travel_blogs/stemmed_travel_blogs.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # başlık atla
    for row in reader:
        if len(row) > 1:
            tokens = row[1].split(", ")
            stem_tokens.extend(tokens)

plot_zipf(stem_tokens, "Stem Edilmiş Veri")
