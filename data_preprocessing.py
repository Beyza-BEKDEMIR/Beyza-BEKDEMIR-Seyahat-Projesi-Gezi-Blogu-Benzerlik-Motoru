import nltk
import re
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# NLTK kaynaklarını indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_sentence(sentence):
    # Tokenize
    tokens = word_tokenize(sentence)

    # Temizle
    cleaned = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    cleaned = [token.lower() for token in cleaned if token]
    cleaned = [token for token in cleaned if token not in stop_words]

    # Lemmatize ve Stem
    lemmatized = [lemmatizer.lemmatize(token) for token in cleaned]
    stemmed = [stemmer.stem(token) for token in cleaned]

    return lemmatized, stemmed, cleaned

# Dosya yolu
file_path = "travel_blogs/all_travel_blogs.txt"

# Ham veri boyutu
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

# Dosyayı oku
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Ham kelime sayısı
raw_tokens = word_tokenize(content)
raw_token_count = len(raw_tokens)

# Cümlelere ayır
sentences = sent_tokenize(content)

# Sayımlar
lemma_token_count = 0
stem_token_count = 0

# CSV dosyaları
lemma_path = "travel_blogs/lemmatized_travel_blogs.csv"
stem_path = "travel_blogs/stemmed_travel_blogs.csv"

with open(lemma_path, "w", newline='', encoding="utf-8") as lemma_file, \
     open(stem_path, "w", newline='', encoding="utf-8") as stem_file:

    lemma_writer = csv.writer(lemma_file)
    stem_writer = csv.writer(stem_file)

    lemma_writer.writerow(["Original Sentence", "Lemmatized Tokens"])
    stem_writer.writerow(["Original Sentence", "Stemmed Tokens"])

    print("Örnek İşlenen Cümleler:\n")

    for i, sentence in enumerate(sentences):
        lemmatized, stemmed, cleaned = preprocess_sentence(sentence)

        lemma_token_count += len(lemmatized)
        stem_token_count += len(stemmed)

        lemma_writer.writerow([sentence, ", ".join(lemmatized)])
        stem_writer.writerow([sentence, ", ".join(stemmed)])

        if i < 3:
            print(f"Cümle: {sentence}")
            print(f"  Lemmatized: {lemmatized}")
            print(f"  Stemmed   : {stemmed}")
            print()

# Dosya boyutlarını ölç
lemma_file_size = os.path.getsize(lemma_path) / (1024 * 1024)
stem_file_size = os.path.getsize(stem_path) / (1024 * 1024)

# Sonuçları yazdır
print("\n--- VERİ ANALİZİ ---")
print(f"Ham Veri Dosya Boyutu (MB): {file_size_mb:.2f} MB")
print(f"Ham Veri Kelime Sayısı: {raw_token_count}")

print(f"\nLemmatized Veri Dosya Boyutu (MB): {lemma_file_size:.2f} MB")
print(f"Lemmatized Kelime Sayısı: {lemma_token_count}")

print(f"\nStemmed Veri Dosya Boyutu (MB): {stem_file_size:.2f} MB")
print(f"Stemmed Kelime Sayısı: {stem_token_count}")

print("\nCSV dosyaları oluşturuldu: lemmatized_travel_blogs.csv ve stemmed_travel_blogs.csv")
