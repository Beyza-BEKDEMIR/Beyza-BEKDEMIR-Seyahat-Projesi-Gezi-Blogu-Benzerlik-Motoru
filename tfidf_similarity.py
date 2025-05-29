import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# === Anlamsal puana dönüştürme fonksiyonu ===
def fixed_similarity_to_score(similarities):
    scores = []
    for sim in similarities:
        if sim > 0.95:
            scores.append(5)
        elif sim > 0.80:
            scores.append(4)
        elif sim > 0.60:
            scores.append(3)
        elif sim > 0.40:
            scores.append(2)
        else:
            scores.append(1)
    return scores

# === TF-IDF benzerlik fonksiyonu ===
def tfidf_similarity(df, sample_text, tfidf_csv_path):
    df = df.dropna(subset=[df.columns[1]])  # NaN'leri temizle
    tfidf_df = pd.read_csv(tfidf_csv_path)  # (Yüklemesen de olabilir)

    vectorizer = TfidfVectorizer()
    texts = df.iloc[:, 1].astype(str).tolist()
    vectors = vectorizer.fit_transform(texts)

    sample_vec = vectorizer.transform([sample_text])
    sim_scores = cosine_similarity(sample_vec, vectors).flatten()
    top_indices = sim_scores.argsort()[-6:-1][::-1]
    return top_indices, sim_scores[top_indices]

# === Veri ve örnek metin yükle ===
sample_index = 13
df_lemma = pd.read_csv("travel_blogs/lemmatized_travel_blogs.csv")
df_stem = pd.read_csv("travel_blogs/stemmed_travel_blogs.csv")
sample_lemma = df_lemma.iloc[sample_index]["Lemmatized Sentence"]
sample_stem = df_stem.iloc[sample_index]["Stemmed Sentence"]

# === Benzerlik analizleri ===
lemma_indices, lemma_scores = tfidf_similarity(df_lemma, sample_lemma, "models/lemma/tfidf_lemmatized.csv")
stem_indices, stem_scores = tfidf_similarity(df_stem, sample_stem, "models/stem/tfidf_stemmed.csv")

lemma_sim_scores = fixed_similarity_to_score(lemma_scores)
stem_sim_scores = fixed_similarity_to_score(stem_scores)

# === Ortalama puanlar ===
avg_lemma_score = np.mean(lemma_sim_scores)
avg_stem_score = np.mean(stem_sim_scores)

# === Yazdır ===
print("\n=== GİRİŞ METNİ (LEMMA) ===")
print(sample_lemma)

print("\nTF-IDF LEMMA - EN BENZER 5 METİN:")
for i, (idx, score, sim_score) in enumerate(zip(lemma_indices, lemma_scores, lemma_sim_scores)):
    print(f"{i+1}. [doc{idx}] {df_lemma.iloc[idx]['Lemmatized Sentence']} -> Skor: {score:.4f} -> Anlamsal Puan: {sim_score}")

print(f"\nOrtalama Anlamsal Puan (LEMMA - TF-IDF): {avg_lemma_score:.2f}")

print("\n=== GİRİŞ METNİ (STEM) ===")
print(sample_stem)

print("\nTF-IDF STEM - EN BENZER 5 METİN:")
for i, (idx, score, sim_score) in enumerate(zip(stem_indices, stem_scores, stem_sim_scores)):
    print(f"{i+1}. [doc{idx}] {df_stem.iloc[idx]['Stemmed Sentence']} -> Skor: {score:.4f} -> Anlamsal Puan: {sim_score}")

print(f"Ortalama Anlamsal Puan (STEM  - TF-IDF): {avg_stem_score:.2f}")
