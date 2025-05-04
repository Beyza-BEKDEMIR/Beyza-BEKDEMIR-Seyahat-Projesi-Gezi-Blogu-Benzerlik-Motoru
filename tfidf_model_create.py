import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

def calculate_matrix_size(matrix):
    """Numpy matrisin yaklaşık bellek boyutunu MB cinsinden döner"""
    return matrix.nbytes / (1024 * 1024)

# === LEMMATIZED ===
start_lemma = time.time()

# CSV'den lemmatized veriyi yükle
df_lemma = pd.read_csv("travel_blogs/lemmatized_travel_blogs.csv")
texts_lemma = df_lemma['Lemmatized Tokens'].astype(str).tolist()

# TF-IDF hesapla
vectorizer_lemma = TfidfVectorizer()
tfidf_matrix_lemma = vectorizer_lemma.fit_transform(texts_lemma)
feature_names_lemma = vectorizer_lemma.get_feature_names_out()
tfidf_df_lemma = pd.DataFrame(tfidf_matrix_lemma.toarray(), columns=feature_names_lemma)

print("\n------- LEMMA -------")
print(tfidf_df_lemma.iloc[14:19]) # 15-18. satırlardaki değerleri yazdırır.

# CSV'ye kaydet
os.makedirs("models/lemma", exist_ok=True)
tfidf_df_lemma.to_csv("models/lemma/tfidf_lemmatized.csv", index=False)

end_lemma = time.time()
duration_lemma = end_lemma - start_lemma
size_lemma_mb = calculate_matrix_size(tfidf_matrix_lemma.toarray())

print(f"\nLEMMA - TF-IDF eğitimi tamamlandı.")
print(f"  -> Eğitim Süresi   : {duration_lemma:.2f} saniye")
print(f"  -> Matris Boyutu   : {size_lemma_mb:.2f} MB")

# İlk cümlede en yüksek 5 TF-IDF skoru (lemma)
top_5_words_lemma = tfidf_df_lemma.iloc[0].sort_values(ascending=False).head(5)
print("\nLEMMA - İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(top_5_words_lemma)

# 'forest' kelimesine en benzer 5 kelime (lemma)
if 'forest' in feature_names_lemma:
    forest_index = list(feature_names_lemma).index('forest')
    forest_vector = tfidf_matrix_lemma[:, forest_index].toarray()
    similarities = cosine_similarity(forest_vector.T, tfidf_matrix_lemma.toarray().T).flatten()
    top_5_indices = similarities.argsort()[-6:][::-1]
    print("\nLEMMA - 'forest' kelimesine en benzer 5 kelime:")
    for idx in top_5_indices:
        print(f"{feature_names_lemma[idx]}: {similarities[idx]:.4f}")
else:
    print("\nLEMMA - 'forest' kelimesi TF-IDF matrisinde yok.")


# === STEMMED ===
start_stem = time.time()

df_stem = pd.read_csv("travel_blogs/stemmed_travel_blogs.csv")
texts_stem = df_stem['Stemmed Tokens'].astype(str).tolist()

vectorizer_stem = TfidfVectorizer()
tfidf_matrix_stem = vectorizer_stem.fit_transform(texts_stem)
feature_names_stem = vectorizer_stem.get_feature_names_out()
tfidf_df_stem = pd.DataFrame(tfidf_matrix_stem.toarray(), columns=feature_names_stem)

print("\n------- STEM -------")
print(tfidf_df_stem.tail()) # son 5 satırı yazdırır.

# CSV'ye kaydet
os.makedirs("models/stem", exist_ok=True)
tfidf_df_stem.to_csv("models/stem/tfidf_stemmed.csv", index=False)

end_stem = time.time()
duration_stem = end_stem - start_stem
size_stem_mb = calculate_matrix_size(tfidf_matrix_stem.toarray())

print(f"\nSTEM - TF-IDF eğitimi tamamlandı.")
print(f"  -> Eğitim Süresi   : {duration_stem:.2f} saniye")
print(f"  -> Matris Boyutu   : {size_stem_mb:.2f} MB")

# İlk cümlede en yüksek 5 TF-IDF skoru (stem)
top_5_words_stem = tfidf_df_stem.iloc[0].sort_values(ascending=False).head(5)
print("\nSTEM - İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(top_5_words_stem)

# 'forest' kelimesine en benzer 5 kelime (stem)
if 'forest' in feature_names_stem:
    forest_index = list(feature_names_stem).index('forest')
    forest_vector = tfidf_matrix_stem[:, forest_index].toarray()
    similarities = cosine_similarity(forest_vector.T, tfidf_matrix_stem.toarray().T).flatten()
    top_5_indices = similarities.argsort()[-6:][::-1]
    print("\nSTEM - 'forest' kelimesine en benzer 5 kelime:")
    for idx in top_5_indices:
        print(f"{feature_names_stem[idx]}: {similarities[idx]:.4f}")
else:
    print("\nSTEM - 'forest' kelimesi TF-IDF matrisinde yok.")
