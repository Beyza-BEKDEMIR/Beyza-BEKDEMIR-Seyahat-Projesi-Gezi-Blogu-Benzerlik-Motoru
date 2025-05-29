import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# --- Jaccard benzerliği (sıralı) ---
def jaccard_similarity_ordered(list1, list2, reference_order):
    set1 = set(list1)
    set2 = set(list2)
    intersection = [x for x in reference_order if x in set1 and x in set2]
    union = [x for x in reference_order if x in set1 or x in set2]
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

# --- Word2Vec için cümle vektörü ortalaması ---
def avg_vector(text, model):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.vector_size)
    tokens = text.split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# --- Word2Vec ile en iyi 5 doc_id döndür ---
def word2vec_top5(df, model, sentence_col, sample_text):
    sample_vec = avg_vector(sample_text, model).reshape(1, -1)
    results = []
    for idx, row in df.iterrows():
        vec = avg_vector(row[sentence_col], model).reshape(1, -1)
        sim = cosine_similarity(sample_vec, vec)[0][0]
        results.append((row['doc_id'], sim))
    top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    return [doc_id for doc_id, _ in top5]

# --- TF-IDF ile en iyi 5 doc_id döndür ---
def tfidf_top5(df, sentence_col, sample_text):
    df_clean = df.dropna(subset=[sentence_col])
    vectorizer = TfidfVectorizer()
    texts = df_clean[sentence_col].astype(str).tolist()
    vectors = vectorizer.fit_transform(texts)
    sample_vec = vectorizer.transform([sample_text])
    sim_scores = cosine_similarity(sample_vec, vectors).flatten()
    # En yüksek 5 skorun indeksleri
    top_indices = sim_scores.argsort()[-5:][::-1]
    top5_doc_ids = df_clean.iloc[top_indices]['doc_id'].tolist()
    return top5_doc_ids

# --- Veri yükleme ---
df_lemma = pd.read_csv("travel_blogs/lemmatized_travel_blogs.csv")
df_stem = pd.read_csv("travel_blogs/stemmed_travel_blogs.csv")

# doc_id sütunu ekle
df_lemma['doc_id'] = ['doc' + str(i+1) for i in range(len(df_lemma))]
df_stem['doc_id'] = ['doc' + str(i+1) for i in range(len(df_stem))]

sample_index = 13
sample_lemma = df_lemma.loc[sample_index, "Lemmatized Sentence"]
sample_stem = df_stem.loc[sample_index, "Stemmed Sentence"]

# Word2Vec modellerinin yolları
lemma_models_paths = [
    "models/lemma/lemmatized_model_cbow_window2_dim100.model",
    "models/lemma/lemmatized_model_cbow_window2_dim300.model",
    "models/lemma/lemmatized_model_cbow_window4_dim100.model",
    "models/lemma/lemmatized_model_cbow_window4_dim300.model",
    "models/lemma/lemmatized_model_skipgram_window2_dim100.model",
    "models/lemma/lemmatized_model_skipgram_window2_dim300.model",
    "models/lemma/lemmatized_model_skipgram_window4_dim100.model",
    "models/lemma/lemmatized_model_skipgram_window4_dim300.model"
]

stem_models_paths = [
    "models/stem/stemmed_model_cbow_window2_dim100.model",
    "models/stem/stemmed_model_cbow_window2_dim300.model",
    "models/stem/stemmed_model_cbow_window4_dim100.model",
    "models/stem/stemmed_model_cbow_window4_dim300.model",
    "models/stem/stemmed_model_skipgram_window2_dim100.model",
    "models/stem/stemmed_model_skipgram_window2_dim300.model",
    "models/stem/stemmed_model_skipgram_window4_dim100.model",
    "models/stem/stemmed_model_skipgram_window4_dim300.model"
]

# Word2Vec modellerini yükle
w2v_lemma_models = [Word2Vec.load(p) for p in lemma_models_paths]
w2v_stem_models = [Word2Vec.load(p) for p in stem_models_paths]

# Sonuçları tutacak dict
models_top5 = {}

# TF-IDF sonuçları (doc_id listesi)
models_top5["tfidf_lemmatized"] = tfidf_top5(df_lemma, "Lemmatized Sentence", sample_lemma)
models_top5["tfidf_stemmed"] = tfidf_top5(df_stem, "Stemmed Sentence", sample_stem)

# Word2Vec lemma modelleri için top5
for path, model in zip(lemma_models_paths, w2v_lemma_models):
    parts = path.split('/')[-1].replace('.model', '').split('_')
    arch = parts[2]
    window = parts[3].replace('window', 'w')
    dim = parts[4].replace('dim', 'd')
    key = f"word2vec_lemmatized_{arch}_{window}_{dim}"
    models_top5[key] = word2vec_top5(df_lemma, model, "Lemmatized Sentence", sample_lemma)

# Word2Vec stem modelleri için top5
for path, model in zip(stem_models_paths, w2v_stem_models):
    parts = path.split('/')[-1].replace('.model', '').split('_')
    arch = parts[2]
    window = parts[3].replace('window', 'w')
    dim = parts[4].replace('dim', 'd')
    key = f"word2vec_stemmed_{arch}_{window}_{dim}"
    models_top5[key] = word2vec_top5(df_stem, model, "Stemmed Sentence", sample_stem)

# Model isimleri listesi
model_names = list(models_top5.keys())

# Tüm top5 doc_id'lerin birleşimi — Jaccard için referans sıra
reference_order = sorted(set().union(*models_top5.values()))

# Jaccard matrisi oluştur
n = len(model_names)
jaccard_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        list_i = models_top5[model_names[i]]
        list_j = models_top5[model_names[j]]
        jaccard_matrix[i, j] = jaccard_similarity_ordered(list_i, list_j, reference_order)

# DataFrame oluştur
jaccard_df = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)

print("\n=== Jaccard Benzerlik Matrisi ===")
print(jaccard_df.round(2))

# CSV olarak kaydet (başlıksız)
jaccard_df.round(2).to_csv("jaccard_similarity_matrix.csv", header=False, index=False)
print("\nJaccard benzerlik matrisi 'jaccard_similarity_matrix.csv' olarak kaydedildi.")
