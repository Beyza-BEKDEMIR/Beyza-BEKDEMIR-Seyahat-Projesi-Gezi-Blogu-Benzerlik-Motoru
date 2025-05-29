from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def avg_vector(text, model):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(model.vector_size)
    tokens = text.split()  # Boşlukla ayırıyoruz
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def similarity_to_score(similarities):
    scores = []
    for s in similarities:
        if s > 0.95:
            scores.append(5)
        elif s > 0.80:
            scores.append(4)
        elif s > 0.60:
            scores.append(3)
        elif s > 0.40:
            scores.append(2)
        else:
            scores.append(1)
    return scores


df_lemma = pd.read_csv("travel_blogs/lemmatized_travel_blogs.csv")
df_stem = pd.read_csv("travel_blogs/stemmed_travel_blogs.csv")

# Her cümleye doc1, doc2... şeklinde ID atıyoruz
df_lemma['doc_id'] = ['doc' + str(i+1) for i in range(len(df_lemma))]
df_stem['doc_id'] = ['doc' + str(i+1) for i in range(len(df_stem))]

sample_index = 13
sample_lemma = df_lemma.iloc[sample_index]["Lemmatized Sentence"]
sample_stem = df_stem.iloc[sample_index]["Stemmed Sentence"]

params = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

def analyze_model(path, sample_text, df, sentence_column):
    model = Word2Vec.load(path)
    sample_vec = avg_vector(sample_text, model).reshape(1, -1)

    results = []
    for idx, row in df.iterrows():
        vec = avg_vector(row[sentence_column], model).reshape(1, -1)
        score = cosine_similarity(sample_vec, vec)[0][0]
        results.append((row['doc_id'], score))  # Burada doc_id ile eşleştiriyoruz

    top = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    return top

for param in params:
    name = f"{param['model_type']}_w{param['window']}_d{param['vector_size']}"

    # Lemmatized modeli ve benzerlik
    path_lemma = f"models/lemma/lemmatized_model_{param['model_type']}_window{param['window']}_dim{param['vector_size']}.model"
    top5_lemma = analyze_model(path_lemma, sample_lemma, df_lemma, "Lemmatized Sentence")

    lemma_scores = [score for _, score in top5_lemma]
    lemma_similarity_scores = similarity_to_score(lemma_scores)
    lemma_avg_score = np.mean(lemma_similarity_scores)

    print(f"\nModel: Lemmatized - {name}")
    for i, ((doc_id, score), sim_score) in enumerate(zip(top5_lemma, lemma_similarity_scores)):
        sentence_preview = df_lemma.loc[df_lemma['doc_id'] == doc_id, 'Lemmatized Sentence'].values[0]
        print(f"{i+1}. [{doc_id}] {sentence_preview} -> Skor: {score:.4f} -> Anlamsal Puan: {sim_score}")
    print(f"Ortalama Anlamsal Puan: {lemma_avg_score:.2f}")

    # Stemmed modeli ve benzerlik
    path_stem = f"models/stem/stemmed_model_{param['model_type']}_window{param['window']}_dim{param['vector_size']}.model"
    top5_stem = analyze_model(path_stem, sample_stem, df_stem, "Stemmed Sentence")

    stem_scores = [score for _, score in top5_stem]
    stem_similarity_scores = similarity_to_score(stem_scores)
    stem_avg_score = np.mean(stem_similarity_scores)

    print(f"\nModel: Stemmed - {name}")
    for i, ((doc_id, score), sim_score) in enumerate(zip(top5_stem, stem_similarity_scores)):
        sentence_preview = df_stem.loc[df_stem['doc_id'] == doc_id, 'Stemmed Sentence'].values[0]
        print(f"{i+1}. [{doc_id}] {sentence_preview} -> Skor: {score:.4f} -> Anlamsal Puan: {sim_score}")

    print(f"Ortalama Anlamsal Puan: {stem_avg_score:.2f}")
