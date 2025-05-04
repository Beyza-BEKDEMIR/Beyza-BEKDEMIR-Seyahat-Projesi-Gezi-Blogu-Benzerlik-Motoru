import time  # Eğitim süresi ölçümü için
import os
from gensim.models import Word2Vec
import pandas as pd

# Word2Vec modeli eğitmek için parametreler
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# CSV'den cümleleri yükle
df_lemma = pd.read_csv("travel_blogs/lemmatized_travel_blogs.csv")
df_stem = pd.read_csv("travel_blogs/stemmed_travel_blogs.csv")

# Tokenize edilmiş cümleleri corpus haline getir
tokenized_corpus_lemmatized = [str(sentence).split(", ") for sentence in df_lemma["Lemmatized Tokens"]]
tokenized_corpus_stemmed = [str(sentence).split(", ") for sentence in df_stem["Stemmed Tokens"]]

# Klasörleri oluştur
os.makedirs("models/lemma", exist_ok=True)
os.makedirs("models/stem", exist_ok=True)

# Model eğitme ve kaydetme fonksiyonu
def train_and_save_model(corpus, params, model_name, folder):
    start_time = time.time()  # Eğitim süresi başlat

    model = Word2Vec(corpus, vector_size=params['vector_size'],
                     window=params['window'], min_count=1,
                     sg=1 if params['model_type'] == 'skipgram' else 0)
    
    file_name = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model"
    save_path = os.path.join("models", folder, file_name)
    model.save(save_path)

    end_time = time.time()
    duration = end_time - start_time
    size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print(f"{file_name} kaydedildi!")
    print(f"  -> Eğitim Süresi   : {duration:.2f} saniye")
    print(f"  -> Model Boyutu    : {size_mb:.2f} MB\n")

# Lemmatize modelleri eğit ve kaydet
for param in parameters:
    train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model", "lemma")

# Stemlenmiş modelleri eğit ve kaydet
for param in parameters:
    train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model", "stem")

# Sabit kullanılacak kelime
word = "forest"

# Benzer kelimeleri yazdıran fonksiyon
def print_similar_words(model, model_name, word):
    try:
        similarity = model.wv.most_similar(word, topn=5)
        print(f"\n{model_name} Modeli - '{word}' kelimesine en yakın 5 kelime:")
        for w, score in similarity:
            print(f"Kelime: {w}, Benzerlik Skoru: {score:.4f}")
    except KeyError:
        print(f"  [!] '{word}' kelimesi modelde bulunamadı: {model_name}")

# Tüm modelleri sırayla yükle ve aynı kelime için benzer kelimeleri yazdır
for param in parameters:
    # Lemmatized model
    model_name_lemma = f"lemmatized_model_{param['model_type']}_window{param['window']}_dim{param['vector_size']}"
    model_path_lemma = os.path.join("models", "lemma", f"{model_name_lemma}.model")
    if os.path.exists(model_path_lemma):
        model = Word2Vec.load(model_path_lemma)
        print_similar_words(model, model_name_lemma, word)
    else:
        print(f"  [!] Model dosyası bulunamadı: {model_path_lemma}")
    
    # Stemmed model
    model_name_stem = f"stemmed_model_{param['model_type']}_window{param['window']}_dim{param['vector_size']}"
    model_path_stem = os.path.join("models", "stem", f"{model_name_stem}.model")
    if os.path.exists(model_path_stem):
        model = Word2Vec.load(model_path_stem)
        print_similar_words(model, model_name_stem, word)
    else:
        print(f"  [!] Model dosyası bulunamadı: {model_path_stem}")