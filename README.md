# Seyahat-Projesi-Gezi-Blogu-Benzerlik-Motoru

Bu projede, Travellerspoint seyahat bloglarından elde edilen metinler üzerinde doğal dil işleme (NLP) teknikleri uygulanarak, ülkelerle ilgili yazıların içerik   analizi gerçekleştirilmiştir.
  
  ---

## Model Nasıl Oluşturulur? 

  1. **Veri Toplama:**
       - `dataset_create.py` dosyası ile web scraping kullanılarak Travellerspoint sitesinden veriler çekilir.
       - Toplanan ham veriler `all_travel_blogs.txt` dosyasında birleştirilir.
  
  2. **Veri Temizleme ve Ön İşleme:**
       - Noktalama işaretleri, özel karakterler ve stopword’ler çıkarılır.
       - Lemmatizasyon ve stemming işlemleri uygulanır.
       - Temiz veriler `.csv` formatında `lemmatized_travel_blogs.csv` ve `stemmed_travel_blogs.csv` dosyalarında saklanır.
  
  3. **Modelleme:**
       - TF-IDF yöntemi kullanılarak, ön işleme adımlarında elde edilen lemmatize ve stem uygulanmış metinlerden iki ayrı kelime matris modeli oluşturulmuştur.
       - Word2Vec algoritması ile bağlamsal benzerlik analizi gerçekleştirilmiştir. Bu kapsamda, her biri CBOW ve Skip-Gram mimarileri ile farklı parametre       
         kombinasyonları kullanılarak toplam 16 ayrı model eğitilmiştir.

---

## Veri Seti Hangi Amaçla Kullanılabilir?

  Bu veri seti, aşağıdaki doğal dil işleme amaçları için kullanılabilir:
  
- Ülkelerle ilgili blog içeriklerinin konu analizi
- TF-IDF ve Word2Vec gibi metin temsillerinin karşılaştırılması
- Kelime benzerliği ve bağlamsal analiz çalışmaları
- Lemmatizasyon ve stemming yöntemlerinin etkilerinin incelenmesi
  
  Veri seti İngilizce'dir ve kültür, doğa, tarih, gezilecek yerler gibi temalara odaklanmaktadır.

---
## Anlamsal Değerlendirme ve Sıralama Tutarlılığı
Bu bölümde, elde edilen TF-IDF ve Word2Vec modelleri üzerinden ülkelerle ilgili cümlelerin benzerlik analizi yapılmış, modellerin bağlamsal başarımı ölçülmüş ve sıralama tutarlılığı değerlendirilmiştir.Bu işlemler aşağıdaki Python dosyaları aracılığıyla gerçekleştirilmiştir:

  **tfidf_similarity.py:** TF-IDF temelli cümle benzerliği hesaplamaları
    
  **word2vec_similarity.py:** Word2Vec (CBOW ve Skip-Gram) tabanlı benzerlik analizleri
    
  **jaccard_similarity.py:** Farklı modellerin sıralama benzerliğini ölçmek için Jaccard benzerlik analizi
  
---
## Cümle Benzerliği Hesaplama
Her model belirli bir giriş metni alarak benzerlikleri cosine similarity yöntemiyle hesaplanmıştır. Word2Vec modelleriyle cümle temsili oluşturulurken, her cümledeki kelime vektörlerinin ortalaması alınmıştır. TF-IDF için klasik kelime matris temsili ile cümleler vektöre dönüştürülmüş ve benzerlik skorları elde edilmiştir.

---
## Anlamsal Değerlendirme (Subjective Evaluation)
Her modelin ürettiği cümle benzerlik skoru, 0 ile 1 arasında normalize edilmiştir.
Model çıktıları, 1 ile 5 arasında puanlanarak değerlendirilmiştir; 1 düşük, 5 ise çok yüksek benzerlik anlamına gelir.
Her bir model için ortalama benzerlik skoru ve ortalama puan hesaplanarak genel başarı karşılaştırması yapılmıştır.

---
## Sıralama Tutarlılığı (Ranking Agreement)
Modellerin cümle benzerliği sıralamaları Jaccard benzerliği kullanılarak karşılaştırılmıştır.
Bu sayede, modellerin benzerlik sıralamalarındaki tutarlılık ölçülmüş ve mimari tercihlerinin etkisi analiz edilmiştir.
Elde edilen Jaccard benzerliği matrisi 'jaccard_similarity_matrix.csv' dosyasına kaydedilmiştir.

--- 

##  Gerekli Kütüphaneler ve Kurulum

  Projeyi çalıştırmadan önce aşağıdaki kütüphanelerin kurulu olması gerekmektedir.
  
    - beautifulsoup4
    - requests
    - nltk
    - gensim
    - pandas
    - scikit-learn
    - matplotlib
    - numpy 
    
### 1. Kurulum

  Aşağıdaki komutla gerekli tüm paketleri yükleyebilirsiniz:
  
  pip install beautifulsoup4 requests nltk gensim pandas scikit-learn matplotlib numpy
  
  # Gerekli olan nltk kaynakları
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


