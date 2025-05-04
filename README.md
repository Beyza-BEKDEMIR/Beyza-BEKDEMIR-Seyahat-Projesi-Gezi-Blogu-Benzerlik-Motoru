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

##  Gerekli Kütüphaneler ve Kurulum

  Projeyi çalıştırmadan önce aşağıdaki kütüphanelerin kurulu olması gerekmektedir.
  
    - beautifulsoup4
    - requests
    - nltk
    - gensim
    - pandas
    - scikit-learn
    - matplotlib
  

### 1. Kurulum

  Aşağıdaki komutla gerekli tüm paketleri yükleyebilirsiniz:
  
  pip install beautifulsoup4 requests nltk gensim pandas scikit-learn matplotlib
  
  # Gerekli olan nltk kaynakları
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


