import requests
from bs4 import BeautifulSoup
import os
import re
import time

# Blog yazılarının kaydedileceği klasör
SAVE_DIR = "travel_blogs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Çekilecek destinasyonlar (daha fazlasını ekleyebilirsin)
destination_urls = {
    "Turkey": "https://www.travellerspoint.com/guide/Turkey/",
    "Australia": "https://www.travellerspoint.com/guide/Australia/",
    "Italy": "https://www.travellerspoint.com/guide/Italy/",
    "France": "https://www.travellerspoint.com/guide/France/",
}

# User-Agent başlığı
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_filename(name):
    """Dosya ismi olarak kullanmak için uygula"""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name.replace(" ", "_"))

def fetch_blog_text(url):
    """Sayfadaki blog paragraf metinlerini çeker ve gereksiz kısmı temizler"""
    try:
        r = requests.get(url, headers=headers)  # Header ekledik
        r.raise_for_status()  # Hata kontrolü
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        
        # Sayfa sonlarındaki "X articles link to this page" gibi metinleri temizle
        text = re.sub(r'\d+\s+articles\s+link\s+to\s+this\s+page\.', '', text)
        text = re.sub(r'Except\s+where\s+otherwise\s+noted,.*?Creative\s+Commons\s+Attribution-ShareAlike\s+3\.0\s+License\.?', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Başka kalıplarla metin varsa temizle
        text = re.sub(r'Creative Commons Attribution-ShareAlike 3.0 License\.', '', text)
        text = re.sub(r'\s+.credits\s+this\s+page.\s*', '', text)
        
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# Tek bir dosya için tüm veriyi birleştir
with open(os.path.join(SAVE_DIR, "all_travel_blogs.txt"), "w", encoding="utf-8") as f:
    for place, url in destination_urls.items():
        print(f"Fetching blog for: {place}")
        blog_text = fetch_blog_text(url)

        if blog_text:
            f.write(f"### {place} ###\n")
            f.write(blog_text + "\n\n")
            print(f"Saved blog for {place}")
        else:
            print(f"No content for {place}")
        
        time.sleep(2)  # Her isteği arasına 2 saniye bekleme ekle