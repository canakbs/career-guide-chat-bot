# Dockerfile

# Adım 1: Slim (zayıf) bir Python imajı ile başla
FROM python:3.11-slim-bullseye AS builder

# Çalışma dizinini ayarla
WORKDIR /app

# Sadece requirements dosyasını kopyala. Bu, Docker'ın katman önbelleğini
# verimli kullanmasını sağlar. Kod değişse bile kütüphaneler tekrar indirilmez.
COPY requirements.txt .

# Kütüphaneleri kur. --no-cache-dir imaj boyutunu küçültür.
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------

# Adım 2: Son imajı oluştur
FROM python:3.11-slim-bullseye

WORKDIR /app

# Bir önceki adımdan sadece kurulu kütüphaneleri kopyala
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Uygulama kodunun tamamını kopyala
COPY . .

# Uygulamanın çalışacağı portu belirt (Railway bunu otomatik algılar)
EXPOSE 8080

# Uygulamayı başlatma komutu. Railway'deki Start Command yerine bu kullanılır.
# --timeout 180, yavaş istekler için 3 dakika beklemesini sağlar.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app"]