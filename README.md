# 📧 Spam Detector API (Machine Learning + Flask)

Bu proje, **gerçek veri setleri** kullanılarak eğitilmiş bir **makine öğrenmesi tabanlı spam tespit sistemi**dir.  
Flask REST API olarak servis edilir ve tekil veya toplu e-posta / mesaj analizi yapabilir.

Model ilk çalıştırmada **otomatik olarak eğitilir** ve daha sonraki çalıştırmalarda kaydedilen model dosyası kullanılır.

---

## 🚀 Özellikler

- Gerçek dünya SMS / e-posta spam veri setleri
- TF-IDF + Machine Learning (Naive Bayes / Logistic Regression / Random Forest)
- Otomatik model eğitimi (ilk çalıştırmada)
- Eğitilmiş modeli `.pkl` olarak kaydetme
- Flask REST API
- Tekli analiz (`/api/analyze`)
- Toplu analiz (`/api/batch`)
- Model durumu kontrolü
- Manuel yeniden eğitme endpoint’i
- CORS destekli (React / Frontend uyumlu)

---

## 🧠 Kullanılan Teknolojiler

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- Flask-CORS

---

## 📊 Kullanılan Veri Seti

- UCI SMS Spam Collection
- Alternatif olarak GitHub üzerinden otomatik indirme
- Dataset repoya dahil değildir
- İlk çalıştırmada otomatik olarak indirilir

---

## 📁 Proje Yapısı
spam-detector-api/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model/
└── spam_model_real.pkl # İlk çalıştırmada otomatik oluşur


## ⚙️ Kurulum

### 1) Depoyu klonla
git clone https://github.com/KULLANICI_ADI/spam-detector-api.git
cd spam-detector-api

shell
Kodu kopyala

### 2) Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

shell
Kodu kopyala

### 3) Gerekli paketleri yükle
pip install -r requirements.txt

yaml
Kodu kopyala

---

## ▶️ Çalıştırma

python app.py

markdown
Kodu kopyala

İlk çalıştırmada sistem:
1. Veri setini indirir  
2. Modeli eğitir  
3. Eğitilmiş modeli `.pkl` olarak kaydeder  

Sonraki çalıştırmalarda model doğrudan yüklenir.

API adresi:
http://localhost:5000

yaml
Kodu kopyala

---

## 🔍 API Endpoint’leri

### Ana Sayfa
GET /

shell
Kodu kopyala

### Model Durumu
GET /api/status

shell
Kodu kopyala

### Tek Mesaj Analizi
POST /api/analyze

css
Kodu kopyala

Body:
```json
{
  "email": "Congratulations! You have won a free prize."
}
Response:

json
Kodu kopyala
{
  "is_spam": true,
  "spam_probability": 0.97,
  "normal_probability": 0.03,
  "confidence": 0.97,
  "success": true
}
Toplu Mesaj Analizi
bash
Kodu kopyala
POST /api/batch
Body:

json
Kodu kopyala
{
  "emails": [
    "Win money now!",
    "Hey, are we meeting tomorrow?"
  ]
}
Modeli Yeniden Eğit
bash
Kodu kopyala
POST /api/retrain
🧪 Model Seçenekleri
Kod içerisinde model kolayca değiştirilebilir:

python
Kodu kopyala
EmailSpamDetector(
    model_type='naive_bayes'  # logistic_regression, random_forest
)
📦 Model Dosyası Hakkında
.pkl dosyası repoya dahil edilmez

.gitignore ile hariç tutulur

Ortama özel olarak ilk çalıştırmada otomatik üretilir

🛡️ Lisans
Bu proje eğitim ve öğrenme amaçlıdır.
Ticari kullanım için veri seti lisanslarını kontrol ediniz.

İsterseniz backend dosyasından test.py den projeyi test edebilirsiniz ya da frontend klasörünü açarak oradan da test edebilirsiniz.

👤 Geliştirici
Çağatay
Machine Learning & Full Stack Development




