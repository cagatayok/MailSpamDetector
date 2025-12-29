import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Flask importları
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


class EmailSpamDetector:
    """Gerçek veri seti ile spam tespit eden makine öğrenimi sınıfı."""

    def __init__(self, model_type='naive_bayes'):
        """
                Args:
                    model_type: 'naive_bayes', 'logistic_regression', 'random_forest'
                """
        self.vectorizer = TfidfVectorizer(  #metin verilerinde hangi kelimelerin onemli oldugunu bulmak icin kullanilir
            max_features=3000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),   #1 ve 2 kelimeden olusan ifadeleri kullanir(unigram ve bigram)
            stop_words='english'
        )
        # Model seçimi
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':  #olasilikk hesaplamasi guclu,daha dengeli sonuclar verir
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':    #birden fazla karar agaci kullanir,karmasik iliskileri yakalar
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError("Geçersiz model tipi!")

        self.model_type = model_type
        self.is_trained = False

    def preprocess_email(self, text):               # makine öğrenmesi algoritmalarının daha verimli çalışabilmesi için
        text = str(text).lower()         # Tüm metinler küçük harfe dönüştürülmektedir.
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        text = re.sub(r'\d+', 'NUM', text)
        text = re.sub(r'[^\w\s]', ' ', text)    # Noktalama işaretleri kaldırılmakta ve gereksiz boşluklar temizlenmektedir.
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_features(self, emails):
        """E-postalardan özellik çıkarımı"""  # Makine öğrenmesi algoritmaları metin verilerini doğrudan anlayamaz.
        # sayısal vektorlere donusturulur
        processed_emails = [self.preprocess_email(email) for email in emails]
        if not self.is_trained:
            features = self.vectorizer.fit_transform(processed_emails)
        else:
            features = self.vectorizer.transform(processed_emails)
        return features

    def load_dataset(self, source='uci'):    #********* VERİ SETİ YÜKLENİR*************

        """
                Veri setini yükle
                Args:
                    source: 'uci' (otomatik indir) veya 'local' (yerel dosya)
                """
        print("Veri seti yükleniyor...")

        if source == 'uci':
            try:

                # UCI'den otomatik indirme
                from ucimlrepo import fetch_ucirepo
                sms_spam = fetch_ucirepo(id=228)
                X = sms_spam.data.features['sms'].values # mesaj girdisi
                y = sms_spam.data.targets['label'].values # mesaj ciktisi
                # spam/ham -> 1/0 dönüşümü makine ögrenme stringle değil sayısal olmalı
                y = np.array([1 if label == 'spam' else 0 for label in y])
                print(f"✓ {len(X)} mesaj UCI'den yüklendi.")
            except Exception as e: # hata yakalanırsa kod devam eder alternatif veri kaynagına yonlendırme
                print(f"UCI Hatası: {e}")
                return self._load_from_kaggle()
        elif source == 'local':
            # Yerel CSV dosyasından yükleme
            df = pd.read_csv('spam.csv', encoding='latin-1') # Bu dosyayı Latin-1 karakter kodlamasıyla oku
            df = df[['v1', 'v2']]
            df.columns = ['label', 'message']
            X = df['message'].values
            y = df['label'].map({'spam': 1, 'ham': 0}).values
            print(f"✓ {len(X)} mesaj yerel dosyadan yüklendi.")
        return X, y

    def _load_from_kaggle(self):
        """Kaggle'dan alternatif yükleme"""
        try:
            url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
            df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
            X = df['message'].values
            y = df['label'].map({'spam': 1, 'ham': 0}).values
            print(f"✓ {len(X)} mesaj GitHub'dan yüklendi.")
            return X, y
        except Exception as e:
            print(f"Alternatif kaynak hatası: {e}")
            raise Exception("Veri seti yüklenemedi!")

    def train_with_dataset(self, test_size=0.2, show_details=True):
        """Gerçek veri seti ile modeli eğit"""
        # Veri setini yükle
        X, y = self.load_dataset()

        # Veri analizi
        if show_details:
            self._show_dataset_stats(X, y)

        # Eğitim/test ayrımı
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\n🔄 Model eğitiliyor (model: {self.model_type})...")
        # Özellikleri çıkar
        X_train_features = self.extract_features(X_train)
        self.is_trained = True
        X_test_features = self.extract_features(X_test)

        # Modeli eğit
        self.model.fit(X_train_features, y_train)

        # Tahmin yap
        y_pred = self.model.predict(X_test_features)
        # Sonuçları göster
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANSI")
        print("=" * 70)
        print(f"Doğruluk: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("\nDetaylı Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

        cm = confusion_matrix(y_test, y_pred)
        print("\nKarışıklık Matrisi:")
        print(f"                Tahmin")
        print(f"              Ham    Spam")
        print(f"Gerçek Ham   {cm[0][0]:4d}   {cm[0][1]:4d}")
        print(f"       Spam  {cm[1][0]:4d}   {cm[1][1]:4d}")

        if show_details:
            print("\n🔄 Cross-validation yapılıyor...")
            cv_scores = cross_val_score(self.model, self.extract_features(X), y, cv=5)
            print(f"CV Skorları: {cv_scores}")
            print(f"Ortalama CV Skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return accuracy

    def _show_dataset_stats(self, X, y):
        print("\n" + "=" * 70)
        print("📈 VERİ SETİ ANALİZİ")
        print("=" * 70)
        print(f"Toplam mesaj: {len(X)}")
        print(f"Spam mesaj: {sum(y)} (%{sum(y) / len(y) * 100:.1f})")
        print(f"Normal mesaj: {len(y) - sum(y)} (%{(len(y) - sum(y)) / len(y) * 100:.1f})")

        spam_lengths = [len(X[i]) for i in range(len(X)) if y[i] == 1]
        ham_lengths = [len(X[i]) for i in range(len(X)) if y[i] == 0]

        print("\nOrtalama mesaj uzunlukları:")
        print(f"  Spam: {np.mean(spam_lengths):.0f} karakter")
        print(f"  Normal: {np.mean(ham_lengths):.0f} karakter")

        print("\n📧 Örnek Spam Mesajı:")
        for msg in [X[i] for i in range(len(X)) if y[i] == 1][:2]:
            print(" -", msg[:120], "...")

        print("\n📧 Örnek Normal Mesaj:")
        for msg in [X[i] for i in range(len(X)) if y[i] == 0][:2]:
            print(" -", msg[:120], "...")

    def predict(self, email):
        if not self.is_trained:
            raise Exception("Model eğitilmedi!")

        features = self.extract_features([email])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        return {
            'is_spam': bool(prediction),
            'spam_probability': float(probability[1]),
            'normal_probability': float(probability[0]),
            'confidence': float(max(probability))
        }

    def predict_batch(self, emails):
        if not self.is_trained:
            raise Exception("Model eğitilmedi!")

        features = self.extract_features(emails)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        results = []
        for i, email in enumerate(emails):
            results.append({
                'email': email[:100] + '...' if len(email) > 100 else email,
                'is_spam': bool(predictions[i]),
                'spam_probability': float(probabilities[i][1]),
                'confidence': float(max(probabilities[i]))
            })
        return results

    def save_model(self, filepath='real_spam_model.pkl'):
        if not self.is_trained:
            raise Exception("Model henüz eğitilmedi!")

        with open(filepath, 'wb') as f:                     # Bu fonksiyon eğitilmiş modeli kalıcı hale getirir
#Yani “öğrenilen her şeyi dosyaya yaz ve sakla” işi yapar.
#Bu fonksiyon, eğitilmiş spam tespit modelini .pkl dosyası olarak diske kaydeder.
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }, f)
        print(f"\n✓ Model başarıyla kaydedildi: {filepath}")

    def load_model(self, filepath='real_spam_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']              # EĞTİLEN DOSYAMIZI AÇAR VE GEREKLİ ÖZELLİKLERİNİ ALARAK BELLEĞE YERLEŞTİRİR.
                                                    # MODELİ HER SEFERİNDE EĞİTMEK YERİNE,
                                                    # HAZIR EĞİTİLMİŞ MODELİN NASIL YÜKLENECEĞİNİ TANIMLAR.
            self.model_type = data['model_type']
            self.is_trained = data['is_trained']
        print(f"✓ Model yüklendi: {filepath}")


# ============================================================================
# FLASK API UYGULAMASI
# ============================================================================

app = Flask(__name__)             #/////////////////// KODU APİ İLE BİRLİKTE BACKEND SUNUCUSUNA ÇEVİRİYOR **********************
CORS(app)  # React'ten gelen isteklere izin ver

# Global model değişkeni
detector = None
MODEL_PATH = 'spam_model_real.pkl'


def init_model():
    """Uygulama başlarken modeli yükle veya eğit"""
    global detector

    print("\n" + "=" * 70)
    print("🚀 SPAM DETECTOR API BAŞLATILIYOR")
    print("=" * 70)

    detector = EmailSpamDetector(model_type='naive_bayes')

    # Eğer kaydedilmiş model varsa onu yükle
    if os.path.exists(MODEL_PATH):
        try:                                              # APİ AYAĞA KALKARKEN MODELİ KONTROL EDER MODEL VAR MI EĞİTİLMİŞ Mİ
            print(f"\n📦 Kaydedilmiş model bulundu: {MODEL_PATH}")
            detector.load_model(MODEL_PATH)
            print("✅ Model başarıyla yüklendi!")
        except Exception as e:
            print(f"❌ Model yüklenemedi: {e}")
            print("🔄 Yeni model eğitiliyor...")
            train_new_model()    # model yoksa model eğitme fonksiyonunu çağırıyor
    else:
        print("\n⚠️  Kaydedilmiş model bulunamadı.")
        print("🔄 Yeni model eğitiliyor...")
        train_new_model()

    print("\n✅ API hazır!")
    print("=" * 70)


def train_new_model():                      # ********MODEL EĞİTME FONKSİYONU**********
    """Yeni model eğit ve kaydet"""
    global detector
    try:
        detector.train_with_dataset(test_size=0.2, show_details=True)     # ***** Model eğitiliyor demek:
        # #Bilgisayarın, spam ve normal e-postalarda hangi kelimelerin ne sıklıkla geçtiğini
        # #öğrenmesi ve bu bilgiyi sayısal kurallar haline getirmesi demektir.***************
        detector.save_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Model eğitimi başarısız: {e}")
        raise


@app.route('/')
def home():
    """API ana sayfası"""
    return jsonify({
        'message': 'Spam Detector API',
        'version': '1.0',                               # APİYE TARAYICIDAN GİRİLİNCE BİLGİLENDİRME DÖNER
        'endpoints': {
            '/api/analyze': 'POST - Tek mesaj analizi',
            '/api/batch': 'POST - Toplu mesaj analizi',
            '/api/status': 'GET - Model durumu',
            '/api/retrain': 'POST - Modeli yeniden eğit'
        }
    })


@app.route('/api/status')
def status():
    """Model durumunu kontrol et"""
    if detector and detector.is_trained:
        return jsonify({
            'status': 'ready',
            'model_type': detector.model_type,              # MODELİN DURUMUNU KONTOL EDER
            'is_trained': detector.is_trained
        })
    else:
        return jsonify({
            'status': 'not_ready',
            'message': 'Model eğitilmedi'
        }), 503


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Tek bir e-posta mesajını analiz et"""
    try:
        data = request.json

        if not data or 'email' not in data:                     # TEK BİR E POSTANIN SPAM OLUP OLMADIĞINI TAHMİN EDER
            return jsonify({
                'error': 'Email metni gerekli',
                'success': False
            }), 400

        email_text = data['email']

        if not email_text.strip():
            return jsonify({
                'error': 'Email metni boş olamaz',
                'success': False
            }), 400

        # Model kontrolü
        if not detector or not detector.is_trained:
            return jsonify({
                'error': 'Model henüz eğitilmedi',
                'success': False
            }), 503

        # Tahmin yap
        result = detector.predict(email_text)
        result['success'] = True                    # GİRİLEN E POSTAYA ÇIKARILAN SONUÇLARA GÖRE TAHMİN YAPAR
        result['timestamp'] = pd.Timestamp.now().isoformat()

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Birden fazla e-postayı analiz et"""              # BURADA DA BİRDEN FAZLA E POSTAYI ANALİZ EDER
    try:
        data = request.json

        if not data or 'emails' not in data:
            return jsonify({
                'error': 'Emails listesi gerekli',
                'success': False
            }), 400

        emails = data['emails']

        if not isinstance(emails, list):
            return jsonify({
                'error': 'Emails bir liste olmalı',
                'success': False
            }), 400

        if len(emails) == 0:
            return jsonify({
                'error': 'Email listesi boş',
                'success': False
            }), 400

        if len(emails) > 100:
            return jsonify({
                'error': 'Maksimum 100 email analiz edilebilir',
                'success': False
            }), 400

        # Model kontrolü
        if not detector or not detector.is_trained:
            return jsonify({
                'error': 'Model henüz eğitilmedi',
                'success': False
            }), 503

        # Toplu tahmin
        results = detector.predict_batch(emails)

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Modeli yeniden eğit"""
    try:
        print("\n🔄 Model yeniden eğitiliyor...")            # MODELİ MANUEL OLARAK EĞİTİLMESİNE OLANAK SAĞLAR
        train_new_model()

        return jsonify({
            'success': True,
            'message': 'Model başarıyla yeniden eğitildi',
            'model_type': detector.model_type
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint bulunamadı',             # HATA YÖNETİMİ YAPILIR
        'success': False
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Sunucu hatası',
        'success': False
    }), 500


# ============================================================================
# UYGULAMA BAŞLATMA
# ============================================================================

if __name__ == "__main__":
    # Modeli başlat
    init_model()

    # Flask sunucusunu başlat
    print("\n🌐 Flask sunucusu başlatılıyor...")
    print("📍 URL: http://localhost:5000")
    print("🛑 Durdurmak için: CTRL+C\n")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Model iki kez yüklenmesin
    )