from spam_detector import EmailSpamDetector

# Kaydedilmiş modeli yükle
detector = EmailSpamDetector()
detector.load_model('spam_model_real.pkl')

# Test et
result = detector.predict("Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/claim now to verify your info.")
print(result)
# {'is_spam': True, 'spam_probability': 0.98, ...}