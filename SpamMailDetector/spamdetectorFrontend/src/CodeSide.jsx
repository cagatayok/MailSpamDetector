import React, { useState, useEffect } from 'react';
import { Mail, Shield, AlertTriangle, CheckCircle, BarChart3, Trash2, Send, RefreshCw, Loader, Server } from 'lucide-react';

const SpamDetectorUI = () => {
  const [emailText, setEmailText] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [stats, setStats] = useState({
    total: 0,
    spam: 0,
    normal: 0
  });

  const API_URL = 'http://localhost:5000';

  useEffect(() => {
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/status`);
      if (response.ok) {
        const data = await response.json();
        setApiStatus(data.status === 'ready' ? 'connected' : 'loading');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('disconnected');
    }
  };

  const analyzeEmail = async () => {
    if (!emailText.trim()) return;

    if (apiStatus !== 'connected') {
      alert('API bağlantısı kurulamadı. Lütfen Flask sunucusunun çalıştığından emin olun.');
      return;
    }

    setIsAnalyzing(true);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: emailText })
      });

      if (!response.ok) {
        throw new Error(`API Hatası: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        const newResult = {
          id: Date.now(),
          text: emailText,
          isSpam: data.is_spam,
          spamProbability: data.spam_probability,
          normalProbability: data.normal_probability,
          confidence: data.confidence,
          timestamp: new Date().toLocaleString('tr-TR')
        };

        setResult(newResult);
        setHistory(prev => [newResult, ...prev].slice(0, 10));
        setStats(prev => ({
          total: prev.total + 1,
          spam: prev.spam + (data.is_spam ? 1 : 0),
          normal: prev.normal + (data.is_spam ? 0 : 1)
        }));
      } else {
        alert('Analiz başarısız: ' + (data.error || 'Bilinmeyen hata'));
      }

    } catch (error) {
      console.error('API Error:', error);
      alert('API ile bağlantı kurulamadı:\n' + error.message + '\n\nFlask sunucusunun çalıştığından emin olun.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    setStats({ total: 0, spam: 0, normal: 0 });
  };

  const exampleEmails = [
    "WINNER!! You have been selected to receive £1000 cash or a £2000 prize!",
    "Hey, are you free for lunch tomorrow?",
    "URGENT! Your mobile number has won a prize worth £5000!",
    "Can you send me the project report by Friday?",
    "FREE entry in weekly competition! Text WIN to 85233",
    "Meeting rescheduled to 3pm. See you there."
  ];

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'connected': return '#10b981';
      case 'loading': return '#f59e0b';
      case 'checking': return '#3b82f6';
      case 'disconnected': return '#ef4444';
      case 'error': return '#f97316';
      default: return '#6b7280';
    }
  };

  const getStatusText = () => {
    switch (apiStatus) {
      case 'connected': return 'Bağlı';
      case 'loading': return 'Model Yükleniyor';
      case 'checking': return 'Kontrol Ediliyor';
      case 'disconnected': return 'Bağlantı Yok';
      case 'error': return 'Hata';
      default: return 'Bilinmiyor';
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.maxWidth}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerTitle}>
            <Shield size={48} color="#2563eb" />
            <h1 style={styles.title}>Spam Tespit Sistemi</h1>
          </div>
          <p style={styles.subtitle}>Makine öğrenimi ile gelişmiş e-posta analizi</p>
          
          {/* API Status Badge */}
          <div style={styles.statusBadge}>
            <div style={{...styles.statusDot, backgroundColor: getStatusColor()}}></div>
            <span style={styles.statusText}>
              API Durumu: {getStatusText()}
            </span>
            <button onClick={checkApiStatus} style={styles.refreshButton} title="Yenile">
              <RefreshCw size={16} color="#4b5563" />
            </button>
          </div>

          {/* Connection Warning */}
          {apiStatus === 'disconnected' && (
            <div style={styles.warningBox}>
              <div style={styles.warningContent}>
                <Server size={20} color="#dc2626" style={styles.warningIcon} />
                <div style={styles.warningText}>
                  <h3 style={styles.warningTitle}>Flask API Bağlantısı Kurulamadı</h3>
                  <p style={styles.warningDescription}>
                    Lütfen Flask sunucusunun çalıştığından emin olun:
                  </p>
                  <code style={styles.codeBlock}>
                    python spam_detector_api.py
                  </code>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Stats Cards */}
        <div style={styles.statsGrid}>
          <div style={{...styles.statCard, borderLeft: '4px solid #3b82f6'}}>
            <div style={styles.statContent}>
              <div>
                <p style={styles.statLabel}>Toplam Analiz</p>
                <p style={styles.statValue}>{stats.total}</p>
              </div>
              <BarChart3 size={40} color="#3b82f6" />
            </div>
          </div>

          <div style={{...styles.statCard, borderLeft: '4px solid #ef4444'}}>
            <div style={styles.statContent}>
              <div>
                <p style={styles.statLabel}>Spam Tespit</p>
                <p style={styles.statValue}>{stats.spam}</p>
              </div>
              <AlertTriangle size={40} color="#ef4444" />
            </div>
          </div>

          <div style={{...styles.statCard, borderLeft: '4px solid #10b981'}}>
            <div style={styles.statContent}>
              <div>
                <p style={styles.statLabel}>Normal E-posta</p>
                <p style={styles.statValue}>{stats.normal}</p>
              </div>
              <CheckCircle size={40} color="#10b981" />
            </div>
          </div>
        </div>

        <div style={styles.mainGrid}>
          {/* Main Analysis Panel */}
          <div style={styles.mainPanel}>
            {/* Input Area */}
            <div style={styles.card}>
              <h2 style={styles.cardTitle}>
                <Mail size={20} color="#2563eb" />
                E-posta Metni
              </h2>
              
              <textarea
                value={emailText}
                onChange={(e) => setEmailText(e.target.value)}
                placeholder="E-posta metnini buraya yapıştırın..."
                style={styles.textarea}
                disabled={isAnalyzing}
              />

              <div style={styles.buttonGroup}>
                <button
                  onClick={analyzeEmail}
                  disabled={isAnalyzing || !emailText.trim() || apiStatus !== 'connected'}
                  style={{
                    ...styles.primaryButton,
                    opacity: (isAnalyzing || !emailText.trim() || apiStatus !== 'connected') ? 0.5 : 1,
                    cursor: (isAnalyzing || !emailText.trim() || apiStatus !== 'connected') ? 'not-allowed' : 'pointer'
                  }}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader size={16} style={styles.spinner} />
                      Analiz Ediliyor...
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      Analiz Et
                    </>
                  )}
                </button>

                <button
                  onClick={() => setEmailText('')}
                  disabled={isAnalyzing}
                  style={{
                    ...styles.secondaryButton,
                    opacity: isAnalyzing ? 0.5 : 1,
                    cursor: isAnalyzing ? 'not-allowed' : 'pointer'
                  }}
                >
                  Temizle
                </button>
              </div>

              {/* Example Emails */}
              <div style={styles.examplesSection}>
                <p style={styles.examplesLabel}>Örnek e-postalar:</p>
                <div style={styles.examplesGrid}>
                  {exampleEmails.map((email, idx) => (
                    <button
                      key={idx}
                      onClick={() => setEmailText(email)}
                      disabled={isAnalyzing}
                      style={{
                        ...styles.exampleButton,
                        opacity: isAnalyzing ? 0.5 : 1,
                        cursor: isAnalyzing ? 'not-allowed' : 'pointer'
                      }}
                    >
                      {email.substring(0, 60)}...
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Result Display */}
            {result && (
              <div style={{...styles.card, ...styles.fadeIn}}>
                <h2 style={styles.cardTitle}>Analiz Sonucu</h2>
                
                <div style={{
                  ...styles.resultBox,
                  backgroundColor: result.isSpam ? '#fef2f2' : '#f0fdf4',
                  border: result.isSpam ? '2px solid #fecaca' : '2px solid #bbf7d0'
                }}>
                  <div style={styles.resultHeader}>
                    {result.isSpam ? (
                      <>
                        <AlertTriangle size={32} color="#dc2626" />
                        <div>
                          <h3 style={{...styles.resultTitle, color: '#dc2626'}}>SPAM TESPİT EDİLDİ</h3>
                          <p style={{...styles.resultSubtitle, color: '#b91c1c'}}>Bu e-posta spam olarak işaretlendi</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <CheckCircle size={32} color="#16a34a" />
                        <div>
                          <h3 style={{...styles.resultTitle, color: '#16a34a'}}>NORMAL E-POSTA</h3>
                          <p style={{...styles.resultSubtitle, color: '#15803d'}}>Bu e-posta güvenli görünüyor</p>
                        </div>
                      </>
                    )}
                  </div>

                  <div style={styles.progressSection}>
                    <div style={styles.progressItem}>
                      <div style={styles.progressLabelRow}>
                        <span style={styles.progressLabel}>Spam Olasılığı</span>
                        <span style={{...styles.progressValue, color: '#dc2626'}}>
                          {(result.spamProbability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={styles.progressBar}>
                        <div style={{
                          ...styles.progressFill,
                          width: `${result.spamProbability * 100}%`,
                          backgroundColor: '#ef4444'
                        }} />
                      </div>
                    </div>

                    <div style={styles.progressItem}>
                      <div style={styles.progressLabelRow}>
                        <span style={styles.progressLabel}>Normal Olasılığı</span>
                        <span style={{...styles.progressValue, color: '#16a34a'}}>
                          {(result.normalProbability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={styles.progressBar}>
                        <div style={{
                          ...styles.progressFill,
                          width: `${result.normalProbability * 100}%`,
                          backgroundColor: '#10b981'
                        }} />
                      </div>
                    </div>

                    <div style={styles.confidenceSection}>
                      <div style={styles.progressLabelRow}>
                        <span style={styles.progressLabel}>Güven Skoru</span>
                        <span style={{...styles.progressValue, color: '#2563eb'}}>
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* History Sidebar */}
          <div style={styles.sidebar}>
            <div style={styles.card}>
              <div style={styles.sidebarHeader}>
                <h2 style={styles.cardTitle}>Geçmiş</h2>
                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    style={styles.deleteButton}
                    title="Geçmişi temizle"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>

              <div style={styles.historyList}>
                {history.length === 0 ? (
                  <p style={styles.emptyHistory}>Henüz analiz yapılmadı</p>
                ) : (
                  history.map((item) => (
                    <div
                      key={item.id}
                      style={{
                        ...styles.historyItem,
                        backgroundColor: item.isSpam ? '#fef2f2' : '#f0fdf4',
                        border: item.isSpam ? '2px solid #fecaca' : '2px solid #bbf7d0'
                      }}
                      onClick={() => setEmailText(item.text)}
                    >
                      <div style={styles.historyContent}>
                        {item.isSpam ? (
                          <AlertTriangle size={16} color="#dc2626" style={styles.historyIcon} />
                        ) : (
                          <CheckCircle size={16} color="#16a34a" style={styles.historyIcon} />
                        )}
                        <p style={styles.historyText}>
                          {item.text}
                        </p>
                      </div>
                      <div style={styles.historyFooter}>
                        <span>{item.timestamp}</span>
                        <span style={styles.historyConfidence}>
                          {(item.confidence * 100).toFixed(0)}% güven
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={styles.footer}>
          <p>💡 Bu uygulama gerçek makine öğrenimi modeli kullanarak spam tespiti yapar</p>
          <p style={{marginTop: '4px'}}>Model: Naive Bayes | Dataset: 5000+ gerçek mesaj</p>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #dbeafe 0%, #e9d5ff 50%, #fce7f3 100%)',
    padding: '24px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  },
  maxWidth: {
    maxWidth: '1280px',
    margin: '0 auto'
  },
  header: {
    textAlign: 'center',
    marginBottom: '32px'
  },
  headerTitle: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    marginBottom: '16px'
  },
  title: {
    fontSize: '36px',
    fontWeight: 'bold',
    color: '#1f2937',
    margin: 0
  },
  subtitle: {
    color: '#4b5563',
    fontSize: '16px',
    margin: 0
  },
  statusBadge: {
    marginTop: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px'
  },
  statusDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    animation: 'pulse 2s ease-in-out infinite'
  },
  statusText: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151'
  },
  refreshButton: {
    marginLeft: '8px',
    padding: '4px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    transition: 'background-color 0.15s'
  },
  warningBox: {
    marginTop: '16px',
    marginLeft: 'auto',
    marginRight: 'auto',
    maxWidth: '672px',
    backgroundColor: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
    padding: '16px'
  },
  warningContent: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '12px'
  },
  warningIcon: {
    flexShrink: 0,
    marginTop: '2px'
  },
  warningText: {
    textAlign: 'left'
  },
  warningTitle: {
    fontWeight: '600',
    color: '#991b1b',
    marginBottom: '4px',
    fontSize: '14px',
    margin: '0 0 4px 0'
  },
  warningDescription: {
    fontSize: '14px',
    color: '#b91c1c',
    marginBottom: '8px',
    margin: '0 0 8px 0'
  },
  codeBlock: {
    fontSize: '12px',
    backgroundColor: '#fee2e2',
    padding: '4px 8px',
    borderRadius: '4px',
    display: 'block',
    fontFamily: 'monospace'
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
    marginBottom: '24px'
  },
  statCard: {
    backgroundColor: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    padding: '24px'
  },
  statContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between'
  },
  statLabel: {
    color: '#6b7280',
    fontSize: '14px',
    margin: '0 0 4px 0'
  },
  statValue: {
    fontSize: '30px',
    fontWeight: 'bold',
    color: '#1f2937',
    margin: 0
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr',
    gap: '24px',
    marginBottom: '32px'
  },
  mainPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px'
  },
  sidebar: {
    display: 'flex',
    flexDirection: 'column'
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    padding: '24px'
  },
  cardTitle: {
    fontSize: '20px',
    fontWeight: '600',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    margin: '0 0 16px 0'
  },
  textarea: {
    width: '100%',
    height: '192px',
    padding: '16px',
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    fontSize: '14px',
    fontFamily: 'inherit',
    resize: 'none',
    transition: 'border-color 0.15s',
    boxSizing: 'border-box'
  },
  buttonGroup: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '16px'
  },
  primaryButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '12px 24px',
    borderRadius: '8px',
    border: 'none',
    fontWeight: '500',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background-color 0.15s'
  },
  secondaryButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: '#e5e7eb',
    color: '#374151',
    padding: '12px 24px',
    borderRadius: '8px',
    border: 'none',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background-color 0.15s'
  },
  spinner: {
    animation: 'spin 1s linear infinite'
  },
  examplesSection: {
    marginTop: '24px'
  },
  examplesLabel: {
    fontSize: '14px',
    color: '#4b5563',
    marginBottom: '8px',
    margin: '0 0 8px 0'
  },
  examplesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '8px'
  },
  exampleButton: {
    textAlign: 'left',
    fontSize: '14px',
    padding: '12px',
    backgroundColor: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.15s'
  },
  fadeIn: {
    animation: 'fadeIn 0.3s ease-out'
  },
  resultBox: {
    padding: '24px',
    borderRadius: '12px'
  },
  resultHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px'
  },
  resultTitle: {
    fontSize: '24px',
    fontWeight: 'bold',
    margin: '0 0 4px 0'
  },
  resultSubtitle: {
    fontSize: '14px',
    margin: 0
  },
  progressSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px'
  },
  progressItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  },
  progressLabelRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '4px'
  },
  progressLabel: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151'
  },
  progressValue: {
    fontSize: '14px',
    fontWeight: 'bold'
  },
  progressBar: {
    width: '100%',
    backgroundColor: '#e5e7eb',
    borderRadius: '9999px',
    height: '12px',
    overflow: 'hidden'
  },
  progressFill: {
    height: '12px',
    borderRadius: '9999px',
    transition: 'width 0.5s ease-in-out'
  },
  confidenceSection: {
    paddingTop: '12px',
    borderTop: '1px solid #d1d5db'
  },
  sidebarHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '16px'
  },
  deleteButton: {
    color: '#dc2626',
    padding: '8px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    transition: 'background-color 0.15s'
  },
  historyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    maxHeight: '600px',
    overflowY: 'auto'
  },
  emptyHistory: {
    color: '#9ca3af',
    textAlign: 'center',
    padding: '32px 0',
    margin: 0
  },
  historyItem: {
    padding: '16px',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'box-shadow 0.15s'
  },
  historyContent: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    marginBottom: '8px'
  },
  historyIcon: {
    flexShrink: 0,
    marginTop: '4px'
  },
  historyText: {
    fontSize: '14px',
    color: '#374151',
    margin: 0,
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden'
  },
  historyFooter: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '12px',
    color: '#6b7280',
    marginTop: '8px'
  },
  historyConfidence: {
    fontWeight: '500'
  },
  footer: {
    textAlign: 'center',
    fontSize: '14px',
    color: '#4b5563'
  }
};

export default SpamDetectorUI;