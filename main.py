import time
import warnings
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- KRİPTO AYARLARI ---
TARAMA_YAPILACAK_PERIYOTLAR = {
    "30 Dakikalık": True,
    "1 Saatlik": True,
    "4 Saatlik": True,
    "Günlük": True,
}

CCI_PERIYOT = 20
RSI_PERIYOT = 14

# --- FİLTRE AKTİFLİK AYARLARI ---
HACIM_FILTRESI_AKTIF = True
HACIM_ORT_PERIYOT = 10
ICHIMOKU_FILTRESI_AKTIF = True  # Ichimoku Bulut Filtresi

# Telegram Bildirim Ayarları
TELEGRAM_AKTIF = True
TELEGRAM_BOT_TOKEN = "8555013735:AAF_kuUHuqqrf7kD_GhXQ2s27CCM2HsXc0M"
TELEGRAM_CHAT_ID = "889982961"

def telegram_mesaj_gonder(mesaj):
    if not TELEGRAM_AKTIF:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mesaj,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        requests.post(url, json=payload, timeout=5)
        time.sleep(0.3)
    except Exception as e:
        print(f"Telegram mesajı gönderilemedi: {e}")

PERIYOT_AYARLARI = {
    "30 Dakikalık": {"interval": "30m", "limit": 150},
    "1 Saatlik": {"interval": "1h", "limit": 150},
    "4 Saatlik": {"interval": "4h", "limit": 150},
    "Günlük": {"interval": "1d", "limit": 150}
}

# Parite Çekme Fonksiyonu
def binance_aktif_usdt_listesini_getir():
    urls = [
        "https://api.binance.com/api/v3/exchangeInfo",
        "https://data-api.binance.vision/api/v3/exchangeInfo",
        "https://api1.binance.com/api/v3/exchangeInfo"
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
                if symbols:
                    return symbols
        except Exception:
            continue
    return []

tickers = binance_aktif_usdt_listesini_getir()
print(f"✅ Binance'ten toplam {len(tickers)} adet aktif USDT paritesi çekildi.")

# 🚀 Mum Çekme Fonksiyonu
def binance_klines_cek(symbol, interval, limit=150):
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://api1.binance.com/api/v3/klines"
    ]
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=4)
            if response.status_code == 200:
                data = response.json()
                min_gerekli = 90 # 52 + 26 + pay
                if not data or len(data) < min_gerekli:
                    return None
                df = pd.DataFrame(data, columns=[
                    'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                    'Taker_buy_base_asset', 'Taker_buy_quote_asset', 'Ignore'
                ])
                df = df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                df['Timestamp'] = pd.to_datetime(df['Open_time'], unit='ms')
                df.set_index('Timestamp', inplace=True)
                return df
        except Exception:
            continue
    return None

results = []

for periyot_adi, aktif_mi in TARAMA_YAPILACAK_PERIYOTLAR.items():
    if not aktif_mi:
        continue

    print(f"\n🔍 '{periyot_adi}' periyodu için tarama başlatılıyor...")
    basarili_sayisi = 0

    for ticker in tqdm(tickers, desc=f"{periyot_adi} Taranıyor"):
        df = binance_klines_cek(ticker, PERIYOT_AYARLARI[periyot_adi]["interval"], PERIYOT_AYARLARI[periyot_adi]["limit"])
        time.sleep(0.04)

        if df is None or df.empty:
            continue
        
        basarili_sayisi += 1

        try:
            curr_vol = float(df['Volume'].iloc[-1])
            if curr_vol == 0:
                continue

            close_curr = float(df['Close'].iloc[-1])

            # --- 1. CCI HESAPLAMA ---
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=CCI_PERIYOT).mean()
            mad = tp.rolling(window=CCI_PERIYOT).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            mad_safe = np.where(mad == 0, 0.0001, mad)
            cci = (tp - sma_tp) / (0.015 * mad_safe)
            curr_cci = float(cci.iloc[-1])
            prev_cci = float(cci.iloc[-2])

            # --- 2. RSI HESAPLAMA ---
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIYOT).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIYOT).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            curr_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2])
            prev_prev_rsi = float(rsi.iloc[-3])

            # --- 3. ICHIMOKU HESAPLAMA (Kijun 52) ---
            tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            kijun_sen = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2

            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()).shift(26) / 2

            curr_span_a = float(senkou_span_a.iloc[-1])
            curr_span_b = float(senkou_span_b.iloc[-1])
            bulut_ust = max(curr_span_a, curr_span_b)

            # --- CHİKOU SPAN VE 26 GÜN ÖNCEKİ BULUT KESİŞİM KONTROLÜ ---
            # Chikou Span (Geciken Çizgi) bugünkü fiyatın 26 mum geriye kaydırılmış halidir.
            # Dolayısıyla bugünkü Chikou seviyesi, 26 mum önceki mumun fiyatına (Close) eşittir.
            # 26 gün önceki bulutun üst seviyesini bulmak için, 26 mum önceki Senkou Span A ve B değerlerine bakılır.
            if len(df) > 52:
                gecmis_span_a = float(senkou_span_a.iloc[-26])
                gecmis_span_b = float(senkou_span_b.iloc[-26])
                gecmis_bulut_ust = max(gecmis_span_a, gecmis_span_b)
                gecmis_bulut_alt = min(gecmis_span_a, gecmis_span_b)

                chikou_degeri = close_curr # Güncel fiyat geriye yansıyan Chikou değeridir
                
                # Şartlar: Chikou 26 gün önceki bulutu yeni yukarı kesti VEYA bulut üst seviyesinin %5 aşağısı/yukarısı bandında
                tam_cikis = (chikou_degeri > gecmis_bulut_ust) and (float(df['Close'].iloc[-2]) <= max(float(senkou_span_a.iloc[-27]), float(senkou_span_b.iloc[-27])))
                band_orani = gecmis_bulut_ust * 0.05
                yuzde_bandi = (gecmis_bulut_ust - band_orani) <= chikou_degeri <= (gecmis_bulut_ust + band_orani)

                chikou_bulut_kosulu = tam_cikis or yuzde_bandi
            else:
                chikou_bulut_kosulu = True

            # --- KOŞUL KONTROLLERİ ---
            cci_kosulu = (curr_cci > -100) and (curr_cci > prev_cci)
            if not cci_kosulu:
                continue

            tam_kesisim = (curr_rsi > 70) and (prev_rsi <= 70)
            bir_mum_once_gecti = (curr_rsi > 70) and (prev_rsi > 70) and (prev_prev_rsi <= 70)
            if not (tam_kesisim or bir_mum_once_gecti):
                continue

            if ICHIMOKU_FILTRESI_AKTIF and (close_curr <= bulut_ust or not chikou_bulut_kosulu):
                continue

            if HACIM_FILTRESI_AKTIF:
                vol_sma = df['Volume'].rolling(window=HACIM_ORT_PERIYOT).mean()
                if curr_vol <= float(vol_sma.iloc[-1]):
                    continue

            bilgi = {
                'Zaman Dilimi': periyot_adi,
                'Coin': ticker,
                'Son Kapanis': round(close_curr, 4),
                'Bulut Ust Seviye': round(bulut_ust, 4),
                'Son RSI': round(curr_rsi, 2),
                'Son CCI': round(curr_cci, 2),
                'Tarih/Saat': str(df.index[-1])
            }
            results.append(bilgi)

            tv_link = f"https://www.tradingview.com/chart/?symbol=BINANCE:{ticker}.P"
            msg = (
                f"🚀 *İCHİMOKU & RSI & CHİKOU BULUT KESİŞİMİ*\n"
                f"*Coin:* `{ticker}`\n"
                f"*Periyot:* {periyot_adi}\n"
                f"*Fiyat:* {close_curr}\n"
                f"*Chikou Eski Bulutu Üstü Geçti:* Evet ☁️📈\n"
                f"*RSI:* {curr_rsi:.2f} (Önceki: {prev_rsi:.2f})\n"
                f"*CCI:* {curr_cci:.2f}\n\n"
                f"📈 [{ticker} Vadeli Grafiğini Aç]({tv_link})"
            )
            telegram_mesaj_gonder(msg)

        except Exception as e:
            pass

    print(f"\nℹ️ Başarıyla taranan geçerli coin sayısı: {basarili_sayisi}")

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=['Zaman Dilimi', 'Coin']).reset_index(drop=True)
    df_results.to_excel("Binance_Ichimoku_RSI_Sonuclari.xlsx", index=False)
    print(f"\n✅ Toplam {len(results)} coin filtrelere ulaştı ve Excel'e kaydedildi.")
else:
    print("\n⚠️ Filtrelere uyan kripto para bulunamadı.")
