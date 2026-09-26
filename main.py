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
    "Haftalık": True,
}

CCI_PERIYOT = 20
RSI_PERIYOT = 14

# --- FİLTRE AKTİFLİK AYARLARI ---
HACIM_FILTRESI_AKTIF = True
HACIM_ORT_PERIYOT = 10
ICHIMOKU_FILTRESI_AKTIF = True

# Telegram Bildirim Ayarları
TELEGRAM_AKTIF = True
TELEGRAM_BOT_TOKEN = "8759153930:AAFcMm17a12TSWhIoMOtGq2s_K3IZKKsuS8"
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
    "Günlük": {"interval": "1d", "limit": 150},
    "Haftalık": {"interval": "1w", "limit": 150}
}

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
                if not data or len(data) < 100:
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

            # --- 3. ICHIMOKU (Kijun 52) HESAPLAMA ---
            tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            kijun_sen = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2

            curr_kijun = float(kijun_sen.iloc[-1])
            close_prev = float(df['Close'].iloc[-2])

            # A) Fiyat Kriteri: Fiyat Kijun-52'nin üstünde olacak ama en fazla %10 yukarısında olacak
            fiyat_kriteri = (curr_kijun < close_curr) and (close_curr <= curr_kijun * 1.10)

            # B) Chikou Span & 52 Periyotluk Geçmiş Kijun Kriteri 
            # (En fazla %2 aşağıda, en fazla %1.5 yukarıda olabilir)
            if len(df) > 78:
                gecmis_kijun = float(kijun_sen.iloc[-26])
                
                alt_limit_chikou = gecmis_kijun * 0.98   # %2 aşağısı
                ust_limit_chikou = gecmis_kijun * 1.015  # %1.5 yukarısı
                
                chikou_kijun_kosulu = alt_limit_chikou <= close_curr <= ust_limit_chikou
            else:
                chikou_kijun_kosulu = True

            # --- KOŞUL KONTROLLERİ ---
            cci_kosulu = (curr_cci > -100) and (curr_cci > prev_cci)
            if not cci_kosulu:
                continue

            tam_kesisim = (curr_rsi > 70) and (prev_rsi <= 70)
            bir_mum_once_gecti = (curr_rsi > 70) and (prev_rsi > 70) and (prev_prev_rsi <= 70)
            if not (tam_kesisim or bir_mum_once_gecti):
                continue

            if ICHIMOKU_FILTRESI_AKTIF and not (fiyat_kriteri and chikou_kijun_kosulu):
                continue

            if HACIM_FILTRESI_AKTIF:
                vol_sma = df['Volume'].rolling(window=HACIM_ORT_PERIYOT).mean()
                if curr_vol <= float(vol_sma.iloc[-1]):
                    continue

            bilgi = {
                'Zaman Dilimi': periyot_adi,
                'Coin': ticker,
                'Son Kapanis': round(close_curr, 4),
                'Kijun-Sen (52)': round(curr_kijun, 4),
                'Son RSI': round(curr_rsi, 2),
                'Son CCI': round(curr_cci, 2),
                'Tarih/Saat': str(df.index[-1])
            }
            results.append(bilgi)

            tv_link = f"https://www.tradingview.com/chart/?symbol=BINANCE:{ticker}.P"
            msg = (
                f"🚀 *ÖZEL İCHİMOKU BAND SİNYALİ*\n"
                f"*Coin:* `{ticker}`\n"
                f"*Periyot:* {periyot_adi}\n"
                f"*Fiyat:* {close_curr}\n"
                f"*Chikou Bant (-%2 ile +%1.5):* Uygun 🎯\n"
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
    df_results.to_excel("Binance_Ozel_Bant_Sonuclari.xlsx", index=False)
    print(f"\n✅ Toplam {len(results)} coin filtrelere ulaştı ve Excel'e kaydedildi.")
else:
    print("\n⚠️ Filtrelere uyan kripto para bulunamadı.")
