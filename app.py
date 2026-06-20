from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

def calculate_adx(df, period=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
    except Exception as e:
        print(f"ADX Error: {e}")
        return 25.0

def calculate_rsi(df, period=14):
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    except Exception as e:
        print(f"RSI Error: {e}")
        return 50.0

def calculate_atr(df, period=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0010
    except Exception as e:
        print(f"ATR Error: {e}")
        return 0.0010

def find_order_blocks(df):
    last = len(df) - 1
    ob_top, ob_bottom = 0, 0
    has_ob = False
    ob_type = "NONE"
    for i in range(last - 10, last - 2):
        candle = df.iloc[i]
        range_val = candle['High'] - candle['Low']
        if range_val == 0:
            continue
        if (candle['Close'] - candle['Open']) / range_val > 0.6:
            for j in range(i+1, min(i+5, last)):
                if df.iloc[j]['Close'] < candle['Low']:
                    ob_top, ob_bottom = candle['High'], candle['Low']
                    has_ob = True
                    ob_type = "BULLISH"
                    return has_ob, ob_top, ob_bottom, ob_type
        if (candle['Open'] - candle['Close']) / range_val > 0.6:
            for j in range(i+1, min(i+5, last)):
                if df.iloc[j]['Close'] > candle['High']:
                    ob_top, ob_bottom = candle['High'], candle['Low']
                    has_ob = True
                    ob_type = "BEARISH"
                    return has_ob, ob_top, ob_bottom, ob_type
    return has_ob, ob_top, ob_bottom, ob_type

def find_fvg(df):
    last = len(df) - 1
    fvg_top, fvg_bottom = 0, 0
    has_fvg = False
    for i in range(last - 10, last - 2):
        if df.iloc[i]['High'] < df.iloc[i+2]['Low']:
            fvg_top, fvg_bottom = df.iloc[i+2]['Low'], df.iloc[i]['High']
            has_fvg = True
            return has_fvg, fvg_top, fvg_bottom
        if df.iloc[i]['Low'] > df.iloc[i+2]['High']:
            fvg_top, fvg_bottom = df.iloc[i]['Low'], df.iloc[i+2]['High']
            has_fvg = True
            return has_fvg, fvg_top, fvg_bottom
    return has_fvg, fvg_top, fvg_bottom

def analyze_market(symbol, bid, ask):
    try:
        yf_symbol = symbol.replace("/", "") + "=X"
        if "USD" in yf_symbol:
            yf_symbol = yf_symbol.replace("USD", "USD=X")
        df = yf.download(yf_symbol, period="2h", interval="1m", progress=False)
        if df.empty:
            df = yf.download(yf_symbol, period="1d", interval="5m", progress=False)
        if df.empty:
            return {"action": "hold", "lot": 0.01, "sl": 0, "tp": 0, "score": 0}
        adx = calculate_adx(df)
        rsi = calculate_rsi(df)
        atr = calculate_atr(df)
        has_ob, ob_top, ob_bottom, ob_type = find_order_blocks(df)
        has_fvg, fvg_top, fvg_bottom = find_fvg(df)
        price = bid
        in_ob = (has_ob and ob_bottom <= price <= ob_top)
        in_fvg = (has_fvg and fvg_bottom <= price <= fvg_top)
        score = 0
        if in_ob:
            score += 2
        if in_fvg:
            score += 1
        if adx > 25:
            score += 1
        if rsi < 40 and rsi > 20:
            score += 1
        if rsi > 60 and rsi < 80:
            score += 1
        bullish_bias = (ob_type == "BULLISH" or (in_fvg and rsi < 40))
        bearish_bias = (ob_type == "BEARISH" or (in_fvg and rsi > 60))
        action = "hold"
        lot = 0.01
        sl = 0
        tp = 0
        risk_amount = 5.0
        risk_distance = atr * 0.5
        if score >= 3 and bullish_bias:
            action = "buy"
            sl = round(bid - risk_distance, 5)
            tp = round(bid + (risk_distance * 1.5), 5)
            lot = round(risk_amount / (risk_distance * 10000), 2)
            if lot < 0.01:
                lot = 0.01
        elif score >= 3 and bearish_bias:
            action = "sell"
            sl = round(ask + risk_distance, 5)
            tp = round(ask - (risk_distance * 1.5), 5)
            lot = round(risk_amount / (risk_distance * 10000), 2)
            if lot < 0.01:
                lot = 0.01
        print(f"📊 {symbol} | Score:{score} | ADX:{adx:.1f} | RSI:{rsi:.1f} | Action:{action}")
        return {
            "action": action,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "score": score,
            "adx": round(adx, 1),
            "rsi": round(rsi, 1)
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"action": "hold", "lot": 0.01, "sl": 0, "tp": 0, "score": 0}

@app.route('/api/signal', methods=['GET'])
def get_signal():
    symbol = request.args.get('symbol')
    bid = float(request.args.get('bid', 0))
    ask = float(request.args.get('ask', 0))
    if not symbol:
        return jsonify({"error": "Missing symbol"}), 400
    result = analyze_market(symbol, bid, ask)
    return jsonify(result)

@app.route('/')
def home():
    return "Trading Brain is ONLINE!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
