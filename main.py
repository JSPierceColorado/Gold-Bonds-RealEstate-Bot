#!/usr/bin/env python3
"""
Multi-ETF RSI + Trend bot (Alpaca)

Buys only: GLD, BND, VNQ
  BUY (15m): RSI(14) < 30  AND  SMA(60,15m) < SMA(240,15m)
    - Uses 10% of buying power per buy

Sells (never sell VIG):
  SELL (15m): RSI(14) >= 70 AND SMA(60,15m) > SMA(240,15m)
              AND last_price >= avg_entry_price * (1 + MIN_PROFIT_PCT)
    - Sells SELL_FRACTION (default 5%) of current position (notional)

Notes:
- Aligns execution to closed 15m bars (:00/:15/:30/:45 + RUN_DELAY_SEC) in America/New_York
- Uses Alpaca positions' avg_entry_price for profit gate
- Uses notional orders for both buys and sells
"""

import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo

from alpaca_trade_api.rest import REST, TimeFrame

# ===== Config (env) =====
BUY_SYMBOLS        = os.getenv("BUY_SYMBOLS", "GLD,BND,VNQ").split(",")
NEVER_SELL         = set((os.getenv("NEVER_SELL", "VIG") or "VIG").split(","))
RSI_LEN            = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_THRESH     = float(os.getenv("RSI_BUY_THRESH", "30"))
RSI_SELL_THRESH    = float(os.getenv("RSI_SELL_THRESH", "70"))
SMA_FAST_LEN       = int(os.getenv("SMA_FAST_LEN", "60"))      # 60 x 15m
SMA_SLOW_LEN       = int(os.getenv("SMA_SLOW_LEN", "240"))     # 240 x 15m
MIN_PROFIT_PCT     = float(os.getenv("MIN_PROFIT_PCT", "0.05"))# 5%
BUY_FRACTION       = float(os.getenv("BUY_FRACTION", "0.10"))  # 10% bp per buy
SELL_FRACTION      = float(os.getenv("SELL_FRACTION", "0.05")) # 5% of position (notional)
DATA_FEED          = os.getenv("DATA_FEED", "iex").lower()     # 'iex' or 'sip'
ALIGN_TZ           = os.getenv("ALIGN_TZ", "America/New_York")
RUN_DELAY_SEC      = int(os.getenv("RUN_DELAY_SEC", "5"))

# History knobs
HISTORY_DAYS_15M   = int(os.getenv("HISTORY_DAYS_15M", "120"))  # ensure we can get >=240 bars
BAR_LIMIT_MAX      = int(os.getenv("BAR_LIMIT_MAX", "10000"))

# Alpaca creds
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL  = os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets")

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY (or APCA_* equivalents).")

api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=APCA_API_BASE_URL)

# ===== Utils & Indicators =====
def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[multietf-rsi15m] {now} | {msg}", flush=True)

def rsi(closes: List[float], length: int = 14) -> float:
    """Wilder RSI on closed bars (closes oldest->newest)."""
    if len(closes) < length + 1:
        return float("nan")
    gains = losses = 0.0
    for i in range(1, length + 1):
        d = closes[i] - closes[i-1]
        gains  += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / length
    avg_loss = losses / length
    for i in range(length + 1, len(closes)):
        d = closes[i] - closes[i-1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def sma(closes: List[float], length: int) -> float:
    if len(closes) < length or length <= 0:
        return float("nan")
    return sum(closes[-length:]) / float(length)

# ===== Data helpers =====
def fetch_closed_15m_closes(symbol: str) -> tuple[list[float], Optional[datetime]]:
    """
    Fetch closed 15m bars for symbol (oldest->newest). Drops the forming bar.
    Returns (closes, last_bar_start_utc)
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=HISTORY_DAYS_15M)

    bars = api.get_bars(
        symbol,
        TimeFrame.Minute,  # will filter to 15m via slice below
        start=start.isoformat(),
        end=now.isoformat(),
        adjustment="raw",
        feed=DATA_FEED,
        limit=BAR_LIMIT_MAX,
    )
    df = getattr(bars, "df", None)
    if df is None or df.empty:
        return [], None

    # If MultiIndex, slice symbol; else assume single index
    try:
        sym_df = df.xs(symbol, level=0)
    except Exception:
        sym_df = df

    sym_df = sym_df.sort_index()

    # Resample to 15-minute bars in case the feed returns 1m
    try:
        # Keep OHLCV shape consistent; use last for close
        r = sym_df.resample("15T").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        sym_df = r
    except Exception:
        # If already 15m from the feed, continue
        pass

    # Drop forming bar (bar timestamps are start times)
    cutoff = now - timedelta(minutes=15)
    if not sym_df.empty and sym_df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) > cutoff:
        sym_df = sym_df.iloc[:-1]

    if sym_df.empty:
        return [], None

    closes = sym_df["close"].astype(float).tolist()
    last_start = sym_df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    return closes, last_start

def get_buying_power_usd() -> float:
    acct = api.get_account()
    try:
        return float(acct.buying_power)
    except Exception:
        return float(acct.cash)

def get_positions_map() -> Dict[str, dict]:
    """
    Returns map[symbol] -> {"qty": float, "avg_entry_price": float, "market_value": float}
    """
    out: Dict[str, dict] = {}
    for p in api.list_positions():
        sym = p.symbol
        out[sym] = {
            "qty": float(p.qty) if hasattr(p, "qty") else float(p.qty_available),
            "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price else float("nan"),
            "market_value": float(p.market_value) if p.market_value else 0.0,
        }
    return out

def get_last_price_from_closes(closes: List[float]) -> float:
    return float(closes[-1]) if closes else float("nan")

# ===== Order helpers (notional) =====
def submit_notional_buy(symbol: str, notional_usd: float):
    notional_usd = round(float(notional_usd), 2)
    order = api.submit_order(
        symbol=symbol, side="buy", type="market", time_in_force="day",
        notional=notional_usd, client_order_id=f"buy-{symbol}-{int(time.time()*1000)}"
    )
    oid = getattr(order, "id", "") or getattr(order, "client_order_id", "")
    status = getattr(order, "status", "submitted")
    log(f"{symbol} | BUY ${notional_usd:.2f} submitted (order {oid}, status {status})")

def submit_notional_sell(symbol: str, notional_usd: float):
    notional_usd = round(float(notional_usd), 2)
    order = api.submit_order(
        symbol=symbol, side="sell", type="market", time_in_force="day",
        notional=notional_usd, client_order_id=f"sell-{symbol}-{int(time.time()*1000)}"
    )
    oid = getattr(order, "id", "") or getattr(order, "client_order_id", "")
    status = getattr(order, "status", "submitted")
    log(f"{symbol} | SELL ${notional_usd:.2f} submitted (order {oid}, status {status})")

# ===== 15m alignment =====
def next_quarter_hour(now_ny: datetime) -> datetime:
    minute = now_ny.minute
    next_min = ((minute // 15) + 1) * 15
    if next_min >= 60:
        base = now_ny.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        base = now_ny.replace(minute=next_min, second=0, microsecond=0)
    return base

def seconds_until_next_run(delay_sec: int) -> float:
    ny = ZoneInfo(ALIGN_TZ)
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(ny)
    target_ny = next_quarter_hour(now_ny) + timedelta(seconds=delay_sec)
    target_utc = target_ny.astimezone(timezone.utc)
    return max(0.0, (target_utc - now_utc).total_seconds())

# ===== Strategy checks =====
def buy_signal_15m(closes: List[float]) -> bool:
    r = rsi(closes, RSI_LEN)
    fast = sma(closes, SMA_FAST_LEN)
    slow = sma(closes, SMA_SLOW_LEN)
    log(f"BUY chk | RSI{RSI_LEN}={r:.2f} | SMA{SMA_FAST_LEN}={fast:.6f} < SMA{SMA_SLOW_LEN}={slow:.6f}? {fast < slow if not math.isnan(fast) and not math.isnan(slow) else None}")
    if math.isnan(r) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r < RSI_BUY_THRESH) and (fast < slow)

def sell_signal_15m(closes: List[float]) -> bool:
    r = rsi(closes, RSI_LEN)
    fast = sma(closes, SMA_FAST_LEN)
    slow = sma(closes, SMA_SLOW_LEN)
    log(f"SELL chk | RSI{RSI_LEN}={r:.2f} | SMA{SMA_FAST_LEN}={fast:.6f} > SMA{SMA_SLOW_LEN}={slow:.6f}? {fast > slow if not math.isnan(fast) and not math.isnan(slow) else None}")
    if math.isnan(r) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r >= RSI_SELL_THRESH) and (fast > slow)

# ===== Main loop =====
def main():
    log(
        f"Start | feed={DATA_FEED} | BUY_SYMBOLS={BUY_SYMBOLS} | NEVER_SELL={sorted(NEVER_SELL)} | "
        f"RSI<{RSI_BUY_THRESH} & SMA{SMA_FAST_LEN}<SMA{SMA_SLOW_LEN} buys | "
        f"RSI>={RSI_SELL_THRESH} & SMA{SMA_FAST_LEN}>SMA{SMA_SLOW_LEN} sells if profit≥{int(MIN_PROFIT_PCT*100)}% | "
        f"buy_fraction={BUY_FRACTION:.2f} | sell_fraction={SELL_FRACTION:.2f} | align_tz={ALIGN_TZ} | delay={RUN_DELAY_SEC}s"
    )

    last_seen_bar: Dict[str, Optional[datetime]] = {s: None for s in BUY_SYMBOLS}

    while True:
        try:
            # Map of positions
            pos = get_positions_map()
            buying_power = get_buying_power_usd()
            log(f"Status | buying_power=${buying_power:.2f} | positions={list(pos.keys())}")

            # ------- SELL checks (all positions except NEVER_SELL) -------
            # Evaluate with same 15m indicators per symbol
            for sym, pdata in pos.items():
                if sym in NEVER_SELL:
                    continue
                closes, last_15 = fetch_closed_15m_closes(sym)
                if not closes or last_15 is None:
                    log(f"{sym} | No closed 15m bars for SELL check.")
                    continue

                px = get_last_price_from_closes(closes)
                avg = pdata.get("avg_entry_price", float("nan"))
                if math.isnan(avg) or avg <= 0:
                    log(f"{sym} | Unknown avg_entry_price; skipping SELL for safety.")
                    continue

                if sell_signal_15m(closes):
                    need = avg * (1.0 + float(MIN_PROFIT_PCT))
                    if px >= need:
                        # Notional amount to sell = SELL_FRACTION * current market value
                        mv = pdata.get("market_value", 0.0)
                        notional = max(0.0, mv * SELL_FRACTION)
                        if notional > 0:
                            log(f"{sym} | SELL gate ok: price={px:.4f} ≥ {need:.4f} | notional=${notional:.2f}")
                            submit_notional_sell(sym, notional)
                        else:
                            log(f"{sym} | SELL gate ok but market_value≈0; skipping.")
                    else:
                        log(f"{sym} | SELL signal true but price {px:.4f} < profit gate {need:.4f}; skipping.")
                else:
                    log(f"{sym} | SELL signal false.")

            # ------- BUY checks (only BUY_SYMBOLS) -------
            for sym in BUY_SYMBOLS:
                closes, last_15 = fetch_closed_15m_closes(sym)
                if not closes or last_15 is None:
                    log(f"{sym} | No closed 15m bars for BUY check.")
                    continue

                is_new_bar = (last_seen_bar.get(sym) is None) or (last_15 != last_seen_bar.get(sym))
                if not is_new_bar:
                    continue  # only act once per new bar per symbol

                px = get_last_price_from_closes(closes)
                log(f"{sym} | Last price={px:.4f} | last_15m_start={last_15.isoformat()} | NEW BAR")

                if buy_signal_15m(closes) and buying_power > 0:
                    notional = buying_power * BUY_FRACTION
                    if notional > 0:
                        submit_notional_buy(sym, notional)
                        # Update our local view of buying power optimistically
                        buying_power = max(0.0, buying_power - notional)
                last_seen_bar[sym] = last_15

        except Exception as e:
            log(f"Error: {type(e).__name__}: {e}")

        # Sleep until next 15m boundary in NY time + delay
        sleep_s = seconds_until_next_run(RUN_DELAY_SEC)
        if sleep_s < 0.5:
            sleep_s = 5.0
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
