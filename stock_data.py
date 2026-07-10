from datetime import datetime, timezone
import logging

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


def _empty_quote(symbol):
    return {
        "symbol": symbol,
        "company_name": "",
        "current_price": None,
        "previous_close": None,
        "price_change": None,
        "change_percent": None,
        "currency": "",
        "market_state": "",
        "data_timestamp": "",
    }


def _safe_number(value):
    if value in (None, "", "None"):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_timestamp(value):
    if not value:
        return ""

    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return ""

    return str(value)


def _get_ticker(symbol):
    if yf is None:
        logger.warning("yfinance is not installed. symbol=%s", symbol)
        return None
    return yf.Ticker(symbol)


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_quote(symbol):
    """Yahoo Finance에서 조회 가격과 등락 정보를 가져옵니다."""
    quote = _empty_quote(symbol)
    ticker = _get_ticker(symbol)
    if ticker is None:
        return quote

    try:
        info = ticker.get_info() or {}
    except Exception as error:
        logger.exception("Failed to load stock info. symbol=%s error=%s", symbol, error)
        info = {}

    try:
        fast_info = ticker.fast_info or {}
    except Exception as error:
        logger.exception("Failed to load fast stock info. symbol=%s error=%s", symbol, error)
        fast_info = {}

    try:
        history = ticker.history(period="5d", interval="1d")
    except Exception as error:
        logger.exception("Failed to load stock quote history. symbol=%s error=%s", symbol, error)
        history = pd.DataFrame()

    current_price = _safe_number(
        fast_info.get("last_price")
        or info.get("currentPrice")
        or info.get("regularMarketPrice")
    )
    previous_close = _safe_number(
        fast_info.get("previous_close")
        or info.get("previousClose")
        or info.get("regularMarketPreviousClose")
    )

    if current_price is None and not history.empty and "Close" in history:
        current_price = _safe_number(history["Close"].dropna().iloc[-1])
    if previous_close is None and not history.empty and "Close" in history and len(history["Close"].dropna()) >= 2:
        previous_close = _safe_number(history["Close"].dropna().iloc[-2])

    price_change = None
    change_percent = None
    if current_price is not None and previous_close not in (None, 0):
        price_change = current_price - previous_close
        change_percent = (price_change / previous_close) * 100

    quote.update(
        {
            "company_name": info.get("shortName") or info.get("longName") or symbol,
            "current_price": current_price,
            "previous_close": previous_close,
            "price_change": price_change,
            "change_percent": change_percent,
            "currency": fast_info.get("currency") or info.get("currency") or "",
            "market_state": info.get("marketState") or "",
            "data_timestamp": _format_timestamp(
                info.get("regularMarketTime")
                or info.get("postMarketTime")
                or info.get("preMarketTime")
            ),
        }
    )
    return quote


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_history(symbol, period="1mo"):
    """최근 기간의 일별 시가, 고가, 저가, 종가, 거래량을 DataFrame으로 반환합니다."""
    ticker = _get_ticker(symbol)
    if ticker is None:
        return pd.DataFrame()

    try:
        history = ticker.history(period=period, interval="1d")
    except Exception as error:
        logger.exception("Failed to load stock history. symbol=%s period=%s error=%s", symbol, period, error)
        return pd.DataFrame()

    if history is None or history.empty or "Close" not in history:
        return pd.DataFrame()

    columns = ["Open", "High", "Low", "Close", "Volume"]
    available_columns = [column for column in columns if column in history.columns]
    if "Close" not in available_columns:
        return pd.DataFrame()

    cleaned_history = history[available_columns].copy()
    for column in available_columns:
        cleaned_history[column] = pd.to_numeric(cleaned_history[column], errors="coerce")

    return cleaned_history.dropna(subset=["Close"])


def _safe_series_last(series):
    if series is None:
        return None
    try:
        clean_series = series.dropna()
        if clean_series.empty:
            return None
        return _safe_number(clean_series.iloc[-1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _calculate_period_return(close_series, trading_days):
    try:
        clean_close = close_series.dropna()
        if len(clean_close) <= trading_days:
            return None
        start_price = _safe_number(clean_close.iloc[-trading_days - 1])
        end_price = _safe_number(clean_close.iloc[-1])
        if start_price in (None, 0) or end_price is None:
            return None
        return ((end_price - start_price) / start_price) * 100
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def _calculate_rsi(close_series, period=14):
    try:
        clean_close = close_series.dropna()
        if len(clean_close) <= period:
            return None
        delta = clean_close.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        latest_gain = _safe_series_last(gain)
        latest_loss = _safe_series_last(loss)
        if latest_loss == 0 and latest_gain is not None and latest_gain > 0:
            return 100.0
        if latest_loss == 0 and latest_gain == 0:
            return 50.0

        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return _safe_series_last(rsi)
    except Exception as error:
        logger.exception("Failed to calculate RSI. error=%s", error)
        return None


def calculate_technical_indicators(history_df):
    """주가 그래프 해석에 필요한 보조 지표를 안전하게 계산합니다."""
    empty_indicators = {
        "current_close": None,
        "return_1mo": None,
        "return_3mo": None,
        "return_6mo": None,
        "ma5": None,
        "ma20": None,
        "ma60": None,
        "avg_volume_20d": None,
        "latest_volume": None,
        "volume_vs_avg_ratio": None,
        "is_volume_above_average": None,
        "high_3mo": None,
        "low_3mo": None,
        "distance_from_3mo_high_percent": None,
        "distance_from_3mo_low_percent": None,
        "volatility_20d": None,
        "rsi_14": None,
        "rsi_status": "확인 불가",
        "data_points": 0,
        "message": "",
    }

    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        empty_indicators["message"] = "분석 가능한 데이터가 부족합니다."
        return empty_indicators

    history = history_df.copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    close = history["Close"].dropna()
    if close.empty:
        empty_indicators["message"] = "종가 데이터가 부족합니다."
        return empty_indicators

    indicators = empty_indicators.copy()
    indicators["data_points"] = int(len(close))
    indicators["current_close"] = _safe_series_last(close)
    indicators["return_1mo"] = _calculate_period_return(close, 21)
    indicators["return_3mo"] = _calculate_period_return(close, 63)
    indicators["return_6mo"] = _calculate_period_return(close, 126)
    indicators["ma5"] = _safe_series_last(close.rolling(window=5).mean())
    indicators["ma20"] = _safe_series_last(close.rolling(window=20).mean())
    indicators["ma60"] = _safe_series_last(close.rolling(window=60).mean())

    if "Volume" in history.columns:
        volume = history["Volume"].dropna()
        indicators["latest_volume"] = _safe_series_last(volume)
        indicators["avg_volume_20d"] = _safe_series_last(volume.rolling(window=20).mean())
        if indicators["latest_volume"] is not None and indicators["avg_volume_20d"] not in (None, 0):
            indicators["volume_vs_avg_ratio"] = indicators["latest_volume"] / indicators["avg_volume_20d"]
            indicators["is_volume_above_average"] = indicators["latest_volume"] > indicators["avg_volume_20d"]

    recent_3mo = history.tail(63)
    if not recent_3mo.empty:
        high_source = recent_3mo["High"] if "High" in recent_3mo.columns else recent_3mo["Close"]
        low_source = recent_3mo["Low"] if "Low" in recent_3mo.columns else recent_3mo["Close"]
        indicators["high_3mo"] = _safe_number(high_source.dropna().max())
        indicators["low_3mo"] = _safe_number(low_source.dropna().min())

    current_close = indicators["current_close"]
    if current_close not in (None, 0):
        if indicators["high_3mo"] not in (None, 0):
            indicators["distance_from_3mo_high_percent"] = ((current_close - indicators["high_3mo"]) / indicators["high_3mo"]) * 100
        if indicators["low_3mo"] not in (None, 0):
            indicators["distance_from_3mo_low_percent"] = ((current_close - indicators["low_3mo"]) / indicators["low_3mo"]) * 100

    try:
        daily_return = close.pct_change().dropna()
        if len(daily_return) >= 10:
            volatility = daily_return.rolling(window=20).std()
            volatility_value = _safe_series_last(volatility)
            indicators["volatility_20d"] = volatility_value * 100 if volatility_value is not None else None
    except Exception as error:
        logger.exception("Failed to calculate volatility. error=%s", error)

    indicators["rsi_14"] = _calculate_rsi(close, 14)
    if indicators["rsi_14"] is not None:
        if indicators["rsi_14"] >= 70:
            indicators["rsi_status"] = "과열 가능성"
        elif indicators["rsi_14"] <= 30:
            indicators["rsi_status"] = "침체 가능성"
        else:
            indicators["rsi_status"] = "중립 구간"

    return indicators


@st.cache_data(ttl=21600, show_spinner=False)
def get_company_info(symbol):
    """기업 기본 정보를 안전하게 조회합니다."""
    empty_info = {
        "company_name": "",
        "sector": "",
        "industry": "",
        "market_cap": None,
        "per": None,
        "pbr": None,
        "dividend_yield": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "description": "",
    }
    ticker = _get_ticker(symbol)
    if ticker is None:
        return empty_info

    try:
        info = ticker.get_info() or {}
    except Exception as error:
        logger.exception("Failed to load company info. symbol=%s error=%s", symbol, error)
        return empty_info

    return {
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "market_cap": _safe_number(info.get("marketCap")),
        "per": _safe_number(info.get("trailingPE") or info.get("forwardPE")),
        "pbr": _safe_number(info.get("priceToBook")),
        "dividend_yield": _safe_number(info.get("dividendYield")),
        "fifty_two_week_high": _safe_number(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _safe_number(info.get("fiftyTwoWeekLow")),
        "description": info.get("longBusinessSummary") or "",
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_news(symbol):
    """최신 뉴스 5개를 제목, 언론사, 게시 시각, 링크 형태로 반환합니다."""
    ticker = _get_ticker(symbol)
    if ticker is None:
        return []

    try:
        raw_news = ticker.news or []
    except Exception as error:
        logger.exception("Failed to load stock news. symbol=%s error=%s", symbol, error)
        return []

    news_items = []
    for item in raw_news[:5]:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        if not isinstance(content, dict):
            content = {}
        provider = content.get("provider", {}) if isinstance(content, dict) else {}
        if not isinstance(provider, dict):
            provider = {}
        canonical_url = content.get("canonicalUrl", {}) if isinstance(content, dict) else {}
        if not isinstance(canonical_url, dict):
            canonical_url = {}
        click_url = content.get("clickThroughUrl", {}) if isinstance(content, dict) else {}
        if not isinstance(click_url, dict):
            click_url = {}

        title = item.get("title") or content.get("title") or ""
        publisher = item.get("publisher") or provider.get("displayName") or ""
        published_at = (
            item.get("providerPublishTime")
            or content.get("pubDate")
            or content.get("displayTime")
            or ""
        )
        link = item.get("link") or canonical_url.get("url") or click_url.get("url") or ""

        if not title:
            continue

        news_items.append(
            {
                "title": title,
                "publisher": publisher,
                "published_at": _format_timestamp(published_at),
                "link": link,
            }
        )

    return news_items
