import os
import csv
import re
import sys
import random
import time
from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from storage import read_json_file, write_json_file
from npc_config import DEFAULT_NPC_ROLE, get_npc_config
from stock_data import calculate_technical_indicators, get_company_info, get_stock_history, get_stock_news, get_stock_quote
from age_trends import SOURCE_REFERENCE_LINKS, build_age_group_trend_context, fetch_age_group_investment_trends

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    chromadb = None


BASE_DIR = Path(__file__).resolve().parent
APP_MODE = os.getenv("NPC_APP_MODE", "all").strip().lower()
if APP_MODE not in {"all", "basic", "stock"}:
    APP_MODE = "all"

ENV_PATH = BASE_DIR / ".env"
CHAT_HISTORIES_PATH = BASE_DIR / {
    "basic": "chat_histories_basic.json",
    "stock": "chat_histories_stock.json",
}.get(APP_MODE, "chat_histories.json")
WATCHLIST_PATH = BASE_DIR / "watchlist.json"
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"
DATA_DIR = BASE_DIR / "data"
DOCUMENT_CHUNKS_PATH = BASE_DIR / "document_chunks.json"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
SQUISHY_FRAMES_DIR = BASE_DIR / "assets" / "squishy_frames"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct:together"
DEFAULT_RAG_TOP_K = 3
DEFAULT_RAG_CHUNK_SIZE = 500
DEFAULT_RAG_OVERLAP = 50
RAG_CHECKLIST_START = "[RAG_CONNECTION_CHECKLIST]"
RAG_CHECKLIST_END = "[/RAG_CONNECTION_CHECKLIST]"
MENTAL_CHARM_START = "[MENTAL_CHARM]"
MENTAL_CHARM_END = "[/MENTAL_CHARM]"
RAG_TEST_QUESTION_TYPES = [
    "사실 확인 질문",
    "요약 질문",
    "비교 질문",
    "수치 확인 질문",
    "확인 범위 질문",
    "근거가 부족한 질문",
    "애매한 질문",
]
DEFAULT_RAG_TEST_QUESTIONS = [
    {"type": "사실 확인 질문", "question": "문서에서 가장 중요한 핵심 사실은 무엇인가요?"},
    {"type": "요약 질문", "question": "이 문서 내용을 초보자도 이해하기 쉽게 요약해줘."},
    {"type": "비교 질문", "question": "문서에서 비교되는 항목이나 차이점을 정리해줘."},
    {"type": "수치 확인 질문", "question": "문서에 나오는 중요한 수치나 기준을 찾아줘."},
    {"type": "근거가 부족한 질문", "question": "문서에서 확인되지 않는 내용은 무엇인지 알려줘."},
]

DATA_DIR.mkdir(exist_ok=True)

NPCS = get_npc_config(BASE_DIR)
BASIC_APP_NPC_ROLES = ["친절한 상담원", "아이디어 기획자", "Python 튜터", "보고서 도우미", "도서관 사서"]
STOCK_APP_NPC_ROLES = ["시장 해설가", "종목 분석가", "초보 투자 튜터", "포트폴리오 코치", "투자 멘탈 코치"]
if APP_MODE == "basic":
    NPCS = {role: NPCS[role] for role in BASIC_APP_NPC_ROLES if role in NPCS}
elif APP_MODE == "stock":
    NPCS = {role: NPCS[role] for role in STOCK_APP_NPC_ROLES if role in NPCS}
NPC_ROLES = list(NPCS.keys())
if DEFAULT_NPC_ROLE not in NPCS and NPC_ROLES:
    DEFAULT_NPC_ROLE = NPC_ROLES[0]
GENERAL_CHAT_ROOM_KEY = "__general__"


def is_stock_app_mode():
    return APP_MODE in {"all", "stock"}


def is_basic_app_mode():
    return APP_MODE in {"all", "basic"}

STOCK_NAME_TO_SYMBOL = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "애플": "AAPL",
    "테슬라": "TSLA",
    "엔비디아": "NVDA",
    "마이크로소프트": "MSFT",
    "아마존": "AMZN",
    "구글": "GOOGL",
}

STOCK_HISTORY_PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
}

INVESTMENT_CHECKLIST_ITEMS = [
    "기업이 어떤 방식으로 돈을 버는가",
    "최근 매출과 이익 흐름은 어떤가",
    "부채와 현금흐름 상태는 어떤가",
    "PER과 PBR을 업종과 비교했는가",
    "최근 주가 상승 또는 하락 요인은 무엇인가",
    "향후 성장 요인은 무엇인가",
    "주요 위험 요소는 무엇인가",
    "경쟁사보다 강한 점과 약한 점은 무엇인가",
    "현재 포트폴리오에서 비중이 지나치게 높지 않은가",
    "예상과 다르게 움직일 경우 대응 기준이 있는가",
]

API_STATUS_NO_RESPONSE = "아직 응답 없음"
API_STATUS_LLM = "Hugging Face LLM 응답 사용 중"
API_STATUS_NO_TOKEN = "Hugging Face API 토큰 없음"
API_STATUS_FALLBACK = "API 호출 실패, fallback 응답 사용"


def sanitize_error_message(error):
    """API 토큰처럼 민감한 값이 화면이나 터미널 로그에 섞이지 않도록 정리합니다."""
    message = str(error or "")
    token = os.getenv("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted-token]")
    message = re.sub(r"hf_[A-Za-z0-9_\-]+", "[redacted-token]", message)
    message = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [redacted-token]", message, flags=re.IGNORECASE)
    return message


def log_app_error(context, error):
    """화면에는 쉬운 안내를 보여주고, 터미널에는 디버깅용 오류 위치를 남깁니다."""
    safe_message = sanitize_error_message(error)
    print(f"[app-error] {context}: {type(error).__name__}: {safe_message}", file=sys.stderr)


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if value and not os.environ.get(key):
            os.environ[key] = value


load_env_file(ENV_PATH)


def has_huggingface_token():
    return bool(os.getenv("HF_TOKEN"))


def get_default_api_status():
    if has_huggingface_token():
        return {
            "label": API_STATUS_NO_RESPONSE,
            "source": "없음",
            "error": "",
        }

    return {
        "label": API_STATUS_NO_TOKEN,
        "source": "없음",
        "error": "",
    }


def get_initial_messages(npc_role=DEFAULT_NPC_ROLE):
    """선택한 NPC에 맞는 첫 인사 메시지를 만듭니다."""
    welcome_npc = NPCS[npc_role]
    return [
        {
            "role": "assistant",
            "npc_role": npc_role,
            "content": (
                f"{welcome_npc['greeting']}\n"
                "질문을 보내면 선택한 NPC의 역할, 스킬, 입력 가이드를 기준으로 답해 드릴게요."
            ),
        }
    ]


def get_initial_conversation_log():
    return []


def create_empty_chat_room(npc_role):
    """NPC와 종목 조합별 대화방 기본 구조를 만듭니다."""
    return {
        "messages": get_initial_messages(npc_role),
        "conversation_log": get_initial_conversation_log(),
    }


def create_empty_chat_histories():
    """NPC마다 일반 대화방을 하나씩 가진 기본 구조를 만듭니다."""
    return {
        npc_role: {
            GENERAL_CHAT_ROOM_KEY: create_empty_chat_room(npc_role),
        }
        for npc_role in NPC_ROLES
    }


def normalize_chat_room(saved_room, npc_role):
    """저장된 대화방 데이터를 안전한 messages/conversation_log 구조로 맞춥니다."""
    empty_room = create_empty_chat_room(npc_role)
    if not isinstance(saved_room, dict):
        return empty_room

    messages = saved_room.get("messages", empty_room["messages"])
    conversation_log = saved_room.get("conversation_log", empty_room["conversation_log"])
    if isinstance(messages, list):
        messages = [clean_saved_chat_record(message) for message in messages]
    if isinstance(conversation_log, list):
        conversation_log = [clean_saved_chat_record(log) for log in conversation_log]

    return {
        "messages": messages if isinstance(messages, list) else empty_room["messages"],
        "conversation_log": conversation_log if isinstance(conversation_log, list) else empty_room["conversation_log"],
    }


def normalize_saved_npc_rooms(saved_npc_data, npc_role):
    """기존 NPC별 기록 또는 새 종목별 기록을 모두 새 구조로 변환합니다."""
    npc_rooms = {
        GENERAL_CHAT_ROOM_KEY: create_empty_chat_room(npc_role),
    }
    if not isinstance(saved_npc_data, dict):
        return npc_rooms

    # 기존 구조: {"messages": [...], "conversation_log": [...]} 는 일반 대화방으로 이전합니다.
    if "messages" in saved_npc_data or "conversation_log" in saved_npc_data:
        npc_rooms[GENERAL_CHAT_ROOM_KEY] = normalize_chat_room(saved_npc_data, npc_role)
        return npc_rooms

    # 새 구조: {"__general__": {...}, "AAPL": {...}} 형태를 방별로 복구합니다.
    for room_key, saved_room in saved_npc_data.items():
        if not isinstance(room_key, str) or not room_key:
            continue
        npc_rooms[room_key] = normalize_chat_room(saved_room, npc_role)

    if GENERAL_CHAT_ROOM_KEY not in npc_rooms:
        npc_rooms[GENERAL_CHAT_ROOM_KEY] = create_empty_chat_room(npc_role)

    return npc_rooms


def load_chat_histories():
    """로컬 JSON 파일에서 NPC별·종목별 대화 기록을 불러옵니다."""
    saved_histories, error = read_json_file(CHAT_HISTORIES_PATH, create_empty_chat_histories())
    if error:
        st.warning("대화 기록 파일을 읽지 못해 빈 대화방으로 복구했습니다.")
        return create_empty_chat_histories()

    if not isinstance(saved_histories, dict):
        return create_empty_chat_histories()

    chat_histories = create_empty_chat_histories()
    for npc_role in NPC_ROLES:
        chat_histories[npc_role] = normalize_saved_npc_rooms(saved_histories.get(npc_role, {}), npc_role)

    return chat_histories


def save_chat_histories():
    """현재 메모리의 전체 NPC 대화방 기록을 JSON 파일에 저장합니다."""
    is_saved, error = write_json_file(CHAT_HISTORIES_PATH, st.session_state.chat_histories)
    if not is_saved:
        log_app_error("chat_histories.json 저장 실패", error)
        st.warning("대화 기록을 파일에 저장하지 못했습니다. 앱은 계속 사용할 수 있습니다.")


def load_watchlist():
    """로컬 watchlist.json에서 관심 종목 목록을 불러옵니다."""
    saved_watchlist, error = read_json_file(WATCHLIST_PATH, [])
    if error:
        st.warning("관심 종목 파일을 읽지 못해 빈 목록으로 복구했습니다.")
        save_watchlist([])
        return []

    if not isinstance(saved_watchlist, list):
        save_watchlist([])
        return []

    watchlist = []
    seen_symbols = set()
    for item in saved_watchlist:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()
        symbol = str(item.get("symbol") or "").strip().upper()
        if not name or not symbol or symbol in seen_symbols:
            continue

        watchlist.append({"name": name, "symbol": symbol})
        seen_symbols.add(symbol)

    if len(watchlist) != len(saved_watchlist):
        save_watchlist(watchlist)

    return watchlist


def save_watchlist(watchlist):
    """관심 종목 목록을 로컬 JSON 파일에 저장합니다."""
    is_saved, error = write_json_file(WATCHLIST_PATH, watchlist)
    if not is_saved:
        log_app_error("watchlist.json 저장 실패", error)
        st.warning("관심 종목을 파일에 저장하지 못했습니다. 앱은 계속 사용할 수 있습니다.")


def load_portfolio():
    """로컬 portfolio.json에서 보유 종목 목록을 불러옵니다."""
    saved_portfolio, error = read_json_file(PORTFOLIO_PATH, [])
    if error:
        st.warning("포트폴리오 파일을 읽지 못해 빈 목록으로 복구했습니다.")
        save_portfolio([])
        return []

    if not isinstance(saved_portfolio, list):
        save_portfolio([])
        return []

    portfolio = []
    for item in saved_portfolio:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()
        symbol = str(item.get("symbol") or "").strip().upper()
        try:
            weight = float(item.get("weight", 0))
        except (TypeError, ValueError):
            continue

        average_price = item.get("average_price")
        try:
            average_price = None if average_price in (None, "") else float(average_price)
        except (TypeError, ValueError):
            average_price = None

        if name and symbol and weight >= 0:
            portfolio.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "weight": weight,
                    "average_price": average_price,
                }
            )

    if len(portfolio) != len(saved_portfolio):
        save_portfolio(portfolio)

    return portfolio


def save_portfolio(portfolio):
    """보유 종목 목록을 로컬 JSON 파일에 저장합니다."""
    is_saved, error = write_json_file(PORTFOLIO_PATH, portfolio)
    if not is_saved:
        log_app_error("portfolio.json 저장 실패", error)
        st.warning("포트폴리오를 파일에 저장하지 못했습니다. 앱은 계속 사용할 수 있습니다.")


def get_current_chat_room_key():
    """현재 선택 종목을 기준으로 사용할 대화방 키를 정합니다."""
    selected_symbol = st.session_state.get("selected_symbol")
    if not selected_symbol:
        return GENERAL_CHAT_ROOM_KEY

    return str(selected_symbol).upper()


def ensure_chat_room(npc_role, room_key=None):
    """NPC와 종목 대화방이 없으면 새로 만들고 반환합니다."""
    if room_key is None:
        room_key = get_current_chat_room_key()

    if npc_role not in st.session_state.chat_histories:
        st.session_state.chat_histories[npc_role] = {
            GENERAL_CHAT_ROOM_KEY: create_empty_chat_room(npc_role),
        }

    npc_rooms = st.session_state.chat_histories[npc_role]
    if not isinstance(npc_rooms, dict) or "messages" in npc_rooms:
        st.session_state.chat_histories[npc_role] = normalize_saved_npc_rooms(npc_rooms, npc_role)
        npc_rooms = st.session_state.chat_histories[npc_role]

    if room_key not in npc_rooms:
        npc_rooms[room_key] = create_empty_chat_room(npc_role)

    return npc_rooms[room_key]


def reset_chat_room(npc_role, room_key=None):
    """현재 NPC와 현재 종목 조합의 대화방만 초기화합니다."""
    if room_key is None:
        room_key = get_current_chat_room_key()

    if npc_role not in st.session_state.chat_histories:
        st.session_state.chat_histories[npc_role] = {}

    st.session_state.chat_histories[npc_role][room_key] = create_empty_chat_room(npc_role)
    save_chat_histories()


def detect_npc_role(user_input, selected_role):
    lowered_input = user_input.lower()

    for role_name, npc in NPCS.items():
        if any(keyword in lowered_input for keyword in npc["keywords"]):
            return role_name

    return selected_role


def normalize_stock_input(raw_input):
    """종목 검색창 입력값의 앞뒤 공백을 정리하고 영문 티커는 대문자로 바꿉니다."""
    normalized_input = (raw_input or "").strip()
    if re.fullmatch(r"[A-Za-z0-9.]+", normalized_input):
        return normalized_input.upper()

    return normalized_input


def resolve_symbol(raw_input):
    """종목명은 Yahoo Finance용 티커로 바꾸고, 직접 입력한 티커는 그대로 사용합니다."""
    normalized_input = normalize_stock_input(raw_input)
    if not normalized_input:
        return None, None, "종목명이나 티커를 입력해 주세요."

    if normalized_input in STOCK_NAME_TO_SYMBOL:
        return STOCK_NAME_TO_SYMBOL[normalized_input], normalized_input, ""

    for stock_name, symbol in STOCK_NAME_TO_SYMBOL.items():
        if normalized_input.lower() == stock_name.lower():
            return symbol, stock_name, ""

    if re.fullmatch(r"[A-Z0-9.]{1,12}", normalized_input):
        return normalized_input, normalized_input, ""

    return None, None, "지원 목록에 없는 종목명이거나 티커 형식이 올바르지 않습니다."


def set_selected_stock(raw_input):
    """선택한 종목을 모든 NPC가 함께 참고할 수 있도록 session_state에 저장합니다."""
    symbol, stock_name, error_message = resolve_symbol(raw_input)
    if error_message:
        return False, error_message

    st.session_state.selected_symbol = symbol
    st.session_state.selected_stock_name = stock_name
    st.session_state.selected_stock_news = {}
    return True, f"{stock_name} ({symbol}) 종목을 선택했습니다."


def is_stock_lookup_available(symbol):
    """Yahoo Finance에서 최소한의 종목 데이터를 조회할 수 있는지 확인합니다."""
    quote = get_stock_quote(symbol)
    company_info = get_company_info(symbol)
    history = get_stock_history(symbol, period="5d")

    has_quote = quote.get("current_price") is not None or quote.get("previous_close") is not None
    has_company = bool(company_info.get("company_name") or company_info.get("sector") or company_info.get("industry"))
    has_history = history is not None and not history.empty
    return has_quote or has_company or has_history


def add_watchlist_stock(raw_input):
    """입력한 종목을 검증한 뒤 관심 종목에 추가합니다."""
    symbol, stock_name, error_message = resolve_symbol(raw_input)
    if error_message:
        return False, error_message

    if not is_stock_lookup_available(symbol):
        return False, "Yahoo Finance에서 조회 가능한 종목인지 확인하지 못했습니다. 종목명이나 티커를 다시 확인해 주세요."

    watchlist = st.session_state.watchlist
    if any(item["symbol"] == symbol for item in watchlist):
        return False, f"{stock_name} ({symbol})는 이미 관심 종목에 있습니다."

    watchlist.append({"name": stock_name, "symbol": symbol})
    st.session_state.watchlist = watchlist
    save_watchlist(watchlist)
    return True, f"{stock_name} ({symbol})를 관심 종목에 추가했습니다."


def select_watchlist_stock(stock_item):
    """관심 종목을 현재 선택 종목으로 바꿉니다. 대화 기록은 변경하지 않습니다."""
    st.session_state.selected_stock_name = stock_item["name"]
    st.session_state.selected_symbol = stock_item["symbol"]


def delete_watchlist_stock(symbol):
    """관심 종목 목록에서 특정 티커를 삭제합니다."""
    st.session_state.watchlist = [
        item for item in st.session_state.watchlist if item["symbol"] != symbol
    ]
    save_watchlist(st.session_state.watchlist)


def resolve_portfolio_asset(raw_input):
    """포트폴리오 입력값을 종목 또는 현금 자산으로 해석합니다."""
    normalized_input = (raw_input or "").strip()
    if normalized_input in {"현금", "cash", "CASH"}:
        return "CASH", "현금", ""

    return resolve_symbol(normalized_input)


def add_portfolio_asset(raw_input, weight, average_price=None):
    """보유 종목과 비중을 포트폴리오에 추가합니다."""
    symbol, name, error_message = resolve_portfolio_asset(raw_input)
    if error_message:
        return False, error_message

    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        return False, "비중은 숫자로 입력해 주세요."

    if weight_value < 0:
        return False, "비중은 0 이상으로 입력해 주세요."

    average_price_value = None
    if average_price not in (None, ""):
        try:
            average_price_value = float(average_price)
        except (TypeError, ValueError):
            return False, "평균 매수가는 숫자로 입력해 주세요."

    if symbol != "CASH" and not is_stock_lookup_available(symbol):
        return False, "Yahoo Finance에서 조회 가능한 종목인지 확인하지 못했습니다."

    portfolio = [item for item in st.session_state.portfolio if item["symbol"] != symbol]
    portfolio.append(
        {
            "name": name,
            "symbol": symbol,
            "weight": weight_value,
            "average_price": average_price_value,
        }
    )
    st.session_state.portfolio = portfolio
    save_portfolio(portfolio)
    return True, f"{name} ({symbol}) {weight_value:g}%를 포트폴리오에 반영했습니다."


def delete_portfolio_asset(symbol):
    """포트폴리오에서 특정 보유 종목을 삭제합니다."""
    st.session_state.portfolio = [
        item for item in st.session_state.portfolio if item["symbol"] != symbol
    ]
    save_portfolio(st.session_state.portfolio)


def get_portfolio_sector_summary(portfolio):
    sector_weights = {}
    for item in portfolio:
        symbol = item["symbol"]
        if symbol == "CASH":
            sector = "현금"
        else:
            company_info = get_company_info(symbol)
            sector = company_info.get("sector") or "업종 정보 없음"

        sector_weights[sector] = sector_weights.get(sector, 0) + item["weight"]

    return sector_weights


def analyze_portfolio(portfolio):
    """단순 참고 기준으로 포트폴리오 집중도와 분산 상태를 계산합니다."""
    if not portfolio:
        return {
            "total_weight": 0,
            "largest": None,
            "top3_weight": 0,
            "cash_weight": 0,
            "sector_weights": {},
            "notes": ["보유 종목과 비중을 입력하면 분산 상태를 점검할 수 있습니다."],
        }

    sorted_assets = sorted(portfolio, key=lambda item: item["weight"], reverse=True)
    total_weight = sum(item["weight"] for item in portfolio)
    largest = sorted_assets[0]
    top3_weight = sum(item["weight"] for item in sorted_assets[:3])
    cash_weight = sum(item["weight"] for item in portfolio if item["symbol"] == "CASH")
    sector_weights = get_portfolio_sector_summary(portfolio)

    notes = []
    if abs(total_weight - 100) > 1:
        notes.append(f"전체 비중 합계가 {total_weight:.1f}%입니다. 100% 기준으로 다시 확인해 보세요.")
    else:
        notes.append("전체 비중 합계가 100%에 가깝습니다.")

    if largest["weight"] >= 40:
        notes.append(f"{largest['name']} 비중이 {largest['weight']:.1f}%로 집중 위험 가능성이 있습니다.")

    if top3_weight >= 80:
        notes.append(f"상위 3개 종목 합계가 {top3_weight:.1f}%로 집중도가 높을 수 있습니다.")

    for sector, weight in sector_weights.items():
        if sector != "현금" and weight >= 50:
            notes.append(f"{sector} 비중이 {weight:.1f}%로 업종 집중 가능성이 있습니다.")

    if cash_weight <= 0:
        notes.append("현금 비중이 입력되지 않았습니다. 변동성 대응 여력을 함께 확인해 보세요.")
    else:
        notes.append(f"현금 비중은 {cash_weight:.1f}%입니다.")

    notes.append("이 기준은 절대적인 투자 규칙이 아니라 참고용 점검 기준입니다.")
    return {
        "total_weight": total_weight,
        "largest": largest,
        "top3_weight": top3_weight,
        "cash_weight": cash_weight,
        "sector_weights": sector_weights,
        "notes": notes,
    }


def get_investor_profile():
    return {
        "experience": st.session_state.get("investor_experience_level", "아직 입력 안 함"),
        "horizon": st.session_state.get("investor_time_horizon", "아직 입력 안 함"),
        "risk_tolerance": st.session_state.get("investor_risk_tolerance", "아직 입력 안 함"),
        "max_loss": st.session_state.get("investor_max_loss_tolerance", 10),
        "strategy": st.session_state.get("investor_preferred_strategy", "아직 입력 안 함"),
        "cash_need": st.session_state.get("investor_cash_need", "아직 입력 안 함"),
    }


def classify_investor_profile(profile):
    risk_label = profile.get("risk_tolerance")
    horizon_label = profile.get("horizon")
    max_loss = profile.get("max_loss")

    if risk_label in {"낮음", "중간 이하"} or max_loss <= 5:
        return "보수형"
    if risk_label in {"높음", "매우 높음"} and horizon_label in {"1년 이상", "3년 이상"} and max_loss >= 20:
        return "공격형 장기 투자자"
    if horizon_label in {"1개월 미만", "3개월 이내"}:
        return "단기 변동성 민감형"
    if risk_label in {"중간", "중간 이하"}:
        return "균형형"
    return "성향 추가 확인 필요"


def build_investor_profile_context():
    if not is_stock_app_mode():
        return ""

    profile = get_investor_profile()
    profile_type = classify_investor_profile(profile)
    return (
        "사용자 투자 성향 참고 자료:\n"
        f"- 추정 성향: {profile_type}\n"
        f"- 투자 경험: {profile['experience']}\n"
        f"- 투자 기간: {profile['horizon']}\n"
        f"- 위험 감수 성향: {profile['risk_tolerance']}\n"
        f"- 감내 가능한 손실폭: {profile['max_loss']}%\n"
        f"- 선호 전략: {profile['strategy']}\n"
        f"- 현금 필요도: {profile['cash_need']}\n"
        "- 위 성향은 사용자가 입력한 자기보고 정보이며 확정 진단이 아니다.\n"
        "- 답변에서는 이 성향과 선택 종목/포트폴리오 리스크가 맞는지 점검한다."
    )


def get_selected_stock_label():
    symbol = st.session_state.get("selected_symbol")
    stock_name = st.session_state.get("selected_stock_name")
    if not symbol:
        return "선택된 종목 없음"

    return f"{stock_name} ({symbol})"


def get_quick_prompt_text(npc_role, prompt_text):
    """빠른 질문 버튼은 전송하지 않고 입력 초안으로 사용할 문장을 만듭니다."""
    if npc_role == "종목 분석가":
        selected_stock = st.session_state.get("selected_stock_name") or st.session_state.get("selected_symbol")
        if not selected_stock:
            return "먼저 종목을 선택해주세요."
        return f"{selected_stock}에 대해 {prompt_text}"

    return prompt_text


def format_optional_number(value, digits=2):
    if value is None:
        return "정보 없음"

    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "정보 없음"


def format_optional_percent(value):
    if value is None:
        return "정보 없음"

    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "정보 없음"


def format_signed_price_change(value, currency):
    if value is None:
        return "정보 없음"

    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return "정보 없음"

    sign = "+" if number_value > 0 else ""
    if currency == "KRW":
        return f"{sign}{number_value:,.0f}원"

    return f"{sign}{number_value:,.2f} {currency or ''}".strip()


def format_market_cap(value):
    if value is None:
        return "정보 없음"

    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return "정보 없음"

    if number_value >= 1_000_000_000_000:
        return f"{number_value / 1_000_000_000_000:,.2f}조"
    if number_value >= 1_000_000_000:
        return f"{number_value / 1_000_000_000:,.2f}B"
    return f"{number_value:,.0f}"


def format_prompt_value(value, suffix=""):
    if value in (None, "", "정보 없음"):
        return "정보 없음"

    if isinstance(value, (int, float)):
        return f"{value:,.2f}{suffix}"

    return f"{value}{suffix}"


def calculate_history_price_change(history):
    if history is None or history.empty or "Close" not in history:
        return "정보 없음"

    close_prices = history["Close"].dropna()
    if len(close_prices) < 2:
        return "정보 없음"

    first_close = float(close_prices.iloc[0])
    last_close = float(close_prices.iloc[-1])
    if first_close == 0:
        return "정보 없음"

    price_change = last_close - first_close
    change_percent = (price_change / first_close) * 100
    sign = "+" if price_change > 0 else ""
    return f"{sign}{price_change:,.2f} ({change_percent:+.2f}%)"


def format_indicator_number(value, digits=2, suffix=""):
    if value is None:
        return "정보 없음"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "정보 없음"


def format_indicator_percent(value):
    if value is None:
        return "정보 없음"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "정보 없음"


def describe_moving_average_position(current_price, average_value, label):
    if current_price is None or average_value is None:
        return f"{label}: 정보 없음"
    relation = "위" if current_price >= average_value else "아래"
    return f"{label}: {format_indicator_number(average_value)} / 현재가는 {label} {relation}"


def describe_volume_signal(indicators):
    ratio = indicators.get("volume_vs_avg_ratio")
    is_above = indicators.get("is_volume_above_average")
    if ratio is None or is_above is None:
        return "정보 없음"
    direction = "많음" if is_above else "적음"
    return f"최근 20일 평균 대비 {ratio:.2f}배로 {direction}"


def describe_high_low_position(indicators):
    high_distance = indicators.get("distance_from_3mo_high_percent")
    low_distance = indicators.get("distance_from_3mo_low_percent")
    if high_distance is None and low_distance is None:
        return "정보 없음"
    return (
        f"3개월 고점 대비 {format_indicator_percent(high_distance)}, "
        f"3개월 저점 대비 {format_indicator_percent(low_distance)}"
    )


def build_technical_indicator_context(symbol):
    """종목 분석가가 참고할 기술적 지표 context를 만듭니다."""
    if not symbol:
        return "종목 분석 보조 지표: 선택 종목 없음"

    history_6mo = get_stock_history(symbol, period="6mo")
    indicators = calculate_technical_indicators(history_6mo)
    if indicators.get("message"):
        return f"종목 분석 보조 지표: {indicators['message']}"

    current_price = indicators.get("current_close")
    return (
        "종목 분석 보조 지표:\n"
        f"- 최근 1개월 수익률: {format_indicator_percent(indicators.get('return_1mo'))}\n"
        f"- 최근 3개월 수익률: {format_indicator_percent(indicators.get('return_3mo'))}\n"
        f"- 최근 6개월 수익률: {format_indicator_percent(indicators.get('return_6mo'))}\n"
        f"- 5일 이동평균선: {describe_moving_average_position(current_price, indicators.get('ma5'), '5일선')}\n"
        f"- 20일 이동평균선: {describe_moving_average_position(current_price, indicators.get('ma20'), '20일선')}\n"
        f"- 60일 이동평균선: {describe_moving_average_position(current_price, indicators.get('ma60'), '60일선')}\n"
        f"- 최근 20일 평균 거래량: {format_indicator_number(indicators.get('avg_volume_20d'), digits=0)}\n"
        f"- 오늘 거래량: {format_indicator_number(indicators.get('latest_volume'), digits=0)}\n"
        f"- 거래량 해석: {describe_volume_signal(indicators)}\n"
        f"- 최근 3개월 고점: {format_indicator_number(indicators.get('high_3mo'))}\n"
        f"- 최근 3개월 저점: {format_indicator_number(indicators.get('low_3mo'))}\n"
        f"- 고점/저점 위치: {describe_high_low_position(indicators)}\n"
        f"- 최근 변동성: {format_indicator_percent(indicators.get('volatility_20d'))}\n"
        f"- RSI 14일: {format_indicator_number(indicators.get('rsi_14'))} ({indicators.get('rsi_status') or '확인 불가'})"
    )


def format_news_for_prompt(news_items):
    news_lines = []
    for news in (news_items or [])[:5]:
        title = news.get("title") or "제목 없음"
        publisher = news.get("publisher") or "언론사 정보 없음"
        published_at = news.get("published_at") or "게시 시각 정보 없음"
        news_lines.append(f"{title} / {publisher} / {published_at}")

    return " | ".join(news_lines) if news_lines else "정보 없음"


def build_portfolio_reference_context():
    """포트폴리오 코치가 참고할 보유 종목과 단순 점검 결과를 만듭니다."""
    portfolio = st.session_state.get("portfolio", [])
    if not portfolio:
        return "포트폴리오 참고 자료: 없음"

    analysis = analyze_portfolio(portfolio)
    holdings = []
    for item in portfolio:
        average_price = item.get("average_price")
        average_price_text = f", 평균 매수가 {average_price:,.2f}" if average_price is not None else ""
        holdings.append(f"{item['name']}({item['symbol']}) {item['weight']:.1f}%{average_price_text}")

    sector_summary = ", ".join(
        f"{sector} {weight:.1f}%" for sector, weight in analysis["sector_weights"].items()
    )
    notes_summary = " | ".join(analysis["notes"])
    return (
        "포트폴리오 참고 자료:\n"
        f"- 보유 목록: {'; '.join(holdings)}\n"
        f"- 전체 비중 합계: {analysis['total_weight']:.1f}%\n"
        f"- 상위 3개 종목 집중도: {analysis['top3_weight']:.1f}%\n"
        f"- 현금 비중: {analysis['cash_weight']:.1f}%\n"
        f"- 업종별 비중: {sector_summary or '정보 없음'}\n"
        f"- 참고 점검: {notes_summary}"
    )


def build_stock_reference_context():
    """LLM이 선택 종목을 기준으로 답하도록 필요한 주식 참고 자료만 짧게 만듭니다."""
    symbol = st.session_state.get("selected_symbol")
    stock_name = st.session_state.get("selected_stock_name")
    if not symbol:
        return "선택 종목 참고 자료: 없음"

    quote = get_stock_quote(symbol)
    company_info = get_company_info(symbol)
    history = get_stock_history(symbol, period="1mo")
    news_items = get_stock_news(symbol)
    news_summary = format_news_for_prompt(news_items)
    change_percent = quote.get("change_percent")
    if isinstance(change_percent, (int, float)):
        direction_label = "상승" if change_percent > 0 else "하락" if change_percent < 0 else "보합"
    else:
        direction_label = "확인 불가"

    return (
        "선택 종목 참고 자료:\n"
        f"- 선택 종목명: {stock_name}\n"
        f"- 티커: {symbol}\n"
        f"- 조회 가격: {format_optional_number(quote.get('current_price'))} {quote.get('currency') or ''}\n"
        f"- 전일 대비 등락률: {format_optional_percent(quote.get('change_percent'))}\n"
        f"- 등락 방향: {direction_label}\n"
        f"- 최근 1개월 가격 변화: {calculate_history_price_change(history)}\n"
        f"- 업종: {company_info.get('sector') or '정보 없음'}\n"
        f"- 산업: {company_info.get('industry') or '정보 없음'}\n"
        f"- PER: {format_prompt_value(company_info.get('per'))}\n"
        f"- PBR: {format_prompt_value(company_info.get('pbr'))}\n"
        f"- 배당수익률: {format_optional_percent(company_info.get('dividend_yield'))}\n"
        f"- 52주 최고가: {format_optional_number(company_info.get('fifty_two_week_high'))}\n"
        f"- 52주 최저가: {format_optional_number(company_info.get('fifty_two_week_low'))}\n"
        f"- 데이터 기준 시각: {quote.get('data_timestamp') or '확인 불가'}\n"
        f"- 최근 뉴스: {news_summary}\n"
        f"{build_technical_indicator_context(symbol)}"
    )


def build_npc_stock_handling_rules(npc_role):
    """선택 종목이 있거나 없을 때 NPC별로 어떻게 답할지 짧게 안내합니다."""
    handling_rules = {
        "시장 해설가": "시장 해설가는 선택 종목이 없어도 시장 분위기와 지수 흐름 중심으로 답변한다. 선택 종목 뉴스가 있으면 제목, 언론사, 게시 시각만 참고하고 시장 영향은 가능성으로 구분한다.",
        "종목 분석가": "종목 분석가는 선택 종목이 없고 질문에도 종목이 없으면 먼저 사이드바에서 종목 선택을 요청한다. 선택 종목이 있으면 사용자가 종목명을 다시 쓰지 않아도 그 종목을 기준으로 답변한다. 가격 흐름, 이동평균선, 거래량, 고점·저점, RSI는 참고 지표로 설명하고 매수·매도 신호처럼 단정하지 않는다. 뉴스 제목만 보고 세부 사실이나 주가 변동 원인을 확정하지 않는다.",
        "초보 투자 튜터": "초보 투자 튜터는 선택 종목이 없어도 일반 투자 용어를 설명할 수 있다. 선택 종목이 있으면 예시로만 참고한다.",
        "포트폴리오 코치": "포트폴리오 코치는 보유 종목과 비중 입력이 없으면 먼저 보유 종목과 비중을 요청한다. 선택 종목은 포트폴리오 일부 예시로만 참고한다.",
        MENTAL_COACH_ROLE: "투자 멘탈 코치는 선택 종목 정보를 투자 판단 지시가 아니라 사용자의 감정과 상황을 정리하는 참고 자료로만 사용한다.",
    }
    return handling_rules.get(npc_role, "선택 종목은 참고 자료로만 사용하고, 질문 의도를 우선한다.")


def get_selected_stock_summary(history_period=None):
    symbol = st.session_state.get("selected_symbol")
    if not symbol:
        return None

    if history_period is None:
        history_period_label = st.session_state.get("stock_history_period_label", "1개월")
        history_period = STOCK_HISTORY_PERIOD_OPTIONS.get(history_period_label, "1mo")

    with st.spinner("주가 데이터를 불러오는 중입니다..."):
        quote = get_stock_quote(symbol)
        history = get_stock_history(symbol, period=history_period)
    with st.spinner("기업 정보를 불러오는 중입니다..."):
        company_info = get_company_info(symbol)
    return {
        "quote": quote,
        "company_info": company_info,
        "news_items": st.session_state.get("selected_stock_news", {}).get(symbol, []),
        "history": history,
        "history_period": history_period,
    }


def get_selected_stock_age_trend_inputs():
    selected_symbol = st.session_state.get("selected_symbol")
    selected_stock_name = st.session_state.get("selected_stock_name")
    selected_sector = None
    if selected_symbol:
        try:
            company_info = get_company_info(selected_symbol)
            selected_sector = company_info.get("sector") or company_info.get("industry")
        except Exception as error:
            log_app_error("연령대별 투자 경향 선택 종목 업종 조회 실패", error)
    return selected_symbol, selected_stock_name, selected_sector


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_cached_age_group_investment_trends(selected_symbol, selected_stock_name, selected_sector, fetch_version="v3"):
    return fetch_age_group_investment_trends(
        selected_symbol=selected_symbol,
        selected_stock_name=selected_stock_name,
        selected_sector=selected_sector,
    )


def get_current_age_group_trend_data():
    selected_symbol, selected_stock_name, selected_sector = get_selected_stock_age_trend_inputs()
    if not selected_symbol and not selected_stock_name:
        return [], "선택된 종목이 없어 연령대별 연관성 분석을 표시할 수 없습니다."

    try:
        return get_cached_age_group_investment_trends(selected_symbol, selected_stock_name, selected_sector, "v3")
    except Exception as error:
        log_app_error("연령대별 투자 경향 자료 조회 실패", error)
        return [], "자료를 불러오지 못했습니다. 출처가 확인된 자료가 없어 임의 분석을 표시하지 않습니다."


def render_age_group_trends_section(trend_data, selected_stock_name=None, selected_symbol=None, error_message=""):
    st.caption("아래 내용은 공개 자료와 뉴스 기반의 참고용 경향 분석이며, 투자 추천이 아닙니다.")

    if not selected_stock_name and not selected_symbol:
        st.info("선택된 종목이 없어 연령대별 연관성 분석을 표시할 수 없습니다.")
        return

    if not trend_data:
        st.info(error_message or "자료를 불러오지 못했습니다.")
        st.caption("출처가 확인된 자료가 없어 임의 분석을 표시하지 않습니다.")
        st.markdown("##### 확인 가능한 출처 후보")
        for source in SOURCE_REFERENCE_LINKS:
            st.markdown(f"- [{source['name']}]({source['url']})")
        st.caption("위 출처에서 연령대별 투자 자료가 확인되면 해당 자료 범위에서만 분석합니다.")
        return

    rows = []
    source_rows = []
    chart_rows = []
    for trend in trend_data:
        rows.append(
            {
                "연령대": trend.get("age_group") or "",
                "선호 업종": ", ".join(trend.get("preferred_sectors") or []) or "",
                "대표 관심 종목": ", ".join(trend.get("representative_stocks") or []) or "",
                "투자 성향": trend.get("investment_style") or "",
                "현재 선택 종목과의 연관성": trend.get("relation_to_selected_stock")
                or "직접 연관 자료는 확인되지 않았습니다",
                "근거 출처": trend.get("source_name") or "",
                "기준일": trend.get("published_at") or "확인 불가",
            }
        )
        source_name = trend.get("source_name") or "자료명 없음"
        source_url = trend.get("source_url") or ""
        if source_url:
            source_rows.append((source_name, source_url))
        numeric_value = trend.get("numeric_value")
        if isinstance(numeric_value, (int, float)):
            chart_rows.append(
                {
                    "연령대": trend.get("age_group") or "",
                    trend.get("numeric_label") or "수치": numeric_value,
                }
            )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows).set_index("연령대")
        st.bar_chart(chart_df)
    else:
        st.caption("시각화 가능한 수치 데이터가 없습니다.")

    if source_rows:
        st.markdown("##### 출처 목록")
        seen_sources = set()
        for source_name, source_url in source_rows:
            source_key = (source_name, source_url)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            st.markdown(f"- [{source_name}]({source_url})")
    else:
        st.info("출처가 확인된 자료가 없어 임의 분석을 표시하지 않습니다.")


def render_stock_answer_graph_analysis(npc_role, message=None):
    if not is_stock_app_mode() or npc_role not in STOCK_NPC_ROLES:
        return

    symbol = (message or {}).get("analysis_symbol") or st.session_state.get("selected_symbol")
    stock_name = (message or {}).get("analysis_stock_name") or st.session_state.get("selected_stock_name") or symbol
    if not symbol:
        with st.expander("선택 종목 전문 그래프 분석", expanded=False):
            st.info("그래프 분석을 보려면 사이드바에서 종목을 먼저 선택해 주세요.")
        return

    try:
        history = get_stock_history(symbol, period="6mo")
        if history is None or history.empty or "Close" not in history.columns or len(history) < 20:
            with st.expander("선택 종목 전문 그래프 분석", expanded=False):
                st.warning("그래프 분석에 필요한 가격 데이터가 부족합니다.")
            return

        chart_data = history.copy()
        chart_data["5일 이동평균"] = chart_data["Close"].rolling(window=5).mean()
        chart_data["20일 이동평균"] = chart_data["Close"].rolling(window=20).mean()
        chart_data["60일 이동평균"] = chart_data["Close"].rolling(window=60).mean()
        indicators = calculate_technical_indicators(history)
        current_price = indicators.get("current_close")

        with st.expander("선택 종목 전문 그래프 분석", expanded=False):
            st.caption("아래 그래프는 Yahoo Finance 가격 데이터를 기반으로 한 참고용 분석입니다. 매수·매도 신호가 아닙니다.")
            st.markdown(f"##### {stock_name} · {symbol} 6개월 가격 흐름")
            st.line_chart(
                chart_data[["Close", "5일 이동평균", "20일 이동평균", "60일 이동평균"]].rename(columns={"Close": "종가"})
            )

            metric_columns = st.columns(4)
            metric_columns[0].metric("1개월 수익률", format_indicator_percent(indicators.get("return_1mo")))
            metric_columns[1].metric("3개월 수익률", format_indicator_percent(indicators.get("return_3mo")))
            metric_columns[2].metric("20일 변동성", format_indicator_percent(indicators.get("volatility_20d")))
            metric_columns[3].metric("RSI 14일", f"{format_indicator_number(indicators.get('rsi_14'))} / {indicators.get('rsi_status') or '확인 불가'}")

            if "Volume" in chart_data.columns:
                st.markdown("##### 거래량 흐름")
                st.bar_chart(chart_data[["Volume"]].rename(columns={"Volume": "거래량"}))

            st.markdown("##### 그래프 해석")
            st.write(f"- 이동평균선 위치: {describe_moving_average_position(current_price, indicators.get('ma5'), '5일선')} / {describe_moving_average_position(current_price, indicators.get('ma20'), '20일선')} / {describe_moving_average_position(current_price, indicators.get('ma60'), '60일선')}")
            st.write(f"- 거래량 해석: {describe_volume_signal(indicators)}")
            st.write(f"- 고점·저점 위치: {describe_high_low_position(indicators)}")
            st.write(f"- 보조지표 해석: RSI는 {format_indicator_number(indicators.get('rsi_14'))}이며, 상태는 {indicators.get('rsi_status') or '확인 불가'}입니다.")
            st.write("- 종합적으로 가격 방향, 이동평균선 배열, 거래량 변화, RSI를 함께 봐야 하며 한 지표만으로 결론을 내리면 안 됩니다.")
    except Exception as error:
        log_app_error("선택 종목 전문 그래프 분석 실패", error)
        with st.expander("선택 종목 전문 그래프 분석", expanded=False):
            st.warning("그래프 분석을 불러오지 못했습니다.")


def build_report_helper_response(user_input, npc):
    lowered_input = user_input.lower()

    if any(keyword in lowered_input for keyword in ["요약", "정리", "보고서", "문서", "회의록"]):
        return (
            f"{npc['example_response']}\n"
            "1. 주제와 목적을 한 줄로 먼저 적어 보세요.\n"
            "2. 중요한 내용을 2~3개 묶음으로 나누면 훨씬 읽기 쉬워집니다.\n"
            "3. 마지막에는 결론이나 요청 사항을 한 줄로 정리하면 좋습니다."
        )

    return (
        f"{npc['example_response']}\n"
        "초안이나 메모를 보내 주시면 주제, 근거, 결론 순서로 다시 정리해 드릴게요."
    )


def build_python_tutor_response(user_input, npc):
    lowered_input = user_input.lower()

    if any(keyword in lowered_input for keyword in ["오류", "에러", "코드", "함수", "반복문", "변수"]):
        return (
            f"{npc['example_response']}\n"
            "1. 먼저 코드가 기대하는 동작을 짚어 볼게요.\n"
            "2. 그다음 실제로 어디에서 값이 달라지는지 확인해 보세요.\n"
            "3. 마지막으로 수정 예시를 적용해서 다시 실행해 보면 됩니다."
        )

    return (
        f"{npc['example_response']}\n"
        "배우고 싶은 Python 주제나 코드 예시를 보내 주시면 쉬운 말로 차근차근 설명해 드릴게요."
    )


def build_counselor_response(user_input, npc):
    lowered_input = user_input.lower()

    if any(keyword in lowered_input for keyword in ["집", "수업", "회의", "못가", "못 가", "계절학기"]):
        return (
            "집에 가고 싶은데 수업과 회의 때문에 못 가는 상황이라 답답하겠어요.\n"
            "지금은 마음을 달래는 말보다, 오늘을 버틸 수 있게 작게 나누는 게 먼저 좋아 보여요.\n\n"
            "1. 먼저 오늘 반드시 해야 하는 일과 미뤄도 되는 일을 나눠 보세요.\n"
            "2. 수업과 회의 사이에 10분이라도 쉬는 시간을 정해 두세요.\n"
            "3. 집에 못 가는 아쉬움은 오늘 끝난 뒤 할 작은 보상으로 바꿔 보세요. 예를 들면 좋아하는 음료 사기, 짧게 산책하기, 일찍 씻고 쉬기처럼요.\n"
            "4. 가능하다면 회의 담당자에게 시간을 줄일 수 있는지 또는 온라인 참여가 가능한지 한 번 확인해 보세요.\n\n"
            "지금 바로 할 일은 '오늘 일정 3개 적기 → 가장 힘든 것 하나 표시하기 → 끝난 뒤 보상 하나 정하기'입니다."
        )

    if any(keyword in lowered_input for keyword in ["불안", "걱정", "스트레스", "고민", "진로", "힘들"]):
        return (
            f"{npc['example_response']}\n"
            "지금은 문제를 한 번에 해결하려고 하기보다, 가장 부담되는 부분 하나만 작게 줄이는 게 좋겠습니다.\n"
            "1. 지금 걱정을 한 문장으로 적습니다.\n"
            "2. 내가 바꿀 수 있는 것과 바꿀 수 없는 것을 나눕니다.\n"
            "3. 바꿀 수 있는 것 중 10분 안에 할 수 있는 행동 하나를 고릅니다."
        )

    return (
        f"{npc['example_response']}\n"
        "우선 상황을 세 부분으로 나눠 보겠습니다.\n"
        "1. 지금 가장 불편한 점\n"
        "2. 오늘 꼭 해야 하는 일\n"
        "3. 바로 줄일 수 있는 부담\n"
        "이 순서로 보면 다음 행동이 더 선명해집니다."
    )


def build_idea_planner_response(user_input, npc):
    lowered_input = user_input.lower()

    if any(keyword in lowered_input for keyword in ["아이디어", "기획", "이름", "브랜드", "서비스", "콘텐츠"]):
        return (
            f"{npc['example_response']}\n"
            "1. 실용적인 방향\n"
            "2. 개성이 강한 방향\n"
            "3. 가볍고 대중적인 방향\n"
            "이렇게 나누어 보면 비교가 훨씬 쉬워집니다."
        )

    return (
        f"{npc['example_response']}\n"
        "주제와 대상, 원하는 분위기를 알려 주시면 여러 갈래 아이디어로 펼쳐 드릴게요."
    )


MENTAL_COACH_ROLE = "투자 멘탈 코치"
STOCK_NPC_ROLES = {"시장 해설가", "종목 분석가", "초보 투자 튜터", "포트폴리오 코치", MENTAL_COACH_ROLE}
INVESTMENT_DISCLAIMER = "※ 본 내용은 투자 권유가 아닌 참고용 정보입니다. 실제 투자 판단과 책임은 사용자에게 있습니다."
MENTAL_COACH_DISCLAIMER = "※ 본 내용은 투자 권유가 아닌 참고용 정보이며, 심리 상담이나 의료적 조언을 대체하지 않습니다. 실제 투자 판단과 책임은 사용자에게 있습니다."
COMMON_INVESTMENT_RULES = (
    "공통 투자 답변 규칙:\n"
    "- 매수, 매도, 목표주가를 단정적으로 추천하지 않습니다.\n"
    "- 수익을 보장하는 표현을 사용하지 않습니다.\n"
    "- 확인 가능한 정보와 해석을 구분해서 설명합니다.\n"
    "- 데이터 기준 시각을 확인할 수 있으면 표시합니다.\n"
    "- 정보가 부족하면 추측하지 말고 부족한 정보를 안내합니다.\n"
    "- 투자 판단은 사용자 투자 성향, 투자 기간, 손실 감내도, 현금 비중, 포트폴리오 집중도를 함께 고려해 설명합니다.\n"
    "- 단일 결론 대신 강세, 중립, 약세 시나리오와 각 시나리오에서 확인할 조건을 구분합니다.\n"
    "- 확률, 기대수익률, 목표가를 임의로 만들지 않습니다.\n"
    f"- 답변 마지막에는 다음 문구를 중복 없이 한 번만 표시합니다: {INVESTMENT_DISCLAIMER}"
)
PROFESSIONAL_STOCK_FRAMEWORK = (
    "전문 분석 프레임워크:\n"
    "- 데이터 품질: 가격, 거래량, 기술지표, 기업 정보, 뉴스 제목, 포트폴리오 자료 중 실제 제공된 자료만 사용한다.\n"
    "- 투자 성향 적합성: 사용자의 투자 기간, 경험 수준, 손실 감내도, 선호 전략과 맞는지 따로 평가한다.\n"
    "- 리스크 관리: 손실 한도, 비중 쏠림, 현금 비중, 섹터 집중, 이벤트 리스크를 점검한다.\n"
    "- 시나리오 분석: 강세/중립/약세 관점에서 확인 조건을 나누되 결과를 예언하지 않는다.\n"
    "- 의사결정 체크리스트: 지금 바로 결론을 내리기 전에 확인해야 할 자료를 제시한다.\n"
    "- 투자 추천이 아니라 사용자가 더 나은 판단을 하도록 돕는 분석으로 표현한다.\n"
)
STOCK_ANSWER_FORMATS = {
    "시장 해설가": [
        "시장 분위기 한 줄 요약",
        "주요 지수 흐름",
        "주요 업종 또는 섹터",
        "영향을 준 뉴스",
        "투자 성향별 해석",
        "강세·중립·약세 시나리오",
        "앞으로 확인할 변수",
        "최종 결론",
    ],
    "종목 분석가": [
        "[종목 분석 리포트]",
        "핵심 요약",
        "차트 흐름",
        "이동평균선 분석",
        "거래량 분석",
        "고점/저점 위치",
        "보조지표 참고",
        "긍정 요인",
        "리스크 요인",
        "투자 성향 적합성",
        "강세·중립·약세 시나리오",
        "다음 확인 포인트",
        "최종 결론",
        "참고 문구",
    ],
    "초보 투자 튜터": [
        "한 문장 정의",
        "쉬운 비유",
        "숫자를 사용한 예시",
        "투자에서 사용하는 이유",
        "흔한 오해",
        "최종 결론",
    ],
    "포트폴리오 코치": [
        "입력된 종목과 비중 요약",
        "가장 비중이 큰 종목",
        "집중 위험",
        "업종 분산 상태",
        "현금 비중",
        "투자 성향 적합성",
        "리밸런싱 전 확인할 조건",
        "사용자가 추가로 확인할 질문",
        "최종 결론",
    ],
    MENTAL_COACH_ROLE: [
        "감정 공감 한 문장",
        "현재 상황 정리",
        "피해야 할 충동 행동",
        "차분히 확인할 체크리스트",
        "다음 질문 1~2개",
        "최종 결론",
        "투자 참고용 문구",
    ],
}


CHICKEN_PRICE = 22000
COFFEE_PRICE = 4500
GUKBAP_PRICE = 10000


def append_investment_disclaimer(response_text, disclaimer=INVESTMENT_DISCLAIMER):
    """투자 안전 문구가 fallback 답변에 한 번만 붙도록 관리합니다."""
    if disclaimer in response_text:
        return response_text

    return f"{response_text.rstrip()}\n\n{disclaimer}"


def extract_stock_keyword(user_input):
    """질문에서 종목명이나 영문 티커처럼 보이는 단어를 간단히 찾습니다."""
    ticker_match = re.search(r"\b[A-Z]{1,5}\b", user_input)
    if ticker_match:
        return ticker_match.group(0)

    known_names = ["삼성전자", "SK하이닉스", "현대차", "네이버", "카카오", "테슬라", "애플", "엔비디아", "마이크로소프트"]
    for name in known_names:
        if name.lower() in user_input.lower():
            return name

    return "입력한 종목"


def build_stock_snapshot(stock_keyword):
    """조회 가능한 경우 Yahoo Finance 기준 종목 정보를 짧게 만듭니다."""
    symbol, stock_name, error_message = resolve_symbol(stock_keyword)
    if error_message and st.session_state.get("selected_symbol"):
        symbol = st.session_state.selected_symbol
        stock_name = st.session_state.selected_stock_name

    if not symbol:
        return (
            f"관심 종목: {stock_keyword}\n"
            "조회 가격/등락률: 종목을 선택하거나 올바른 티커를 입력하면 확인할 수 있습니다.\n"
            "확인 포인트: 최근 가격 흐름, 실적 발표, 업종 뉴스, 거래량 변화"
        )

    quote = get_stock_quote(symbol)
    return (
        f"관심 종목: {stock_name} ({symbol})\n"
        f"조회 가격: {format_optional_number(quote.get('current_price'))} {quote.get('currency') or ''}\n"
        f"전일 종가: {format_optional_number(quote.get('previous_close'))}\n"
        f"등락률: {format_optional_percent(quote.get('change_percent'))}\n"
        f"데이터 기준 시각: {quote.get('data_timestamp') or '확인 불가'}\n"
        "확인 포인트: 최근 가격 흐름, 실적 발표, 업종 뉴스, 거래량 변화"
    )


def build_market_commentary_response(user_input, npc):
    response_text = (
        f"{npc['example_response']}\n\n"
        "데이터 기준 시각: 실시간 시장 데이터 API 연결 전이라 앱 안에서는 별도 기준 시각을 확인할 수 없습니다.\n\n"
        "1. 시장 분위기 한 줄 요약: 오늘 분위기는 지수 방향만 보지 말고 금리, 환율, 실적 기대를 함께 봐야 합니다.\n"
        "2. 주요 지수 흐름: 코스피·코스닥·나스닥처럼 대표 지수가 같은 방향인지, 서로 엇갈리는지 확인해 보세요.\n"
        "3. 주요 업종 또는 섹터: 반도체, 2차전지, 금융, 성장주처럼 강한 섹터와 약한 섹터를 나누면 흐름이 더 선명해집니다.\n"
        "4. 영향을 준 뉴스: 큰 뉴스가 있어도 실제로 어느 섹터에 돈이 몰리는지 확인해야 합니다.\n"
        "5. 앞으로 확인할 변수: 금리, 환율, 실적 발표, 주요 경제지표, 정책 이슈를 함께 확인해 보세요."
    )
    return append_investment_disclaimer(response_text)


def build_stock_analyst_response(user_input, npc):
    stock_keyword = extract_stock_keyword(user_input)
    if stock_keyword == "입력한 종목" and st.session_state.get("selected_stock_name"):
        stock_keyword = st.session_state.selected_stock_name

    selected_symbol = st.session_state.get("selected_symbol")
    technical_context = build_technical_indicator_context(selected_symbol) if selected_symbol else "종목 분석 보조 지표: 선택 종목 없음"
    profile_type = classify_investor_profile(get_investor_profile())
    profile_context = build_investor_profile_context()
    response_text = (
        "[종목 분석 리포트]\n\n"
        f"{npc['example_response']}\n\n"
        f"{build_stock_snapshot(stock_keyword)}\n\n"
        f"{technical_context}\n\n"
        f"{profile_context}\n\n"
        "1. 핵심 요약\n"
        "- 현재 주가 흐름은 가격 흐름, 이동평균선, 거래량을 함께 보며 판단해야 합니다.\n\n"
        "2. 차트 흐름\n"
        "- 최근 1개월과 3개월 흐름을 비교해 상승 추세, 하락 추세, 횡보 가능성을 구분해 보세요. 데이터가 부족하면 부족하다고 봐야 합니다.\n\n"
        "3. 이동평균선 분석\n"
        "- 현재가가 5일, 20일, 60일 이동평균선 위인지 아래인지로 단기 흐름과 중기 흐름을 나누어 확인합니다.\n\n"
        "4. 거래량 분석\n"
        "- 최근 거래량이 20일 평균보다 많은지 적은지, 가격 변화와 함께 나타났는지 확인해야 합니다.\n\n"
        "5. 고점/저점 위치\n"
        "- 최근 3개월 고점과 저점 대비 현재 위치를 보고 고점 근처, 저점 근처, 중간 구간인지 확인합니다.\n\n"
        "6. 보조지표 참고\n"
        "- RSI가 있으면 과열 또는 침체 가능성을 쉬운 말로 참고하되, 단독 판단 기준으로 쓰지 않습니다.\n\n"
        "7. 긍정 요인\n"
        "- 데이터와 뉴스에서 확인 가능한 실적 개선, 업종 흐름, 거래량 동반 상승 가능성을 확인합니다.\n\n"
        "8. 리스크 요인\n"
        "- 주가 변동성, 실적 둔화, 부정적 뉴스, 업종 약세, 이동평균선 이탈 가능성을 함께 봅니다.\n\n"
        "9. 투자 성향 적합성\n"
        f"- 현재 입력된 성향은 {profile_type}으로 분류됩니다. 이 종목의 변동성, 보유 기간, 손실 감내 범위가 성향과 맞는지 먼저 확인해야 합니다.\n\n"
        "10. 강세·중립·약세 시나리오\n"
        "- 강세: 가격 흐름, 거래량, 업종 뉴스가 함께 개선되는지 확인합니다.\n"
        "- 중립: 방향성이 뚜렷하지 않으면 실적 발표와 주요 뉴스까지 기다리는 관점이 필요합니다.\n"
        "- 약세: 이동평균선 이탈, 거래량 동반 하락, 부정적 뉴스가 겹치는지 확인합니다.\n\n"
        "11. 다음 확인 포인트\n"
        "- 실적 발표, 주요 뉴스, 업종 흐름, 거래량 변화, 이동평균선 이탈 여부를 계속 확인해 보세요.\n\n"
        "12. 최종 결론\n"
        f"- {build_stock_direction_judgement(selected_symbol)}\n\n"
        "13. 참고 문구\n"
        "- 이 분석은 매수·매도 추천이 아니라 참고용 정보입니다."
    )
    return append_investment_disclaimer(response_text)


def build_beginner_investment_tutor_response(user_input, npc):
    lowered_input = user_input.lower()

    if "per" in lowered_input:
        concept = "PER은 주가가 회사의 이익에 비해 어느 정도 평가받는지 보는 지표입니다."
        example = "예를 들어 PER이 높으면 성장 기대가 크다는 뜻일 수도 있지만, 이미 비싸게 평가됐다는 뜻일 수도 있습니다."
    elif "pbr" in lowered_input:
        concept = "PBR은 주가가 회사의 순자산에 비해 어느 정도인지 보는 지표입니다."
        example = "PBR이 낮다고 무조건 싼 것은 아니고, 회사의 수익성이나 산업 특성도 같이 봐야 합니다."
    elif "etf" in lowered_input:
        concept = "ETF는 여러 종목을 한 바구니에 담아 거래하는 상품입니다."
        example = "개별주 하나보다 분산 효과가 있지만, 어떤 지수나 섹터를 따라가는지 확인해야 합니다."
    elif "배당" in lowered_input:
        concept = "배당은 회사가 이익 일부를 주주에게 나누어 주는 것입니다."
        example = "배당률만 높다고 좋은 것은 아니고, 배당이 계속 유지될 수 있는지도 중요합니다."
    else:
        concept = "투자 지표는 종목을 한쪽 면만 보지 않도록 도와주는 도구입니다."
        example = "PER, PBR, 배당, 차트, 실적을 함께 보면 한 가지 숫자에만 흔들릴 가능성이 줄어듭니다."

    response_text = (
        f"{npc['example_response']}\n\n"
        f"1. 한 문장 정의: {concept}\n"
        "2. 쉬운 비유: 투자 지표는 자동차 계기판처럼 현재 상태를 빠르게 살피는 도구라고 볼 수 있습니다.\n"
        f"3. 숫자를 사용한 예시: {example}\n"
        "4. 투자에서 사용하는 이유: 같은 업종의 다른 기업과 비교하거나, 한 가지 숫자에만 흔들리지 않기 위해 사용합니다.\n"
        "5. 흔한 오해: 지표가 낮거나 높다고 해서 무조건 좋거나 나쁜 것은 아니며 실적, 뉴스, 시장 분위기를 함께 봐야 합니다.\n\n"
        "6. 선택 종목 공통 판단 기준\n"
        f"{build_shared_stock_direction_context() or '- 선택된 종목이 없어 공통 방향성 판단은 표시하지 않습니다.'}"
    )
    return append_investment_disclaimer(response_text)


def build_portfolio_coach_response(user_input, npc):
    percentages = [int(value) for value in re.findall(r"(\d{1,3})\s*%", user_input)]
    saved_portfolio = st.session_state.get("portfolio", [])
    concentration_note = "비중 숫자가 있으면 가장 큰 비중이 40%를 넘는지 먼저 확인해 보세요."
    if percentages:
        max_weight = max(percentages)
        concentration_note = (
            f"입력한 비중 중 가장 큰 값은 {max_weight}%입니다. "
            "40%를 넘는 자산이 있다면 한 종목 또는 한 섹터에 너무 집중됐는지 점검해 보세요."
        )
    elif saved_portfolio:
        analysis = analyze_portfolio(saved_portfolio)
        largest = analysis["largest"]
        concentration_note = (
            f"저장된 포트폴리오 기준 가장 큰 비중은 {largest['name']} {largest['weight']:.1f}%입니다. "
            f"상위 3개 종목 합계는 {analysis['top3_weight']:.1f}%입니다."
        )

    profile_type = classify_investor_profile(get_investor_profile())
    profile_context = build_investor_profile_context()
    response_text = (
        f"{npc['example_response']}\n\n"
        f"{profile_context}\n\n"
        "1. 입력된 종목과 비중 요약: 입력한 보유 종목과 비중을 기준으로 분산 상태를 점검합니다.\n"
        f"2. 가장 비중이 큰 종목: {concentration_note}\n"
        "3. 집중 위험: 종목 수만 많아도 같은 섹터에 몰려 있으면 분산 효과가 약할 수 있습니다.\n"
        "4. 업종 분산 상태: 반도체, 플랫폼, 금융, 헬스케어처럼 업종이 나뉘어 있는지 확인해 보세요.\n"
        "5. 현금 비중: 급락 때 대응할 여유 자금이 있는지도 함께 확인하세요.\n"
        f"6. 투자 성향 적합성: 현재 입력된 성향은 {profile_type}입니다. 보유 비중과 변동성이 이 성향에 맞는지 확인해야 합니다.\n"
        "7. 리밸런싱 전 확인할 조건: 비중이 커진 이유가 가격 상승인지 추가 매수인지, 세금과 수수료가 있는지, 현금 필요도가 높은지 확인하세요.\n"
        "8. 사용자가 추가로 확인할 질문: 보유 종목, 비중, 투자 기간, 현금 비중을 한 줄로 정리하면 더 구체적으로 점검할 수 있습니다."
    )
    return append_investment_disclaimer(response_text)


def get_selected_stock_change_context():
    """선택 종목의 등락률이 있으면 멘탈 코치 답변에 쓸 맥락을 반환합니다."""
    symbol = st.session_state.get("selected_symbol")
    stock_name = st.session_state.get("selected_stock_name")
    if not symbol:
        return None

    try:
        quote = get_stock_quote(symbol)
    except Exception as error:
        log_app_error("투자 멘탈 코치 선택 종목 등락률 조회 실패", error)
        return None

    change_percent = quote.get("change_percent")
    if not isinstance(change_percent, (int, float)):
        return None

    return {
        "stock_name": stock_name or symbol,
        "symbol": symbol,
        "change_percent": change_percent,
        "formatted_change": format_optional_percent(change_percent),
        "data_timestamp": quote.get("data_timestamp") or "확인 불가",
    }


def pick_mental_charm(user_input):
    """사용자 감정에 맞는 그림 부적 데이터를 고릅니다."""
    lowered_input = str(user_input or "").lower()
    if any(keyword in lowered_input for keyword in ["패닉", "매도", "팔", "불안", "하락", "급락"]):
        return {
            "title": "패닉셀 봉인 부적",
            "seal": "STOP",
            "lines": ["급한 손가락은 오늘 휴식", "차트보다 호흡 먼저", "결정은 10초 뒤 다시 보기"],
        }
    if any(keyword in lowered_input for keyword in ["fomo", "조급", "놓칠", "추가 매수", "불타기"]):
        return {
            "title": "FOMO 냉각 부적",
            "seal": "COOL",
            "lines": ["남의 수익률은 내 매수 버튼이 아님", "급등 열차는 또 옵니다", "원칙 먼저 클릭은 나중"],
        }
    if any(keyword in lowered_input for keyword in ["복수", "분노", "화", "만회"]):
        return {
            "title": "복수 매매 차단 부적",
            "seal": "PAUSE",
            "lines": ["시장은 내 화풀이 상대가 아님", "화난 클릭은 냉장 보관", "물 한 모금 후 원인 메모"],
        }
    if any(keyword in lowered_input for keyword in ["후회", "그때", "살걸", "팔걸"]):
        return {
            "title": "지난 봉 복기 부적",
            "seal": "LOG",
            "lines": ["지나간 캔들은 못 돌림", "다음 원칙은 오늘 만들 수 있음", "후회 대신 기록 한 줄"],
        }
    return {
        "title": "멘탈 안정 부적",
        "seal": "CALM",
        "lines": ["불안은 잠시 내려놓기", "호흡 한 번 물 한 모금", "큰 결정은 내일의 나에게"],
    }


def build_mental_charm_block(charm):
    lines = "\n".join(charm["lines"])
    return (
        f"{MENTAL_CHARM_START}\n"
        f"title: {charm['title']}\n"
        f"seal: {charm['seal']}\n"
        f"{lines}\n"
        f"{MENTAL_CHARM_END}"
    )


def build_investment_mental_coach_response(user_input, npc):
    """투자 감정 정리용 fallback 답변을 만듭니다."""
    lowered_input = user_input.lower()
    stock_change_context = get_selected_stock_change_context()

    if stock_change_context and stock_change_context["change_percent"] > 0:
        empathy = "오늘은 상승 흐름이네요. 축하드려요. 다만 수익이 났을 때도 계획 없이 비중을 늘리는 것은 조심해야 합니다."
        situation = (
            f"선택 종목 {stock_change_context['stock_name']}({stock_change_context['symbol']})의 "
            f"전일 대비 등락률은 {stock_change_context['formatted_change']}입니다. "
            f"데이터 기준 시각은 {stock_change_context['data_timestamp']}입니다."
        )
        avoid_action = "기분이 좋아진 상태에서 계획에 없던 추가 매수나 과도한 비중 확대를 바로 결정하는 행동은 피하는 편이 좋습니다."
    elif stock_change_context and stock_change_context["change_percent"] < 0:
        empathy = "하락을 보면 불안한 게 자연스럽습니다. 지금은 바로 결론을 내리기보다 하락 이유와 내 투자 기간을 먼저 확인해보는 게 좋습니다."
        situation = (
            f"선택 종목 {stock_change_context['stock_name']}({stock_change_context['symbol']})의 "
            f"전일 대비 등락률은 {stock_change_context['formatted_change']}입니다. "
            f"데이터 기준 시각은 {stock_change_context['data_timestamp']}입니다."
        )
        avoid_action = "공포 매도, 복수 매매, 손실을 한 번에 만회하려는 큰 베팅은 잠시 멈추고 점검해 보세요."
    elif stock_change_context and stock_change_context["change_percent"] == 0:
        empathy = "오늘은 큰 등락이 확인되지 않는 흐름입니다. 그래도 마음이 흔들린다면 행동보다 기준을 먼저 확인해보면 좋겠습니다."
        situation = (
            f"선택 종목 {stock_change_context['stock_name']}({stock_change_context['symbol']})의 "
            f"전일 대비 등락률은 {stock_change_context['formatted_change']}입니다. "
            f"데이터 기준 시각은 {stock_change_context['data_timestamp']}입니다."
        )
        avoid_action = "뚜렷한 변화가 없는데도 불안감만으로 매수·매도를 결정하는 행동은 잠시 미뤄보세요."
    elif any(keyword in lowered_input for keyword in ["수익", "올랐", "상승", "익절", "플러스"]):
        empathy = "수익이 났다면 정말 반가운 순간이에요. 차분히 기뻐하되, 그 기분이 다음 판단을 너무 밀어붙이지 않게 살펴보면 좋겠습니다."
        situation = "선택 종목의 등락률을 확인할 수 없어, 현재 문장에 나타난 수익 감정을 기준으로 정리합니다."
        avoid_action = "기분이 좋아진 상태에서 계획에 없던 추가 매수나 과도한 비중 확대를 바로 결정하는 행동은 피하는 편이 좋습니다."
    elif any(keyword in lowered_input for keyword in ["손실", "하락", "급락", "불안", "패닉", "멘탈", "매도", "후회", "망했다", "물렸다"]):
        empathy = "손실이나 급락을 보면 불안하고 마음이 흔들리는 게 자연스럽습니다. 지금은 판단보다 감정을 먼저 낮추는 시간이 필요할 수 있어요."
        situation = "선택 종목의 등락률을 확인할 수 없어, 현재 문장에 나타난 불안이나 손실 감정을 기준으로 정리합니다."
        avoid_action = "공포 매도, 복수 매매, 손실을 한 번에 만회하려는 큰 베팅은 잠시 멈추고 점검해 보세요."
    else:
        empathy = "투자 중 마음이 흔들리는 순간은 누구에게나 생길 수 있습니다. 먼저 지금 느끼는 감정을 말로 정리해 보는 것부터 시작해도 좋습니다."
        situation = "현재 상황은 투자 판단과 감정 반응이 섞여 있을 수 있는 상태로 보입니다."
        avoid_action = "감정이 강한 상태에서 즉시 매수·매도 결정을 내리기보다 확인할 기준을 먼저 적어보는 것이 좋습니다."

    wants_charm = any(keyword in lowered_input for keyword in ["부적", "액땜"])
    charm_notice = (
        "\n\n6. 액땜 부적 안내:\n"
        "   부적은 대화 답변 안에서 자동 생성하지 않습니다. 멘탈 케어 도구의 '액땜 부적 생성기'를 눌러 만들어 주세요."
        if wants_charm
        else ""
    )

    response_text = (
        f"{npc['example_response']}\n\n"
        f"1. 감정 공감 한 문장: {empathy}\n"
        f"2. 현재 상황 정리: {situation}\n"
        f"3. 피해야 할 충동 행동: {avoid_action}\n"
        "4. 차분히 확인할 체크리스트:\n"
        "   - 원래 투자 이유가 아직 유지되는지 확인하기\n"
        "   - 보유 비중이 내 감당 범위를 넘었는지 확인하기\n"
        "   - 오늘 가격 변화가 기업 자체 변화인지, 시장 분위기 영향인지 나누어 보기\n"
        "   - 지금 결정하지 않아도 되는 일인지 10분만 미뤄보기\n"
        "   - 수면, 식사, 일상에 영향을 줄 정도로 불안한지 확인하기\n"
        "5. 다음 질문:\n"
        "   - 지금 가장 하고 싶은 행동은 매수, 매도, 추가 확인 중 무엇인가요?\n"
        "   - 그 행동은 미리 세운 기준에 따른 것인가요, 아니면 지금 감정에 가까운가요?"
        f"{charm_notice}"
    )
    response_text = append_investment_disclaimer(response_text, MENTAL_COACH_DISCLAIMER)
    return response_text


ROLE_RESPONSE_BUILDERS = {
    "친절한 상담원": build_counselor_response,
    "아이디어 기획자": build_idea_planner_response,
    "Python 튜터": build_python_tutor_response,
    "보고서 도우미": build_report_helper_response,
    "시장 해설가": build_market_commentary_response,
    "종목 분석가": build_stock_analyst_response,
    "초보 투자 튜터": build_beginner_investment_tutor_response,
    "포트폴리오 코치": build_portfolio_coach_response,
    MENTAL_COACH_ROLE: build_investment_mental_coach_response,
}


def get_npc_response(user_input, npc_role):
    npc = NPCS.get(npc_role, NPCS[DEFAULT_NPC_ROLE])
    response_builder = ROLE_RESPONSE_BUILDERS.get(npc_role)

    if response_builder is None:
        return (
            f"{npc_role}입니다. {npc['personality']}으로 답해 볼게요.\n"
            "질문을 조금 더 자세히 보내 주시면 더 잘 맞는 답변을 드릴 수 있어요."
        )

    return response_builder(user_input, npc)


@st.cache_resource(show_spinner=False)
def get_huggingface_client(token_fingerprint=""):
    if OpenAI is None:
        log_app_error("LLM API 패키지 없음", RuntimeError("openai package is not installed"))
        return None

    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        return None

    try:
        return OpenAI(base_url=HF_BASE_URL, api_key=api_key, timeout=45.0)
    except Exception as error:
        log_app_error("LLM API 클라이언트 생성 실패", error)
        return None


def build_system_prompt(npc_role):
    npc = NPCS.get(npc_role, NPCS[DEFAULT_NPC_ROLE])
    response_rules = "\n".join(f"- {rule}" for rule in npc["response_rules"])
    stock_prompt_rules = ""
    if npc_role == MENTAL_COACH_ROLE:
        answer_format = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(STOCK_ANSWER_FORMATS.get(npc_role, []), start=1)
        )
        stock_prompt_rules = (
            "투자 멘탈 코치 답변 규칙:\n"
            "- 사용자의 감정을 먼저 인정하고 공감한다.\n"
            "- 수익 상황에서는 과도한 자신감이나 무리한 추가 매수를 부추기지 않는다.\n"
            "- 손실 상황에서는 공포 매도나 복수 매매를 부추기지 않는다.\n"
            "- 선택 종목의 전일 대비 등락률이 양수이면 상승 흐름을 축하하되 과도한 자신감과 계획 없는 비중 확대를 경계하도록 돕는다.\n"
            "- 선택 종목의 전일 대비 등락률이 음수이면 하락으로 인한 불안을 공감하되 공포 매도와 충동매매를 막도록 돕는다.\n"
            "- 선택 종목 등락률이 없거나 숫자가 아니면 임의로 상승·하락을 판단하지 말고 사용자 문장의 감정을 기준으로 답한다.\n"
            "- 매수, 매도, 보유를 단정적으로 지시하지 않는다.\n"
            "- '반드시 오른다', '무조건 회복된다', '지금 사야 한다', '지금 팔아야 한다' 같은 표현을 사용하지 않는다.\n"
            "- 투자 조언이 아니라 감정 정리와 의사결정 보조라는 점을 명확히 한다.\n"
            "- 심각한 불안, 수면장애, 일상생활 어려움, 자해 위험이 보이면 전문가나 주변 사람에게 도움을 요청하라고 안내한다.\n"
            f"답변 구조:\n{answer_format}\n"
            f"답변 마지막에는 다음 문구를 중복 없이 한 번만 표시한다: {MENTAL_COACH_DISCLAIMER}\n"
        )
    elif npc_role in STOCK_NPC_ROLES:
        answer_format = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(STOCK_ANSWER_FORMATS.get(npc_role, []), start=1)
        )
        analyst_extra_rules = ""
        if npc_role == "종목 분석가":
            analyst_extra_rules = (
                "종목 분석가 추가 규칙:\n"
                "- 답변은 반드시 '[종목 분석 리포트]' 제목으로 시작한다.\n"
                "- 아래 10개 섹션명을 그대로 사용한다: 1. 핵심 요약, 2. 차트 흐름, 3. 이동평균선 분석, 4. 거래량 분석, 5. 고점/저점 위치, 6. 보조지표 참고, 7. 긍정 요인, 8. 리스크 요인, 9. 다음 확인 포인트, 10. 참고 문구.\n"
                "- 핵심 요약은 현재 주가 흐름을 한 문장으로 정리한다.\n"
                "- 최근 1개월/3개월/6개월 수익률을 비교해 단기와 중기 흐름을 나누어 설명한다.\n"
                "- 차트 흐름에서는 상승 추세, 하락 추세, 횡보 가능성을 데이터 기준으로 구분하되 단정하지 않는다.\n"
                "- 5일/20일/60일 이동평균선 기준으로 현재가의 위치를 해석하되 매수·매도 신호로 단정하지 않는다.\n"
                "- 거래량이 최근 20일 평균보다 많은지 적은지 설명하고, 가격 움직임의 신뢰도는 가능성으로만 표현한다.\n"
                "- 최근 3개월 고점/저점과 현재가의 거리를 참고해 부담 또는 회복 위치를 설명한다.\n"
                "- RSI와 변동성은 과열·침체 가능성을 보는 보조 지표라고 설명한다.\n"
                "- RSI 값이 없으면 RSI 설명은 생략하거나 정보 없음으로 표시한다.\n"
                "- 긍정 요인과 리스크 요인은 데이터, 뉴스, 업종 관점으로 나누어 확인 가능한 범위에서만 정리한다.\n"
                "- 다음 확인 포인트에는 실적 발표, 주요 뉴스, 업종 흐름, 거래량 변화, 이동평균선 이탈 여부를 포함한다.\n"
                "- 숫자가 없으면 임의로 만들지 말고 정보 없음 또는 추가 확인 필요라고 말한다.\n"
                "- 매수하세요, 매도하세요, 무조건 오릅니다, 지금 들어가야 합니다 같은 표현을 사용하지 않는다.\n"
                "- 목표주가를 단정적으로 제시하지 않는다.\n"
            )
        stock_prompt_rules = (
            f"{COMMON_INVESTMENT_RULES}\n"
            f"NPC별 답변 구조:\n{answer_format}\n"
            f"{analyst_extra_rules}"
            "모든 주식 NPC 답변에는 반드시 '최종 결론' 섹션을 포함한다.\n"
            "사용자가 오를지 내릴지, 흐름이 어떤지, 지금 어떤 판단이 맞는지 물으면 현재 제공된 가격·거래량·뉴스·투자 성향 정보를 기준으로 '상승 우위', '중립', '하락 우위' 중 하나를 조건부 판단으로 제시한다.\n"
            "최종 결론에는 판단 근거 2~4개와 판단 신뢰도(낮음/보통/높음)를 함께 적는다.\n"
            "데이터가 부족하면 결론을 회피하지 말고 '현재 자료 기준으로는 중립 또는 판단 보류'라고 말한 뒤 부족한 자료를 구체적으로 밝힌다.\n"
            "최종 결론은 투자 추천이 아니라 참고용 판단으로 표현하고, 매수·매도·보유를 지시하지 않는다.\n"
            "답변 마지막의 투자 안전 문구는 중복 없이 한 번만 출력한다.\n"
        )
    portfolio_prompt_rules = ""
    if npc_role == "포트폴리오 코치":
        portfolio_prompt_rules = (
            f"{build_portfolio_reference_context()}\n"
            "포트폴리오 점검은 참고용이며 매수·매도 명령을 하지 않는다.\n"
            "적정 비중을 하나의 정답처럼 제시하지 않는다.\n"
            "사용자 투자 기간과 위험 감수 수준을 추가로 질문한다.\n"
        )
    stock_reference_context = ""
    selected_stock_rule = ""
    stock_data_rules = ""
    if npc_role in STOCK_NPC_ROLES:
        stock_reference_context = build_stock_reference_context()
        selected_stock_rule = f"선택 종목 처리 규칙: {build_npc_stock_handling_rules(npc_role)}\n"
        stock_data_rules = (
            f"{PROFESSIONAL_STOCK_FRAMEWORK}\n"
            f"{build_investor_profile_context()}\n"
            "뉴스 원문 전체를 수집하거나 읽은 것처럼 말하지 않는다. LLM에는 뉴스 제목, 언론사, 게시 시각만 참고 자료로 제공된다.\n"
            "주가 데이터와 뉴스 제목은 사실 정보로만 참고하고, 상승과 하락의 원인을 확정적으로 단정하지 않는다.\n"
            "뉴스와 주가 흐름을 연결할 때는 '영향을 주었을 가능성이 있습니다'처럼 가능성과 해석을 구분한다.\n"
            "숫자나 데이터가 없으면 임의로 만들지 말고 정보 없음 또는 추가 확인 필요라고 말한다.\n"
        )
    skills = ", ".join(npc["skills"])
    input_guide = ", ".join(npc["input_guide"])
    quick_prompts = ", ".join(npc["quick_prompts"])
    response_length_rule = (
        "응답은 전문 분석 보고서처럼 작성한다. 핵심 요약 뒤에 근거 데이터, 그래프 해석 관점, 리스크, 강세·중립·약세 시나리오, 다음 확인 포인트, 최종 결론을 충분히 설명한다. 단, 매수·매도 지시나 수익 보장 표현은 금지한다."
        if npc_role in STOCK_NPC_ROLES
        else "응답은 텍스트로만 작성하고, 과하게 길지 않게 핵심 위주로 설명한다."
    )

    return (
        f"{npc['system_prompt']}\n"
        f"현재 NPC 이름: {npc_role}\n"
        f"역할: {npc['role']}\n"
        f"성격: {npc['personality']}\n"
        f"설명: {npc['description']}\n"
        f"첫 인사: {npc['greeting']}\n"
        f"{stock_reference_context}\n"
        f"{selected_stock_rule}"
        f"{stock_data_rules}"
        "사용자가 입력한 질문을 임의로 다시 쓰거나 바꾸지 않는다.\n"
        f"입력 가이드: {input_guide}\n"
        f"NPC Skill: {skills}\n"
        f"응답 규칙:\n{response_rules}\n"
        f"{stock_prompt_rules}"
        f"{portfolio_prompt_rules}"
        f"예시 톤: {npc['example_response']}\n"
        f"빠른 요청 예시: {quick_prompts}\n"
        f"{response_length_rule}"
    )


def build_user_prompt(user_input, npc_role):
    npc = NPCS.get(npc_role, NPCS[DEFAULT_NPC_ROLE])
    stock_line = f"현재 선택 종목: {get_selected_stock_label()}\n" if npc_role in STOCK_NPC_ROLES else ""
    investor_profile_line = f"{build_investor_profile_context()}\n" if npc_role in STOCK_NPC_ROLES else ""
    age_group_context = ""
    if npc_role in STOCK_NPC_ROLES:
        trend_data, _ = get_current_age_group_trend_data()
        context_text = build_age_group_trend_context(trend_data)
        if context_text:
            age_group_context = f"\n{context_text}\n"
    shared_direction_context = f"{build_shared_stock_direction_context()}\n" if npc_role in STOCK_NPC_ROLES else ""
    return (
        f"선택한 NPC 역할: {npc_role}\n"
        f"NPC 성격: {npc['personality']}\n"
        f"{stock_line}"
        f"{investor_profile_line}"
        f"{shared_direction_context}"
        f"{age_group_context}"
        f"사용자 질문: {user_input}"
    )


def build_chat_messages(user_input, npc_role):
    current_room = ensure_chat_room(npc_role)
    recent_messages = current_room["messages"][-6:]
    chat_messages = [{"role": "system", "content": build_system_prompt(npc_role)}]

    for message in recent_messages:
        role = message["role"]
        if role not in {"user", "assistant"}:
            continue
        chat_messages.append({"role": role, "content": message["content"]})

    chat_messages.append({"role": "user", "content": build_user_prompt(user_input, npc_role)})
    return chat_messages


def summarize_api_error(error_message):
    error_message = sanitize_error_message(error_message)
    if not error_message:
        return ""

    lowered_message = error_message.lower()

    if "401" in lowered_message or "unauthorized" in lowered_message:
        return "인증에 실패했습니다."
    if "403" in lowered_message or "forbidden" in lowered_message:
        return "권한이 없어 호출할 수 없습니다."
    if "429" in lowered_message or "rate limit" in lowered_message:
        return "호출 한도에 걸렸습니다."
    if "timeout" in lowered_message:
        return "응답 시간이 초과되었습니다."
    if "connection" in lowered_message or "network" in lowered_message:
        return "네트워크 연결에 문제가 있습니다."
    if "empty response" in lowered_message:
        return "응답이 비어 있습니다."

    return "API 호출 중 문제가 발생했습니다."


def set_api_status(label, source, error=""):
    st.session_state.api_status = {
        "label": label,
        "source": source,
        "error": error,
    }


def extract_llm_message(completion):
    """OpenAI 호환 응답에서 assistant 메시지를 안전하게 꺼냅니다."""
    if completion is None:
        raise ValueError("Empty completion object")

    if isinstance(completion, dict):
        choices = completion.get("choices") or []
        if not choices:
            raise ValueError("Missing choices in completion")
        first_choice = choices[0] or {}
        message = first_choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
    else:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ValueError("Missing choices in completion")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)

    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    if not content or not str(content).strip():
        raise ValueError("Empty response from Hugging Face Router")
    return str(content).strip()


def get_llm_response(user_input, npc_role, prompt_override=None):
    token = os.getenv("HF_TOKEN") or ""
    token_fingerprint = f"configured:{len(token)}" if token else ""
    client = get_huggingface_client(token_fingerprint)
    if client is None:
        log_app_error("LLM API 토큰 없음", RuntimeError("HF_TOKEN is not configured"))
        reply = get_fallback_response(user_input, npc_role)
        set_api_status(API_STATUS_NO_TOKEN, "Fallback", "API 토큰이 없어 fallback 응답을 사용했습니다.")
        return reply, "Fallback"

    try:
        if prompt_override:
            messages = [
                {"role": "system", "content": build_system_prompt(npc_role)},
                {"role": "user", "content": prompt_override},
            ]
        else:
            messages = build_chat_messages(user_input, npc_role)
        completion = client.chat.completions.create(
            model=HF_MODEL_NAME,
            messages=messages,
        )
        message = extract_llm_message(completion)
    except Exception as error:
        log_app_error("LLM API 응답 실패", error)
        reply = get_fallback_response(user_input, npc_role)
        set_api_status(API_STATUS_FALLBACK, "Fallback", summarize_api_error(str(error)))
        return reply, "Fallback"

    set_api_status(API_STATUS_LLM, "LLM")
    return message, "LLM"


def get_fallback_response(user_input, npc_role):
    if npc_role == MENTAL_COACH_ROLE:
        return build_investment_mental_coach_response(user_input, NPCS[MENTAL_COACH_ROLE])

    fallback_reply = get_npc_response(user_input, npc_role)
    if fallback_reply:
        return fallback_reply

    return "지금은 답변을 바로 만들지 못했어요. 질문을 조금 더 구체적으로 보내 주시면 이어서 도와드릴게요."


@st.cache_data(show_spinner=False)
def get_image_data_uri(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        fallback_path = BASE_DIR / image_path.name
        if fallback_path.exists():
            image_path = fallback_path
        else:
            image_path = BASE_DIR / "assets" / "counselor.png"
    if not image_path.exists():
        fallback_path = BASE_DIR / "counselor.png"
        if fallback_path.exists():
            image_path = fallback_path
    image_bytes = Path(image_path).read_bytes()
    encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_file_to_data_url(image_path):
    image_path = Path(image_path)
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_squishy_frame_paths():
    frame_paths = [SQUISHY_FRAMES_DIR / f"squishy_{index:02d}.png" for index in range(1, 8)]
    if not all(path.exists() for path in frame_paths):
        frame_paths = [BASE_DIR / f"squishy_{index:02d}.png" for index in range(1, 8)]
    if not all(path.exists() for path in frame_paths):
        return []
    return frame_paths


def render_squishy_image_html(image_url):
    return f"""
        <div class="squishy-stage" style="background-image: url('{image_url}');" aria-label="말랑이"></div>
    """


def render_squishy_stop_motion(animate: bool):
    frame_paths = get_squishy_frame_paths()
    if not frame_paths:
        st.info("말랑이 이미지 파일을 assets/squishy_frames 폴더에 추가해 주세요.")
        return

    placeholder = st.empty()
    sequence = list(range(7)) + [5, 4, 3, 2, 1, 0]
    if not animate:
        sequence = [0]

    for frame_index in sequence:
        image_url = image_file_to_data_url(frame_paths[frame_index])
        placeholder.markdown(render_squishy_image_html(image_url), unsafe_allow_html=True)
        if animate:
            time.sleep(0.07)


def render_malrang_float(count=16):
    float_items = []
    for index in range(count):
        left = random.randint(8, 88)
        delay = random.uniform(0, 0.7)
        duration = random.uniform(1.3, 2.1)
        size = random.randint(18, 30)
        drift = random.randint(-36, 36)
        float_items.append(
            "<span class='malrang-float' "
            f"style='left:{left}%; animation-delay:{delay:.2f}s; "
            f"animation-duration:{duration:.2f}s; font-size:{size}px; --drift:{drift}px;'>"
            "말랑</span>"
        )
    st.markdown(
        f"<div class='malrang-float-field'>{''.join(float_items)}</div>",
        unsafe_allow_html=True,
    )


def clean_visible_message_text(text):
    content = str(text or "").replace("\r\n", "\n")
    closing_tag_pattern = r"</(?:div|section|article|main|span|p)>"
    escaped_closing_tag_pattern = r"&lt;/(?:div|section|article|main|span|p)&gt;"
    artifact_body_pattern = rf"(?:{closing_tag_pattern}|{escaped_closing_tag_pattern})"
    content = re.sub(
        rf"```(?:html)?\s*(?:{artifact_body_pattern}\s*)+```",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned_lines = []
    fence_lines = []
    in_fence = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lines = [line]
                continue

            fence_lines.append(line)
            fence_body_lines = fence_lines[1:-1]
            fence_body = "\n".join(fence_body_lines).strip()
            if not re.fullmatch(rf"(?:{artifact_body_pattern}\s*)*", fence_body, flags=re.IGNORECASE):
                cleaned_lines.extend(fence_body_lines)
            in_fence = False
            fence_lines = []
            continue

        if in_fence:
            fence_lines.append(line)
            continue

        if re.fullmatch(artifact_body_pattern, stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    if in_fence:
        fence_body_lines = fence_lines[1:]
        fence_body = "\n".join(fence_body_lines).strip()
        if not re.fullmatch(rf"(?:{artifact_body_pattern}\s*)*", fence_body, flags=re.IGNORECASE):
            cleaned_lines.extend(fence_body_lines)

    content = "\n".join(cleaned_lines)
    content = re.sub(r"(?im)^\s*```(?:[a-zA-Z0-9_-]+)?\s*$", "", content)
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if re.fullmatch(artifact_body_pattern, stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def format_message_html(text):
    return escape(clean_visible_message_text(text)).replace("\n", "<br>")


def clean_saved_chat_record(record):
    if not isinstance(record, dict):
        return record

    cleaned_record = dict(record)
    if "content" in cleaned_record:
        cleaned_record["content"] = clean_visible_message_text(cleaned_record.get("content", ""))
    return cleaned_record


def build_stock_direction_judgement(symbol=None):
    if not symbol:
        return "현재 선택 종목 데이터가 없어 상승·하락 방향성 판단은 보류입니다. 종목을 먼저 선택하면 가격 흐름과 보조지표를 기준으로 다시 판단할 수 있습니다."

    try:
        history = get_stock_history(symbol, period="6mo")
    except Exception as error:
        log_app_error("최종 결론용 주가 데이터 조회 실패", error)
        history = None

    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return "현재 확보된 가격 데이터가 부족해 상승·하락 방향성 판단은 보류입니다. 추가 가격 데이터와 최근 뉴스를 확인한 뒤 판단해야 합니다."

    close = history["Close"].dropna()
    if len(close) < 20:
        return "가격 데이터가 충분하지 않아 단기 방향성 결론은 보류입니다. 최소 20거래일 이상의 흐름을 확인하는 것이 좋습니다."

    last_price = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None
    recent_return = ((last_price / float(close.iloc[-20])) - 1) * 100 if len(close) >= 20 and close.iloc[-20] else 0
    mid_return = ((last_price / float(close.iloc[-60])) - 1) * 100 if len(close) >= 60 and close.iloc[-60] else None

    bullish = 0
    bearish = 0
    reasons = []

    if last_price > ma20:
        bullish += 1
        reasons.append("현재가가 20일 이동평균선 위에 있습니다")
    else:
        bearish += 1
        reasons.append("현재가가 20일 이동평균선 아래에 있습니다")

    if ma60 is not None:
        if last_price > ma60:
            bullish += 1
            reasons.append("현재가가 60일 이동평균선 위에 있습니다")
        else:
            bearish += 1
            reasons.append("현재가가 60일 이동평균선 아래에 있습니다")

    if recent_return > 3:
        bullish += 1
        reasons.append(f"최근 약 1개월 수익률이 {recent_return:.1f}%로 양호합니다")
    elif recent_return < -3:
        bearish += 1
        reasons.append(f"최근 약 1개월 수익률이 {recent_return:.1f}%로 약합니다")
    else:
        reasons.append(f"최근 약 1개월 수익률이 {recent_return:.1f}%로 방향성이 강하지 않습니다")

    if mid_return is not None:
        if mid_return > 5:
            bullish += 1
            reasons.append(f"최근 약 3개월 수익률이 {mid_return:.1f}%로 중기 흐름이 우호적입니다")
        elif mid_return < -5:
            bearish += 1
            reasons.append(f"최근 약 3개월 수익률이 {mid_return:.1f}%로 중기 흐름이 약합니다")

    if bullish > bearish:
        bias = "상승 우위"
    elif bearish > bullish:
        bias = "하락 우위"
    else:
        bias = "중립"

    confidence = "보통" if abs(bullish - bearish) >= 2 else "낮음"
    reason_text = "; ".join(reasons[:4])
    return (
        f"현재 확보된 6개월 가격 흐름 기준 최종 판단은 '{bias}'입니다. "
        f"판단 신뢰도는 {confidence}이며, 근거는 {reason_text}입니다. "
        "다만 이는 매수·매도 지시가 아니라 현재 데이터로 본 조건부 판단이므로 실적, 뉴스, 시장 지수 흐름을 함께 확인해야 합니다."
    )


def build_shared_stock_direction_context():
    if not st.session_state.get("selected_symbol"):
        return ""

    stock_label = get_selected_stock_label()
    judgement = build_stock_direction_judgement(st.session_state.get("selected_symbol"))
    return (
        "[공통 주식 방향성 판단 기준]\n"
        f"- 대상 종목: {stock_label}\n"
        f"- 앱의 모든 주식 NPC는 아래 공통 판단을 최종 결론 기준으로 사용해야 합니다.\n"
        f"- {judgement}\n"
        "- NPC별 설명 방식은 달라도 상승/중립/하락 최종 방향성은 이 기준과 충돌하면 안 됩니다.\n"
    )


def replace_stock_final_conclusion(response_text, npc_role):
    disclaimer = MENTAL_COACH_DISCLAIMER if npc_role == MENTAL_COACH_ROLE else INVESTMENT_DISCLAIMER
    content = str(response_text or "").replace(disclaimer, "").rstrip()
    content = re.sub(r"\n*#{1,6}\s*최종 결론\s*\n.*$", "", content, flags=re.DOTALL)
    content = re.sub(r"\n*최종 결론\s*\n.*$", "", content, flags=re.DOTALL)

    selected_symbol = st.session_state.get("selected_symbol")
    judgement = build_stock_direction_judgement(selected_symbol)
    unified_conclusion = (
        "### 최종 결론\n"
        f"{judgement}\n\n"
        "공통 기준 안내: 시장 해설가, 종목 분석가, 초보 투자 튜터, 포트폴리오 코치, 투자 멘탈 코치는 "
        "역할과 설명 방식은 다르지만 같은 선택 종목에 대해서는 이 공통 데이터 기준 결론을 사용합니다."
    )
    return f"{content.rstrip()}\n\n{unified_conclusion}".strip()


def finalize_response_text(response_text, npc_role):
    cleaned_text = clean_visible_message_text(response_text)
    if npc_role not in STOCK_NPC_ROLES:
        return cleaned_text

    cleaned_text = replace_stock_final_conclusion(cleaned_text, npc_role)

    disclaimer = MENTAL_COACH_DISCLAIMER if npc_role == MENTAL_COACH_ROLE else INVESTMENT_DISCLAIMER
    return append_investment_disclaimer(cleaned_text, disclaimer)


def split_mental_charm_block(text):
    """대화 내용에서 그림 부적 블록을 분리합니다."""
    content = str(text or "")
    if MENTAL_CHARM_START not in content or MENTAL_CHARM_END not in content:
        return content, None

    before, remainder = content.split(MENTAL_CHARM_START, 1)
    charm_block, after = remainder.split(MENTAL_CHARM_END, 1)
    charm_data = parse_mental_charm_block(charm_block)
    visible_content = (before + after).strip()
    return visible_content, charm_data


def parse_mental_charm_block(charm_block):
    title = "멘탈 안정 부적"
    seal = "CALM"
    lines = []
    for raw_line in str(charm_block or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
        elif line.startswith("seal:"):
            seal = line.split(":", 1)[1].strip() or seal
        else:
            lines.append(line)
    return {"title": title, "seal": seal, "lines": lines[:4]}


def render_mental_charm_html(charm):
    if not charm:
        return ""
    line_items = "".join(
        f"<div class='mental-charm-line'>{escape(line)}</div>"
        for line in charm.get("lines", [])
    )
    return f"""
        <div class="mental-charm-card">
            <div class="mental-charm-top">MENTAL LUCKY CHARM</div>
            <div class="mental-charm-title">{escape(charm.get('title') or '멘탈 안정 부적')}</div>
            <div class="mental-charm-body">
                <div class="mental-charm-side">止<br>衝<br>動</div>
                <div class="mental-charm-center">
                    <div class="mental-charm-seal">{escape(charm.get('seal') or 'CALM')}</div>
                    {line_items}
                </div>
                <div class="mental-charm-side">十<br>秒<br>停</div>
            </div>
            <div class="mental-charm-footer">매수·매도 전 10초 정지 · 재미용 감정 정리 카드</div>
        </div>
    """


def split_rag_checklist_block(text):
    """대화 저장 내용에서 RAG 점검 블록을 분리해 expander로 따로 표시할 수 있게 합니다."""
    content = str(text or "")
    if RAG_CHECKLIST_START not in content or RAG_CHECKLIST_END not in content:
        return content, ""

    before, remainder = content.split(RAG_CHECKLIST_START, 1)
    checklist, after = remainder.split(RAG_CHECKLIST_END, 1)
    visible_content = (before + after).strip()
    return visible_content, checklist.strip()


def get_npc_meta_html(npc_role):
    npc = NPCS[npc_role]
    rules_html = "".join(f"<li>{escape(rule)}</li>" for rule in npc["response_rules"])
    skills_html = "".join(f"<span class='skill-chip'>{escape(skill)}</span>" for skill in npc["skills"])
    guide_html = "".join(f"<li>{escape(item)}</li>" for item in npc["input_guide"])
    return (
        f"<div class='npc-role'>{escape(npc['role'])}</div>"
        f"<div class='npc-role'>{escape(npc['title'])}</div>"
        f"<div class='npc-personality'>성격: {escape(npc['personality'])}</div>"
        f"<div class='npc-card-desc'>{escape(npc['description'])}</div>"
        f"<div class='skill-panel-title'>현재 NPC 스킬</div>"
        f"<div class='skill-chip-row'>{skills_html}</div>"
        f"<div class='skill-panel-title'>입력 가이드</div>"
        f"<ul class='guide-list'>{guide_html}</ul>"
        f"<ul class='npc-rule-list'>{rules_html}</ul>"
    )


def render_npc_skill_panel(npc_role):
    """선택한 NPC가 사용할 수 있는 스킬을 화면에 태그로 보여줍니다."""
    npc = NPCS[npc_role]
    skills_html = "".join(f"<span class='skill-chip'>{escape(skill)}</span>" for skill in npc["skills"])
    st.markdown(
        f"""
        <div class="skill-panel">
            <div class="skill-panel-title">현재 NPC 스킬</div>
            <div class="skill-chip-row">{skills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_guide_panel(npc_role):
    """질문을 어떻게 입력하면 좋은지 NPC별 가이드를 보여줍니다."""
    npc = NPCS[npc_role]
    guide_html = "".join(f"<li>{escape(item)}</li>" for item in npc["input_guide"])
    st.markdown(
        f"""
        <div class="skill-panel">
            <div class="skill-panel-title">입력 가이드</div>
            <ul class="guide-list">{guide_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stock_context_panel(npc_role):
    """주식 NPC를 선택했을 때 현재 MVP 범위와 입력 예시를 짧게 보여줍니다."""
    if npc_role not in STOCK_NPC_ROLES:
        return

    selected_stock_label = escape(get_selected_stock_label())
    st.markdown(
        f"""
        <div class="api-status-card">
            <div class="api-status-label">STOCK MVP</div>
            <div class="api-status-value">선택 종목: {selected_stock_label}</div>
            <div class="api-status-value">종목명·티커를 질문에 입력하면 분석 흐름을 잡아 줍니다.</div>
            <div class="api-status-meta">
                조회 가격/등락률: Yahoo Finance 기준 | 예시: AAPL 최근 흐름 알려줘, PER이 뭐야?
            </div>
            <div class="api-status-error">Yahoo Finance 데이터는 지연되거나 일부 항목이 없을 수 있습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_money_burst(amount):
    fly_count = min(36, max(8, int(amount // 50000)))
    money_items = []
    for index in range(fly_count):
        left = random.randint(5, 92)
        delay = random.uniform(0, 0.55)
        duration = random.uniform(1.0, 1.8)
        size = random.randint(20, 34)
        drift = random.randint(-80, 80)
        rotate = random.randint(-45, 45)
        emoji = random.choice(["💸", "💵", "💰"])
        money_items.append(
            "<span class='money-fly' "
            f"style='left:{left}%; animation-delay:{delay:.2f}s; "
            f"animation-duration:{duration:.2f}s; font-size:{size}px; "
            f"--drift:{drift}px; --rotate:{rotate}deg;'>"
            f"{emoji}</span>"
        )

    st.markdown(
        f"""
        <div class="money-burst">
            {''.join(money_items)}
        </div>
        <style>
            .money-burst {{
                position: relative;
                height: 0;
                width: 100%;
                pointer-events: none;
                z-index: 10;
            }}
            .money-fly {{
                position: absolute;
                bottom: -24px;
                opacity: 0;
                animation-name: money-rise;
                animation-timing-function: ease-out;
                animation-fill-mode: forwards;
                filter: drop-shadow(0 8px 14px rgba(30, 72, 120, 0.18));
            }}
            @keyframes money-rise {{
                0% {{
                    opacity: 0;
                    transform: translate(0, 0) scale(0.7) rotate(0deg);
                }}
                14% {{
                    opacity: 1;
                }}
                100% {{
                    opacity: 0;
                    transform: translate(var(--drift), -180px) scale(1.25) rotate(var(--rotate));
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_spending_simulator():
    st.markdown("##### 손실 금액 생활 에너지 환산")
    st.caption("손실 금액을 입력해 보세요. 숫자를 생활감 있는 단위로 바꿔서 잠깐 웃고 숨을 고르는 기능입니다.")

    amount = st.number_input(
        "손실 금액",
        min_value=0,
        value=0,
        step=10000,
        key="mental_care_loss_amount",
    )

    if "mental_care_spending_animated_amount" not in st.session_state:
        st.session_state.mental_care_spending_animated_amount = 0

    if amount <= 0:
        st.info("손실 금액을 입력해 보세요.")
        return

    chicken_count = int(amount // CHICKEN_PRICE)
    coffee_count = int(amount // COFFEE_PRICE)
    gukbap_count = int(amount // GUKBAP_PRICE)

    if st.session_state.mental_care_spending_animated_amount != amount:
        render_money_burst(amount)
        st.session_state.mental_care_spending_animated_amount = amount

    with st.container(border=True):
        st.markdown(
            f"""
            그 돈이면 치킨을 **{chicken_count:,}마리**, 아메리카노를 **{coffee_count:,}잔**,
            국밥을 **{gukbap_count:,}그릇** 먹을 수 있는 금액입니다.

            계좌에서는 사라졌지만, 이렇게 보면 꽤 큰 생활 에너지였네요.
            오늘은 숫자보다 몸을 먼저 챙겨주세요.
            """
        )
        st.caption(
            f"계산 기준: 치킨 1마리 {CHICKEN_PRICE:,}원 · 스타벅스 아메리카노 1잔 {COFFEE_PRICE:,}원 · 국밥 1그릇 {GUKBAP_PRICE:,}원"
        )


def render_mental_care_special_result(active_feature=None):
    active_feature = active_feature or st.session_state.get("mental_care_active_feature")
    if active_feature == "spending_simulator":
        render_spending_simulator()


def render_virtual_loss_simulator():
    """투자 멘탈 코치 전용 가상 손실 체험 도구를 표시합니다."""
    st.markdown("##### 가상 탕진 시뮬레이터")
    st.caption(
        "투자 습관 점검용 교육 시뮬레이션입니다. 실제 시장 예측, 실제 종목 가격, 매수·매도 판단과 연결되지 않습니다."
    )
    st.session_state.mental_care_active_feature = "spending_simulator"
    render_mental_care_special_result("spending_simulator")
    st.divider()

    virtual_amount = st.number_input(
        "가상 투자금",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=10000,
        key="mental_virtual_amount",
    )
    current_emotion = st.selectbox(
        "현재 감정",
        ["불안함", "후회됨", "조급함", "분노", "멘탈 붕괴", "다시 시작하고 싶음"],
        key="mental_virtual_emotion",
    )
    mistake_type = st.selectbox(
        "실수 유형 선택",
        list(MENTAL_SIMULATION_SCENARIOS.keys()),
        key="mental_virtual_mistake_type",
    )
    investment_period = st.selectbox(
        "투자 기간 선택",
        ["하루 안에 급하게", "1주일", "1개월", "3개월", "6개월 이상"],
        key="mental_virtual_period",
    )

    if st.button("가상으로 망해보기", use_container_width=True, key="mental_virtual_run"):
        scenario = MENTAL_SIMULATION_SCENARIOS[mistake_type]
        loss_rate = scenario["loss_rate"]
        loss_amount = int(virtual_amount * loss_rate / 100)
        remaining_amount = int(virtual_amount - loss_amount)
        st.session_state.mental_virtual_result = {
            "scenario": scenario["scenario_name"],
            "mistake_type": mistake_type,
            "emotion_selected": current_emotion,
            "period": investment_period,
            "loss_rate": loss_rate,
            "loss_amount": loss_amount,
            "remaining_amount": remaining_amount,
            "emotion": scenario["emotion"],
            "root_cause": scenario["root_cause"],
            "principles": scenario["principles"],
            "checklist": scenario["checklist"],
        }

    result = st.session_state.get("mental_virtual_result")
    if result:
        st.info(
            "아래 손실률은 교육용 예시값입니다. 실제 시장 예측이나 실제 종목의 미래 수익률이 아닙니다."
        )
        principles_html = "".join(f"<li>{escape(item)}</li>" for item in result["principles"])
        checklist_html = "".join(f"<li>{escape(item)}</li>" for item in result["checklist"])
        st.markdown(
            f"""
            <div class="api-status-card">
                <div class="api-status-label">VIRTUAL HABIT CHECK</div>
                <div class="api-status-value">{escape(result['scenario'])}</div>
                <div class="api-status-meta">
                    현재 감정: {escape(result['emotion_selected'])} · 투자 기간: {escape(result['period'])} · 실수 유형: {escape(result['mistake_type'])}
                </div>
                <div class="stock-metric-grid">
                    <div class="stock-metric"><span>가상 손실률</span><strong>-{result['loss_rate']}%</strong></div>
                    <div class="stock-metric"><span>가상 손실 금액</span><strong>-{result['loss_amount']:,}원</strong></div>
                    <div class="stock-metric"><span>남은 가상 금액</span><strong>{result['remaining_amount']:,}원</strong></div>
                </div>
                <div class="selected-stock-note">이때 흔히 드는 감정: {escape(result['emotion'])}</div>
                <div class="selected-stock-note">이 실수의 핵심 원인: {escape(result['root_cause'])}</div>
                <div class="skill-panel-title">다음에 피하기 위한 원칙 3개</div>
                <ul class="guide-list">{principles_html}</ul>
                <div class="skill-panel-title">실제 매매 전 체크리스트 5개</div>
                <ul class="guide-list">{checklist_html}</ul>
                <div class="api-status-error">
                    이 결과는 투자 습관 점검용 시뮬레이션이며 실제 투자 판단을 대신하지 않습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_10_second_meditation():
    """충동적인 매매 전 잠깐 멈추는 10초 가이드를 표시합니다."""
    st.markdown("##### 10초 명상 가이드")
    st.caption("의료적 치료가 아니라 매매 버튼을 누르기 전 감정을 잠깐 정리하는 일시적 안정 보조입니다.")

    if st.button("10초 진정하기", use_container_width=True, key="mental_meditation_run"):
        guide_placeholder = st.empty()
        progress_placeholder = st.empty()

        for second in range(1, 11):
            if second <= 3:
                phase = "들이마시기"
                message = "천천히 숨을 들이마십니다. 지금은 판단보다 호흡을 먼저 봅니다."
            elif second <= 5:
                phase = "멈추기"
                message = "잠깐 멈춥니다. 바로 매수·매도하지 않아도 괜찮습니다."
            else:
                phase = "내쉬기"
                message = "천천히 내쉽니다. 처음 이 종목을 본 이유를 차분히 떠올립니다."

            guide_placeholder.info(f"{second}/10초 · {phase}\n\n{message}")
            progress_placeholder.progress(second / 10, text=f"{phase} 중입니다")
            time.sleep(1)

        st.session_state.mental_meditation_visible = True

    if st.session_state.get("mental_meditation_visible"):
        st.success("10초 멈춤이 끝났습니다. 지금도 바로 결론을 내리지 않아도 됩니다.")
        steps = [
            "1단계: 숨을 천천히 들이마십니다.",
            "2단계: 잠깐 멈추고 어깨 힘을 풉니다.",
            "3단계: 천천히 내쉬며 차트에서 눈을 잠시 뗍니다.",
            "4단계: 지금 바로 매수·매도하지 않아도 됩니다.",
            "5단계: 내가 처음 이 종목을 산 이유를 떠올립니다.",
        ]
        for step in steps:
            st.write(step)
        st.warning(
            "불안이 심하거나 수면장애, 일상생활의 어려움이 계속된다면 전문가나 주변 사람에게 도움을 요청해 주세요."
        )


def render_lucky_charm_generator():
    """재미용 액땜 부적 문구를 생성합니다."""
    st.markdown("##### 액땜 부적 생성기")
    st.caption("재미용 문구입니다. 실제 미신 효과나 수익 회복을 보장하지 않습니다.")

    mood = st.selectbox(
        "현재 기분",
        ["불안함", "후회됨", "조급함", "분노", "멘탈 붕괴", "다시 시작하고 싶음"],
        key="mental_charm_mood",
    )
    charm_templates = {
        "불안함": [
            {
                "title": "패닉셀 봉인 부적",
                "lines": ["급한 손가락은 오늘 휴식.", "차트보다 호흡 먼저.", "결정은 10초 뒤에 다시 보기."],
            },
            {
                "title": "심장 쿵쾅 진정 부적",
                "lines": ["빨간 숫자는 숫자일 뿐.", "내 심장은 HTS가 아닙니다.", "숨 한 번, 물 한 모금, 판단은 천천히."],
            },
        ],
        "후회됨": [
            {
                "title": "이미 지난 봉 차트 복기 부적",
                "lines": ["지나간 캔들은 되돌릴 수 없지만,", "다음 원칙은 오늘 만들 수 있습니다.", "후회 대신 기록 한 줄."],
            },
            {
                "title": "아 그때 살걸 봉인 부적",
                "lines": ["그때의 나는 그때 정보로 최선을 다했습니다.", "오늘의 나는 기록을 남깁니다.", "다음에는 원칙 먼저, 클릭은 나중."],
            },
        ],
        "조급함": [
            {
                "title": "FOMO 냉각 부적",
                "lines": ["남의 수익률은 내 매수 버튼이 아닙니다.", "급등 열차는 매일 다른 역에 옵니다.", "10초 정지 후 다시 보기."],
            },
            {
                "title": "급한 손가락 휴가 부적",
                "lines": ["손가락은 쉬고, 머리는 일합니다.", "오늘 놓친 종목이 인생 마지막 종목은 아닙니다.", "호흡 후 체크리스트 확인."],
            },
        ],
        "분노": [
            {
                "title": "복수 매매 차단 부적",
                "lines": ["시장은 내 화풀이 상대가 아닙니다.", "화난 클릭은 잠시 냉장 보관.", "화면을 닫고 물부터 마시기."],
            },
            {
                "title": "빨간불 심판 금지 부적",
                "lines": ["분노 상태의 판단은 휴장입니다.", "복수 매매 대신 원인 메모.", "내 계좌에게 소리치기 전에 10초 정지."],
            },
        ],
        "멘탈 붕괴": [
            {
                "title": "오늘 결론 금지 부적",
                "lines": ["멘탈이 퇴근했으면 판단도 퇴근.", "오늘은 결론보다 회복이 먼저.", "작은 행동 하나만 체크하기."],
            },
            {
                "title": "계좌 안부 확인 부적",
                "lines": ["내 멘탈은 종가보다 소중합니다.", "큰 결론은 내일의 나에게 양보.", "지금은 숨, 물, 메모 세 가지만."],
            },
        ],
        "다시 시작하고 싶음": [
            {
                "title": "새 원칙 작성 부적",
                "lines": ["새 출발은 몰빵이 아니라 기록부터.", "비중은 작게, 원칙은 선명하게.", "다음 클릭 전에 이유 한 줄."],
            },
            {
                "title": "리셋 버튼 착각 방지 부적",
                "lines": ["계좌에는 리셋 버튼이 없지만,", "원칙표는 다시 만들 수 있습니다.", "오늘은 비중과 기록부터 정리."],
            },
        ],
    }

    if st.button("액땜 부적 만들기", use_container_width=True, key="mental_charm_run"):
        charm = random.choice(charm_templates[mood])
        charm_body = "\n".join(charm["lines"])
        st.session_state.mental_charm_data = {
            "title": charm["title"],
            "seal": "CALM",
            "lines": charm["lines"],
        }
        st.session_state.mental_charm_text = (
            "━━━━━━━━━━━━━━\n"
            f"[{charm['title']}]\n"
            f"현재 기분: {mood}\n"
            f"{charm_body}\n"
            "매수·매도 전 10초 정지.\n"
            "재미용 부적입니다. 실제 투자 판단과 책임은 사용자에게 있습니다.\n"
            "━━━━━━━━━━━━━━"
        )

    if st.session_state.get("mental_charm_text"):
        st.markdown(
            render_mental_charm_html(st.session_state.get("mental_charm_data")),
            unsafe_allow_html=True,
        )


def render_squishy_touch():
    st.markdown("##### 말랑이 만지기")
    st.caption("비누 말랑이를 눌러 긴장을 조금 풀어보세요.")

    frame_paths = get_squishy_frame_paths()
    if not frame_paths:
        st.info("말랑이 이미지 파일을 assets/squishy_frames 폴더에 추가해 주세요.")
        return

    frame_urls = [image_file_to_data_url(path) for path in frame_paths]
    frame_list = ", ".join(f'"{frame_url}"' for frame_url in frame_urls)
    components.html(
        f"""
        <div class="malrang-widget">
            <button id="squishyButton" class="squishy-touch" type="button" aria-label="말랑이 만지기"></button>
            <div id="floatLayer" class="float-layer"></div>
            <div class="touch-count">말랑이 터치 횟수: <span id="touchCount">0</span></div>
        </div>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: sans-serif;
            }}
            .malrang-widget {{
                position: relative;
                width: min(100%, 620px);
                margin: 0.75rem auto 0 auto;
                padding-bottom: 2.2rem;
            }}
            .squishy-touch {{
                display: block;
                width: 100%;
                height: 360px;
                border: 1px solid #d9e4f2;
                border-radius: 16px;
                background-color: #ffffff;
                background-image: url("{frame_urls[0]}");
                background-position: center center;
                background-size: contain;
                background-repeat: no-repeat;
                box-shadow: 0 12px 26px rgba(30, 72, 120, 0.08);
                cursor: pointer;
                transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
            }}
            .squishy-touch:hover {{
                border-color: #9fc7ff;
                box-shadow: 0 16px 30px rgba(47, 128, 255, 0.13);
                transform: translateY(-1px);
            }}
            .squishy-touch:active {{
                transform: translateY(1px);
            }}
            .float-layer {{
                position: absolute;
                inset: 0;
                pointer-events: none;
                overflow: hidden;
            }}
            .malrang-word {{
                position: absolute;
                bottom: 34px;
                color: #39a887;
                font-weight: 900;
                text-shadow: 0 2px 0 #ffffff, 0 8px 18px rgba(57, 168, 135, 0.22);
                opacity: 0;
                white-space: nowrap;
                animation: floatMalrang var(--duration) ease-out forwards;
            }}
            .touch-count {{
                margin-top: 0.75rem;
                color: #6b7280;
                font-size: 0.95rem;
            }}
            @keyframes floatMalrang {{
                0% {{
                    opacity: 0;
                    transform: translate(0, 0) scale(0.72) rotate(-8deg);
                }}
                14% {{
                    opacity: 1;
                }}
                100% {{
                    opacity: 0;
                    transform: translate(var(--drift), -230px) scale(1.15) rotate(8deg);
                }}
            }}
        </style>
        <script>
            const frames = [{frame_list}];
            const sequence = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0];
            const button = document.getElementById("squishyButton");
            const floatLayer = document.getElementById("floatLayer");
            const touchCount = document.getElementById("touchCount");
            let count = 0;
            let isAnimating = false;

            function showFrame(index) {{
                button.style.backgroundImage = `url("${{frames[index]}}")`;
            }}

            function spawnMalrangWords() {{
                floatLayer.innerHTML = "";
                for (let i = 0; i < 16; i += 1) {{
                    const word = document.createElement("span");
                    word.className = "malrang-word";
                    word.textContent = "말랑";
                    word.style.left = `${{8 + Math.random() * 80}}%`;
                    word.style.fontSize = `${{18 + Math.random() * 13}}px`;
                    word.style.animationDelay = `${{Math.random() * 0.35}}s`;
                    word.style.setProperty("--duration", `${{1.25 + Math.random() * 0.7}}s`);
                    word.style.setProperty("--drift", `${{-36 + Math.random() * 72}}px`);
                    floatLayer.appendChild(word);
                }}
            }}

            function animateSquishy() {{
                if (isAnimating) return;
                isAnimating = true;
                count += 1;
                touchCount.textContent = String(count);
                spawnMalrangWords();
                sequence.forEach((frameIndex, step) => {{
                    window.setTimeout(() => {{
                        showFrame(frameIndex);
                        if (step === sequence.length - 1) {{
                            window.setTimeout(() => {{
                                showFrame(0);
                                isAnimating = false;
                            }}, 70);
                        }}
                    }}, step * 70);
                }});
            }}

            button.addEventListener("click", animateSquishy);
            showFrame(0);
        </script>
        """,
        height=430,
    )


def render_mental_care_tools(npc_role):
    """투자 멘탈 코치 선택 시에만 감정 정리용 도구를 표시합니다."""
    if npc_role != MENTAL_COACH_ROLE:
        return

    st.markdown("#### 투자 멘탈 케어 도구")
    st.caption("아래 기능은 재미와 감정 정리용입니다. 실제 투자, 도박, 매수·매도 추천 기능이 아닙니다.")

    simulator_tab, meditation_tab, charm_tab, squishy_tab = st.tabs(
        ["가상 탕진 시뮬레이터", "10초 명상 가이드", "액땜 부적 생성기", "말랑이 만지기"]
    )
    with simulator_tab:
        render_virtual_loss_simulator()
    with meditation_tab:
        render_10_second_meditation()
    with charm_tab:
        render_lucky_charm_generator()
    with squishy_tab:
        render_squishy_touch()


def render_stock_selector():
    """사이드바에서 공통으로 참고할 종목명 또는 티커를 선택합니다."""
    st.subheader("종목 검색")
    stock_input = st.text_input(
        "종목명 또는 티커",
        placeholder="AAPL, TSLA, 삼성전자, NAVER",
        key="stock_search_input",
    )

    if st.button("종목 선택", use_container_width=True):
        is_selected, message = set_selected_stock(stock_input)
        if is_selected:
            st.success(message)
            st.rerun()
        else:
            st.warning(message)

    st.caption(f"현재 선택 종목: {get_selected_stock_label()}")

    if st.button("관심 종목 추가", use_container_width=True):
        is_added, message = add_watchlist_stock(stock_input)
        if is_added:
            st.success(message)
            st.rerun()
        else:
            st.warning(message)


def render_watchlist_panel():
    """로컬 JSON에 저장된 관심 종목 목록을 사이드바에 표시합니다."""
    st.subheader("관심 종목")
    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        st.caption("아직 등록된 관심 종목이 없습니다.")
        return

    selected_symbol = st.session_state.get("selected_symbol")
    for index, stock_item in enumerate(watchlist):
        name = stock_item.get("name") or "이름 없음"
        symbol = stock_item.get("symbol") or ""
        is_selected = symbol == selected_symbol
        label = f"{'✓ ' if is_selected else ''}{name} · {symbol}"

        select_col, delete_col = st.columns([0.78, 0.22])
        with select_col:
            if st.button(label, use_container_width=True, key=f"watch_select_{symbol}_{index}"):
                select_watchlist_stock(stock_item)
                st.rerun()
        with delete_col:
            if st.button("삭제", use_container_width=True, key=f"watch_delete_{symbol}_{index}"):
                delete_watchlist_stock(symbol)
                st.rerun()


def render_portfolio_panel(npc_role):
    """포트폴리오 코치 전용 보유 종목 입력과 점검 결과를 표시합니다."""
    if npc_role != "포트폴리오 코치":
        return

    st.subheader("보유 종목 입력")
    asset_input = st.text_input(
        "종목명 또는 티커",
        placeholder="AAPL, NVDA, 005930.KS, 현금",
        key="portfolio_asset_input",
    )
    weight_input = st.number_input(
        "비중(%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="portfolio_weight_input",
    )
    average_price_input = st.text_input(
        "평균 매수가(선택)",
        placeholder="예: 180.5",
        key="portfolio_average_price_input",
    )

    if st.button("종목 추가", use_container_width=True):
        is_added, message = add_portfolio_asset(asset_input, weight_input, average_price_input)
        if is_added:
            st.success(message)
            st.rerun()
        else:
            st.warning(message)

    portfolio = st.session_state.get("portfolio", [])
    st.subheader("보유 목록")
    if not portfolio:
        st.caption("아직 입력된 보유 종목이 없습니다.")
        return

    for index, item in enumerate(portfolio):
        average_price = item.get("average_price")
        average_text = f" / 평균 {average_price:,.2f}" if average_price is not None else ""
        label = f"{item['name']} · {item['symbol']} · {item['weight']:.1f}%{average_text}"
        item_col, delete_col = st.columns([0.78, 0.22])
        with item_col:
            st.caption(label)
        with delete_col:
            if st.button("삭제", use_container_width=True, key=f"portfolio_delete_{item['symbol']}_{index}"):
                delete_portfolio_asset(item["symbol"])
                st.rerun()

    analysis = analyze_portfolio(portfolio)
    st.subheader("포트폴리오 점검")
    st.write(f"전체 비중 합계: {analysis['total_weight']:.1f}%")
    if analysis["largest"]:
        largest = analysis["largest"]
        st.write(f"가장 큰 비중: {largest['name']} {largest['weight']:.1f}%")
    st.write(f"상위 3개 종목 집중도: {analysis['top3_weight']:.1f}%")
    st.write(f"현금 비중: {analysis['cash_weight']:.1f}%")
    sector_text = ", ".join(
        f"{sector} {weight:.1f}%" for sector, weight in analysis["sector_weights"].items()
    )
    st.write(f"업종 분산: {sector_text or '정보 없음'}")
    st.write(f"종목 수: {len([item for item in portfolio if item['symbol'] != 'CASH'])}개")
    for note in analysis["notes"]:
        st.caption(f"- {note}")


def build_stock_compare_item(raw_input, period="6mo"):
    symbol, stock_name, error_message = resolve_symbol(raw_input)
    if error_message:
        return None, error_message

    quote = get_stock_quote(symbol)
    history = get_stock_history(symbol, period=period)
    if history is None or history.empty or "Close" not in history.columns:
        return None, f"{stock_name} ({symbol})의 주가 데이터를 불러오지 못했습니다."

    analysis_period = "1y" if period in {"3mo", "6mo"} else period
    analysis_history = get_stock_history(symbol, period=analysis_period)
    if analysis_history is None or analysis_history.empty or "Close" not in analysis_history.columns:
        analysis_history = history

    indicators = calculate_technical_indicators(analysis_history)
    company_info = get_company_info(symbol)
    display_name = stock_name or company_info.get("name") or symbol
    return {
        "name": display_name,
        "symbol": symbol,
        "quote": quote,
        "history": history,
        "indicators": indicators,
        "company_info": company_info,
    }, ""


def render_stock_compare_sidebar_panel(npc_role=None):
    if npc_role == MENTAL_COACH_ROLE:
        return

    st.subheader("A/B 종목 비교")
    st.caption("두 회사의 수익률, 변동성, RSI, 거래량을 나란히 비교합니다.")
    st.text_input("A 회사", placeholder="예: 삼성전자, AAPL", key="compare_stock_a")
    st.text_input("B 회사", placeholder="예: SK하이닉스, TSLA", key="compare_stock_b")
    st.selectbox("비교 기간", ["3개월", "6개월", "1년"], key="compare_stock_period_label")

    if st.button("두 종목 비교", use_container_width=True):
        if not st.session_state.compare_stock_a.strip() or not st.session_state.compare_stock_b.strip():
            st.session_state.compare_stock_error = "A 회사와 B 회사를 모두 입력해 주세요."
            st.session_state.show_stock_compare = False
        else:
            st.session_state.compare_stock_error = ""
            st.session_state.show_stock_compare = True
        st.rerun()

    if st.session_state.get("show_stock_compare"):
        if st.button("비교 결과 숨기기", use_container_width=True):
            st.session_state.show_stock_compare = False
            st.rerun()

    if st.session_state.get("compare_stock_error"):
        st.warning(st.session_state.compare_stock_error)


def get_compare_metric_rows(stock_a, stock_b):
    rows = []
    for item in [stock_a, stock_b]:
        indicators = item["indicators"]
        quote = item["quote"]
        rows.append(
            {
                "종목": f"{item['name']} ({item['symbol']})",
                "현재가": f"{format_optional_number(quote.get('current_price'))} {quote.get('currency') or ''}".strip(),
                "1개월 수익률": format_indicator_percent(indicators.get("return_1mo")),
                "3개월 수익률": format_indicator_percent(indicators.get("return_3mo")),
                "6개월 수익률": format_indicator_percent(indicators.get("return_6mo")),
                "20일 변동성": format_indicator_percent(indicators.get("volatility_20d")),
                "RSI 14일": f"{format_indicator_number(indicators.get('rsi_14'))} / {indicators.get('rsi_status') or '확인 불가'}",
                "거래량 해석": describe_volume_signal(indicators),
            }
        )
    return rows


def render_stock_compare_chart(stock_a, stock_b):
    close_a = stock_a["history"]["Close"].dropna()
    close_b = stock_b["history"]["Close"].dropna()
    if close_a.empty or close_b.empty:
        st.info("비교 그래프를 그릴 수 있는 종가 데이터가 부족합니다.")
        return

    series_a = (close_a / close_a.iloc[0] - 1) * 100
    series_b = (close_b / close_b.iloc[0] - 1) * 100
    series_a.name = f"{stock_a['name']} 누적 수익률(%)"
    series_b.name = f"{stock_b['name']} 누적 수익률(%)"
    comparison_df = pd.concat([series_a, series_b], axis=1).dropna(how="all")
    if comparison_df.empty:
        st.info("두 종목의 기간이 겹치지 않아 비교 그래프를 표시할 수 없습니다.")
        return

    st.line_chart(comparison_df)


def build_stock_compare_summary(stock_a, stock_b):
    indicators_a = stock_a["indicators"]
    indicators_b = stock_b["indicators"]
    summary_lines = []

    return_3mo_a = indicators_a.get("return_3mo")
    return_3mo_b = indicators_b.get("return_3mo")
    if isinstance(return_3mo_a, (int, float)) and isinstance(return_3mo_b, (int, float)):
        stronger = stock_a if return_3mo_a > return_3mo_b else stock_b
        diff = abs(return_3mo_a - return_3mo_b)
        summary_lines.append(
            f"- 최근 3개월 수익률은 {stronger['name']}이 상대적으로 높습니다. 차이는 약 {diff:.2f}%p입니다."
        )
    else:
        summary_lines.append("- 최근 3개월 수익률은 한쪽 또는 양쪽 데이터가 부족해 비교하지 않았습니다.")

    volatility_a = indicators_a.get("volatility_20d")
    volatility_b = indicators_b.get("volatility_20d")
    if isinstance(volatility_a, (int, float)) and isinstance(volatility_b, (int, float)):
        steadier = stock_a if volatility_a < volatility_b else stock_b
        summary_lines.append(f"- 20일 변동성 기준으로는 {steadier['name']}이 상대적으로 안정적인 흐름입니다.")
    else:
        summary_lines.append("- 20일 변동성은 한쪽 또는 양쪽 데이터가 부족해 비교하지 않았습니다.")

    rsi_a = indicators_a.get("rsi_14")
    rsi_b = indicators_b.get("rsi_14")
    if isinstance(rsi_a, (int, float)) and isinstance(rsi_b, (int, float)):
        summary_lines.append(
            f"- RSI는 {stock_a['name']} {rsi_a:.2f}, {stock_b['name']} {rsi_b:.2f}입니다. 70 이상은 과열, 30 이하는 침체 가능성을 참고합니다."
        )
    else:
        summary_lines.append("- RSI는 한쪽 또는 양쪽 데이터가 부족해 비교하지 않았습니다.")

    summary_lines.append("- 최종 판단은 수익률, 변동성, RSI, 뉴스, 실적, 투자 성향을 함께 확인해야 합니다.")
    return "\n".join(summary_lines)


def render_stock_compare_section(npc_role=None):
    if npc_role == MENTAL_COACH_ROLE:
        return

    if not st.session_state.get("show_stock_compare"):
        return

    period_map = {"3개월": "3mo", "6개월": "6mo", "1년": "1y"}
    period_label = st.session_state.get("compare_stock_period_label", "6개월")
    period = period_map.get(period_label, "6mo")
    stock_a, error_a = build_stock_compare_item(st.session_state.get("compare_stock_a", ""), period)
    stock_b, error_b = build_stock_compare_item(st.session_state.get("compare_stock_b", ""), period)

    st.subheader("A/B 종목 비교")
    if error_a or error_b:
        if error_a:
            st.warning(error_a)
        if error_b:
            st.warning(error_b)
        st.info("실제 조회 가능한 데이터가 있을 때만 비교 결과를 표시합니다.")
        return

    with st.container(border=True):
        st.markdown(f"#### {stock_a['name']} vs {stock_b['name']}")
        st.caption(f"비교 기간: {period_label} · Yahoo Finance 기준 데이터")
        st.dataframe(pd.DataFrame(get_compare_metric_rows(stock_a, stock_b)), use_container_width=True, hide_index=True)
        st.markdown("##### 누적 수익률 비교 그래프")
        render_stock_compare_chart(stock_a, stock_b)
        st.markdown("##### 비교 해석")
        st.markdown(build_stock_compare_summary(stock_a, stock_b))
        st.caption("이 비교는 투자 추천이 아니라 참고용 분석입니다. 실제 투자 판단과 책임은 사용자에게 있습니다.")


INVESTOR_PROFILE_PRESETS = {
    "저투자형": {
        "experience": "처음 시작",
        "horizon": "1년 이상",
        "risk_tolerance": "낮음",
        "max_loss": 5,
        "strategy": "배당/안정",
        "cash_need": "높음",
    },
    "균형형": {
        "experience": "1~3년",
        "horizon": "1년 이상",
        "risk_tolerance": "중간",
        "max_loss": 10,
        "strategy": "ETF/분산",
        "cash_need": "중간",
    },
    "고투자형": {
        "experience": "3년 이상",
        "horizon": "3년 이상",
        "risk_tolerance": "높음",
        "max_loss": 25,
        "strategy": "성장주",
        "cash_need": "낮음",
    },
    "단기형": {
        "experience": "1~3년",
        "horizon": "3개월 이내",
        "risk_tolerance": "중간",
        "max_loss": 10,
        "strategy": "단기 트레이딩",
        "cash_need": "중간",
    },
}


def apply_investor_profile_preset(preset_name):
    preset = INVESTOR_PROFILE_PRESETS.get(preset_name)
    if not preset:
        return

    st.session_state.investor_experience_level = preset["experience"]
    st.session_state.investor_time_horizon = preset["horizon"]
    st.session_state.investor_risk_tolerance = preset["risk_tolerance"]
    st.session_state.investor_max_loss_tolerance = preset["max_loss"]
    st.session_state.investor_preferred_strategy = preset["strategy"]
    st.session_state.investor_cash_need = preset["cash_need"]
    st.session_state.investor_profile_preset = preset_name


def render_investor_profile_preset_buttons():
    st.markdown("##### 빠른 성향 선택")
    st.caption("처음 시작할 때는 아래 버튼으로 기본값을 빠르게 채울 수 있습니다.")
    preset_names = list(INVESTOR_PROFILE_PRESETS.keys())
    preset_columns = st.columns(2)
    for index, preset_name in enumerate(preset_names):
        with preset_columns[index % 2]:
            if st.button(preset_name, use_container_width=True, key=f"investor_preset_{preset_name}"):
                apply_investor_profile_preset(preset_name)
                st.rerun()

    selected_preset = st.session_state.get("investor_profile_preset")
    if selected_preset:
        st.caption(f"선택된 빠른 성향: {selected_preset}")


def render_investor_profile_panel():
    st.subheader("투자 성향")
    with st.expander("투자 성향 입력", expanded=False):
        render_investor_profile_preset_buttons()
        st.selectbox(
            "투자 경험",
            ["아직 입력 안 함", "처음 시작", "1년 미만", "1~3년", "3년 이상"],
            key="investor_experience_level",
        )
        st.selectbox(
            "투자 기간",
            ["아직 입력 안 함", "1개월 미만", "3개월 이내", "1년 이상", "3년 이상"],
            key="investor_time_horizon",
        )
        st.selectbox(
            "위험 감수 성향",
            ["아직 입력 안 함", "낮음", "중간 이하", "중간", "높음", "매우 높음"],
            key="investor_risk_tolerance",
        )
        st.slider(
            "감내 가능한 손실폭",
            min_value=0,
            max_value=50,
            value=st.session_state.get("investor_max_loss_tolerance", 10),
            step=5,
            format="%d%%",
            key="investor_max_loss_tolerance",
        )
        st.selectbox(
            "선호 전략",
            ["아직 입력 안 함", "배당/안정", "성장주", "가치주", "단기 트레이딩", "ETF/분산"],
            key="investor_preferred_strategy",
        )
        st.selectbox(
            "현금 필요도",
            ["아직 입력 안 함", "낮음", "중간", "높음"],
            key="investor_cash_need",
        )
        profile_type = classify_investor_profile(get_investor_profile())
        st.info(f"현재 추정 성향: {profile_type}")
        st.caption("이 성향은 답변의 리스크 해석에만 사용되며 수익을 보장하지 않습니다.")


def sanitize_upload_filename(filename):
    """저장할 때 위험한 경로 문자를 빼고 파일명만 안전하게 남깁니다."""
    safe_name = Path(filename or "uploaded_file").name
    safe_name = re.sub(r"[^0-9A-Za-z가-힣._ -]", "_", safe_name).strip()
    return safe_name or "uploaded_file"


def save_library_document(uploaded_file):
    """도서관 사서가 받은 파일을 data 폴더에 저장합니다. 같은 파일명이 있으면 덮어쓰지 않습니다."""
    safe_name = sanitize_upload_filename(uploaded_file.name)
    target_path = DATA_DIR / safe_name
    if target_path.exists():
        return True, f"{safe_name} 파일은 이미 data 폴더에 있어 기존 파일을 사용합니다."

    try:
        target_path.write_bytes(uploaded_file.getbuffer())
        return True, f"{safe_name} 파일을 data 폴더에 저장했습니다."
    except OSError as error:
        log_app_error("문서 업로드 저장 실패", error)
        return False, f"{safe_name} 파일을 저장하지 못했습니다. 폴더 권한을 확인해 주세요."


def list_library_documents():
    """data 폴더에 저장된 PDF, TXT, CSV 파일 목록을 반환합니다."""
    DATA_DIR.mkdir(exist_ok=True)
    allowed_suffixes = {".pdf", ".txt", ".csv"}
    documents = []
    for file_path in sorted(DATA_DIR.iterdir(), key=lambda path: path.name.lower()):
        if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes:
            documents.append(file_path)
    return documents


def read_text_file(file_path):
    """TXT 파일을 흔한 인코딩 순서로 읽습니다."""
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def build_text_segments(text):
    """TXT는 페이지/행 번호가 없으므로 위치 정보를 비워 둡니다."""
    return [{"text": text, "page_number": None, "row_number": None}]


def read_csv_file(file_path):
    """CSV 파일을 행 단위 문자열로 바꿉니다."""
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with file_path.open("r", encoding=encoding, newline="") as csv_file:
                rows = []
                for row in csv.reader(csv_file):
                    rows.append(", ".join(str(cell) for cell in row))
            return "\n".join(rows)
        except UnicodeDecodeError:
            continue
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as csv_file:
        rows = [", ".join(str(cell) for cell in row) for row in csv.reader(csv_file)]
    return "\n".join(rows)


def read_csv_segments(file_path):
    """CSV 각 행을 row_number가 있는 출처 단위로 읽습니다."""
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with file_path.open("r", encoding=encoding, newline="") as csv_file:
                return [
                    {"text": ", ".join(str(cell) for cell in row), "page_number": None, "row_number": row_index}
                    for row_index, row in enumerate(csv.reader(csv_file), start=1)
                ]
        except UnicodeDecodeError:
            continue
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as csv_file:
        return [
            {"text": ", ".join(str(cell) for cell in row), "page_number": None, "row_number": row_index}
            for row_index, row in enumerate(csv.reader(csv_file), start=1)
        ]


def read_pdf_file(file_path):
    """PDF에서 추출 가능한 텍스트만 가져옵니다."""
    if PdfReader is None:
        raise RuntimeError("PDF를 읽으려면 pypdf 패키지가 필요합니다. pip install -r requirements.txt를 실행해 주세요.")

    reader = PdfReader(str(file_path))
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts).strip()


def read_pdf_segments(file_path):
    """PDF 각 페이지를 page_number가 있는 출처 단위로 읽습니다."""
    if PdfReader is None:
        raise RuntimeError("PDF를 읽으려면 pypdf 패키지가 필요합니다. pip install -r requirements.txt를 실행해 주세요.")

    reader = PdfReader(str(file_path))
    return [
        {"text": page.extract_text() or "", "page_number": page_index, "row_number": None}
        for page_index, page in enumerate(reader.pages, start=1)
    ]


def read_library_document(file_path):
    """파일 형식에 맞춰 텍스트를 읽고 성공/실패 상태를 반환합니다."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = read_pdf_file(file_path)
            segments = read_pdf_segments(file_path)
        elif suffix == ".txt":
            text = read_text_file(file_path)
            segments = build_text_segments(text)
        elif suffix == ".csv":
            text = read_csv_file(file_path)
            segments = read_csv_segments(file_path)
        else:
            return {
                "success": False,
                "text": "",
                "segments": [],
                "char_count": 0,
                "message": "지원하지 않는 파일 형식입니다.",
            }

        if not text.strip():
            return {
                "success": False,
                "text": "",
                "segments": [],
                "char_count": 0,
                "message": "파일은 읽었지만 추출된 텍스트가 없습니다.",
            }

        return {
            "success": True,
            "text": text,
            "segments": [segment for segment in segments if segment.get("text", "").strip()],
            "char_count": len(text),
            "message": "문서 읽기 성공",
        }
    except Exception as error:
        log_app_error(f"문서 읽기 실패: {file_path.name}", error)
        return {
            "success": False,
            "text": "",
            "segments": [],
            "char_count": 0,
            "message": f"문서 읽기 실패: {error}",
        }


def split_text_with_overlap(text, chunk_size=DEFAULT_RAG_CHUNK_SIZE, overlap=DEFAULT_RAG_OVERLAP):
    """긴 텍스트를 검색하기 쉬운 작은 조각으로 나눕니다."""
    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned_text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunks.append(cleaned_text[start:end])
        if end >= len(cleaned_text):
            break
        start += step
    return chunks


def build_document_chunks(file_path, read_result, chunk_size=DEFAULT_RAG_CHUNK_SIZE, overlap=DEFAULT_RAG_OVERLAP):
    """읽기 성공한 문서를 출처 정보가 있는 chunk 리스트로 변환합니다."""
    if not read_result.get("success"):
        return []

    file_type = file_path.suffix.upper().lstrip(".")
    document_chunks = []
    chunk_number = 1
    for segment in read_result.get("segments", []):
        if segment.get("page_number") is not None:
            page_or_row = f"page:{segment.get('page_number')}"
        elif segment.get("row_number") is not None:
            page_or_row = f"row:{segment.get('row_number')}"
        else:
            page_or_row = "없음"
        for chunk_text in split_text_with_overlap(segment.get("text", ""), chunk_size, overlap):
            chunk_id = f"{file_path.stem}_{page_or_row}_cs:{chunk_size}_ov:{overlap}_chunk:{chunk_number}"
            document_chunks.append(
                {
                    "file_name": file_path.name,
                    "filename": file_path.name,
                    "file_type": file_type,
                    "page_or_row": page_or_row,
                    "page_number": segment.get("page_number"),
                    "row_number": segment.get("row_number"),
                    "chunk_id": chunk_id,
                    "chunk_number": chunk_number,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "text": chunk_text,
                }
            )
            chunk_number += 1
    return document_chunks


def save_document_chunks(chunks_by_document):
    """문서별 chunk 리스트를 로컬 JSON 파일로 저장합니다."""
    try:
        write_json_file(DOCUMENT_CHUNKS_PATH, chunks_by_document)
        return True, f"문서 조각을 {DOCUMENT_CHUNKS_PATH.name}에 저장했습니다."
    except Exception as error:
        log_app_error("문서 조각 저장 실패", error)
        return False, "문서 조각 저장에 실패했습니다. 터미널 로그를 확인해 주세요."


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """sentence-transformers 모델은 무거우므로 한 번만 불러와 재사용합니다."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers 패키지가 필요합니다. pip install -r requirements.txt를 실행해 주세요.")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def flatten_document_chunks(chunks_by_document):
    """문서별 chunk 딕셔너리를 임베딩 처리용 단일 리스트로 펼칩니다."""
    flattened_chunks = []
    for chunks in chunks_by_document.values():
        flattened_chunks.extend(chunks)
    return flattened_chunks


def build_embeddings_from_chunks(chunks_by_document):
    """문서 조각마다 숫자 벡터를 만들고 저장 가능한 구조로 반환합니다."""
    chunks = flatten_document_chunks(chunks_by_document)
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=False)

    embedding_rows = []
    for chunk, vector in zip(chunks, vectors):
        embedding = [float(value) for value in vector]
        embedding_rows.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "source": chunk.get("file_name") or chunk.get("filename"),
                "page": chunk.get("page_or_row"),
                "file_type": chunk.get("file_type"),
                "chunk_number": chunk.get("chunk_number"),
                "chunk_size": chunk.get("chunk_size"),
                "overlap": chunk.get("overlap"),
                "embedding": embedding,
            }
        )
    return embedding_rows


def save_embeddings(embedding_rows):
    """임베딩 결과를 embeddings.json에 저장합니다."""
    try:
        write_json_file(EMBEDDINGS_PATH, embedding_rows)
        return True, ""
    except Exception as error:
        log_app_error("embeddings.json 저장 실패", error)
        return False, "임베딩 결과를 embeddings.json에 저장하지 못했습니다."


def load_saved_embeddings():
    """이전 단계에서 만든 embeddings.json을 안전하게 불러옵니다."""
    saved_embeddings, error = read_json_file(EMBEDDINGS_PATH, [])
    if error:
        log_app_error("embeddings.json 읽기 실패", error)
        return []
    return saved_embeddings if isinstance(saved_embeddings, list) else []


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """로컬 Chroma 저장소와 documents 컬렉션을 준비합니다."""
    if chromadb is None:
        raise RuntimeError("ChromaDB 패키지가 필요합니다. requirements.txt에 chromadb를 설치 목록으로 추가했습니다.")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(name="documents")


def build_chroma_metadata(row):
    """Chroma에 저장할 출처 metadata를 만듭니다. Chroma metadata는 단순 타입만 사용합니다."""
    return {
        "chunk_id": str(row.get("chunk_id") or ""),
        "file_name": str(row.get("source") or ""),
        "file_type": str(row.get("file_type") or ""),
        "page_or_row": str(row.get("page") or ""),
        "chunk_number": int(row.get("chunk_number") or 0),
        "chunk_size": int(row.get("chunk_size") or 0),
        "overlap": int(row.get("overlap") or 0),
    }


def save_embeddings_to_chroma(embedding_rows):
    """임베딩 결과를 Chroma documents 컬렉션에 저장하고 중복 chunk_id는 건너뜁니다."""
    valid_rows = [
        row for row in embedding_rows
        if row.get("chunk_id") and row.get("text") and row.get("embedding")
    ]
    if not valid_rows:
        return {
            "stored_count": 0,
            "skipped_count": 0,
            "sample": None,
            "message": "Chroma에 저장할 임베딩 결과가 없습니다.",
        }

    collection = get_chroma_collection()
    candidate_ids = [str(row["chunk_id"]) for row in valid_rows]
    existing = collection.get(ids=candidate_ids)
    existing_ids = set(existing.get("ids", []))
    new_rows = [row for row in valid_rows if str(row["chunk_id"]) not in existing_ids]

    if new_rows:
        collection.add(
            ids=[str(row["chunk_id"]) for row in new_rows],
            documents=[row["text"] for row in new_rows],
            embeddings=[row["embedding"] for row in new_rows],
            metadatas=[build_chroma_metadata(row) for row in new_rows],
        )

    sample = new_rows[0] if new_rows else valid_rows[0]
    return {
        "stored_count": len(new_rows),
        "skipped_count": len(valid_rows) - len(new_rows),
        "sample": sample,
        "message": "Chroma 저장이 완료되었습니다.",
    }


def format_chroma_result_message(result):
    """Chroma 저장 결과를 도서관 사서 응답 형태로 요약합니다."""
    sample = result.get("sample")
    if not sample:
        return (
            "Chroma 저장 결과입니다.\n\n"
            f"- 저장소 경로: {CHROMA_DB_PATH}\n"
            "- 컬렉션 이름: documents\n"
            "- 저장된 청크 수: 0개\n"
            "- 저장된 항목: chunk_id, text, embedding, metadata\n\n"
            f"{result.get('message')}\n\n"
            "다음 단계: 임베딩 결과를 먼저 만든 뒤 Chroma 저장을 다시 시도할 수 있습니다."
        )

    vector_preview = ", ".join(f"{value:.4f}" for value in sample.get("embedding", [])[:6])
    metadata = build_chroma_metadata(sample)
    return (
        "Chroma 저장 결과입니다.\n\n"
        f"- 저장소 경로: {CHROMA_DB_PATH}\n"
        "- 컬렉션 이름: documents\n"
        f"- 새로 저장된 청크 수: {result.get('stored_count', 0):,}개\n"
        f"- 중복으로 건너뛴 청크 수: {result.get('skipped_count', 0):,}개\n"
        "- 저장된 항목: chunk_id, text, embedding, metadata\n\n"
        "샘플 chunk 1개:\n"
        f"- chunk_id: {sample.get('chunk_id')}\n"
        f"- text: {sample.get('text', '')[:180]}\n"
        f"- embedding 일부: [{vector_preview}, ...]\n\n"
        "샘플 chunk 출처 정보:\n"
        f"- 파일명: {metadata['file_name']}\n"
        f"- 파일 형식: {metadata['file_type']}\n"
        f"- 페이지 또는 행: {metadata['page_or_row']}\n"
        f"- 문서 조각 번호: {metadata['chunk_number']}\n\n"
        "다음 단계: 저장된 Chroma 컬렉션에서 질문과 가까운 문서 조각을 검색하는 기능을 연결할 수 있습니다."
    )


def encode_question_embedding(question):
    """사용자 질문을 문서 조각과 같은 모델로 벡터화합니다."""
    model = get_embedding_model()
    vector = model.encode([question], show_progress_bar=False)[0]
    return [float(value) for value in vector]


def normalize_page_label(metadata):
    """검색 결과의 페이지/행 정보를 사람이 읽기 쉽게 정리합니다."""
    page_or_row = str((metadata or {}).get("page_or_row") or "").strip()
    if not page_or_row or page_or_row.lower() in {"none", "row:none", "없음"}:
        return "없음"
    return page_or_row


def search_chroma_top_k(question, top_k=3):
    """질문 임베딩으로 Chroma documents 컬렉션에서 관련 chunk 후보를 찾습니다."""
    question_embedding = encode_question_embedding(question)
    collection = get_chroma_collection()
    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    documents = result.get("documents", [[]])[0] if result.get("documents") else []
    metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
    distances = result.get("distances", [[]])[0] if result.get("distances") else []

    matches = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        matches.append(
            {
                "rank": index + 1,
                "chunk_id": metadata.get("chunk_id") or chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "source": metadata.get("file_name") or metadata.get("source") or "",
                "score": distances[index] if index < len(distances) else None,
                "metadata": metadata,
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return {
        "question": question,
        "question_embedding": question_embedding,
        "matches": matches,
        "top_k": top_k,
    }


def build_rag_context(search_results):
    """Chroma search results list -> RAG context string for Qwen."""
    if isinstance(search_results, dict):
        results = search_results.get("matches", [])
    elif isinstance(search_results, list):
        results = search_results
    else:
        results = []

    if not results:
        return (
            "[RAG_CONTEXT]\n"
            "검색된 문서 조각이 없습니다.\n"
            "문서를 업로드하고, 임베딩을 생성한 뒤 Chroma 저장까지 완료했는지 확인해 주세요.\n"
            "[/RAG_CONTEXT]"
        )

    context_lines = [
        "[RAG_CONTEXT]",
        "아래 내용은 Chroma Top-k 검색으로 찾은 문서 조각입니다.",
        "답변을 만들 때는 이 근거 안에서만 사용하고, 근거에 없는 내용은 추측하지 마세요.",
        "",
    ]

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        text = " ".join(str(result.get("text") or "").split())
        if len(text) > 700:
            text = text[:700].rstrip() + "..."

        source = (
            result.get("source")
            or metadata.get("file_name")
            or metadata.get("source")
            or "출처 정보 없음"
        )
        chunk_id = result.get("chunk_id") or metadata.get("chunk_id") or f"chunk-{index}"
        score = result.get("score")
        if score is None:
            score = result.get("distance")
        if isinstance(score, (int, float)):
            score_text = f"{score:.4f}"
        else:
            score_text = str(score) if score is not None else "점수 정보 없음"

        page_or_row = normalize_page_label(metadata)
        context_lines.extend(
            [
                f"[문서 조각 {index}]",
                f"source: {source}",
                f"chunk_id: {chunk_id}",
                f"score: {score_text}",
                f"page_or_row: {page_or_row}",
                f"text: {text or '내용 없음'}",
                "",
            ]
        )

    context_lines.append("[/RAG_CONTEXT]")
    return "\n".join(context_lines)


def build_rag_prompt(question, context):
    """사용자 질문과 검색 근거 context를 분리한 RAG 프롬프트를 만듭니다."""
    safe_question = str(question or "").strip()
    safe_context = str(context or "").strip()
    if not safe_context:
        safe_context = "검색된 근거 자료가 없습니다."

    return (
        "[역할]\n"
        "너는 제공된 근거를 참고해 답변하는 RAG 챗봇입니다.\n\n"
        "[답변 규칙]\n"
        "- context에 있는 내용을 중심으로 답변합니다.\n"
        "- 문서에서 확인되지 않는 내용은 추측하지 말고 \"문서에서 확인되지 않습니다.\"라고 말합니다.\n"
        "- 답변은 초보자가 이해하기 쉽게 작성합니다.\n"
        "- 필요한 경우 근거 source와 chunk_id를 함께 표시합니다.\n\n"
        "[context]\n"
        f"{safe_context}\n\n"
        "[question]\n"
        f"{safe_question}\n\n"
        "[answer]\n"
        "Qwen 답변 생성 전"
    )


def get_rag_search_settings():
    """도서관 사서 RAG 검색/청킹 설정을 session_state에서 안전하게 읽습니다."""
    top_k = int(st.session_state.get("rag_top_k", DEFAULT_RAG_TOP_K) or DEFAULT_RAG_TOP_K)
    chunk_size = int(st.session_state.get("rag_chunk_size", DEFAULT_RAG_CHUNK_SIZE) or DEFAULT_RAG_CHUNK_SIZE)
    overlap = int(st.session_state.get("rag_overlap", DEFAULT_RAG_OVERLAP) or DEFAULT_RAG_OVERLAP)
    top_k = min(10, max(1, top_k))
    chunk_size = min(2000, max(100, chunk_size))
    overlap = min(max(0, overlap), chunk_size - 1)
    return top_k, chunk_size, overlap


def format_top_k_search_message(search_result):
    """Top-k 검색 결과를 도서관 사서 NPC 답변 형태로 정리합니다."""
    question = search_result.get("question", "")
    question_embedding = search_result.get("question_embedding", [])
    matches = search_result.get("matches", [])
    vector_preview = ", ".join(f"{value:.4f}" for value in question_embedding[:8])

    if not matches:
        return (
            "문서 조각 검색 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- 사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}\n"
            f"- 기본 Top-k: {search_result.get('top_k', 3)}개\n"
            f"- 질문 임베딩 일부 샘플: [{vector_preview}, ...]\n"
            "- 검색 결과: 관련 청크 후보를 찾지 못했습니다.\n\n"
            "다음 단계: 문서를 업로드하고, 임베딩을 만든 뒤 Chroma에 저장했는지 확인해 주세요."
        )

    lines = [
        "문서 조각 검색 결과입니다.",
        "",
        f"- 입력 질문: {question}",
        f"- 사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}",
        f"- 기본 Top-k: {search_result.get('top_k', 3)}개",
        f"- 질문 임베딩 일부 샘플: [{vector_preview}, ...]",
        "- 이번 단계에서는 최종 답변을 만들지 않고 관련 청크 후보만 보여줍니다.",
        "",
        "관련 청크 후보:",
    ]
    for match in matches:
        metadata = match.get("metadata") or {}
        distance = match.get("distance")
        distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "확인 불가"
        lines.extend(
            [
                "",
                f"{match['rank']}. chunk_id: {match.get('chunk_id')}",
                f"   파일명: {metadata.get('file_name') or '정보 없음'}",
                f"   파일 형식: {metadata.get('file_type') or '정보 없음'}",
                f"   페이지/행 번호: {normalize_page_label(metadata)}",
                f"   거리 점수: {distance_text}",
                f"   청크 내용 일부: {match.get('text', '')[:240]}",
            ]
        )
    lines.append("")
    lines.append("다음 단계: 이 후보 청크를 근거 context로 사용해 답변 생성 단계에 연결할 수 있습니다.")
    return "\n".join(lines)


def build_librarian_answer_from_search(search_result):
    """검색된 문서 조각을 사용자가 읽기 쉬운 답변 초안으로 정리합니다."""
    question = search_result.get("question", "")
    matches = search_result.get("matches", [])
    top_k = search_result.get("top_k", 3)

    if not matches:
        return (
            "Top-k 검색 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- 검색 방식: Chroma 유사도 검색\n"
            f"- Top-k: {top_k}개\n"
            "- 검색 결과: 0개\n\n"
            "문서 기반 답변:\n"
            "관련 문서 조각을 찾지 못했습니다.\n"
            "문서를 업로드한 뒤 문서 읽기, 문서 나누기, 임베딩 생성, Chroma 저장까지 완료했는지 확인해 주세요.\n\n"
            "※ 이 답변은 저장된 문서 조각 검색 결과를 바탕으로 생성됩니다."
        )

    lines = [
        "Top-k 검색 결과입니다.",
        "",
        f"- 입력 질문: {question}",
        "- 검색 방식: Chroma 유사도 검색",
        f"- Top-k: {top_k}개",
        f"- 검색 결과: {len(matches)}개 청크 발견",
        "",
        "관련 문서 조각 후보:",
    ]

    for match in matches:
        metadata = match.get("metadata") or {}
        distance = match.get("distance")
        distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "확인 불가"
        text = " ".join((match.get("text") or "").split())
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        lines.extend(
            [
                "",
                f"[Top-k 결과 {match['rank']}]",
                f"- 출처: {metadata.get('file_name') or '파일명 없음'} / 위치: {normalize_page_label(metadata)} / chunk_id: {match.get('chunk_id')}",
                f"- 거리 점수: {distance_text}",
                f"- 내용 일부: {snippet}",
            ]
        )

    lines.extend(
        [
            "",
            "문서 기반 답변:",
            "검색된 Top-k 문서 조각을 기준으로 관련 내용을 아래처럼 정리할 수 있습니다.",
            "",
            "핵심 내용:",
        ]
    )

    for match in matches:
        text = " ".join((match.get("text") or "").split())
        snippet = text[:280] + ("..." if len(text) > 280 else "")
        lines.append(f"{match['rank']}. {snippet}")

    lines.extend(
        [
            "",
            "참고한 출처:",
        ]
    )

    for match in matches:
        metadata = match.get("metadata") or {}
        lines.append(
            "- "
            f"{metadata.get('file_name') or '파일명 없음'} "
            f"({metadata.get('file_type') or '형식 없음'}, 위치: {normalize_page_label(metadata)}, "
            f"chunk_id: {match.get('chunk_id')})"
        )

    lines.extend(
        [
            "",
            "주의:",
            "위 내용은 검색된 문서 조각만 바탕으로 정리한 초안입니다. 문서 전체의 결론과 다를 수 있으니 원문과 출처를 함께 확인해 주세요.",
        ]
    )
    return "\n".join(lines)


def build_rag_qwen_fallback_response(question, matches):
    """Qwen 호출 실패 시에도 검색 근거를 잃지 않도록 도서관 사서 전용 안내를 만듭니다."""
    lines = [
        "현재 Qwen 답변 생성에 문제가 있어 검색 결과를 기반으로 기본 안내를 제공합니다.",
        "",
        f"- 입력 질문: {question}",
    ]

    if not matches:
        lines.extend(
            [
                "",
                "관련 문서를 찾지 못했습니다.",
                "문서를 Chroma에 저장했는지, 질문 표현을 조금 더 구체적으로 바꿔볼 수 있는지 확인해 주세요.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "검색된 문서 조각의 핵심 내용:",
        ]
    )

    for match in matches:
        metadata = match.get("metadata") or {}
        score = match.get("score")
        if score is None:
            score = match.get("distance")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "점수 정보 없음"
        source = match.get("source") or metadata.get("file_name") or "출처 정보 없음"
        chunk_id = match.get("chunk_id") or metadata.get("chunk_id") or "chunk_id 없음"
        text = " ".join((match.get("text") or "").split())
        snippet = text[:260] + ("..." if len(text) > 260 else "")

        lines.extend(
            [
                "",
                f"{match.get('rank', '-')}. {snippet or '내용 미리보기가 없습니다.'}",
                f"   - source: {source}",
                f"   - chunk_id: {chunk_id}",
                f"   - score: {score_text}",
            ]
        )

    lines.extend(
        [
            "",
            "이 안내는 Qwen이 생성한 최종 답변이 아니라, 검색된 근거를 바탕으로 한 기본 요약입니다.",
            "문서의 정확한 의미는 원문과 출처를 함께 확인해 주세요.",
        ]
    )
    return "\n".join(lines)


def build_rag_connection_checklist(question, search_result, matches, rag_context, rag_prompt, qwen_answer, answer_source):
    """Chroma 검색과 Qwen 답변 연결 상태를 사용자가 확인할 수 있는 체크리스트로 만듭니다."""
    matches = matches or []
    search_result = search_result or {}
    has_metadata = any(
        (
            (match.get("source") or (match.get("metadata") or {}).get("file_name"))
            and match.get("chunk_id")
            and (match.get("score") is not None or match.get("distance") is not None)
        )
        for match in matches
    )
    context_has_results = bool(
        matches
        and "source:" in rag_context
        and "chunk_id:" in rag_context
        and "score:" in rag_context
    )
    prompt_has_sections = bool(
        rag_prompt
        and "[context]" in rag_prompt
        and "[question]" in rag_prompt
        and str(question or "").strip() in rag_prompt
    )

    checklist_items = [
        (
            "질문을 입력했는가",
            bool(str(question or "").strip()),
            "질문 입력창에 확인할 내용을 입력해 주세요.",
        ),
        (
            "Chroma 검색이 실행되었는가",
            isinstance(search_result, dict) and "matches" in search_result,
            "문서 임베딩과 Chroma 저장이 완료되었는지 확인해 주세요.",
        ),
        (
            "Top-k 검색 결과가 표시되었는가",
            bool(matches),
            "관련 문서를 찾지 못했습니다. 문서 저장 상태나 질문 표현을 확인해 주세요.",
        ),
        (
            "source, chunk_id, score가 보이는가",
            has_metadata,
            "검색 결과 metadata에 source, chunk_id, score 정보가 있는지 확인해 주세요.",
        ),
        (
            "검색 결과가 context에 들어갔는가",
            context_has_results,
            "검색 결과가 없거나 context 구성에 필요한 필드가 부족합니다.",
        ),
        (
            "RAG 프롬프트가 생성되었는가",
            prompt_has_sections,
            "context와 question이 포함된 RAG 프롬프트 생성 여부를 확인해 주세요.",
        ),
        (
            "Qwen 답변이 표시되었는가",
            bool(str(qwen_answer or "").strip()),
            "Qwen 응답이 없으면 fallback 안내가 표시되는지 확인해 주세요.",
        ),
        (
            "오류 시 fallback 응답이 나오는가",
            answer_source in {"LLM", "Fallback"},
            "API 오류가 발생하면 fallback 응답으로 검색 근거 요약이 표시되어야 합니다.",
        ),
    ]

    lines = [
        "아래 항목으로 Chroma 검색 결과와 Qwen 답변 연결 상태를 확인합니다.",
        "",
    ]
    for label, passed, guide in checklist_items:
        marker = "x" if passed else " "
        state = "정상" if passed else f"확인 필요: {guide}"
        lines.append(f"- [{marker}] {label} - {state}")

    if answer_source == "Fallback":
        lines.extend(
            [
                "",
                "현재 응답은 Qwen 대신 fallback 경로로 표시되었습니다. 검색 결과와 근거 정보는 유지됩니다.",
            ]
        )
    elif answer_source == "LLM":
        lines.extend(
            [
                "",
                "현재 응답은 Qwen 경로로 표시되었습니다. 오류 상황에서는 fallback 경로가 사용됩니다.",
            ]
        )

    return "\n".join(lines)


def build_librarian_document_answer(question, top_k=3):
    """질문과 가까운 문서 조각을 찾고, 출처가 있는 답변 형태로 정리합니다."""
    try:
        search_result = search_chroma_top_k(question, top_k)
        matches = search_result.get("matches", [])
        rag_context = build_rag_context(search_result.get("matches", []))
        rag_prompt = build_rag_prompt(question, rag_context)
        top_k_lines = []
        if matches:
            for match in matches:
                metadata = match.get("metadata") or {}
                score = match.get("score")
                if score is None:
                    score = match.get("distance")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "점수 정보 없음"
                text = " ".join((match.get("text") or "").split())
                snippet = text[:220] + ("..." if len(text) > 220 else "")
                top_k_lines.extend(
                    [
                        f"{match.get('rank')}. source: {match.get('source') or metadata.get('file_name') or '출처 정보 없음'}",
                        f"   chunk_id: {match.get('chunk_id')}",
                        f"   score: {score_text}",
                        f"   text: {snippet or '내용 없음'}",
                    ]
                )
        else:
            top_k_lines.append("검색된 문서 조각이 없습니다.")

        qwen_answer, answer_source = get_llm_response(question, "도서관 사서", prompt_override=rag_prompt)
        if answer_source == "Fallback":
            qwen_answer = build_rag_qwen_fallback_response(question, matches)

        evidence_lines = []
        if matches:
            for match in matches:
                metadata = match.get("metadata") or {}
                score = match.get("score")
                if score is None:
                    score = match.get("distance")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "점수 정보 없음"
                evidence_lines.append(
                    "- "
                    f"source: {match.get('source') or metadata.get('file_name') or '출처 정보 없음'} / "
                    f"chunk_id: {match.get('chunk_id')} / "
                    f"score: {score_text}"
                )
        else:
            evidence_lines.append("- 검색된 근거가 없습니다.")

        rag_checklist = build_rag_connection_checklist(
            question=question,
            search_result=search_result,
            matches=matches,
            rag_context=rag_context,
            rag_prompt=rag_prompt,
            qwen_answer=qwen_answer,
            answer_source=answer_source,
        )

        return (
            "RAG 기반 도서관 사서 답변입니다.\n\n"
            "## 1. 사용자 질문\n"
            f"{question}\n\n"
            "## 2. 검색 결과 Top-k\n"
            f"- 검색 방식: Chroma Top-k\n"
            f"- Top-k: {top_k}개\n"
            f"- 검색 결과 수: {len(matches)}개\n\n"
            "```text\n"
            f"{chr(10).join(top_k_lines)}\n"
            "```\n\n"
            "## 3. context 구성 확인\n"
            "context 안에는 각 근거의 text, source, chunk_id, score 정보가 포함됩니다.\n\n"
            "```text\n"
            f"{rag_context}\n"
            "```\n\n"
            "## 4. RAG 프롬프트 미리보기\n"
            "아래 프롬프트를 Qwen에게 전달했습니다.\n\n"
            "```text\n"
            f"{rag_prompt}\n"
            "```\n\n"
            "## 5. Qwen 답변\n"
            f"- 응답 출처: {answer_source}\n\n"
            f"{qwen_answer}\n\n"
            "## 6. 답변 근거\n"
            "```text\n"
            f"{chr(10).join(evidence_lines)}\n"
            "```\n\n"
            f"{RAG_CHECKLIST_START}\n"
            f"{rag_checklist}\n"
            f"\n{RAG_CHECKLIST_END}"
        ), answer_source
    except Exception as error:
        log_app_error("도서관 사서 문서 기반 답변 생성 실패", error)
        safe_error = sanitize_error_message(error)
        rag_checklist = build_rag_connection_checklist(
            question=question,
            search_result={},
            matches=[],
            rag_context="",
            rag_prompt="",
            qwen_answer="문서 검색 또는 답변 정리 중 문제가 발생했습니다.",
            answer_source="ChromaTopKAnswerError",
        )
        return (
            "Top-k 검색 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- Top-k: {top_k}개\n"
            "- 검색 상태: 실패\n"
            f"- 오류 안내: {safe_error}\n\n"
            "문서 기반 답변:\n"
            "문서 검색 또는 답변 정리 중 문제가 발생했습니다.\n"
            "문서 업로드, 문서 읽기, 임베딩 생성, Chroma 저장이 완료되었는지 확인해 주세요.\n\n"
            f"{RAG_CHECKLIST_START}\n"
            f"{rag_checklist}\n"
            f"{RAG_CHECKLIST_END}"
        ), "ChromaTopKAnswerError"


def build_top_k_search_message(question, top_k=3):
    """질문을 임베딩하고 Chroma Top-k 후보를 찾아 대화창용 메시지를 만듭니다."""
    try:
        return format_top_k_search_message(search_chroma_top_k(question, top_k)), "ChromaSearch"
    except Exception as error:
        log_app_error("Chroma Top-k 검색 실패", error)
        return (
            "문서 조각 검색 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- 사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}\n"
            "- 검색 상태: 실패\n"
            f"- 오류 안내: {error}\n\n"
            "다음 단계: sentence-transformers와 chromadb 설치 여부, Chroma 저장 여부를 확인해 주세요."
        ), "ChromaSearchError"


def build_question_embedding_message(question):
    """사용자 질문을 같은 임베딩 모델로 벡터화하고 대화창용 요약을 만듭니다."""
    try:
        embedding = encode_question_embedding(question)
        preview = ", ".join(f"{value:.4f}" for value in embedding[:8])
        return (
            "질문 임베딩 생성 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- 사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}\n"
            f"- 벡터 차원 수: {len(embedding)}\n"
            "- 생성 상태: 성공\n"
            f"- 질문 임베딩 일부 샘플: [{preview}, ...]\n\n"
            "다음 단계: 이 질문 벡터를 Chroma에 저장된 문서 조각 벡터와 비교해 가까운 조각을 찾을 수 있습니다."
        ), "QuestionEmbedding"
    except Exception as error:
        log_app_error("질문 임베딩 생성 실패", error)
        return (
            "질문 임베딩 생성 결과입니다.\n\n"
            f"- 입력 질문: {question}\n"
            f"- 사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}\n"
            "- 생성 상태: 실패\n"
            f"- 오류 안내: {error}\n\n"
            "sentence-transformers 설치 여부와 모델 로딩 상태를 확인해 주세요."
        ), "QuestionEmbeddingError"


def format_embedding_result_message(embedding_rows):
    """대화 영역에 보여줄 임베딩 요약 메시지를 만듭니다."""
    if not embedding_rows:
        return "임베딩할 문서 조각이 없습니다. 먼저 읽기 성공한 PDF, TXT, CSV 문서를 준비해 주세요."

    sample_lines = []
    for row in embedding_rows[:3]:
        vector_preview = ", ".join(f"{value:.4f}" for value in row["embedding"][:6])
        sample_lines.append(
            "\n".join(
                [
                    f"- chunk_id: {row.get('chunk_id')}",
                    f"  source: {row.get('source')}",
                    f"  page: {row.get('page')}",
                    f"  text: {row.get('text', '')[:120]}",
                    f"  embedding 일부: [{vector_preview}, ...]",
                ]
            )
        )

    return (
        f"임베딩 추출이 완료되었습니다.\n\n"
        f"- 저장 파일: {EMBEDDINGS_PATH.name}\n"
        f"- 임베딩된 문서 조각 수: {len(embedding_rows):,}개\n"
        f"- 각 항목 저장 정보: text, source, page, embedding\n\n"
        f"샘플 임베딩:\n" + "\n\n".join(sample_lines)
    )


def add_assistant_message_to_current_room(npc_role, content):
    """현재 대화방에 assistant 메시지를 추가해 일반 답변처럼 보이게 합니다."""
    current_room = ensure_chat_room(npc_role, get_current_chat_room_key())
    current_room["messages"].append(
        {
            "role": "assistant",
            "npc_role": npc_role,
            "content": content,
        }
    )
    current_room["conversation_log"].append(
        {
            "speaker": "assistant",
            "npc_role": npc_role,
            "content": content,
        }
    )
    save_chat_histories()


def run_rag_quality_test(question, question_type, top_k=3):
    """선택한 테스트 질문으로 검색, context 구성, Qwen 답변 흐름을 한 번 점검합니다."""
    search_result = search_chroma_top_k(question, top_k)
    matches = search_result.get("matches", [])
    rag_context = build_rag_context(matches)
    rag_prompt = build_rag_prompt(question, rag_context)
    qwen_answer, answer_source = get_llm_response(question, "도서관 사서", prompt_override=rag_prompt)
    if answer_source == "Fallback":
        qwen_answer = build_rag_qwen_fallback_response(question, matches)

    evidence_lines = []
    for match in matches:
        metadata = match.get("metadata") or {}
        score = match.get("score")
        if score is None:
            score = match.get("distance")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "점수 정보 없음"
        source = match.get("source") or metadata.get("file_name") or "출처 정보 없음"
        chunk_id = match.get("chunk_id") or metadata.get("chunk_id") or "chunk_id 없음"
        text = " ".join((match.get("text") or "").split())
        evidence_lines.append(
            {
                "rank": match.get("rank"),
                "source": source,
                "chunk_id": chunk_id,
                "score": score_text,
                "text": text[:350] + ("..." if len(text) > 350 else ""),
            }
        )

    checklist = build_rag_connection_checklist(
        question=question,
        search_result=search_result,
        matches=matches,
        rag_context=rag_context,
        rag_prompt=rag_prompt,
        qwen_answer=qwen_answer,
        answer_source=answer_source,
    )
    return {
        "question_type": question_type,
        "question": question,
        "top_k": top_k,
        "matches_count": len(matches),
        "answer_source": answer_source,
        "answer": qwen_answer,
        "evidence": evidence_lines,
        "context": rag_context,
        "prompt": rag_prompt,
        "checklist": checklist,
    }


def get_rag_quality_review_point(question_type):
    """질문 유형별로 사용자가 답변 품질을 볼 때 확인할 기준을 알려줍니다."""
    review_points = {
        "사실 확인 질문": "검색된 문서 조각의 사실만 답변에 반영됐는지 확인합니다.",
        "요약 질문": "여러 근거를 무리 없이 묶고, 없는 내용을 덧붙이지 않았는지 확인합니다.",
        "비교 질문": "비교 대상과 차이점이 context 안의 근거와 맞게 연결됐는지 확인합니다.",
        "수치 확인 질문": "숫자, 단위, 기준 시점이 검색 결과와 정확히 일치하는지 확인합니다.",
        "확인 범위 질문": "문서로 확인 가능한 내용과 확인 불가능한 내용을 구분했는지 확인합니다.",
        "근거가 부족한 질문": "근거가 부족할 때 추측하지 않고 부족하다고 말하는지 확인합니다.",
        "애매한 질문": "질문이 불명확할 때 단정하지 않고 필요한 уточ질문이나 범위를 제안하는지 확인합니다.",
    }
    return review_points.get(question_type, "검색 결과, context, Qwen 답변이 서로 맞게 이어졌는지 확인합니다.")


def render_rag_quality_test_result(result):
    """RAG 품질 테스트 결과를 검색 결과, 답변, 근거, context 순서로 보여줍니다."""
    if not result:
        return

    st.markdown("##### 테스트 결과")
    st.write(f"질문 유형: {result.get('question_type')}")
    st.write(f"질문: {result.get('question')}")
    st.caption(get_rag_quality_review_point(result.get("question_type")))
    st.write(f"응답 출처: {result.get('answer_source')}")
    st.write(f"Top-k 검색 결과 수: {result.get('matches_count')}개")

    st.markdown("**검색 결과**")
    if not result.get("evidence"):
        st.info("검색 결과가 없습니다.")
    else:
        for item in result["evidence"]:
            with st.expander(f"{item.get('rank')}. {item.get('source')} · {item.get('chunk_id')}", expanded=False):
                st.write(f"score: {item.get('score')}")
                st.write(item.get("text") or "내용 미리보기가 없습니다.")

    st.markdown("**Qwen 답변**")
    st.info("Qwen 호출 실패 시 fallback 응답이 표시됩니다.")
    st.markdown(result.get("answer") or "답변이 없습니다.")

    st.markdown("**사용된 근거**")
    if not result.get("context"):
        st.info("사용된 근거 context가 없습니다.")
    else:
        st.caption("아래 context를 함께 보면 Qwen 답변이 근거에 없는 내용을 말했는지 직접 확인할 수 있습니다.")

    with st.expander("사용된 근거 context 확인", expanded=False):
        st.text_area(
            "context",
            value=result.get("context") or "",
            height=260,
            disabled=True,
            key="rag_quality_context_preview",
        )

    with st.expander("RAG 프롬프트 확인", expanded=False):
        st.text_area(
            "prompt",
            value=result.get("prompt") or "",
            height=320,
            disabled=True,
            key="rag_quality_prompt_preview",
        )

    with st.expander("RAG 연결 점검", expanded=False):
        st.markdown(result.get("checklist") or "점검 결과가 없습니다.")


def summarize_rag_answer_for_table(answer, max_length=120):
    """표에서 보기 좋도록 Qwen 또는 fallback 답변을 짧게 요약합니다."""
    summary = " ".join(str(answer or "").split())
    if not summary:
        return "답변 없음"
    return summary[:max_length] + ("..." if len(summary) > max_length else "")


def classify_rag_test_status(result):
    """검색 결과와 답변 상태를 간단한 품질 상태로 분류합니다."""
    if not result:
        return "확인 필요"
    if result.get("answer_source") == "TestError":
        return "답변 오류"
    matches_count = int(result.get("matches_count") or 0)
    answer = str(result.get("answer") or "")
    if matches_count <= 0:
        return "검색 결과 부족"
    if "문서에서 확인되지 않습니다" in answer or "관련 문서를 찾지 못했습니다" in answer:
        return "근거 부족"
    if result.get("answer_source") == "Fallback":
        return "확인 필요"
    return "성공"


def build_rag_quality_table_row(result):
    """RAG 품질 테스트 결과 1개를 비교 표의 한 행으로 변환합니다."""
    evidence = result.get("evidence") or []
    representative = evidence[0] if evidence else {}
    used_evidence = "; ".join(
        f"{item.get('source')} / {item.get('chunk_id')}"
        for item in evidence[:3]
    )
    return {
        "테스트 질문": result.get("question") or "",
        "질문 유형": result.get("question_type") or "",
        "Chroma 검색 결과 수": result.get("matches_count", 0),
        "대표 source": representative.get("source") or "없음",
        "대표 chunk_id": representative.get("chunk_id") or "없음",
        "score": representative.get("score") or "없음",
        "Qwen 답변 요약": summarize_rag_answer_for_table(result.get("answer")),
        "사용된 근거": used_evidence or "없음",
        "결과 상태": classify_rag_test_status(result),
    }


def append_rag_quality_table_result(result):
    """새로고침 전까지 비교할 수 있도록 테스트 결과 행을 session_state에 누적합니다."""
    if "rag_quality_test_rows" not in st.session_state:
        st.session_state.rag_quality_test_rows = []
    st.session_state.rag_quality_test_rows.append(build_rag_quality_table_row(result))


def render_rag_quality_results_table():
    """누적된 RAG 품질 테스트 결과를 표로 표시합니다."""
    rows = st.session_state.get("rag_quality_test_rows", [])
    st.markdown("##### RAG 품질 테스트 결과 표")
    if not rows:
        st.caption("아직 누적된 테스트 결과가 없습니다. 테스트 질문을 실행하면 여기에 표로 쌓입니다.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.button("테스트 결과 초기화", use_container_width=True, key="reset_rag_quality_results"):
        st.session_state.rag_quality_test_rows = []
        st.session_state.rag_quality_test_result = None
        st.success("테스트 결과를 초기화했습니다.")
        st.rerun()


def render_rag_quality_test_panel():
    """도서관 사서 전용 RAG 품질 테스트 도구를 표시합니다."""
    if "rag_quality_test_questions" not in st.session_state:
        st.session_state.rag_quality_test_questions = list(DEFAULT_RAG_TEST_QUESTIONS)
    if "rag_quality_test_rows" not in st.session_state:
        st.session_state.rag_quality_test_rows = []

    with st.expander("RAG 품질 테스트", expanded=False):
        st.caption(
            "여러 질문으로 Chroma 검색 결과, context, RAG 프롬프트, Qwen 답변이 잘 연결되는지 확인합니다. "
            "일반 채팅 입력창과는 별도로 동작합니다."
        )

        question_types = [
            "사실 확인 질문",
            "요약 질문",
            "비교 질문",
            "수치 확인 질문",
            "확인 범위 질문",
            "근거가 부족한 질문",
            "애매한 질문",
        ]
        add_col1, add_col2 = st.columns([1, 2])
        with add_col1:
            custom_type = st.selectbox("추가할 질문 유형", question_types, key="rag_quality_custom_type")
        with add_col2:
            custom_question = st.text_input("테스트 질문 직접 추가", key="rag_quality_custom_question")

        if st.button("테스트 질문 추가", use_container_width=True, key="add_rag_quality_question"):
            cleaned_question = custom_question.strip()
            if cleaned_question:
                st.session_state.rag_quality_test_questions.append(
                    {"type": custom_type, "question": cleaned_question}
                )
                st.success("테스트 질문을 추가했습니다.")
            else:
                st.warning("추가할 질문을 입력해 주세요.")

        question_options = [
            f"[{item['type']}] {item['question']}"
            for item in st.session_state.rag_quality_test_questions
        ]
        selected_label = st.selectbox(
            "실행할 테스트 질문",
            question_options,
            key="rag_quality_selected_question",
        )
        selected_index = question_options.index(selected_label)
        selected_item = st.session_state.rag_quality_test_questions[selected_index]
        current_top_k, _, _ = get_rag_search_settings()
        top_k = st.number_input(
            "Top-k",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("rag_quality_top_k", current_top_k)),
            step=1,
            key="rag_quality_top_k",
        )

        if st.button("선택한 질문으로 RAG 테스트 실행", use_container_width=True, key="run_rag_quality_test"):
            try:
                with st.status("RAG 품질 테스트를 실행하고 있습니다...", expanded=True) as status:
                    st.write("Chroma에서 관련 문서 조각을 검색합니다.")
                    result = run_rag_quality_test(
                        selected_item["question"],
                        selected_item["type"],
                        top_k=int(top_k),
                    )
                    st.session_state.rag_quality_test_result = result
                    append_rag_quality_table_result(result)
                    status.update(label="RAG 품질 테스트가 완료되었습니다.", state="complete")
            except Exception as error:
                log_app_error("RAG 품질 테스트 실패", error)
                st.session_state.rag_quality_test_result = {
                    "question_type": selected_item["type"],
                    "question": selected_item["question"],
                    "top_k": int(top_k),
                    "matches_count": 0,
                    "answer_source": "TestError",
                    "answer": "RAG 품질 테스트 중 오류가 발생했습니다. Chroma 저장 상태와 임베딩 모델 설치 상태를 확인해 주세요.",
                    "evidence": [],
                    "context": "",
                    "prompt": "",
                    "checklist": "- [ ] 테스트 실행 실패 - 확인 필요: 터미널 로그를 확인해 주세요.",
                }
                append_rag_quality_table_result(st.session_state.rag_quality_test_result)

        render_rag_quality_results_table()
        render_rag_quality_test_result(st.session_state.get("rag_quality_test_result"))


def render_document_chunks_summary(chunks_by_document):
    """문서별 chunk 개수와 샘플 내용을 화면에 표시합니다."""
    st.markdown("##### 문서 조각 저장 결과")
    if not chunks_by_document:
        st.info("읽기 성공한 문서가 없어 만들 수 있는 문서 조각이 없습니다.")
        return

    total_chunks = sum(len(chunks) for chunks in chunks_by_document.values())
    st.write(f"전체 문서 조각 수: {total_chunks:,}개")
    for filename, chunks in chunks_by_document.items():
        with st.expander(f"{filename} · {len(chunks):,}개 chunk", expanded=False):
            if not chunks:
                st.caption("이 문서에서 생성된 조각이 없습니다.")
                continue
            for sample in chunks[:3]:
                source_label = sample.get("page_or_row") or (
                    f"page:{sample['page_number']}"
                    if sample.get("page_number") is not None
                    else f"row:{sample.get('row_number')}"
                )
                st.caption(
                    f"chunk_id: {sample.get('chunk_id')} · "
                    f"file_name: {sample.get('file_name') or sample.get('filename')} · "
                    f"file_type: {sample.get('file_type')} · "
                    f"page_or_row: {source_label}"
                )
                st.text(sample["text"][:300])


def render_document_read_status(file_path):
    """파일별 읽기 결과, 글자 수, 미리보기를 화면에 표시합니다."""
    result = read_library_document(file_path)
    file_size_kb = file_path.stat().st_size / 1024
    status_icon = "성공" if result["success"] else "실패"
    summary = f"{status_icon} · {file_path.name} · {file_path.suffix.upper().lstrip('.')} · {file_size_kb:,.1f} KB"

    with st.expander(summary, expanded=result["success"]):
        if result["success"]:
            st.success(result["message"])
        else:
            st.warning(result["message"])
        st.write(f"추출된 글자 수: {result['char_count']:,}자")
        preview_text = result["text"][:800] if result["text"] else "미리보기로 표시할 텍스트가 없습니다."
        st.text_area(
            "텍스트 일부 미리보기",
            value=preview_text,
            height=180,
            disabled=True,
            key=f"library_preview_{file_path.name}",
        )
    return result


def render_librarian_upload_panel(npc_role):
    """도서관 사서 NPC를 선택했을 때만 문서 업로드 영역을 표시합니다."""
    if npc_role != "도서관 사서":
        return

    st.markdown("#### 문서 업로드")
    st.caption(
        "PDF, TXT, CSV 파일을 업로드하고 data 폴더의 기존 파일까지 함께 불러옵니다. "
        "문서 읽기, 청킹, 임베딩, Chroma 저장, Top-k 검색, RAG 답변까지 이어서 확인할 수 있습니다."
    )

    st.markdown("##### 검색 설정")
    if int(st.session_state.get("rag_overlap", DEFAULT_RAG_OVERLAP)) >= int(
        st.session_state.get("rag_chunk_size", DEFAULT_RAG_CHUNK_SIZE)
    ):
        st.session_state.rag_overlap = max(
            0,
            int(st.session_state.get("rag_chunk_size", DEFAULT_RAG_CHUNK_SIZE)) - 1,
        )
    setting_col1, setting_col2, setting_col3 = st.columns(3)
    with setting_col1:
        st.number_input(
            "Top-k",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("rag_top_k", DEFAULT_RAG_TOP_K)),
            step=1,
            key="rag_top_k",
            help="질문과 가까운 문서 조각을 몇 개 가져올지 정합니다.",
        )
    with setting_col2:
        st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=int(st.session_state.get("rag_chunk_size", DEFAULT_RAG_CHUNK_SIZE)),
            step=50,
            key="rag_chunk_size",
            help="문서를 나눌 때 한 조각에 넣는 최대 글자 수입니다.",
        )
    with setting_col3:
        max_overlap = max(0, int(st.session_state.get("rag_chunk_size", DEFAULT_RAG_CHUNK_SIZE)) - 1)
        st.number_input(
            "Overlap",
            min_value=0,
            max_value=max_overlap,
            value=min(int(st.session_state.get("rag_overlap", DEFAULT_RAG_OVERLAP)), max_overlap),
            step=10,
            key="rag_overlap",
            help="인접한 문서 조각 사이에 겹쳐 넣는 글자 수입니다.",
        )
    top_k, chunk_size, overlap = get_rag_search_settings()
    st.caption(f"현재 설정: Top-k {top_k}개 · Chunk Size {chunk_size}자 · Overlap {overlap}자")

    uploaded_files = st.file_uploader(
        "문서 파일 선택",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        key="librarian_document_uploader",
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            is_saved, message = save_library_document(uploaded_file)
            if is_saved:
                st.success(message)
            else:
                st.warning(message)

    st.markdown("##### data 폴더 문서 목록")
    documents = list_library_documents()
    if not documents:
        st.info("data 폴더에 PDF, TXT, CSV 문서가 아직 없습니다.")
        return

    chunks_by_document = {}
    for file_path in documents:
        read_result = render_document_read_status(file_path)
        if read_result.get("success"):
            document_chunks = build_document_chunks(
                file_path,
                read_result,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if document_chunks:
                chunks_by_document[file_path.name] = document_chunks

    is_saved, chunk_message = save_document_chunks(chunks_by_document)
    if is_saved:
        st.success(chunk_message)
    else:
        st.warning(chunk_message)
    render_document_chunks_summary(chunks_by_document)

    if st.button("임베딩 결과 보기", use_container_width=True, key="show_embedding_result"):
        try:
            with st.status("임베딩 추출 중입니다...", expanded=True) as embedding_status:
                st.write("문서 조각을 sentence-transformers 모델에 전달하고 있습니다.")
                embedding_rows = build_embeddings_from_chunks(chunks_by_document)
                is_embedding_saved, embedding_error = save_embeddings(embedding_rows)
                if not is_embedding_saved:
                    raise RuntimeError(embedding_error)
                embedding_status.update(label="임베딩 추출이 완료되었습니다.", state="complete")
            add_assistant_message_to_current_room(
                npc_role,
                format_embedding_result_message(embedding_rows),
            )
        except Exception as error:
            log_app_error("임베딩 추출 실패", error)
            add_assistant_message_to_current_room(
                npc_role,
                f"임베딩 추출 중 오류가 발생했습니다.\n\n오류 내용: {sanitize_error_message(error)}",
            )
        st.rerun()

    if st.button("Chroma에 저장", use_container_width=True, key="save_to_chroma"):
        try:
            with st.status("Chroma 저장 중입니다...", expanded=True) as chroma_status:
                embedding_rows = load_saved_embeddings()
                if not embedding_rows:
                    st.write("저장된 임베딩이 없어 현재 문서 조각에서 임베딩을 먼저 생성합니다.")
                    embedding_rows = build_embeddings_from_chunks(chunks_by_document)
                    is_embedding_saved, embedding_error = save_embeddings(embedding_rows)
                    if not is_embedding_saved:
                        raise RuntimeError(embedding_error)
                st.write("문서 조각과 임베딩을 Chroma documents 컬렉션에 저장하고 있습니다.")
                chroma_result = save_embeddings_to_chroma(embedding_rows)
                chroma_status.update(label="Chroma 저장이 완료되었습니다.", state="complete")
            add_assistant_message_to_current_room(
                npc_role,
                format_chroma_result_message(chroma_result),
            )
        except Exception as error:
            log_app_error("Chroma 저장 실패", error)
            add_assistant_message_to_current_room(
                npc_role,
                f"Chroma 저장 중 오류가 발생했습니다.\n\n오류 내용: {sanitize_error_message(error)}",
            )
        st.rerun()

    render_rag_quality_test_panel()


def render_investment_checklist_panel(npc_role):
    """종목 분석가 전용 투자 판단 체크리스트를 표시합니다."""
    if npc_role != "종목 분석가":
        return

    st.markdown("#### 투자 판단 체크리스트")
    if not st.session_state.get("selected_symbol"):
        st.info("체크리스트를 사용하려면 사이드바에서 종목을 먼저 선택해주세요.")
        return

    symbol = st.session_state.selected_symbol
    stock_name = st.session_state.get("selected_stock_name") or symbol
    if st.button("투자 판단 체크리스트 보기", use_container_width=True):
        st.session_state.investment_checklist_visible[symbol] = True

    if not st.session_state.investment_checklist_visible.get(symbol):
        st.caption("버튼을 누르면 현재 선택 종목 기준 체크리스트가 표시됩니다.")
        return

    st.markdown(f"**{stock_name} · {symbol}**")
    symbol_checks = st.session_state.investment_checklist_checks.setdefault(symbol, {})
    checked_count = 0

    for index, item in enumerate(INVESTMENT_CHECKLIST_ITEMS, start=1):
        checkbox_key = f"investment_checklist_{symbol}_{index}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = bool(symbol_checks.get(str(index), False))

        checked = st.checkbox(f"{index}. {item}", key=checkbox_key)
        symbol_checks[str(index)] = checked
        if checked:
            checked_count += 1

    st.session_state.investment_checklist_checks[symbol] = symbol_checks
    st.caption(f"체크 진행: {checked_count}/{len(INVESTMENT_CHECKLIST_ITEMS)}")
    st.caption("※ 이 체크리스트는 매수 또는 매도 신호를 계산하지 않으며, 투자 판단 전 확인할 참고용 항목입니다.")


def render_selected_stock_summary_card(stock_summary):
    """메인 화면 상단에 선택 종목의 핵심 조회 정보를 보여줍니다."""
    if not st.session_state.get("selected_symbol"):
        st.info("사이드바에서 종목을 선택해주세요.")
        return

    if not stock_summary:
        st.warning("선택한 종목 정보를 불러오지 못했습니다. 종목명이나 티커를 다시 확인해 주세요.")
        return

    quote = stock_summary["quote"]
    company_name = st.session_state.get("selected_stock_name") or quote.get("company_name") or "기업명 정보 없음"
    symbol = quote.get("symbol") or st.session_state.get("selected_symbol") or ""
    currency = quote.get("currency") or ""
    current_price = quote.get("current_price")
    previous_close = quote.get("previous_close")
    price_change = quote.get("price_change")
    change_percent = quote.get("change_percent")
    data_timestamp = quote.get("data_timestamp") or "확인 불가"

    if current_price is None:
        st.warning("조회 가격을 불러오지 못했습니다. 인터넷 연결, 잘못된 티커, 또는 yfinance 데이터 없음 가능성이 있습니다.")
        return

    st.markdown(
        f"""
        <div class="selected-stock-card">
            <div class="selected-stock-title">{escape(company_name)} · {escape(symbol)}</div>
            <div class="selected-stock-grid">
                <div>
                    <span class="selected-stock-label">조회 가격</span>
                    <strong>{escape(format_optional_number(current_price))} {escape(currency)}</strong>
                </div>
                <div>
                    <span class="selected-stock-label">전일 대비</span>
                    <strong>{escape(format_signed_price_change(price_change, currency))}</strong>
                </div>
                <div>
                    <span class="selected-stock-label">등락률</span>
                    <strong>{escape(format_optional_percent(change_percent))}</strong>
                </div>
                <div>
                    <span class="selected-stock-label">통화</span>
                    <strong>{escape(currency or "정보 없음")}</strong>
                </div>
            </div>
            <div class="selected-stock-meta">
                전일 종가: {escape(format_optional_number(previous_close))} · 기준 시각: {escape(data_timestamp)}
            </div>
            <div class="selected-stock-note">
                Yahoo Finance 데이터는 지연되거나 일부 항목이 없을 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_stock_news_panel(stock_summary):
    """Show recent news for the selected stock without exposing raw HTML."""
    symbol = st.session_state.get("selected_symbol")
    if not symbol:
        return

    if not stock_summary:
        st.info("최근 뉴스를 불러오지 못했습니다.")
        return

    st.markdown("#### 최근 뉴스")
    news_cache = st.session_state.setdefault("selected_stock_news", {})
    news_items = news_cache.get(symbol, [])

    if not news_items:
        st.caption("종목 선택 속도를 위해 뉴스는 필요할 때만 불러옵니다.")
        if st.button("최근 뉴스 불러오기", use_container_width=True, key=f"load_news_{symbol}"):
            with st.spinner("최근 뉴스를 불러오는 중입니다..."):
                news_cache[symbol] = get_stock_news(symbol)
                st.session_state.selected_stock_news = news_cache
            st.rerun()
        return

    # Streamlit 기본 요소로 그려야 HTML 코드가 화면에 그대로 노출되지 않습니다.
    with st.container(border=True):
        visible_news = news_items[:5]
        for index, news in enumerate(visible_news):
            title = news.get("title") or "제목 없음"
            publisher = news.get("publisher") or "언론사 정보 없음"
            published_at = news.get("published_at") or "게시 시각 정보 없음"
            link = news.get("link") or ""

            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")
            st.caption(f"{publisher} · {published_at}")
            if index < len(visible_news) - 1:
                st.divider()

        st.caption(
            "뉴스 제목과 출처만 표시합니다. 제목만으로 주가 변동 원인을 단정할 수 없습니다."
        )
        if st.button("뉴스 새로고침", use_container_width=True, key=f"refresh_news_{symbol}"):
            with st.spinner("최근 뉴스를 다시 불러오는 중입니다..."):
                news_cache[symbol] = get_stock_news(symbol)
                st.session_state.selected_stock_news = news_cache
            st.rerun()


def render_selected_stock_news_panel_legacy_unused(stock_summary):
    """선택 종목 카드 아래에 최근 뉴스 제목과 출처를 표시합니다."""
    if not st.session_state.get("selected_symbol"):
        return

    if not stock_summary:
        st.info("최근 뉴스를 불러오지 못했습니다.")
        return

    news_items = stock_summary.get("news_items") or []
    st.markdown("#### 최근 뉴스")
    if not news_items:
        st.caption("조회 가능한 뉴스가 없습니다.")
        return

    news_html = []
    for news in news_items[:5]:
        title = escape(news.get("title") or "제목 없음")
        publisher = escape(news.get("publisher") or "언론사 정보 없음")
        published_at = escape(news.get("published_at") or "게시 시각 정보 없음")
        link = news.get("link") or ""
        if link:
            title_html = f'<a href="{escape(link)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        else:
            title_html = f"<span>{title}</span>"

        news_html.append(
            f"""
            <div class="stock-news-item">
                <div class="stock-news-title">{title_html}</div>
                <div class="stock-news-meta">{publisher} · {published_at}</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="stock-news-panel">
            {''.join(news_html)}
            <div class="selected-stock-note">
                뉴스 제목과 출처만 표시합니다. 제목만으로 주가 변동 원인을 단정할 수 없습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_stock_history_chart(stock_summary, npc_role=None):
    """선택된 종목의 종가 흐름을 기간별 선 그래프로 보여줍니다."""
    if not st.session_state.get("selected_symbol"):
        return

    period_labels = list(STOCK_HISTORY_PERIOD_OPTIONS.keys())
    st.selectbox(
        "주가 흐름 기간",
        period_labels,
        key="stock_history_period_label",
    )

    if not stock_summary:
        st.info("주가 흐름을 불러오지 못했습니다. 종목명이나 티커를 다시 확인해 주세요.")
        return

    quote = stock_summary["quote"]
    history = stock_summary["history"]
    selected_period_label = st.session_state.get("stock_history_period_label", "1개월")
    company_name = st.session_state.get("selected_stock_name") or quote.get("company_name") or "선택 종목"

    st.markdown(f"#### {company_name} 최근 {selected_period_label} 종가 흐름")
    if history is None or history.empty or len(history) < 2:
        st.caption("차트를 표시할 만큼 충분한 종가 데이터가 없습니다.")
        return

    st.line_chart(history[["Close"]].rename(columns={"Close": "종가"}))

    if npc_role != "종목 분석가":
        return

    st.markdown("#### 종목 분석가 전문 차트")
    st.caption("3개월·6개월 수익률과 60일 이동평균 계산을 위해 전문 지표는 선택 기간과 별개로 최소 6개월 데이터를 사용합니다.")
    selected_analysis_period = STOCK_HISTORY_PERIOD_OPTIONS.get(selected_period_label, "6mo")
    if selected_analysis_period in {"5d", "1mo", "3mo"}:
        selected_analysis_period = "6mo"
    analysis_history = get_stock_history(st.session_state.selected_symbol, period=selected_analysis_period)
    if analysis_history is None or analysis_history.empty or "Close" not in analysis_history.columns or len(analysis_history) < 20:
        st.info("분석 가능한 데이터가 부족합니다.")
        return

    chart_data = analysis_history.copy()
    chart_data["5일 이동평균"] = chart_data["Close"].rolling(window=5).mean()
    chart_data["20일 이동평균"] = chart_data["Close"].rolling(window=20).mean()
    chart_data["60일 이동평균"] = chart_data["Close"].rolling(window=60).mean()

    st.caption("종가와 이동평균선입니다. 5일선은 단기, 20일선은 중기, 60일선은 더 긴 흐름을 볼 때 참고합니다.")
    st.line_chart(
        chart_data[["Close", "5일 이동평균", "20일 이동평균", "60일 이동평균"]].rename(columns={"Close": "종가"})
    )

    if "Volume" in chart_data.columns:
        st.caption("거래량입니다. 가격 변화와 함께 거래량이 늘었는지 확인할 때 참고합니다.")
        st.bar_chart(chart_data[["Volume"]].rename(columns={"Volume": "거래량"}))

    indicators = calculate_technical_indicators(analysis_history)
    metric_columns = st.columns(4)
    metric_columns[0].metric("1개월 수익률", format_indicator_percent(indicators.get("return_1mo")))
    metric_columns[1].metric("3개월 수익률", format_indicator_percent(indicators.get("return_3mo")))
    metric_columns[2].metric("6개월 수익률", format_indicator_percent(indicators.get("return_6mo")))
    metric_columns[3].metric("RSI 14일", f"{format_indicator_number(indicators.get('rsi_14'))} / {indicators.get('rsi_status')}")

    position_columns = st.columns(3)
    position_columns[0].metric("3개월 고점", format_indicator_number(indicators.get("high_3mo")))
    position_columns[1].metric("3개월 저점", format_indicator_number(indicators.get("low_3mo")))
    position_columns[2].metric("20일 변동성", format_indicator_percent(indicators.get("volatility_20d")))

    st.info(
        "거래량 해석: "
        f"{describe_volume_signal(indicators)} / "
        f"고점·저점 위치: {describe_high_low_position(indicators)}"
    )

    if "rsi_14" in indicators and indicators.get("rsi_14") is not None:
        rsi_data = analysis_history[["Close"]].copy()
        delta = rsi_data["Close"].dropna().diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi_data["RSI 14일"] = 100 - (100 / (1 + rs))
        st.caption("RSI는 과열 또는 침체 가능성을 참고하는 보조 지표입니다. 단독 매수·매도 기준으로 사용하지 않습니다.")
        st.line_chart(rsi_data[["RSI 14일"]].dropna())


def render_selected_stock_data_panel(stock_summary=None):
    """선택된 종목의 Yahoo Finance 조회 정보를 사이드바에 표시합니다."""
    symbol = st.session_state.get("selected_symbol")
    if not symbol:
        st.info("종목을 선택하면 Yahoo Finance 조회 정보가 여기에 표시됩니다.")
        return

    if stock_summary is None:
        stock_summary = get_selected_stock_summary()
    if not stock_summary:
        st.warning("선택한 종목 정보를 불러오지 못했습니다. 인터넷 연결 또는 티커를 확인해 주세요.")
        return

    quote = stock_summary["quote"]
    company_info = stock_summary["company_info"]
    news_items = stock_summary["news_items"]
    history = stock_summary["history"]
    selected_period_label = st.session_state.get("stock_history_period_label", "1개월")

    st.subheader("선택 종목 정보")
    st.caption("Yahoo Finance 데이터는 지연되거나 일부 항목이 없을 수 있습니다.")
    st.metric(
        "조회 가격",
        f"{format_optional_number(quote.get('current_price'))} {quote.get('currency') or ''}",
        format_optional_percent(quote.get("change_percent")),
    )
    st.write(f"회사명: {quote.get('company_name') or company_info.get('company_name') or '정보 없음'}")
    st.write(f"전일 종가: {format_optional_number(quote.get('previous_close'))}")
    st.write(f"시장 상태: {quote.get('market_state') or '정보 없음'}")
    st.write(f"데이터 기준 시각: {quote.get('data_timestamp') or '확인 불가'}")

    if history is not None and not history.empty:
        st.line_chart(history[["Close"]].rename(columns={"Close": "종가"}))
    else:
        st.caption(f"최근 {selected_period_label} 종가 데이터가 없습니다.")

    with st.expander("기업 기본 정보", expanded=False):
        st.write(f"업종: {company_info.get('sector') or '정보 없음'}")
        st.write(f"산업: {company_info.get('industry') or '정보 없음'}")
        st.write(f"시가총액: {format_market_cap(company_info.get('market_cap'))}")
        st.write(f"PER: {format_optional_number(company_info.get('per'))}")
        st.write(f"PBR: {format_optional_number(company_info.get('pbr'))}")
        st.write(f"배당수익률: {format_optional_percent(company_info.get('dividend_yield'))}")
        st.write(f"52주 최고가: {format_optional_number(company_info.get('fifty_two_week_high'))}")
        st.write(f"52주 최저가: {format_optional_number(company_info.get('fifty_two_week_low'))}")
        description = company_info.get("description") or "기업 설명 정보가 없습니다."
        st.caption(description[:500])

    with st.expander("최신 뉴스", expanded=False):
        if not news_items:
            st.caption("표시할 뉴스가 없습니다.")
        for news in news_items:
            title = news.get("title") or "제목 없음"
            publisher = news.get("publisher") or "언론사 정보 없음"
            published_at = news.get("published_at") or "게시 시각 정보 없음"
            link = news.get("link") or ""
            if link:
                st.markdown(f"- [{title}]({link})  \n  {publisher} · {published_at}")
            else:
                st.markdown(f"- {title}  \n  {publisher} · {published_at}")


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --panel: #f5f9ff;
            --line: #cfe0ff;
            --line-strong: #7dafff;
            --text: #17325c;
            --muted: #5f7aa6;
            --user: #2f80ff;
            --shadow: 0 12px 28px rgba(44, 102, 198, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(72, 145, 255, 0.12), transparent 24%),
                linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        }

        .stApp [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1280px;
            padding-top: 8.5rem;
            padding-bottom: 13rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(10px);
        }

        .st-key-npc_switch_bar {
            position: fixed;
            top: 3.2rem;
            left: min(32rem, 24vw);
            right: 1.25rem;
            z-index: 9999;
            max-width: 1280px;
            margin: 0 auto;
            padding: 0.65rem 1rem 0.8rem 1rem;
            background: rgba(248, 251, 255, 0.97);
            border: 1px solid rgba(207, 224, 255, 0.95);
            border-radius: 0 0 18px 18px;
            box-shadow: 0 12px 28px rgba(44, 102, 198, 0.08);
            backdrop-filter: blur(12px);
        }

        .st-key-npc_switch_bar [data-testid="stSelectbox"] {
            margin-bottom: 0;
        }

        .st-key-npc_switch_bar label {
            font-weight: 800;
            color: var(--text);
        }

        .st-key-chat_input_bar {
            position: fixed;
            bottom: 0;
            left: min(32rem, 24vw);
            right: 1.25rem;
            z-index: 10000;
            max-width: 1280px;
            margin: 0 auto;
            padding: 0.7rem 1rem 0.85rem 1rem;
            background: rgba(248, 251, 255, 0.96);
            border-top: 1px solid rgba(207, 224, 255, 0.95);
            border-left: 1px solid rgba(207, 224, 255, 0.75);
            border-right: 1px solid rgba(207, 224, 255, 0.75);
            border-radius: 18px 18px 0 0;
            box-shadow: 0 -12px 28px rgba(44, 102, 198, 0.08);
            backdrop-filter: blur(12px);
        }

        .st-key-chat_input_bar textarea {
            min-height: 76px !important;
        }

        .quick-prompt-spacer {
            height: 10.5rem;
        }

        @media (max-width: 900px) {
            .stApp [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 8.8rem;
            }

            .st-key-npc_switch_bar {
                left: 0;
                right: 0;
                top: 3rem;
                border-radius: 0 0 14px 14px;
            }

            .st-key-chat_input_bar {
                left: 0;
                right: 0;
                border-radius: 14px 14px 0 0;
            }

            .quick-prompt-spacer {
                height: 11.5rem;
            }
        }

        .hero-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(244,249,255,0.98) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.4rem 1.35rem 1.2rem 1.35rem;
            margin-bottom: 0.9rem;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: rgba(47, 128, 255, 0.10);
            color: var(--user);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        .hero-title {
            margin-top: 0.8rem;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            color: var(--text);
        }

        .hero-subtitle {
            margin-top: 0.45rem;
            color: var(--muted);
            line-height: 1.6;
            font-size: 0.98rem;
        }

        .user-row {
            display: flex;
            justify-content: flex-end;
            margin: 0.2rem 0 0.35rem 0;
        }

        .user-bubble {
            max-width: min(78%, 560px);
            padding: 0.85rem 1rem;
            border-radius: 20px 20px 6px 20px;
            background: linear-gradient(180deg, #4a97ff 0%, #2f80ff 100%);
            color: #ffffff;
            box-shadow: 0 10px 24px rgba(47, 128, 255, 0.22);
        }

        .user-label {
            display: inline-block;
            margin-bottom: 0.35rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }

        .user-text {
            line-height: 1.55;
            font-size: 0.98rem;
            word-break: break-word;
        }

        .assistant-turn {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: var(--shadow);
            padding: 1rem;
            margin-bottom: 0.6rem;
        }

        .npc-layout {
            display: flex;
            gap: 1rem;
            align-items: stretch;
        }

        .npc-portrait-wrap {
            width: min(32%, 220px);
            min-width: 150px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
        }

        .npc-dialogue-wrap {
            flex: 1;
        }

        .npc-art {
            width: 100%;
            max-width: 220px;
            aspect-ratio: 1 / 1;
            border-radius: 24px;
            object-fit: cover;
            display: block;
            margin: 0 auto;
            background: var(--panel);
            border: 2px solid var(--line);
            box-shadow: 0 14px 26px rgba(47, 128, 255, 0.12);
        }

        .npc-dialogue-card {
            position: relative;
            background: #ffffff;
            border: 2px solid var(--line-strong);
            border-radius: 24px;
            padding: 1.1rem 1.1rem 1rem 1.1rem;
            min-height: 100%;
            box-shadow: 0 10px 24px rgba(47, 128, 255, 0.10);
        }

        .npc-dialogue-card::before {
            content: "";
            position: absolute;
            left: -10px;
            top: 32px;
            width: 18px;
            height: 18px;
            background: #ffffff;
            border-left: 2px solid var(--line-strong);
            border-bottom: 2px solid var(--line-strong);
            transform: rotate(45deg);
        }

        .npc-tag {
            display: inline-flex;
            align-items: center;
            margin-bottom: 0.6rem;
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            background: var(--panel);
            color: var(--user);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }

        .npc-name {
            color: var(--text);
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .npc-role {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.2rem;
            line-height: 1.5;
        }

        .npc-personality {
            margin-top: 0.35rem;
            color: #21457b;
            font-size: 0.9rem;
            line-height: 1.45;
            font-weight: 600;
        }

        .npc-rule-list {
            margin: 0.7rem 0 0 1.1rem;
            padding: 0;
            color: #4d6994;
            line-height: 1.55;
            font-size: 0.88rem;
        }

        .npc-text {
            color: #1f2f49;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 0.95rem;
            word-break: break-word;
        }

        .mental-charm-card {
            width: min(100%, 420px);
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 18px;
            border: 2px solid #d64b4b;
            background:
                linear-gradient(135deg, rgba(214,75,75,0.10), transparent 32%),
                linear-gradient(180deg, #fff7df 0%, #ffe8b8 100%);
            box-shadow: 0 14px 28px rgba(153, 55, 34, 0.14);
            color: #6d2020;
            text-align: center;
        }

        .mental-charm-top {
            display: inline-flex;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #d64b4b;
            color: #fff8e8;
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }

        .mental-charm-title {
            margin-top: 0.65rem;
            font-size: 1.35rem;
            font-weight: 900;
            line-height: 1.25;
        }

        .mental-charm-body {
            display: grid;
            grid-template-columns: 42px 1fr 42px;
            gap: 0.6rem;
            align-items: stretch;
            margin-top: 0.75rem;
        }

        .mental-charm-side {
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            border: 1px solid rgba(214,75,75,0.36);
            background: rgba(255,255,255,0.45);
            font-size: 1rem;
            font-weight: 900;
            line-height: 1.45;
        }

        .mental-charm-center {
            min-width: 0;
            border-radius: 14px;
            border: 1px dashed rgba(109,32,32,0.42);
            background: rgba(255,255,255,0.45);
            padding: 0.75rem;
        }

        .mental-charm-seal {
            width: 82px;
            height: 82px;
            margin: 0 auto 0.7rem auto;
            border-radius: 50%;
            border: 4px double #d64b4b;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #d64b4b;
            background: #fffaf0;
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: 0.03em;
        }

        .mental-charm-line {
            margin-top: 0.34rem;
            color: #522020;
            font-size: 0.96rem;
            font-weight: 800;
            line-height: 1.4;
            overflow-wrap: anywhere;
        }

        .mental-charm-footer {
            margin-top: 0.75rem;
            color: #9b4a37;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .squishy-stage {
            position: relative;
            width: min(100%, 620px);
            height: 360px;
            margin: 0.75rem auto 1rem auto;
            border: 1px solid #d9e4f2;
            border-radius: 16px;
            background-color: #ffffff;
            background-position: center center;
            background-size: contain;
            background-repeat: no-repeat;
            box-shadow: 0 12px 26px rgba(30, 72, 120, 0.08);
            overflow: hidden;
        }

        .malrang-float-field {
            position: relative;
            width: min(100%, 620px);
            height: 0;
            margin: 0 auto;
            pointer-events: none;
            z-index: 5;
        }

        .malrang-float {
            position: absolute;
            bottom: -320px;
            color: #39a887;
            font-weight: 900;
            text-shadow: 0 2px 0 #ffffff, 0 8px 18px rgba(57, 168, 135, 0.22);
            opacity: 0;
            animation-name: malrang-rise;
            animation-timing-function: ease-out;
            animation-fill-mode: forwards;
            white-space: nowrap;
        }

        @keyframes malrang-rise {
            0% {
                opacity: 0;
                transform: translate(0, 0) scale(0.72) rotate(-8deg);
            }
            15% {
                opacity: 1;
            }
            100% {
                opacity: 0;
                transform: translate(var(--drift), -210px) scale(1.15) rotate(8deg);
            }
        }

        .suggestion-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .suggestion-chip {
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--panel);
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .skill-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: var(--shadow);
            margin: 0.9rem 0 1rem 0;
            padding: 0.95rem 1rem;
        }

        .skill-panel-title {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .skill-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .skill-chip {
            background: #eaf2ff;
            border: 1px solid #d5e6ff;
            border-radius: 999px;
            color: #2367c7;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.2;
            padding: 0.42rem 0.75rem;
        }

        .guide-list {
            color: #4d6994;
            line-height: 1.6;
            margin: 0.15rem 0 0 1.1rem;
            padding: 0;
        }

        .npc-card {
            display: flex;
            gap: 0.95rem;
            align-items: flex-start;
            padding: 1rem;
            border-radius: 22px;
            background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
            border: 1px solid #c9ddff;
            box-shadow: 0 12px 28px rgba(47, 128, 255, 0.09);
            margin-bottom: 1rem;
        }

        .npc-card img {
            width: 86px;
            height: 86px;
            border-radius: 18px;
            object-fit: cover;
            border: 2px solid #bdd6ff;
            background: #ffffff;
            flex-shrink: 0;
            box-shadow: 0 8px 18px rgba(47, 128, 255, 0.12);
        }

        .npc-card-body {
            min-width: 0;
            flex: 1;
        }

        .npc-card-label {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: #edf5ff;
            border: 1px solid #d5e6ff;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: var(--user);
        }

        .npc-card-name {
            font-size: 1.32rem;
            font-weight: 800;
            color: var(--text);
            margin-top: 0.42rem;
            line-height: 1.18;
            word-break: keep-all;
        }

        .npc-card-title {
            font-size: 0.98rem;
            color: #21457b;
            margin-top: 0.38rem;
            line-height: 1.45;
            word-break: keep-all;
        }

        .npc-card-desc {
            font-size: 0.92rem;
            color: #5677a9;
            margin-top: 0.38rem;
            line-height: 1.55;
            word-break: keep-all;
        }

        .sidebar-log {
            padding-top: 0.25rem;
            color: #24416f;
            line-height: 1.5;
        }

        .api-status-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(47, 128, 255, 0.08);
        }

        .selected-stock-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--line-strong);
            border-radius: 18px;
            box-shadow: var(--shadow);
            margin: 0.75rem 0 1rem 0;
            padding: 1rem;
        }

        .selected-stock-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 0.85rem;
        }

        .selected-stock-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .selected-stock-grid > div {
            min-width: 0;
        }

        .selected-stock-label {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.22rem;
        }

        .selected-stock-grid strong {
            color: var(--text);
            display: block;
            font-size: 0.98rem;
            overflow-wrap: anywhere;
        }

        .selected-stock-meta {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.9rem;
        }

        .selected-stock-note {
            color: #7a8dab;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }

        .stock-news-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(47, 128, 255, 0.08);
            margin: 0.25rem 0 1rem 0;
            padding: 0.9rem 1rem;
        }

        .stock-news-item {
            border-bottom: 1px solid #e8f0ff;
            padding: 0.65rem 0;
        }

        .stock-news-item:last-of-type {
            border-bottom: 0;
        }

        .stock-news-title,
        .stock-news-title a {
            color: var(--text);
            font-size: 0.94rem;
            font-weight: 800;
            line-height: 1.45;
            overflow-wrap: anywhere;
            text-decoration: none;
        }

        .stock-news-title a:hover {
            color: var(--user);
            text-decoration: underline;
        }

        .stock-news-meta {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .api-status-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--user);
            letter-spacing: 0.06em;
        }

        .api-status-value {
            margin-top: 0.45rem;
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1.45;
        }

        .api-status-meta {
            margin-top: 0.45rem;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .api-status-error {
            margin-top: 0.45rem;
            color: #a94b4b;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        @media (max-width: 900px) {
            .stApp [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 4rem;
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            .hero-title {
                font-size: 1.7rem;
            }

            .user-bubble {
                max-width: 88%;
            }
        }

        @media (max-width: 640px) {
            .hero-card {
                padding: 1.1rem 1rem;
                border-radius: 20px;
            }

            .user-bubble {
                max-width: 94%;
                padding: 0.8rem 0.9rem;
            }

            .assistant-turn {
                padding: 0.85rem;
                border-radius: 20px;
            }

            .npc-dialogue-card::before {
                display: none;
            }

            .npc-layout {
                flex-direction: column;
            }

            .npc-portrait-wrap {
                width: 100%;
                min-width: 0;
            }

            .npc-dialogue-card {
                margin-top: 0.8rem;
            }

            .npc-card {
                padding: 0.9rem;
                border-radius: 18px;
            }

            .npc-card img {
                width: 78px;
                height: 78px;
            }

            .npc-card-name {
                font-size: 1.18rem;
            }

            .selected-stock-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .npc-card-title,
            .npc-card-desc {
                font-size: 0.88rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(selected_npc_role):
    npc = NPCS[selected_npc_role]
    if APP_MODE == "stock":
        title = "주식 NPC 챗봇"
        subtitle = (
            "시장 해설, 종목 분석, 투자 용어, 포트폴리오 점검을 "
            "선택한 NPC 역할에 맞춰 쉽게 설명합니다. "
            "응답은 투자 조언이 아니라 학습과 점검을 위한 참고용 정보입니다."
        )
    elif APP_MODE == "basic":
        title = "NPC 미니 챗봇"
        subtitle = (
            "상담, 아이디어 기획, Python 학습, 보고서 정리, 문서 RAG 검색을 "
            "선택한 NPC 역할에 맞춰 도와줍니다."
        )
    else:
        title = "통합 NPC 챗봇"
        subtitle = "기본 NPC와 주식 NPC를 함께 실행하는 통합 모드입니다."
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">NPC CHAT PLAYGROUND</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-subtitle">
                {escape(subtitle)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"현재 선택한 NPC: {selected_npc_role} | {npc['personality']}")


def render_sidebar_npc_card(npc_role):
    npc = NPCS[npc_role]
    image_uri = get_image_data_uri(npc["image"])
    st.markdown(
        f"""
        <div class="npc-card">
            <img src="{image_uri}" alt="{escape(npc_role)}">
            <div class="npc-card-body">
                <div class="npc-card-label">CURRENT NPC</div>
                <div class="npc-card-name">{escape(npc_role)}</div>
                <div class="npc-card-title">{escape(npc['title'])}</div>
                <div class="npc-personality">성격: {escape(npc['personality'])}</div>
                <div class="npc-card-desc">{escape(npc['description'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_status():
    api_status = st.session_state.api_status
    error_html = ""
    if api_status["error"]:
        error_html = f"<div class='api-status-error'>오류 원인: {escape(api_status['error'])}</div>"

    st.markdown(
        f"""
        <div class="api-status-card">
            <div class="api-status-label">API STATUS</div>
            <div class="api-status-value">{escape(api_status['label'])}</div>
            <div class="api-status-meta">마지막 응답 출처: {escape(api_status['source'])}</div>
            {error_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_message(message):
    st.markdown(
        f"""
        <div class="user-row">
            <div class="user-bubble">
                <div class="user-label">YOU</div>
                <div class="user-text">{format_message_html(message['content'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_message(message):
    npc_role = message.get("npc_role", DEFAULT_NPC_ROLE)
    npc = NPCS.get(npc_role, NPCS[DEFAULT_NPC_ROLE])
    image_uri = get_image_data_uri(npc["image"])
    visible_content, charm_data = split_mental_charm_block(message.get("content", ""))
    visible_content, rag_checklist = split_rag_checklist_block(visible_content)
    charm_html = render_mental_charm_html(charm_data)
    assistant_html = (
        '<div class="assistant-turn">'
        '<div class="npc-layout">'
        '<div class="npc-portrait-wrap">'
        f'<img class="npc-art" src="{image_uri}" alt="{escape(npc_role)}">'
        "</div>"
        '<div class="npc-dialogue-wrap">'
        '<div class="npc-dialogue-card">'
        '<div class="npc-tag">NPC DIALOGUE</div>'
        f'<div class="npc-name">{escape(npc_role)}</div>'
        f"{get_npc_meta_html(npc_role)}"
        f'<div class="npc-text">{format_message_html(visible_content)}</div>'
        f"{charm_html}"
        "</div>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(assistant_html, unsafe_allow_html=True)
    if rag_checklist:
        with st.expander("RAG 연결 점검", expanded=True):
            st.markdown(rag_checklist)

    if is_stock_app_mode() and npc_role in STOCK_NPC_ROLES:
        render_stock_answer_graph_analysis(npc_role, message)
        selected_symbol, selected_stock_name, _ = get_selected_stock_age_trend_inputs()
        trend_data, trend_error = get_current_age_group_trend_data()
        with st.expander("연령대별 투자 경향 분석", expanded=False):
            render_age_group_trends_section(
                trend_data,
                selected_stock_name=selected_stock_name,
                selected_symbol=selected_symbol,
                error_message=trend_error,
            )


def render_quick_prompt_buttons(npc_role, key_prefix):
    quick_prompts = NPCS[npc_role]["quick_prompts"]
    st.markdown("#### 빠른 요청")
    quick_prompt_columns = st.columns(len(quick_prompts))
    for index, suggestion in enumerate(quick_prompts):
        with quick_prompt_columns[index]:
            if st.button(suggestion, use_container_width=True, key=f"quick_{key_prefix}_{npc_role}_{index}"):
                st.session_state.chat_draft_text = get_quick_prompt_text(npc_role, suggestion)
                st.session_state.chat_draft_version += 1
                st.rerun()
    if key_prefix == "after_answer":
        st.markdown('<div class="quick-prompt-spacer"></div>', unsafe_allow_html=True)


st.set_page_config(
    page_title={
        "basic": "기본 NPC RAG 챗봇",
        "stock": "주식 NPC 챗봇",
    }.get(APP_MODE, "통합 NPC 챗봇"),
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MENTAL_SIMULATION_SCENARIOS = {
    "FOMO 매수": {
        "scenario_name": "놓칠까 봐 급하게 올라탄 FOMO 매수",
        "loss_rate": 18,
        "emotion": "처음에는 놓칠까 봐 급했지만, 결과를 보고 나니 마음이 더 흔들릴 수 있습니다.",
        "root_cause": "남의 수익률과 급등 차트를 보고 내 계획보다 감정이 먼저 움직인 상황입니다.",
        "principles": ["급등한 이유를 확인하기 전에는 금액을 키우지 않기", "남의 수익 인증을 내 매수 근거로 쓰지 않기", "진입 이유와 손실 허용 범위를 먼저 적기"],
        "checklist": ["지금 사려는 이유를 한 문장으로 썼는가?", "실적이나 뉴스의 사실 정보를 확인했는가?", "내가 감당 가능한 비중인가?", "가격이 이미 많이 오른 뒤인지 확인했는가?", "10초 멈춘 뒤에도 같은 판단인가?"],
    },
    "패닉셀": {
        "scenario_name": "하락을 보고 바로 던지는 패닉셀",
        "loss_rate": 12,
        "emotion": "불안을 줄이려고 행동했지만, 결정 직후에도 후회가 남을 수 있습니다.",
        "root_cause": "하락의 원인과 내 투자 기간을 구분하기 전에 불안을 줄이려는 행동이 먼저 나온 상황입니다.",
        "principles": ["하락 이유를 확인하기 전에는 결론 내리지 않기", "투자 기간과 원래 계획을 다시 보기", "비중이 불안을 키우는지 점검하기"],
        "checklist": ["하락 원인이 기업 문제인지 시장 전체 문제인지 봤는가?", "처음 투자한 이유가 바뀌었는가?", "오늘 반드시 결정해야 하는가?", "내 비중이 너무 커서 불안한 것은 아닌가?", "호흡 후 다시 보아도 같은 판단인가?"],
    },
    "복수 매매": {
        "scenario_name": "손실을 한 번에 되찾으려는 복수 매매",
        "loss_rate": 25,
        "emotion": "빨리 회복하고 싶은 마음이 커질수록 판단이 더 급해질 수 있습니다.",
        "root_cause": "손실 자체보다 손실을 빨리 없애고 싶은 마음이 판단 기준을 흐린 상황입니다.",
        "principles": ["손실 회복을 목표로 매매하지 않기", "한 번의 거래로 만회하려는 생각 멈추기", "거래 전 쉬는 시간을 정해두기"],
        "checklist": ["지금 매매 이유가 복구 욕심인가?", "새 거래의 근거가 이전 손실과 분리되어 있는가?", "비중을 키우려는 이유가 합리적인가?", "쉬는 시간을 가졌는가?", "오늘 거래를 하지 않아도 되는가?"],
    },
    "몰빵": {
        "scenario_name": "확신 하나로 비중을 몰아넣는 몰빵",
        "loss_rate": 30,
        "emotion": "한 종목 움직임에 하루 기분 전체가 끌려가는 느낌이 들 수 있습니다.",
        "root_cause": "좋은 종목이라는 생각과 적절한 비중이라는 문제를 분리하지 못한 상황입니다.",
        "principles": ["확신과 비중은 따로 판단하기", "한 종목이 계좌 감정을 지배하지 않게 하기", "분산 기준을 먼저 정하기"],
        "checklist": ["이 종목 비중이 40%를 넘는가?", "같은 업종에 몰려 있지는 않은가?", "하락 시 버틸 기간을 정했는가?", "현금 비중이 있는가?", "다른 선택지가 있어도 이 비중이 적절한가?"],
    },
    "뉴스만 보고 매수": {
        "scenario_name": "뉴스 제목만 보고 들어가는 충동 매수",
        "loss_rate": 15,
        "emotion": "뉴스 제목은 강하게 느껴지지만, 실제 판단 근거는 부족할 수 있습니다.",
        "root_cause": "뉴스 제목의 분위기를 기업 가치나 가격 위치 확인보다 앞세운 상황입니다.",
        "principles": ["뉴스 제목과 실제 내용을 구분하기", "이미 가격에 반영됐는지 확인하기", "뉴스 하나만으로 판단하지 않기"],
        "checklist": ["뉴스 원문이나 핵심 내용을 확인했는가?", "주가가 이미 반응했는가?", "실적과 연결되는 뉴스인가?", "출처가 신뢰 가능한가?", "내 투자 계획과 맞는가?"],
    },
    "손절 기준 없이 버티기": {
        "scenario_name": "기준 없이 버티다 마음만 지치는 버티기",
        "loss_rate": 22,
        "emotion": "기다리면 괜찮아질 것 같지만, 기준이 없으면 불안이 계속 커질 수 있습니다.",
        "root_cause": "처음부터 확인 기준과 대응 기준을 정하지 않아 시간이 지날수록 판단이 더 어려워진 상황입니다.",
        "principles": ["기다리는 이유와 확인 기준을 구분하기", "버티는 기간을 정해두기", "추가 확인 지표를 미리 정하기"],
        "checklist": ["기다리는 이유가 명확한가?", "어떤 지표가 바뀌면 다시 볼지 정했는가?", "손실 규모가 감당 가능한가?", "추가 매수로 평균가만 낮추려는 건 아닌가?", "기록 없이 버티고 있지는 않은가?"],
    },
    "수익 났다고 무리하게 추가 매수": {
        "scenario_name": "수익의 기세에 취한 무리한 추가 매수",
        "loss_rate": 14,
        "emotion": "수익이 나면 자신감이 커지지만, 그 자신감이 계획보다 커질 수 있습니다.",
        "root_cause": "수익이 났다는 결과를 앞으로도 계속 맞을 것이라는 확신으로 착각한 상황입니다.",
        "principles": ["수익 이후에도 처음 계획을 다시 보기", "추가 매수 근거를 새로 확인하기", "자신감과 비중 확대를 분리하기"],
        "checklist": ["추가 매수 이유가 새 근거인가?", "현재 가격 위치를 확인했는가?", "비중이 과해지지 않는가?", "수익이 사라져도 괜찮은 계획인가?", "흥분이 가라앉은 뒤에도 같은 판단인가?"],
    },
}

inject_styles()

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = load_chat_histories()

for npc_role in NPC_ROLES:
    if npc_role not in st.session_state.chat_histories:
        st.session_state.chat_histories[npc_role] = {
            GENERAL_CHAT_ROOM_KEY: create_empty_chat_room(npc_role),
        }
    else:
        st.session_state.chat_histories[npc_role] = normalize_saved_npc_rooms(
            st.session_state.chat_histories[npc_role],
            npc_role,
        )

if "chat_histories_schema_saved" not in st.session_state:
    save_chat_histories()
    st.session_state.chat_histories_schema_saved = True

if "selected_npc_role" not in st.session_state:
    st.session_state.selected_npc_role = DEFAULT_NPC_ROLE
if st.session_state.selected_npc_role not in NPC_ROLES:
    st.session_state.selected_npc_role = DEFAULT_NPC_ROLE

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "selected_stock_name" not in st.session_state:
    st.session_state.selected_stock_name = None
if "selected_stock_news" not in st.session_state:
    st.session_state.selected_stock_news = {}
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()
if "stock_history_period_label" not in st.session_state:
    st.session_state.stock_history_period_label = "1개월"
if "chat_draft_text" not in st.session_state:
    st.session_state.chat_draft_text = ""
if "chat_draft_version" not in st.session_state:
    st.session_state.chat_draft_version = 0
if "investment_checklist_checks" not in st.session_state:
    st.session_state.investment_checklist_checks = {}
if "investment_checklist_visible" not in st.session_state:
    st.session_state.investment_checklist_visible = {}
if "show_stock_compare" not in st.session_state:
    st.session_state.show_stock_compare = False
if "compare_stock_a" not in st.session_state:
    st.session_state.compare_stock_a = ""
if "compare_stock_b" not in st.session_state:
    st.session_state.compare_stock_b = ""
if "compare_stock_period_label" not in st.session_state:
    st.session_state.compare_stock_period_label = "6개월"
if "compare_stock_error" not in st.session_state:
    st.session_state.compare_stock_error = ""

if "api_status" not in st.session_state:
    st.session_state.api_status = get_default_api_status()
elif has_huggingface_token() and st.session_state.api_status.get("label") == API_STATUS_NO_TOKEN:
    st.session_state.api_status = get_default_api_status()

render_header(st.session_state.selected_npc_role)

with st.container(key="npc_switch_bar"):
    selected_npc_role = st.selectbox("NPC 역할 선택", NPC_ROLES, key="selected_npc_role")
current_room_key = get_current_chat_room_key()
current_room = ensure_chat_room(selected_npc_role, current_room_key)
current_messages = current_room["messages"]
current_conversation_log = current_room["conversation_log"]
current_stock_summary = None
if is_stock_app_mode():
    current_history_period = STOCK_HISTORY_PERIOD_OPTIONS.get(st.session_state.stock_history_period_label, "1mo")
    current_stock_summary = get_selected_stock_summary(current_history_period)
    render_selected_stock_summary_card(current_stock_summary)
    if selected_npc_role != MENTAL_COACH_ROLE:
        render_selected_stock_news_panel(current_stock_summary)
        render_selected_stock_history_chart(current_stock_summary, selected_npc_role)
    render_investment_checklist_panel(selected_npc_role)
    render_stock_compare_section(selected_npc_role)
render_npc_skill_panel(selected_npc_role)
render_input_guide_panel(selected_npc_role)
if is_stock_app_mode():
    render_stock_context_panel(selected_npc_role)
    render_mental_care_tools(selected_npc_role)
if is_basic_app_mode():
    render_librarian_upload_panel(selected_npc_role)

render_quick_prompt_buttons(selected_npc_role, "top")

with st.sidebar:
    render_sidebar_npc_card(selected_npc_role)
    if is_stock_app_mode():
        render_stock_selector()
        render_watchlist_panel()
        render_stock_compare_sidebar_panel(selected_npc_role)
        render_investor_profile_panel()
        render_portfolio_panel(selected_npc_role)
        render_selected_stock_data_panel(current_stock_summary)
    render_npc_skill_panel(selected_npc_role)
    render_input_guide_panel(selected_npc_role)
    if is_stock_app_mode():
        render_stock_context_panel(selected_npc_role)
    st.subheader("API 상태")
    render_api_status()

    st.subheader("대화 설정")
    st.write(f"현재 선택한 기본 역할: {selected_npc_role}")
    st.write(f"현재 대화방: {current_room_key}")
    st.write(f"대화 수: {len(current_conversation_log) // 2}")

    if st.button("새 대화 시작"):
        reset_chat_room(selected_npc_role, current_room_key)
        st.session_state.api_status = get_default_api_status()
        st.rerun()

    st.subheader("대화 기록")
    if current_conversation_log:
        for log in current_conversation_log:
            if log["speaker"] == "user":
                st.markdown(
                    f"<div class='sidebar-log'><strong>사용자</strong>: {escape(clean_visible_message_text(log['content']))}</div>",
                    unsafe_allow_html=True,
                )
            else:
                npc_role = log["npc_role"]
                personality = NPCS.get(npc_role, NPCS[DEFAULT_NPC_ROLE])["personality"]
                st.markdown(
                    (
                        f"<div class='sidebar-log'><strong>{escape(npc_role)}</strong> "
                        f"({escape(personality)}): {escape(clean_visible_message_text(log['content']))}</div>"
                    ),
                    unsafe_allow_html=True,
                )
    else:
        st.caption("아직 대화 기록이 없습니다.")

chat_container = st.container()
with chat_container:
    for message in current_messages:
        if message["role"] == "user":
            render_user_message(message)
        else:
            render_assistant_message(message)

if current_messages and current_messages[-1]["role"] == "assistant":
    render_quick_prompt_buttons(selected_npc_role, "after_answer")

draft_key = f"chat_draft_input_{st.session_state.chat_draft_version}"
with st.container(key="chat_input_bar"):
    draft_input = st.text_area(
        "질문 입력",
        value=st.session_state.chat_draft_text,
        placeholder="질문을 입력해 주세요. 빠른 요청 버튼을 누르면 여기에 초안이 들어갑니다.",
        key=draft_key,
        height=88,
    )
    send_clicked = st.button("전송", type="primary", use_container_width=True)
user_input = draft_input.strip() if send_clicked else ""

if user_input:
    current_messages.append({"role": "user", "content": user_input})
    current_conversation_log.append({"speaker": "user", "content": user_input})
    save_chat_histories()

    response_npc_role = selected_npc_role
    applied_skills = ", ".join(NPCS[response_npc_role]["skills"][:3])

    if response_npc_role == "도서관 사서":
        with st.status("RAG 프롬프트를 생성하고 있습니다...", expanded=True) as response_status:
            top_k, chunk_size, overlap = get_rag_search_settings()
            st.write(f"현재 NPC: **{response_npc_role}**")
            st.write(f"사용한 임베딩 모델: {EMBEDDING_MODEL_NAME}")
            st.write(f"검색 설정: Top-k {top_k}개 · Chunk Size {chunk_size}자 · Overlap {overlap}자")
            st.write("입력 질문과 검색 근거 context를 분리해 Qwen에게 전달할 프롬프트로 정리하고 있습니다.")
            reply, response_source = build_librarian_document_answer(user_input, top_k=top_k)
            response_status.update(label="RAG 프롬프트 생성이 완료되었습니다.", state="complete")
    else:
        with st.status("답변을 생성하고 있습니다...", expanded=True) as response_status:
            st.write(f"현재 NPC: **{response_npc_role}**")
            if response_npc_role in STOCK_NPC_ROLES:
                st.write(f"현재 선택 종목: {get_selected_stock_label()}")
            st.write(f"역할: {NPCS[response_npc_role]['role']}")
            st.write(f"적용 Skill: {applied_skills}")
            st.write("AI가 선택된 NPC 역할로 답변을 작성하는 중입니다.")
            reply, response_source = get_llm_response(user_input, response_npc_role)
            response_status.update(label="답변 생성이 완료되었습니다.", state="complete")

    reply = finalize_response_text(reply, response_npc_role)

    assistant_message = {
        "role": "assistant",
        "npc_role": response_npc_role,
        "content": reply,
        "analysis_symbol": st.session_state.get("selected_symbol") if response_npc_role in STOCK_NPC_ROLES else None,
        "analysis_stock_name": st.session_state.get("selected_stock_name") if response_npc_role in STOCK_NPC_ROLES else None,
    }
    current_messages.append(assistant_message)
    current_conversation_log.append(
        {
            "speaker": "assistant",
            "npc_role": response_npc_role,
            "content": reply,
        }
    )
    save_chat_histories()

    if response_source == "LLM" and st.session_state.api_status["label"] != API_STATUS_LLM:
        set_api_status(API_STATUS_LLM, "LLM")

    st.session_state.chat_draft_text = ""
    st.session_state.chat_draft_version += 1
    st.rerun()
