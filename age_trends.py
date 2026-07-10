from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


AGE_GROUPS = ["10대", "20대", "30대", "40대", "50대 이상"]

SOURCE_CANDIDATES = [
    {
        "name": "네이버 뉴스 검색",
        "url": "https://search.naver.com/search.naver?where=news&query={query}",
    },
    {
        "name": "네이버 뉴스 검색: 연령대별 주식 투자",
        "url": "https://search.naver.com/search.naver?where=news&query={base_query}",
    },
    {
        "name": "네이버 웹 검색: 연령대별 투자자",
        "url": "https://search.naver.com/search.naver?where=web&query={base_query}",
    },
    {
        "name": "네이버 뉴스 검색: 한국예탁결제원 연령별 투자자",
        "url": "https://search.naver.com/search.naver?where=news&query={ksd_query}",
    },
    {
        "name": "네이버 뉴스 검색: 금융투자협회 연령별 투자",
        "url": "https://search.naver.com/search.naver?where=news&query={kofia_query}",
    },
]

SOURCE_REFERENCE_LINKS = [
    {"name": "한국예탁결제원 증권정보포털", "url": "https://seibro.or.kr/"},
    {"name": "금융투자협회", "url": "https://www.kofia.or.kr/"},
    {"name": "금융위원회 보도자료", "url": "https://www.fsc.go.kr/no010101"},
    {"name": "공공데이터포털", "url": "https://www.data.go.kr/"},
]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_get(url: str, timeout: int = 8) -> tuple[str, str]:
    if requests is None:
        raise RuntimeError("requests package is not installed")
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    return response.text, response.url


def _extract_page_text(html: str) -> str:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 package is not installed")
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return _clean_text(soup.get_text(" "))


def _age_patterns(age_group: str) -> list[str]:
    if age_group == "50대 이상":
        return ["50대 이상", "50대", "60대", "70대", "고령층"]
    return [age_group]


def _find_age_mentions(text: str) -> list[str]:
    mentions = []
    for age_group in AGE_GROUPS:
        if any(pattern in text for pattern in _age_patterns(age_group)):
            mentions.append(age_group)
    return mentions


def _extract_nearby_keywords(text: str, age_group: str, keywords: list[str]) -> list[str]:
    matches = []
    for pattern in _age_patterns(age_group):
        for match in re.finditer(re.escape(pattern), text):
            start = max(0, match.start() - 160)
            end = min(len(text), match.end() + 220)
            window = text[start:end]
            for keyword in keywords:
                if keyword in window and keyword not in matches:
                    matches.append(keyword)
    return matches


def _extract_nearby_date(text: str, age_group: str) -> str:
    date_patterns = [
        r"\b20\d{2}[.-]\d{1,2}[.-]\d{1,2}\b",
        r"\b20\d{2}년\s*\d{1,2}월\s*\d{1,2}일\b",
        r"\b20\d{2}년\s*\d{1,2}월\b",
        r"\b\d{1,2}[.-]\d{1,2}\b",
    ]
    for pattern in _age_patterns(age_group):
        for match in re.finditer(re.escape(pattern), text):
            start = max(0, match.start() - 260)
            end = min(len(text), match.end() + 320)
            window = text[start:end]
            for date_pattern in date_patterns:
                date_match = re.search(date_pattern, window)
                if date_match:
                    return _clean_text(date_match.group(0))
    for date_pattern in date_patterns:
        date_match = re.search(date_pattern, text[:1200])
        if date_match:
            return _clean_text(date_match.group(0))
    return ""


def _build_trend_from_text(
    text: str,
    source_name: str,
    source_url: str,
    selected_stock_name: str | None,
    selected_symbol: str | None,
    selected_sector: str | None,
) -> list[dict[str, Any]]:
    age_mentions = _find_age_mentions(text)
    if not age_mentions:
        return []

    sector_keywords = [
        selected_sector,
        "반도체",
        "AI",
        "전기차",
        "2차전지",
        "바이오",
        "금융",
        "자동차",
        "플랫폼",
        "게임",
        "헬스케어",
        "ETF",
    ]
    stock_keywords = [selected_stock_name, selected_symbol, "삼성전자", "SK하이닉스", "현대차", "NAVER", "카카오"]
    style_keywords = ["성장주", "가치주", "장기투자", "단기투자", "공격적", "안정형", "배당", "ETF", "해외주식"]
    sector_keywords = [keyword for keyword in sector_keywords if keyword]
    stock_keywords = [keyword for keyword in stock_keywords if keyword]

    trends = []
    for age_group in AGE_GROUPS:
        if age_group not in age_mentions:
            continue
        sectors = _extract_nearby_keywords(text, age_group, sector_keywords)
        stocks = _extract_nearby_keywords(text, age_group, stock_keywords)
        styles = _extract_nearby_keywords(text, age_group, style_keywords)
        published_at = _extract_nearby_date(text, age_group)
        trend = {
            "age_group": age_group,
            "preferred_sectors": sectors,
            "representative_stocks": stocks,
            "investment_style": ", ".join(styles),
            "relation_to_selected_stock": "",
            "source_name": source_name,
            "source_url": source_url,
            "published_at": published_at,
            "numeric_value": None,
            "numeric_label": "",
        }
        trends.append(trend)
    return trends


def match_stock_to_age_trend(
    trend_data: list[dict[str, Any]],
    selected_stock_name: str | None,
    selected_symbol: str | None,
    selected_sector: str | None,
) -> list[dict[str, Any]]:
    matched_trends = []
    stock_tokens = [token for token in [selected_stock_name, selected_symbol] if token]
    sector_token = selected_sector or ""

    for trend in trend_data:
        representative_stocks = trend.get("representative_stocks") or []
        preferred_sectors = trend.get("preferred_sectors") or []
        relation = "직접 연관 자료는 확인되지 않았습니다"

        if any(token in representative_stocks for token in stock_tokens):
            relation = "선택 종목이 출처 문맥의 대표 관심 종목으로 언급되었습니다."
        elif sector_token and sector_token in preferred_sectors:
            relation = "선택 종목의 업종 키워드가 출처 문맥의 선호 업종과 함께 언급되었습니다."

        updated = dict(trend)
        updated["relation_to_selected_stock"] = relation
        matched_trends.append(updated)

    return matched_trends


def fetch_age_group_investment_trends(
    selected_symbol: str | None = None,
    selected_stock_name: str | None = None,
    selected_sector: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    query_parts = ["연령대별", "투자", "주식", "경향"]
    if selected_stock_name:
        query_parts.append(selected_stock_name)
    elif selected_symbol:
        query_parts.append(selected_symbol)
    if selected_sector:
        query_parts.append(selected_sector)
    query = quote_plus(" ".join(query_parts))
    base_query = quote_plus("연령대별 주식 투자 경향 20대 30대 40대 50대")
    ksd_query = quote_plus("한국예탁결제원 연령별 주식 투자자 보유 종목")
    kofia_query = quote_plus("금융투자협회 연령별 투자자 주식 투자 성향")

    errors = []
    collected: list[dict[str, Any]] = []
    for source in SOURCE_CANDIDATES:
        source_name = source["name"]
        source_url = source["url"].format(
            query=query,
            base_query=base_query,
            ksd_query=ksd_query,
            kofia_query=kofia_query,
        )
        try:
            html, final_url = _safe_get(source_url)
            page_text = _extract_page_text(html)
            trends = _build_trend_from_text(
                page_text,
                source_name,
                final_url,
                selected_stock_name,
                selected_symbol,
                selected_sector,
            )
            collected.extend(trends)
        except Exception as error:
            errors.append(f"{source_name}: {error}")

    if not collected:
        error_message = "자료를 불러오지 못했습니다."
        if errors:
            error_message = f"{error_message} 출처가 확인된 자료가 없어 임의 분석을 표시하지 않습니다."
        return [], error_message

    return match_stock_to_age_trend(
        collected,
        selected_stock_name=selected_stock_name,
        selected_symbol=selected_symbol,
        selected_sector=selected_sector,
    ), ""


def build_age_group_trend_context(trend_data: list[dict[str, Any]]) -> str:
    if not trend_data:
        return ""

    lines = [
        "연령대별 투자 경향 참고 context:",
        "- 아래 내용은 출처가 확인된 공개 자료와 뉴스 문맥에서만 사용한다.",
        "- 자료에 없는 연령대별 선호 종목이나 성향은 추측하지 않는다.",
        "- 선택 종목과의 연관성이 불명확하면 불명확하다고 말한다.",
        "- 투자 추천처럼 표현하지 않는다.",
    ]
    for trend in trend_data:
        sectors = ", ".join(trend.get("preferred_sectors") or []) or "확인된 값 없음"
        stocks = ", ".join(trend.get("representative_stocks") or []) or "확인된 값 없음"
        lines.append(
            f"- {trend.get('age_group')}: 선호 업종={sectors}; "
            f"대표 관심 종목={stocks}; "
            f"선택 종목 연관성={trend.get('relation_to_selected_stock') or '직접 연관 자료는 확인되지 않았습니다'}; "
            f"출처={trend.get('source_name') or '출처명 없음'} "
            f"{trend.get('source_url') or ''}; 기준일={trend.get('published_at') or '확인 불가'}"
        )
    return "\n".join(lines)
