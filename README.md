# NPC 챗봇 Streamlit 프로젝트

Streamlit으로 실행하는 로컬 NPC 챗봇입니다. 기본/RAG NPC 앱과 주식 NPC 앱을 분리해 따로 실행할 수 있습니다. 기본/RAG 앱에서는 `친절한 상담원`, `아이디어 기획자`, `Python 튜터`, `보고서 도우미`, `도서관 사서` NPC를 사용하고, 주식 앱에서는 시장·종목·포트폴리오 관련 NPC를 사용합니다.

## 1. 프로젝트 소개

이 프로젝트는 초보자도 로컬에서 바로 실행해 볼 수 있는 NPC 챗봇 시작 프로젝트입니다. 기본 대화는 NPC별 역할과 스킬에 맞춰 답변하고, `도서관 사서`는 업로드한 PDF, TXT, CSV 문서를 검색해 출처가 있는 답변을 만듭니다.

## 2. 실행 환경

- Python 3.10 이상 권장
- Windows PowerShell 또는 Anaconda Prompt
- 인터넷 연결: 패키지 설치, 최초 임베딩 모델 다운로드, Hugging Face API 호출에 필요
- 주요 패키지: Streamlit, openai, pypdf, sentence-transformers, chromadb

## 3. 설치 방법

```bat
cd /d "C:\Users\user\Desktop\1\2-1\2-1여름 계절\agent"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Conda 환경을 사용한다면 원하는 환경을 먼저 활성화한 뒤 아래 명령만 실행해도 됩니다.

```bat
pip install -r requirements.txt
```

## 4. 환경변수 설정 방법

LLM API 답변을 사용하려면 `.env.example`을 복사해 `.env` 파일을 만들고 `HF_TOKEN` 값을 입력합니다. 실제 API 키나 토큰 값은 README와 소스 코드에 직접 적지 않습니다.

```bat
copy .env.example .env
notepad .env
```

`.env` 형식:

```env
HF_TOKEN=your_huggingface_token_here
```

토큰이 없거나 API 호출에 실패하면 앱은 로컬 fallback 답변 또는 검색 근거 요약을 표시합니다.

## 5. 실행 방법

기본/RAG NPC 앱:

```bat
streamlit run basic_npc_app\app.py
```

주식 NPC 앱:

```bat
streamlit run stock_npc_app\app.py
```

통합 모드가 필요하면 기존 진입점도 사용할 수 있습니다.

```bat
streamlit run app.py
```

실행 후 터미널에 표시되는 로컬 주소로 접속합니다. 보통 `http://localhost:8501` 입니다.

## 6. 주요 기능

- 앱 분리 실행: 기본/RAG NPC 앱과 주식 NPC 앱을 별도 폴더에서 실행
- 기본/RAG NPC 선택: 친절한 상담원, 아이디어 기획자, Python 튜터, 보고서 도우미, 도서관 사서
- NPC별 빠른 질문 버튼과 역할별 답변
- 대화 기록을 `chat_histories.json`에 로컬 저장
- 도서관 사서 전용 PDF, TXT, CSV 문서 업로드
- 문서 텍스트 추출, chunk 분할, `document_chunks.json` 저장
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 임베딩 생성
- `embeddings.json` 저장 및 로컬 `chroma_db` ChromaDB 저장
- Chroma Top-k 검색 기반 RAG 답변 생성
- RAG 품질 테스트에서 검색 결과, context, prompt, 답변 근거 확인

NPC별 역할:

- 친절한 상담원: 상황을 정리하고 다음 행동을 제안합니다.
- 아이디어 기획자: 아이디어를 넓히고 여러 안을 비교합니다.
- Python 튜터: Python과 Streamlit 코드, 오류 원인을 쉽게 설명합니다.
- 보고서 도우미: 메모와 초안을 요약하고 보고서 형식으로 다듬습니다.
- 도서관 사서: 업로드 문서에서 관련 근거를 찾아 출처와 함께 정리합니다.

## 7. 사용 예시

일반 대화:

```text
친절한 상담원: 지금 해야 할 일을 우선순위로 정리해줘.
아이디어 기획자: 발표 주제 아이디어를 5개 추천해줘.
Python 튜터: 이 오류 메시지가 무슨 뜻인지 쉽게 설명해줘.
보고서 도우미: 이 메모를 보고서 문장으로 다듬어줘.
```

도서관 사서 RAG 사용:

1. NPC 역할에서 `도서관 사서`를 선택합니다.
2. PDF, TXT, CSV 문서를 업로드합니다.
3. 문서 읽기 결과와 chunk 생성 결과를 확인합니다.
4. `임베딩 결과 보기`를 눌러 임베딩을 생성합니다.
5. `Chroma에 저장`을 눌러 검색 가능한 상태로 만듭니다.
6. 채팅창에 문서 기반 질문을 입력합니다.

```text
이 문서의 핵심 내용을 요약해줘.
문서에서 중요한 근거를 출처와 함께 알려줘.
이 내용은 어느 페이지에서 확인할 수 있어?
```

## 8. 문서 업로드 방법

- `도서관 사서` NPC를 선택하면 문서 업로드 영역이 표시됩니다.
- 지원 형식은 PDF, TXT, CSV입니다.
- 업로드한 파일은 프로젝트의 `data` 폴더에 저장됩니다.
- 같은 이름의 파일이 이미 있으면 덮어쓰지 않고 기존 파일을 사용합니다.
- PDF는 텍스트 추출이 가능한 파일이어야 합니다. 이미지로만 된 스캔 PDF는 내용이 비어 있을 수 있습니다.

## 9. RAG 답변 생성 흐름

1. `data` 폴더의 문서를 불러옵니다.
2. PDF, TXT, CSV에서 텍스트를 추출합니다.
3. 텍스트를 chunk 단위로 나눕니다.
4. 각 chunk에 `file_name`, `page_or_row`, `chunk_id` 같은 출처 정보를 붙입니다.
5. chunk를 임베딩 벡터로 변환합니다.
6. 임베딩과 메타데이터를 ChromaDB `documents` 컬렉션에 저장합니다.
7. 사용자 질문을 임베딩합니다.
8. Chroma Top-k 검색으로 질문과 가까운 chunk를 찾습니다.
9. 검색된 chunk를 context로 묶어 LLM 프롬프트에 넣습니다.
10. 답변과 함께 source, chunk_id, score를 표시합니다.

## 10. 검색 설정 설명

- Top-k: 질문과 가장 가까운 문서 조각을 몇 개 가져올지 정합니다. 기본 답변 흐름은 3개를 사용하고, RAG 품질 테스트에서는 1~10 사이로 조절할 수 있습니다.
- Chunk Size: 문서를 나눌 때 한 조각에 넣는 최대 글자 수입니다. 기본값은 500입니다. 값이 크면 문맥은 넓어지지만 검색이 덜 정밀할 수 있습니다.
- Overlap: 인접 chunk 사이에 겹쳐 넣는 글자 수입니다. 기본값은 50입니다. 문장이 잘리는 문제를 줄이지만 chunk 수와 저장량이 늘어날 수 있습니다.

## 11. 오류 해결

- `streamlit` 명령을 찾을 수 없음: 가상환경을 활성화하고 `pip install -r requirements.txt`를 다시 실행합니다.
- `.env`를 설정했는데 API가 실패함: `HF_TOKEN` 값과 네트워크 연결을 확인합니다.
- PDF 내용이 비어 있음: 텍스트 선택이 가능한 PDF인지 확인합니다.
- 임베딩 생성 실패: `sentence-transformers` 설치 여부와 인터넷 연결을 확인합니다.
- Chroma 저장 실패: `chromadb` 설치 여부와 `chroma_db` 폴더 쓰기 권한을 확인합니다.
- 답변 근거가 부족함: 문서가 Chroma에 저장되었는지, 질문이 문서 내용과 관련 있는지 확인합니다.

## 12. 주의 사항

- API 키와 토큰은 `.env`에만 저장합니다.
- `.env`, `chat_histories.json`, `document_chunks.json`, `embeddings.json`, `chroma_db`에는 개인 대화나 문서 내용이 포함될 수 있으므로 공유에 주의합니다.
- 업로드 문서의 저작권과 개인정보 포함 여부를 확인한 뒤 사용합니다.
- RAG 답변은 업로드 문서를 바탕으로 한 보조 결과입니다. 중요한 내용은 원문 근거를 함께 확인합니다.

## 13. 개발 기록

- 기본 NPC 5종 구성: 친절한 상담원, 아이디어 기획자, Python 튜터, 보고서 도우미, 도서관 사서
- NPC별 역할, 스킬, 빠른 질문 버튼 구성
- 대화 기록 로컬 저장
- Day 4: 도서관 사서 NPC의 RAG 기능 구현
- 문서 업로드 및 `data` 폴더 저장
- PDF, TXT, CSV 텍스트 추출
- chunk 분할과 `document_chunks.json` 저장
- 임베딩 생성 및 `embeddings.json` 저장
- ChromaDB 저장과 Top-k 검색
- 검색 context 기반 RAG 프롬프트 생성
- 답변 근거와 RAG 품질 테스트 화면 추가
