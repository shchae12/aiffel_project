# api.py
# 1. 라이브러리 임포트
import os
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import time
import re

# 2. 공공데이터 API에서 건강기능식품 수집 (XML 파싱)
def fetch_health_product_data(start_page=1, num_pages=1, delay=0.2):
    url = "http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"
    service_key = st.secrets["PUBLIC_API_KEY"]

    all_items = []
    for page in range(start_page, start_page + num_pages):
        params = {
            'serviceKey': service_key,
            'pageNo': page,
            'numOfRows': 100,
            'type': 'xml'
        }
        try:
            response = requests.get(url, params=params, verify=False)
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            if not items:
                print(f"🚫 {page}페이지 데이터 없음, 중단합니다.")
                break
            for item in items:
                row = {child.tag: child.text for child in item}
                all_items.append(row)
            
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ {page}페이지 에러: {e}")
            continue
    print(f"✅ {start_page}~{start_page + num_pages - 1} 페이지 수집 완료! 총 {len(all_items)}개")
    return all_items

# 3. 각 item → LangChain Document로 변환
def item_to_document(item):
    text = f"""제품명: {item.get('PRDUCT', '')}
업체명: {item.get('ENTRPS', '')}
기능성: {item.get('MAIN_FNCTN', '')}
주의사항: {item.get('INTAKE_HINT1', '')}
섭취방법: {item.get('SRV_USE', '')}
표준성분: {item.get('BASE_STANDARD', '')}
"""
    return Document(page_content=text, metadata={"product": item.get("PRDUCT", "")})

# 4. 벡터 임베딩 및 FAISS 벡터 DB 초기화
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# all_documents = []
db = None  # 최초 FAISS 인스턴스는 첫 500개로 생성

# 안전하게 문서 100개씩 나눠서 FAISS에 추가 - 임베딩 토큰 초과 에러 발생
def add_documents_in_chunks(db, documents, chunk_size=100):
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        db.add_documents(chunk)

# 5. 전체 10000개 데이터를 500개 단위로 수집, 임베딩
TOTAL_PAGES = 100
BATCH_SIZE = 500  # → 500개씩 임베딩
PAGES_PER_BATCH = BATCH_SIZE // 100  # 5페이지씩 = 500개

for i in range(0, TOTAL_PAGES, PAGES_PER_BATCH):
    start_page = i + 1
    print(f"\n📦 Batch {i // PAGES_PER_BATCH + 1} (페이지 {start_page}~{start_page + PAGES_PER_BATCH - 1}) 수집 중...")
    
    items = fetch_health_product_data(start_page=start_page, num_pages=PAGES_PER_BATCH)
    documents = [item_to_document(item) for item in items]

    if not documents:
        print("❌ 문서 없음 → 루프 종료")
        break

    if db is None:
        db = FAISS.from_documents(documents, embedding)
        print(f"🧠 초기 벡터 저장소 생성! (문서 수: {len(documents)})")
    else:
        add_documents_in_chunks(db, documents, chunk_size=100)
        print(f"➕ 벡터 저장소에 추가 완료! 누적 문서 수: {len(db.docstore._dict)}")

db.save_local("./faiss_index/db")
print(f"\n✅ 전체 벡터 DB 생성 완료! 총 문서 수: {len(db.docstore._dict)}개")

# 쿼리 -> 유사도 검색 -> 결과 리턴
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
    return FAISS.load_local("faiss_index/db", embeddings)

def search_products(ingredient_query, avoid=None, top_k: int = 5):
    """
    ingredient_query : 추천 성분(리스트 또는 쉼표 문자열)
    avoid           : 제외할 성분 리스트(옵션)
    top_k           : 최종 반환 개수
    """
    
    db = load_vectorstore()

    # ── [A] 입력 전처리 & 로그 ──────────────────────────────
    if isinstance(ingredient_query, list):
        ingredient_query = ", ".join(ingredient_query)
    # avoid가 문자열이면 쉼표 기준 분리
    if isinstance(avoid, str):
        avoid = [a.strip() for a in avoid.split(",") if a.strip()]

    print(f"\n[DEBUG] ingredient_query = {ingredient_query}")
    print(f"[DEBUG] avoid            = {avoid}")

    # ── [B] 후보군 검색 ───────────────────────────────────
    candidate_docs = db.similarity_search(ingredient_query, k=max(1, top_k * 4))
    print(f"[DEBUG] candidate_docs   = {len(candidate_docs)}개")

    # ── [C] ‘avoid’ 성분 필터 ─────────────────────────────
    if avoid:
        avoid_lower = [a.lower() for a in avoid]
        filtered_docs = []

        for doc in candidate_docs:
            text_lower = doc.page_content.lower()
            # 현재 문서에서 발견된 피해야 할 성분 목록
            matched = [a for a in avoid_lower if a in text_lower]

            if matched:
                # 어떤 성분 때문에 제외됐는지 표시
                product_name = doc.metadata.get("product", "이름없음")
                print(f"  ↪︎ [FILTER] {product_name}  (제외 이유: {', '.join(matched)})")
                continue

            filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break

        print(f"[DEBUG] filtered_docs    = {len(filtered_docs)}개 최종 반환\n")
        return filtered_docs

    # ── [D] 필터링 불필요 시 ───────────────────────────────
    print(f"[DEBUG] filtered_docs    = {top_k}개 최종 반환 (필터 없음)\n")
    return candidate_docs[:top_k]
