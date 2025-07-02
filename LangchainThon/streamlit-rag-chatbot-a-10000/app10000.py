# app.py
# ─────────────────────────────
# Ⅰ. Import & 환경 설정
# ─────────────────────────────
import os
import json
import streamlit as st
from api10000 import search_products
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="영양제 Check 챗봇", page_icon="💊")

# ─────────────────────────────
# Ⅱ. PDF 로드 & 벡터스토어
# ─────────────────────────────
@st.cache_resource
# PDF를 카테고리 폴더 단위로 로드하고 LangChain Document로 변환
def load_and_split_pdfs_by_category(base_path: str) -> dict:
    base_path = os.path.join(os.path.dirname(__file__), base_path)
    
    all_docs_by_category = {}
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            docs = []
            for f in os.listdir(category_path):
                if f.endswith(".pdf"):
                    docs.extend(PyPDFLoader(os.path.join(category_path, f)).load())
            all_docs_by_category[category] = docs
    return all_docs_by_category


@st.cache_resource
# 문서를 chunk 단위로 분할 (1000자, 중복 150자)
# OpenAI Embedding으로 벡터 생성
# FAISS 인덱스를 카테고리 단위로 저장하거나 로드
# -> 운영 환경 배포 시에는 .pkl 제거 및 faiss + json 방식 권장
def create_vectorstore_per_category(_docs_by_category: dict) -> dict:
    vectorstores = {}
    for category, docs in _docs_by_category.items():
        persist_dir = f"./faiss_index/{category}"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

        if os.path.exists(os.path.join(persist_dir, "index.faiss")):
            # 이미 저장된 인덱스가 있다면 로드
            vectorstores[category] = FAISS.load_local(
                persist_dir,
                embeddings,
                allow_dangerous_deserialization=True  # 보안을 위해 기본적으로 Pickle 사용을 금지
            )
        else:
            # 없으면 새로 생성 후 저장
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            faiss_index = FAISS.from_documents(split_docs, embeddings)
            faiss_index.save_local(persist_dir)
            vectorstores[category] = faiss_index

    return vectorstores

# 사용자 쿼리에 포함된 키워드를 기반으로 관련 카테고리 선택
# 다중 카테고리 대응 (Case A~D 전용)
def categorize_user_query(query: str) -> List[str]:
    """시연용: 사용자 질문에서 케이스 A~D에 맞는 키워드 기반 카테고리 추출"""
    mapping = {
        "category_A": ["술", "담배", "커피", "종합비타민", "눈 건강", "간 회복"],
        "category_B": ["운동", "식습관", "종합비타민", "눈 건강", "근육 회복"],
        "category_C": ["당뇨병", "고혈압", "심혈관", "혈압"],
        "category_D": ["관절염", "치매", "아리셉트", "뇌 건강"],
    }

    query_lower = query.lower()
    selected = []

    for category, keywords in mapping.items():
        if all(keyword.lower() in query_lower for keyword in keywords):
            selected.append(category)

    return list(set(selected))


# 선택된 카테고리의 retriever들을 EnsembleRetriever로 통합
# -> create_history_aware_retriever를 통해 대화 기반 질문 리포맷
# -> QA Prompt 구성 후 create_retrieval_chain으로 RAG 완성
def build_rag_chain_from_categories(categories: List[str], vectorstores: dict):
    # combined_docs = []
    for cat in categories:
        retriever = vectorstores[cat].as_retriever()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    retrievers = [
    vectorstores[cat].as_retriever(search_kwargs={"k": 8})
    for cat in categories
]
    retriever = EnsembleRetriever(retrievers=retrievers)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question relating to dietary supplements, diseases, or schedules, reformulate a standalone question. Do NOT answer the question."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",  """당신은 개인 맞춤 영양제 코치입니다. 아래 사용자 정보를 분석해 사용자에게 필요한 영양제 성분과 피해야 할 영양제 성분을 알려주세요.

[성분 선정 원칙]
추천 영양제 성분 목록, 피해야 할 영양제 성분 목록은 문헌에 근거하여 작성해야 하며, 명시적 출처를 제시하세요.
문헌 내 언급이 없거나 명확하지 않은 경우, “정보 부족” 또는 “근거 없음”이라고 명시해주세요.
추천 성분 목록에 “영양제의 성분”에 집중하여 분석하고, 제품명이나 생활습관 요소(예: 커피, 알코올, 녹차 등)는 언급하지 마세요.
피해야 할 성분도 반드시 “영양제의 성분”에 집중하여 분석하고, 제품명이나 생활습관 요소(예: 커피, 알코올, 녹차, 고구마 등)는 언급하지 마세요.
성분명은 한국어로 작성하고, 성분명에는 공백이 없어야 합니다.
주의사항은 사용자에게 건강 관련 주의할 사항을 한, 두 문장으로 설명해주세요

[출력 형식(순수 JSON)]
{{
  "recommended": ["성분1", "성분2", ...],
  "avoid":       ["성분A", "성분B", ...],
  "cautions":    ["주의사항 ...", ...],
  "sources":     ["출처1", "출처2", ...]
}}

__마크다운이나 부가 설명 없이 위 JSON 한 덩어리만 반환__하세요.

{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ──────────────────────────────────────────────
# Ⅲ. LangChain 컴포넌트
# ──────────────────────────────────────────────
# @st.cache_resource
# : query에 따라 동적으로 벡터스토어를 고르기 때문에 캐시하면 이전 질의와 결과가 섞여 잘못된 결과를 줄 수 임음
# RAG 체인 초기화
# : 문서 로딩 + 벡터 생성/로드 + 카테고리 선택 + 체인 생성
def initialize_components( query: str):
    """영양제 문서 기반 RAG 체인 초기화 (카테고리 기반)"""
    docs_by_cat = load_and_split_pdfs_by_category("./data/supplement_knowledge")
    vectorstores = create_vectorstore_per_category(docs_by_cat)
    categories = categorize_user_query(query)
    return build_rag_chain_from_categories(categories, vectorstores)


def build_user_query(age: int, gender: str, disease: str, taking: str, prefers: str) -> str:
    return (
        f"다음은 개인 맞춤 영양제를 추천받고자 하는 사용자 정보입니다.\n\n"
        f"[사용자 정보]\n"
        f"- 나이/성별: {age}세 {gender}\n"
        f"- 질환/증상/특이사항: {disease}\n"
        f"- 현재 복용 중인 약제 또는 영양제: {taking}\n"
        f"- 원하는 영양제 기능 또는 효과: {prefers}\n\n"
        "이 정보를 바탕으로 어떤 성분을 추천하거나 피해야 할지를 분석해 주세요."
    )


# ──────────────────────────────────────────────
# Ⅳ. Streamlit UI
# ──────────────────────────────────────────────
st.header("💊 영양제 Check!")

chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 입력 폼
with st.form("user_profile_form"):
    st.markdown("#### 👤 사용자 건강 정보 입력")

    # 첫 줄: 왼쪽 = 나이·성별, 오른쪽 = 질환·증상
    top_left, top_right = st.columns(2)
    with top_left:
        age = st.number_input("나이", min_value=0, max_value=120, step=1, placeholder="예: 35")
        gender = st.selectbox("성별", ["여성", "남성", "기타"])
    with top_right:
        disease = st.text_area("① 질환·증상·특이사항", placeholder="예) 만성 위염, 고지혈증")

    # 두 번째 줄: 왼쪽 = 복용 중, 오른쪽 = 원하는 특징
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        taking = st.text_area("② 현재 복용 중인 영양제·약품", placeholder="예) 종합비타민")
    with bottom_right:
        prefers = st.text_area("③ 원하는 영양제 특징", placeholder="예) 피로 회복, 간 보호, 하루 한 알")

    submitted = st.form_submit_button("추천받기")

# 기존 대화 출력
for m in chat_history.messages:
    st.chat_message(m.type).write(m.content)

# 제출 이벤트
if submitted:
    if not all([age, gender, disease, taking, prefers]):
        st.warning("항목을 모두 입력해 주세요.")
        st.stop()

    user_query = build_user_query(age, gender, disease, taking, prefers)
    rag_chain = initialize_components(user_query)
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
    )

    st.chat_message("human").write(user_query)

    with st.chat_message("ai"):
        with st.spinner("분석 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": user_query}, config)
            # answer = response["answer"]
            # st.write(answer)
            answer_raw = response["answer"]

            # ── JSON 파싱 ─────────────────────────────
            try:
                answer_json = json.loads(answer_raw)
            except json.JSONDecodeError:
                st.error("⚠️ LLM 답변을 JSON으로 해석하지 못했습니다.")
                st.stop()
            
            # JSON 파싱
            recommended = answer_json.get("recommended", [])
            avoid = answer_json.get("avoid", [])
            cautions = answer_json.get("cautions", {})
            sources = answer_json.get("sources", [])

            # 구조화 출력
            st.subheader("📍사용자 건강 기반 분석")

            with st.expander("✅ 추천 성분"):
                if recommended:
                    st.markdown("\n".join([f"- {item}" for item in recommended]))
                else:
                    st.write("추천 성분 없음.")

            with st.expander("🚫 피해야 할 성분"):
                if avoid:
                    st.markdown("\n".join([f"- {item}" for item in avoid]))
                else:
                    st.write("피해야 할 성분 없음.")

            with st.expander("⚠️ 주의사항"):
                if cautions:
                    if isinstance(cautions, list):
                            for line in cautions:
                                st.markdown(f"- {line}")
                    elif isinstance(cautions, str):
                            st.markdown(f"- {cautions}")
                    else:
                            st.write("주의사항 형식을 확인할 수 없습니다.")
                else:
                    st.write("특별한 주의사항 없음.")

            with st.expander("📚 출처"):
                if sources:
                    for s in sources:
                        st.markdown(f"- {s}")
                else:
                    st.write("출처 없음.")

            # 추천 성분이 있으면 → api.py 로 전달 → 관련 제품 5개 검색
            if recommended:
                product_docs = search_products(recommended, avoid=avoid, top_k=5)

                # ──────────────────────────────
                #   Streamlit 화면에 결과 표시
                # ──────────────────────────────
                if product_docs:
                    st.subheader("💡 맞춤 영양제 추천")

                    for idx, doc in enumerate(product_docs, 1):
                        name = doc.metadata.get("product", f"제품 {idx}")
                        content = doc.page_content.strip()

                        with st.expander(f"{idx}. {name}"):
                            # 줄바꿈 기준으로 쪼개서 한 줄씩 출력
                            for line in content.split("\n"):
                                line = line.strip()
                                if line:
                                    st.write(line)
                else:
                    st.info("추천 성분과 연관된 제품을 찾을 수 없습니다.")

# 면책 고지
st.caption("⚠️ 본 서비스는 일반적인 건강 정보 제공을 목적으로 하며, "
           "개인 처방이 아닙니다. 복용 전 반드시 전문가와 상담하세요.")

