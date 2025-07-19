import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

st.title("💬AIチャット")

# --- APIキー入力欄 ---
api_key = st.text_input(
    "OpenAI APIキーを入力してください（入力後、エンターでセット）",
    type="password",
    value=st.session_state.get("api_key", ""),
)
if api_key:
    st.session_state["api_key"] = api_key

if "api_key" not in st.session_state or not st.session_state["api_key"]:
    st.info("OpenAI APIキーを入力してください。")
    st.stop()

# --- チャット履歴用メモリ ---
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="history",
)

# --- プロンプトの指定（history埋め込みはhumanの前に） ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なAIチャットアシスタントです。必ず日本語で簡潔に回答してください。"),
    ("human", "{history}\n{input}")
])

# --- LangChain LLM ---
llm = ChatOpenAI(
    openai_api_key=st.session_state["api_key"],
    model="gpt-3.5-turbo",
    temperature=0.7,
)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False,
)

# --- チャット履歴表示 ---
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# --- 入力 ---
user_input = st.chat_input("メッセージを入力…")

if user_input:
    st.chat_message("user").write(user_input)
    output = chain.run(user_input)
    st.chat_message("assistant").write(output)
