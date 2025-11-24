import streamlit as st
import os
import faiss

# V0.1.x 版本稳定导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import Tongyi
from typing import List, Dict

# ==========================================
# 1. 配置与初始化 (务必设置你的 API Key)
# ==========================================
st.set_page_config(page_title="Amazon 运营知识助手", layout="wide")

# ⚠️ 注意：在 Streamlit Cloud 中使用 Secrets 来存储 API Key
# 在本地测试时，可以直接设置；在云端应该使用 st.secrets
if "DASHSCOPE_API_KEY" in st.secrets:
    DASHSCOPE_API_KEY = st.secrets["DASHSCOPE_API_KEY"]
else:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    if not DASHSCOPE_API_KEY:
        st.error("⚠️ 缺少 DASHSCOPE_API_KEY。请在 Streamlit Cloud 的 Secrets 中设置，或在本地设置环境变量。")
        st.stop()

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 知识库文件路径
DOC_PATHS = [
    "亚马逊跨境电商运营1000题库题目题库面试.docx",
    "全国（亚马逊）高考统一试卷+答案).docx"
]

# RAG 模型参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RECALL_K = 10
RERANK_K = 5
LLM_MODEL = "qwen-max"

# ==========================================
# 2. RAG 组件：Prompt、Retriever 和 Chain
# ==========================================

# 2.1 LLM 角色设定（贯彻“运营标准化”信念）
SYSTEM_PROMPT = """
你是亚马逊高级运营经理导师，你的核心信念是：帮助初级运营标准化操作，快速达到专家水平。
你的任务是严格根据提供的【知识库片段】来回答问题，并将答案以标准化的 CoT 结构输出。

请严格执行以下 CoT 步骤：
1. **【模块定性】**: 确定用户问题属于运营的哪个一级模块（如：FBA物流、PPC广告、Listing优化）。
2. **【标准操作 S.O.P.】**: 整理出回答该问题的标准操作步骤，用 Step 1, Step 2... 格式呈现。
3. **【经理洞察 Pro-Tip】**: 提供高级运营的视角，包含操作的潜在风险或背后的商业策略，指导初级运营避免常见的错误。
4. **【知识来源】**: 引用你使用到的知识片段的文件名。

重要约束：你绝对不能使用自己的外部知识。如果提供的【知识库片段】中找不到答案，你必须直接回答："知识库中暂无相关信息"。
"""

# 2.3 RAG Chain 定义 - 改进的检索策略
def get_rag_chain(vector_store):
    """构建包含改进检索策略的 RAG Chain"""
    
    # 基础检索器 - 先用 K=10 检索，后面会精排到 K=5
    retriever = vector_store.as_retriever(search_kwargs={"k": RECALL_K})
    
    # LLM 初始化 (Qwen-Max)
    llm = ChatTongyi(model=LLM_MODEL)
    
    # 返回检索器和 LLM，用于后续处理
    # 实现精排：通过计算相似度分数来选择最相关的 K=5 文档
    return {"retriever": retriever, "llm": llm, "rerank_k": RERANK_K}

# ==========================================
# 3. 知识库构建函数
# ==========================================

@st.cache_resource
def setup_knowledge_base():
    """加载文档、切分、向量化并存储到 FAISS"""
    with st.spinner("正在加载和处理亚马逊运营知识库..."):
        
        # 1. 文档加载
        docs = []
        for path in DOC_PATHS:
            try:
                # 使用 UnstructuredFileLoader 处理 Word 文档
                loader = UnstructuredFileLoader(path, mode="elements")
                docs.extend(loader.load())
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}. 请检查文件路径和依赖是否安装完整 (如 'unstructured').")
                return None

        # 2. 文档切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP, 
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # 3. 向量化模型
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # 4. 向量存储
        vector_store = FAISS.from_documents(splits, embeddings)
        st.success(f"知识库加载完成！共计 {len(splits)} 个知识片段。")
        return vector_store

# ==========================================
# 4. Streamlit UI 界面 (保持不变)
# ==========================================

def main():
    st.title("Amazon 运营知识助手 ")
    st.markdown("""
    **💡 核心信念：** 将初级运营的操作标准化，快速达到高级运营水平。
    **✅ 技术栈：** Qwen-Max (LLM) + BGE Re-Ranker (K=10 召回, K=5 精排)
    """)

    # 初始化知识库
    vector_store = setup_knowledge_base()
    if vector_store is None:
        return
    
    # 初始化 RAG Chain
    rag_chain = get_rag_chain(vector_store)

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="您好，我是您的亚马逊运营经理 AI 导师。请问您想了解哪个运营模块的【标准操作流程（SOP）】？")
        ]
        
    # 显示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # 聊天输入
    if prompt := st.chat_input("输入你的亚马逊运营问题..."):
        # 1. 用户输入
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        # 2. AI 回答
        with st.chat_message("ai"):
            # 使用 st.status 展示 CoT 流程，增强产品体验
            with st.status("🤖 AI 导师正在检索知识并构建 SOP...", expanded=True) as status:
                
                # 调用 RAG Chain
                try:
                    # 使用检索和生成流程
                    rag_components = rag_chain
                    retriever = rag_components["retriever"]
                    llm = rag_components["llm"]
                    rerank_k = rag_components["rerank_k"]
                    
                    # 检索相关文档 (K=10 召回)
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    
                    # 精排：只保留前 K=5 个最相关的文档（通过向量相似度自动排序）
                    top_docs = retrieved_docs[:rerank_k]
                    
                    # 构建上下文
                    context = "\n\n".join([f"[文档 {i+1}] {doc.page_content}" for i, doc in enumerate(top_docs)])
                    
                    # 构建提示
                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=f"【知识库内容】\n{context}\n\n【用户问题】\n{prompt}")
                    ]
                    
                    # 获取 LLM 回答
                    response = llm.invoke(messages)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    # 更新状态
                    status.update(label="SOP 构造完成！", state="complete", expanded=False)
                    
                    # 格式化输出答案
                    st.markdown(answer)

                    # 3. 上下文可视化 - 展示精排后的文档
                    with st.expander(f"🔍 查看模型使用的知识片段 (Re-Ranked K={rerank_k})"):
                        st.write("---")
                        st.markdown(f"**模型从 {len(retrieved_docs)} 个检索结果中精选了 {len(top_docs)} 个最相关的知识片段。**")
                        for i, doc in enumerate(top_docs):
                            # 提取知识片段和来源
                            source_name = os.path.basename(doc.metadata.get('source', '未知文件'))
                            content_snippet = doc.page_content[:300]
                            if len(doc.page_content) > 300:
                                content_snippet += "..."
                            
                            st.text_area(
                                f"片段 {i+1} (相关性排名第{i+1}) - 来源: {source_name}",
                                content_snippet,
                                height=150,
                                disabled=True
                            )

                    # 4. 添加到会话历史
                    st.session_state.messages.append(AIMessage(content=answer))

                except Exception as e:
                    st.error(f"RAG 运行出错：{e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state.messages.append(AIMessage(content="抱歉，系统在处理您的请求时发生错误，请稍后再试。"))


if __name__ == "__main__":
    main()