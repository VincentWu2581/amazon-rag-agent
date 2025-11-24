# Amazon 运营标准化 AI 导师 (RAG Pro)

一个基于 RAG（检索增强生成）技术的亚马逊运营知识库系统，集成了大型语言模型和向量数据库。

## 🚀 快速开始

### 本地运行

1. **克隆项目**
```bash
git clone <your-repo-url>
cd 亚马逊知识rag
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置 API Key**
在 `.streamlit/secrets.toml` 中添加：
```toml
DASHSCOPE_API_KEY = "sk-your-api-key-here"
```

5. **运行应用**
```bash
streamlit run streamlit_app.py
```

## 🌐 部署到 Streamlit Community Cloud

### 前置条件
- GitHub 账号
- Streamlit Community Cloud 账号（免费注册：https://streamlit.io/cloud）
- 所有知识库文件已上传到 GitHub

### 部署步骤

#### 1. 上传到 GitHub

```bash
# 初始化 Git 仓库（如果尚未）
git init
git add .
git commit -m "初始提交：Amazon RAG 应用"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/amazon-rag-agent.git
git push -u origin main
```

#### 2. 连接 Streamlit Cloud

1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 点击 **"New app"**
3. 选择：
   - **Repository**: 选择你的 GitHub 仓库
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. 点击 **"Deploy"**

#### 3. 配置 Secrets（API Key）

1. 在 Streamlit Cloud 应用页面，点击右上角的 **"☰"** 菜单
2. 选择 **"Settings"** → **"Secrets"**
3. 在文本框中添加：
```toml
DASHSCOPE_API_KEY = "sk-your-dashscope-api-key"
```
4. 点击 **"Save"** - 应用会自动重新启动

#### 4. 验证部署

- 等待应用加载完成
- 测试聊天功能是否正常

## ⚙️ 系统架构

```
用户输入
   ↓
知识库检索 (FAISS + BGE Embeddings)
   ↓
K=10 召回 → K=5 精排
   ↓
LLM 生成 (Qwen-Max)
   ↓
结构化输出 (CoT 推理链)
```

## 📋 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CHUNK_SIZE | 800 | 文档分块大小 |
| CHUNK_OVERLAP | 100 | 块之间的重叠 |
| EMBEDDING_MODEL_NAME | BAAI/bge-small-zh-v1.5 | 中文嵌入模型 |
| RECALL_K | 10 | 初始检索数量 |
| RERANK_K | 5 | 精排后的文档数 |
| LLM_MODEL | qwen-max | 使用的语言模型 |

## 🔧 常见问题

### 部署后应用无法加载
- **原因**: 知识库文件缺失或 API Key 未正确配置
- **解决**: 确保所有 `.docx` 文件已上传到 GitHub，且 Secrets 中的 API Key 正确

### "缺少 DASHSCOPE_API_KEY" 错误
- **原因**: 未在 Streamlit Cloud 的 Secrets 中配置
- **解决**: 按步骤 3 配置 Secrets，应用会自动重启

### 模型加载缓慢
- **原因**: 首次加载时需要下载嵌入模型和知识库处理
- **解决**: 这是正常的，等待 2-3 分钟。第二次访问会更快（使用 `@st.cache_resource` 缓存）

## 🛠️ 技术栈

- **LLM**: Qwen-Max (通义千问)
- **嵌入模型**: BGE-Small-ZH
- **向量数据库**: FAISS
- **框架**: Streamlit
- **文本处理**: LangChain

## 📝 许可证

MIT License

## 👨‍💻 作者

Amazon RAG Team

---

**需要帮助？** 查看 [Streamlit 文档](https://docs.streamlit.io) 或 [LangChain 文档](https://python.langchain.com)
