# AI Knowledge Assistant

基于 Flask + DeepSeek API + FAISS 的 AI 知识库问答系统。

## 功能

- PDF 上传与文本解析
- 文本切分与本地向量化
- FAISS 向量检索
- RAG 知识库问答
- AI 性格选择
- 中英文界面切换
- 英文模式下强制英文回答
- 本地 JSON 长期记忆，重启后仍可加载历史对话

## 运行方法

```bash
pip install -r requirements.txt
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

## 环境变量

复制 `.env.example` 为 `.env`，填写 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

请不要把 `.env` 上传到 GitHub。

## 目录说明

```text
app.py                 Flask 后端接口
llm.py                 DeepSeek API 调用
pdf_utils.py           PDF 文本解析与切分
embedding_utils.py     本地文本向量化
vector_db.py           FAISS 向量检索与知识库持久化
templates/index.html   前端页面
static/style.css       页面样式
data/                  本地记忆与知识库文本块
uploads/               上传的 PDF 文件
```
