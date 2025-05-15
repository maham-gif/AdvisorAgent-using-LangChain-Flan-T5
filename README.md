# AdvisorAgent-using-LangChain-Flan-T5
For this project, you can name it:

### 🔹 **Project Name:**

**AdvisorAgent: A University Guidance Chatbot using LangChain & Flan-T5**

---

### 📝 **README.md**

Here’s a professional and concise `README.md` you can include on GitHub:

---

````markdown
# AdvisorAgent: University Guidance Chatbot

A session-aware conversational assistant built with Hugging Face's Flan-T5 model and LangChain, designed to help university students get quick and informative academic advice.

## 🧠 Features

- Powered by `google/flan-t5-small` via Hugging Face Transformers
- Uses LangChain's `RunnableWithMessageHistory` for session memory
- Dynamic responses based on conversation history
- Simulates a helpful, concise university advisor
- Simple and modular Python implementation

## 🚀 Technologies Used

- Python 3.9+
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- [LangChain](https://www.langchain.com/)
- Chat Memory: `ChatMessageHistory`

## 📦 Installation

```bash
git clone https://github.com/yourusername/advisoragent.git
cd advisoragent
pip install -r requirements.txt
````

### `requirements.txt`

```
transformers
torch
langchain
langchain-core
langchain-huggingface
```

## 🧪 How to Run

```bash
python advisor_chatbot.py
```

## 📘 Example

```
===== Prompt 1 =====
User: Can you explain how credit hours work?
Bot: Credit hours are units that represent how much time you'll spend in a class each week. Typically, 3 credit hours = 3 hours of class/week.

===== Prompt 2 =====
User: How many do I need to graduate?
Bot: Most undergraduate programs require around 120 credit hours to graduate. Check with your department for exact details.
```

## 🔒 Session Memory

The chatbot remembers user conversations using session IDs, allowing follow-up questions to retain context.

```python
config={"configurable": {"session_id": "student123"}}
```

## 📄 License

MIT License. Feel free to use and modify.

## 👤 Author

\[Maham] – Final Year Student at AUST


Let me know if you'd like me to generate:
- The actual `requirements.txt`
- A `GitHub` repository structure
- Add Flask or a frontend interface for deployment

I'm happy to assist with uploading as well.
