# 📄 Conversational RAG with PDF Uploads & Chat History  

## 🚀 Overview  
Conversational RAG-based Q&A application that allows users to upload PDFs and interact with their content in real-time using LangChain, ChromaDB, and Groq's ChatGroq model.

## ✨ Features  
- **Upload PDFs** and chat with their content  
- **Retrieval-Augmented Generation (RAG)** for precise answers  
- **Persistent Chat History** using LangChain  
- **Groq LLM Integration** for smart responses  
- **ChromaDB** for efficient vector storage  

## 📦 Installation  
### 1️⃣ Clone this repository  

git clone https

://github.com/yourusername/your-repo-name.git
cd your-repo-name

2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt


3️⃣ Set up environment variables
Create a .env file and add your API keys:

env
Copy
Edit
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key


4️⃣ Run the application
bash
Copy
Edit
streamlit run app.py


## 🛠 Technologies Used
- **Streamlit**: For UI
- **LangChain**: For chaining and retrieval
- **ChromaDB**: For vector storage
- **HuggingFace Embeddings**: For document processing
- **Groq ChatGroq**: For AI-powered responses

## 📜 Usage
1. Upload one or multiple PDF files.
2. Enter a **Session ID** for history tracking.
3. Ask questions related to the PDF content.
4. View answers and previous chat history.

## 🤝 Contributing
Feel free to submit issues and pull requests. Any contributions are welcome!

## 📄 License
This project is licensed under the MIT License.
