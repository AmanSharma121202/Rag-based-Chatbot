import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from typing import List, Dict, Any
from langchain_core.runnables import RunnableMap
from langchain_core.runnables.base import RunnableLambda
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.llms import HuggingFaceHub
import google.generativeai as genai


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if not GOOGLE_KEY:
#     raise EnvironmentError("Missing Google API Key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env")
# if not HUGGINGFACEHUB_API_TOKEN:
#     logger.warning("HUGGINGFACEHUB_API_TOKEN not found. Required for remote Hugging Face models.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# Replace the direct genai.GenerativeModel with LangChain wrapper

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    top_p=0.9,
    google_api_key=GOOGLE_KEY
)

class EnhancedRetriever:
    """Enhanced retriever with semantic chunking integration"""
    
    def __init__(self, persist_directory: str = "./vahei_db", collection_name: str = "dgft_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.retriever = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store and retriever"""
        try:
            logger.info(f"Initializing Chroma vector store with collection: {self.collection_name}, persist_directory: {self.persist_directory}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("Chroma vector store initialized.")
            # Enhanced retriever with metadata filtering and better search
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": 7,  # Retrieve more chunks initially
                    "fetch_k": 10,  # Fetch more for MMR selection
                    "lambda_mult": 0.7  # Balance between relevance and diversity
                }
            )
            logger.info("Vector store and retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def get_retriever_with_source_filtering(self, source_types: List[str] = None) -> Any:
        """Get retriever with optional source type filtering"""
        if source_types:
            logger.info(f"Creating filtered retriever for source_types: {source_types}")
            def filtered_retriever(query: str) -> List[Document]:
                logger.info(f"Filtered retrieval for query: '{query}' with source_types: {source_types}")
                docs = self.vector_store.similarity_search(
                    query, 
                    k=5,
                    filter={"source_type": {"$in": source_types}}
                )
                logger.info(f"Filtered retriever returned {len(docs)} documents.")
                return docs
            return filtered_retriever
        logger.info("Returning default retriever (no source filtering)")
        return self.retriever

    def debug_retrieval(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Debug function to inspect retrieved documents"""
        logger.info(f"Running debug retrieval for query: '{query}' with k={k}")
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Retrieved {len(docs)} documents with scores.")
        debug_info = []
        for doc, score in docs:
            logger.info(f"Doc source: {doc.metadata.get('source', 'unknown')}, Score: {score}")
            debug_info.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": score,
                "metadata": doc.metadata,
                "source": doc.metadata.get('source', 'unknown'),
                "chunk_info": {
                    "chunk_index": doc.metadata.get('chunk_index', 'N/A'),
                    "word_count": doc.metadata.get('word_count', 'N/A'),
                    "file_type": doc.metadata.get('file_type', 'N/A')
                }
            })
        return debug_info

# Initialize the enhanced retriever
enhanced_retriever = EnhancedRetriever()
retriever = enhanced_retriever.retriever


# Alternative for remote inference (uncomment to use):
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def initialize_llm():
#     return HuggingFaceHub(
#         repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
#         model_kwargs={"temperature": 0.1, "max_length": 500},
#         huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
#     )
# try:
#     llm = initialize_llm()
#     logger.info("Successfully initialized Hugging Face Hub model: mistralai/Mixtral-8x7B-Instruct-v0.1")
# except Exception as e:
#     logger.error(f"Failed to initialize Hugging Face Hub model: {e}")
#     raise


# Enhanced system prompt that works better with semantic chunks
system_prompt = (
"You are a highly knowledgeable and policy-compliant assistant specializing in DGFT (Directorate General of Foreign Trade) regulations, policies, and procedures."
"Your sole responsibility is to interpret, clarify, and communicate official information strictly based on the provided **DGFT document context**. You must not infer, assume, or introduce external data."
" TASK INSTRUCTIONS"
"1. Carefully review the entire `{context}` block containing excerpts from official DGFT notifications, trade circulars, handbooks, or FTP (Foreign Trade Policy)."
"2. Assess whether the `{input}` (user query) can be **answered precisely and fully** using only the supplied context."
"3. Based on the evaluation:"
   "**IF** the context contains sufficient information to answer the question:"
    "-  Provide a **clear, concise, and accurate** response in human-friendly language."
    "-  Reference specific DGFT notifications, policy clauses, chapters, procedures, or form names/numbers if available."
    "-  Avoid complex legal jargon where possible; explain technical terms simply."
    "-  Use a polite, professional tone that builds trust."
   "**ELSE IF** the context is **insufficient or unrelated** to the users query:"
     "-  Respond exactly with: "
     "- I cannot find sufficient information in the available DGFT documents to answer your question."
     "- Do not attempt to answer based on assumed or external knowledge."
" OUTPUT FORMAT"
"**Answer:**"
"{{your well-structured answer or fallback message here}}"
" BEST PRACTICES"
"- Avoid repeating users question unless needed for clarity."
"- Do not invent policy numbers or form names."
"- Maintain high fidelity to the source context at all times."
"- Be transparent if information is missing."
" INPUT BLOCK"
"Context:"
"{context}"
"Question:"
"{input}"
)

# Enhanced contextualization prompt for better question reformulation
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "Given a chat history and the latest user question about DGFT policies or foreign trade, "
    "reformulate the question to be standalone and clear. "
    "Fix any grammar or spelling mistakes. "
    "If the question references previous context (like 'that policy' or 'the same procedure'), "
    "make it specific based on the chat history. "
    "Do NOT answer the question, just reformulate it if needed."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Enhanced QA prompt with few-shot examples
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    # Few-shot examples for better performance
    ("human", "What is the weather today?"),
    ("ai", "I cannot find sufficient information in the available DGFT documents to answer your question."),
    ("human", "What are the export procedures for textiles?"),
    ("ai", "According to the DGFT guidelines, textile exports require the following procedures: [specific procedures from context]"),
    ("human", "{input}"),
])

# Enhanced follow-up question generation prompt
followup_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Based on the context, suggest 2 highly relevant follow-up questions the user might ask next. "
     "Questions should be short and informative. Do not repeat the original question. "
     "Only include questions if the context is relevant."),
    ("human", "Context:\n{context}\nOriginal Question:\n{input}")
])

# Create enhanced history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# Function to generate follow-up questions based on context
followup_chain = followup_prompt | llm | StrOutputParser()

def retrieve_and_split_docs(inputs: dict) -> dict:
    all_docs = history_aware_retriever.invoke(inputs)
    primary_docs = all_docs[:5]
    followup_docs = all_docs[5:7]
    return {"primary_docs": primary_docs, "followup_docs": followup_docs}

split_docs_chain = RunnableLambda(retrieve_and_split_docs)

def generate_followups(query, docs):
    if not docs:
        return []
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return followup_chain.invoke({"context": context_text, "input": query}).split("\n")

def rag_with_followups(input_text: str, chat_history: List[dict]) -> dict:
    inputs = {"input": input_text, "chat_history": chat_history}
    doc_dict = split_docs_chain.invoke(inputs)
    answer = question_answer_chain.invoke({
        "input": input_text,
        "chat_history": chat_history,
        "context": doc_dict["primary_docs"]
    })
    fallback = "I cannot find sufficient information" in answer
    followups = []
    if not fallback:
        followups = generate_followups(input_text, doc_dict["followup_docs"])
    return {
        "answer": answer,
        "followup_questions": followups
    }

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Enhanced debug function with better formatting
def debug_retriever(inputs):
    """Enhanced debug function to inspect retrieved documents"""
    logger.info(f"Debug retriever invoked with inputs: {inputs}")
    docs = history_aware_retriever.invoke(inputs)
    logger.info(f"Debug retriever returned {len(docs)} documents.")
    print("\n" + "="*50)
    print("RETRIEVED DOCUMENTS DEBUG")
    print("="*50)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"File Type: {doc.metadata.get('file_type', 'Unknown')}")
        print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
        print(f"Word Count: {doc.metadata.get('word_count', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:300]}...")
        print("-" * 40)
    return docs

# Enhanced debug function for similarity search
def debug_similarity_search(query: str, k: int = 3):
    """Debug similarity search with scores"""
    logger.info(f"Debug similarity search for query: '{query}' with k={k}")
    debug_info = enhanced_retriever.debug_retrieval(query, k)
    logger.info(f"Similarity search returned {len(debug_info)} results.")
    print(f"\n{'='*50}")
    print(f"SIMILARITY SEARCH DEBUG: '{query}'")
    print(f"{'='*50}")
    for i, info in enumerate(debug_info, 1):
        print(f"\n--- Result {i} (Score: {info['score']:.4f}) ---")
        print(f"Source: {info['source']}")
        print(f"Chunk Index: {info['chunk_info']['chunk_index']}")
        print(f"Word Count: {info['chunk_info']['word_count']}")
        print(f"File Type: {info['chunk_info']['file_type']}")
        print(f"Content: {info['content']}")
        print("-" * 40)
    return debug_info

# Create the retrieval chain
rag_chain_hist = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Enhanced session store with better management
class SessionManager:
    def __init__(self):
        self.store = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logger.info(f"Created new session: {session_id}")
        return self.store[session_id]
    
    def clear_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        return list(self.store.keys())

session_manager = SessionManager()

# Enhanced guardrail system
def enhanced_context_guard(inputs: dict, output: dict) -> dict:
    """Enhanced guardrail with better context validation"""
    answer = output.get("answer", "")
    context = inputs.get("context", [])
    if not context or (isinstance(context, list) and len(context) == 0):
        output["answer"] = "I cannot find sufficient information in the available DGFT documents to answer your question."
        logger.info("Guardrail triggered: No context available")
        return output
    generic_phrases = [
        "in general", "typically", "usually", "most likely", 
        "it is common", "generally speaking"
    ]
    if any(phrase in answer.lower() for phrase in generic_phrases):
        if "according to" not in answer.lower() and "based on" not in answer.lower():
            output["answer"] = "I cannot find sufficient information in the available DGFT documents to answer your question."
            logger.info("Guardrail triggered: Generic answer detected")
    return output

# Create the enhanced conversational RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_hist.with_config(run_postprocess_fn=enhanced_context_guard),
    session_manager.get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Utility Functions
def get_collection_stats():
    """Get statistics about the vector store collection"""
    try:
        logger.info("Getting collection stats...")
        collection = enhanced_retriever.vector_store._collection
        count = collection.count()
        logger.info(f"Collection '{enhanced_retriever.collection_name}' has {count} documents.")
        return {
            "document_count": count,
            "collection_name": enhanced_retriever.collection_name,
            "persist_directory": enhanced_retriever.persist_directory
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return None

def search_by_metadata(metadata_filter: Dict[str, Any], limit: int = 5) -> List[Document]:
    """Search documents by metadata filters"""
    try:
        logger.info(f"Searching by metadata: {metadata_filter}, limit: {limit}")
        docs = enhanced_retriever.vector_store.similarity_search(
            "", 
            k=limit,
            filter=metadata_filter
        )
        logger.info(f"Metadata search returned {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Failed to search by metadata: {e}")
        return []

# Exported items
__all__ = [
    "conversational_rag_chain", 
    "retriever", 
    "session_manager",
    "debug_retriever",
    "debug_similarity_search",
    "get_collection_stats",
    "search_by_metadata",
    "rag_with_followups"
]