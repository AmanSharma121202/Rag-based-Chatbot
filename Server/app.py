from fastapi import FastAPI, Request
from pydantic import BaseModel
from generate import ChatbotInterface
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VAHEI 2.0 Chatbot API")

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chatbot instance
chatbot = ChatbotInterface()

# Request/Response Models
class QueryInput(BaseModel):
    query: str

class BotResponse(BaseModel):
    response: str
    followups: list[str] = []

@app.post("/chat", response_model=BotResponse)
async def post_user_query(input_data: QueryInput):
    """
    Accepts user query and returns chatbot response.
    Handles special commands and regular queries.
    """
    user_input = input_data.query.strip()

    if user_input.lower() in ["exit", "quit"]:
        return {"response": "Session terminated. Thank you for using VAHEI 2.0.", "followups": []}

    # Handle special commands (debug, stats, etc.)
    special_response = chatbot.handle_special_commands(user_input)
    if special_response:
        return {"response": special_response, "followups": []}

    # Generate response from RAG pipeline
    response = chatbot.generate_response(user_input)
    
    # Generate follow-up suggestions
    followups = chatbot.suggest_followups(user_input, response)
    if followups is None:
        followups = []
    
    return {"response": response, "followups": followups}


@app.get("/health")
async def health_check():
    """
        Health check endpoint to ensure the server is running.
    """
    return {"status": "VAHEI 2.0 backend is operational."}
