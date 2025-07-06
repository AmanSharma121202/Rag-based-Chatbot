# main_chatbot.py
import pyttsx3
import speech_recognition as sr
import logging
import uuid
from datetime import datetime
from typing import Optional
from rag_pipline import (
    conversational_rag_chain, 
    session_manager, 
    debug_retriever,
    debug_similarity_search,
    get_collection_stats,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceManager:
    """Manages text-to-speech and speech recognition functionality"""
    
    def __init__(self):
        self.tts_engine = None
        self.recognizer = None
        self.microphone = None
        self.voice_available = False
        self._initialize_voice_components()
    
    def _initialize_voice_components(self):
        """Initialize TTS and speech recognition components"""
        try:
            # Initialize TTS
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            # Set to first male voice found, or default if not found
            male_voice = None
            for voice in voices:
                if 'male' in voice.name.lower() or 'male' in voice.id.lower():
                    male_voice = voice
                    break
            if male_voice:
                self.tts_engine.setProperty('voice', male_voice.id)
                print(f"[INFO] Male voice selected: {male_voice.name}")
            else:
                print("[INFO] Default voice selected.")
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Slower for better understanding
            self.tts_engine.setProperty('volume', 0.9)
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                print("[INFO] Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
            self.voice_available = True
            print("[SUCCESS] Voice components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            print("[ERROR] Voice functionality not available. Text-only mode.")
            self.voice_available = False
    
    def speak_text(self, text: str):
        """Convert text to speech"""
        if not self.voice_available or not self.tts_engine:
            print("[WARNING] Text-to-speech not available")
            return
        
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print("[ERROR] Failed to speak text")
    
    def listen_to_user(self, timeout: int = 10) -> str:
        """Listen to user speech and convert to text"""
        if not self.voice_available or not self.recognizer:
            return "Voice recognition not available"
        
        try:
            with self.microphone as source:
                print("🎤 Listening... Please speak now.")
                # Listen with timeout and phrase time limit
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
            
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your speech. Please try again."
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return "Speech recognition service is unavailable."
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except Exception as e:
            logger.error(f"Unexpected speech recognition error: {e}")
            return "An error occurred during speech recognition."

class ChatbotInterface:
    """Main chatbot interface with enhanced functionality"""
    
    def __init__(self):
        self.voice_manager = VoiceManager()
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.debug_mode = False
        
        print(f"[INFO] Session ID: {self.session_id}")
        
        # Get collection stats
        stats = get_collection_stats()
        if stats:
            print(f"[INFO] Loaded {stats['document_count']} documents from collection")
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using the enhanced RAG system"""
        try:
            # Log the interaction
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.conversation_history.append({
                "timestamp": timestamp,
                "user_input": user_input,
                "response": None
            })
            
            # Debug mode - show retrieved documents
            if self.debug_mode:
                print("\n[DEBUG] Retrieved documents:")
                debug_retriever({"input": user_input})
            
            # Generate response
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )["answer"]
            
            # Update conversation history
            self.conversation_history[-1]["response"] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def handle_special_commands(self, user_input: str) -> Optional[str]:
        """Handle special commands like debug, stats, etc."""
        command = user_input.lower().strip()
        
        if command == "debug on":
            self.debug_mode = True
            return "Debug mode enabled. Retrieved documents will be shown."
        
        elif command == "debug off":
            self.debug_mode = False
            return "Debug mode disabled."
        
        elif command == "stats":
            stats = get_collection_stats()
            if stats:
                return f"Collection Stats:\n- Documents: {stats['document_count']}\n- Collection: {stats['collection_name']}"
            return "Unable to retrieve collection statistics."
        
        elif command == "history":
            if not self.conversation_history:
                return "No conversation history available."
            
            history_text = "Recent Conversation History:\n"
            for i, entry in enumerate(self.conversation_history[-5:], 1):  # Last 5 entries
                history_text += f"{i}. [{entry['timestamp']}]\n"
                history_text += f"   You: {entry['user_input'][:50]}...\n"
                history_text += f"   Bot: {entry['response'][:50] if entry['response'] else 'No response'}...\n\n"
            
            return history_text
        
        elif command == "clear":
            session_manager.clear_session(self.session_id)
            self.conversation_history = []
            return "Conversation history cleared."
        
        elif command == "help":
            return self.get_help_text()
        
        elif command.startswith("search "):
            query = command[7:]  # Remove "search " prefix
            debug_info = debug_similarity_search(query, k=3)
            result = f"Search Results for '{query}':\n"
            for i, info in enumerate(debug_info, 1):
                result += f"{i}. {info['source']} (Score: {info['score']:.3f})\n"
                result += f"   {info['content'][:100]}...\n\n"
            return result
        
        return None
    
    def get_help_text(self) -> str:
        """Get help text with available commands"""
        help_text = """
Available Commands:
- 'voice': Switch to voice input
- 'debug on/off': Enable/disable debug mode
- 'stats': Show collection statistics
- 'history': Show recent conversation history
- 'clear': Clear conversation history
- 'search <query>': Search documents directly
- 'help': Show this help message
- 'exit' or 'quit': Exit the chatbot

Voice Commands:
- Say your question normally
- The bot can read responses aloud

Tips:
- Ask specific questions about DGFT policies
- Mention document types (like "export procedures")
- Use follow-up questions for clarification
        """
        return help_text.strip()
    
    def run(self):
        """Main chatbot loop"""
        print("\n" + "="*50)
        print("🤖 Welcome to VAHEI 2.0 - DGFT Assistant")
        print("="*50)
        print("I can help you with DGFT policies, export procedures, and foreign trade regulations.")
        print("Type 'help' for available commands or 'voice' for voice input.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("💬 Your input: ").strip()
                
                # Handle exit commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Thank you for using VAHEI 2.0! Goodbye!")
                    break
                
                # Handle voice input
                if user_input.lower() == "voice":
                    if self.voice_manager.voice_available:
                        user_input = self.voice_manager.listen_to_user()
                        print(f"🎤 You said: {user_input}")
                    else:
                        print("❌ Voice functionality not available. Please type your message.")
                        continue
                
                # Skip empty input
                if not user_input or user_input.strip() == "":
                    continue
                
                # Handle special commands
                special_response = self.handle_special_commands(user_input)
                if special_response:
                    print(f"🔧 {special_response}\n")
                    continue
                
                # Generate bot response
                print("🤔 Thinking...")
                bot_response = self.generate_response(user_input)
                print(f"🤖 Bot: {bot_response}\n")
                
                # Offer text-to-speech
                if self.voice_manager.voice_available:
                    play_audio = input("🔊 Play audio response? (y/n): ").strip().lower()
                    if play_audio == "y":
                        self.voice_manager.speak_text(bot_response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Session ended by user.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print("❌ An unexpected error occurred. Please try again.")

def main():
    """Main function to start the chatbot"""
    try:
        chatbot = ChatbotInterface()
        chatbot.run()
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print("❌ Failed to start VAHEI 2.0. Please check your setup.")

if __name__ == "__main__":
    main()