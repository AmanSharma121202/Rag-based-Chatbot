
import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Minimize2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatInterfaceProps {
  onSendMessage?: (message: string) => Promise<{ response: string; followups: string[] }>;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onSendMessage }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Namaste! I will be your virtual assistant to help you with your queries related to Foreign Trade Policy of India and DGFT.",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const [followupQuestions, setFollowupQuestions] = useState<string[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let botResponse = "Thank you for your message. I'm here to help with Foreign Trade Policy queries.";
      let followups: string[] = [];
      
      if (onSendMessage) {
        const result = await onSendMessage(inputValue);
        botResponse = result.response;
        followups = result.followups || [];
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: botResponse,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
      setFollowupQuestions(followups);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm having trouble connecting right now. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      setFollowupQuestions([]);
    }

    setIsLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };


  const handleUserQuery = (query: string) => {
    setInputValue(query);
    setFollowupQuestions([]); // Clear current followups when user selects one
    // Automatically send the query
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };


  return (
    <div className={`fixed bottom-4 right-4 z-50 transition-all duration-300 ${
      isMinimized ? 'w-16 h-16' : 'w-96 h-[600px]'
    }`}>
      {isMinimized ? (
        <Button
          onClick={() => setIsMinimized(false)}
          className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 shadow-lg"
        >
          <Bot className="w-6 h-6 text-white" />
        </Button>
      ) : (
        <div className="bg-white rounded-lg shadow-2xl border border-gray-200 flex flex-col h-full">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white p-4 rounded-t-lg flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                <Bot className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-lg">VAHEI</h3>
                <p className="text-xs text-purple-100">Virtual Assistant To Help Exporters and Importers</p>
              </div>
            </div>
            <div className="flex space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMinimized(true)}
                className="text-white hover:bg-white hover:bg-opacity-20 p-1 h-8 w-8"
              >
                <Minimize2 className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex items-start space-x-2 max-w-[80%] ${
                    message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.sender === 'user' 
                        ? 'bg-gray-300' 
                        : 'bg-gradient-to-r from-purple-600 to-purple-700'
                    }`}>
                      {message.sender === 'user' ? (
                        <User className="w-4 h-4 text-gray-600" />
                      ) : (
                        <Bot className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div className={`rounded-lg p-3 ${
                      message.sender === 'user'
                        ? 'bg-gray-100 text-gray-800'
                        : 'bg-gradient-to-r from-purple-600 to-purple-700 text-white'
                    }`}>
                      {message.sender === 'bot' ? (
                        <div className="text-sm prose prose-invert max-w-none">
                          <ReactMarkdown
                            components={{
                              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                              ul: ({ children }) => <ul className="mb-2 last:mb-0 list-disc list-inside">{children}</ul>,
                              ol: ({ children }) => <ol className="mb-2 last:mb-0 list-decimal list-inside">{children}</ol>,
                              li: ({ children }) => <li className="mb-1">{children}</li>,
                              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                              em: ({ children }) => <em className="italic">{children}</em>,
                              code: ({ children }) => <code className="bg-purple-800 px-1 py-0.5 rounded text-sm">{children}</code>,
                              pre: ({ children }) => <pre className="bg-purple-800 p-2 rounded text-sm overflow-x-auto">{children}</pre>,
                              h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                              h2: ({ children }) => <h2 className="text-base font-semibold mb-2">{children}</h2>,
                              h3: ({ children }) => <h3 className="text-sm font-medium mb-1">{children}</h3>,
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="text-sm">{message.content}</p>
                      )}
                      <p className={`text-xs mt-1 ${
                        message.sender === 'user' ? 'text-gray-500' : 'text-purple-100'
                      }`}>
                        {message.timestamp.toLocaleTimeString([], { 
                          hour: '2-digit', 
                          minute: '2-digit' 
                        })}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-purple-700 flex items-center justify-center">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg p-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <Input
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  // Clear followups when user starts typing a new message
                  if (e.target.value !== inputValue && followupQuestions.length > 0) {
                    setFollowupQuestions([]);
                  }
                }}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1"
                disabled={isLoading}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
            

            {/* Followup Questions */}
            {followupQuestions.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-2">Quick questions:</p>
                <div className="flex flex-wrap gap-2">
                  {followupQuestions.slice(0, 3).map((q, index) => (
                    <Button
                      key={index}
                      onClick={() => handleUserQuery(q)}
                      variant="outline"
                      size="sm"
                      className="text-xs px-2 py-1 h-auto bg-white hover:bg-purple-50 hover:border-purple-300 border-gray-300 text-gray-700 hover:text-purple-700 transition-colors duration-200 max-w-full text-left whitespace-normal"
                      disabled={isLoading}
                      title={q} // Show full text on hover
                    >
                      <div className="prose prose-sm max-w-none text-inherit">
                        <ReactMarkdown
                          components={{
                            p: ({ children }) => <span>{children}</span>,
                            strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                            em: ({ children }) => <em className="italic">{children}</em>,
                            code: ({ children }) => <code className="bg-gray-200 px-1 py-0.5 rounded text-xs">{children}</code>,
                          }}
                        >
                          {q.length > 50 ? `${q.substring(0, 47)}...` : q}
                        </ReactMarkdown>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>
            )}
            
            
            {/* Footer */}
            <div className="mt-3 text-center">
              <p className="text-xs text-gray-400 mt-2 leading-tight">
                *Responses provided by chatbot are informative in nature. No legal claims can be made on the basis of chatbot responses. Users should refer policy documents for legal position.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;
