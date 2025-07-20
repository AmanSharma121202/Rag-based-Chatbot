import React, { useState } from "react";
import ChatInterface from "@/components/ChatInterface";
import { Bot, MessageSquare, Shield, Zap } from "lucide-react";

const Index = () => {
  const [showChat, setShowChat] = useState(false); // 👈 control visibility

  const API_ENDPOINT = "http://127.0.0.1:8000/chat";
  const handleSendMessage = async (message: string): Promise<{ response: string; followups: string[] }> => {
    try {
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: message }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      return {
        response: data.response || "No response from backend.",
        followups: data.followups || []
      };
    } catch (err) {
      console.error("Backend error:", err);
      return {
        response: "Failed to connect to backend. Please make sure the server is running.",
        followups: []
      };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-purple-50">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-600 to-purple-700 rounded-full mb-6">
            <Bot className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 to-purple-800 bg-clip-text text-transparent mb-4">
            VAHEI
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-2">
            Virtual Assistant To Help Exporters and Importers
          </p>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Your intelligent companion for navigating Foreign Trade Policy of
            India and DGFT regulations
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="text-center p-6 rounded-lg bg-white shadow-sm border border-purple-100">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Smart Conversations</h3>
            <p className="text-gray-600">
              Engage in natural conversations about trade policies and regulations
            </p>
          </div>

          <div className="text-center p-6 rounded-lg bg-white shadow-sm border border-purple-100">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Shield className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Reliable Information</h3>
            <p className="text-gray-600">
              Get accurate information based on official DGFT guidelines
            </p>
          </div>

          <div className="text-center p-6 rounded-lg bg-white shadow-sm border border-purple-100">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Zap className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Instant Responses</h3>
            <p className="text-gray-600">
              Get quick answers to your export-import queries 24/7
            </p>
          </div>
        </div>
      </div>

      {/* Chat Icon - opens chat interface on click */}
      {!showChat && (
        <button
          onClick={() => setShowChat(true)}
          className="fixed bottom-6 right-6 p-4 rounded-full bg-purple-600 hover:bg-purple-700 text-white shadow-lg z-50"
        >
          <MessageSquare className="w-6 h-6" />
        </button>
      )}

      {/* Chat Interface - renders only after icon is clicked */}
      {showChat && (
        <ChatInterface onSendMessage={handleSendMessage} />
      )}
    </div>
  );
};

export default Index;
