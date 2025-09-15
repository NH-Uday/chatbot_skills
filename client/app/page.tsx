'use client'

import { useState, useEffect } from "react";
import axios from "axios";

// 🔒 Extend window to support MathJax types
declare global {
  interface Window {
    MathJax?: {
      typeset: () => void;
    };
  }
}

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function ChatPage() {
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  // ✅ Re-render MathJax equations when messages change
  useEffect(() => {
    if (window.MathJax && typeof window.MathJax.typeset === "function") {
      window.MathJax.typeset();
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/chat", {
        question: input,
      });

      const reply: Message = {
        role: "assistant",
        content: response.data.answer,
      };

      setMessages([...newMessages, reply]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-blue-50 p-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-blue-800 mb-6">
          Chatbot companion for Lab
        </h1>

        <div className="space-y-6">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`p-4 rounded-xl shadow-md ${
                msg.role === "user"
                  ? "bg-white text-right"
                  : "bg-blue-100 text-left"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="space-y-3">
                  {msg.content
                    .split(/\n(?=🧠|⚖️|🚀)/)
                    .map((section: string, idx: number) => (
                      <div key={idx} className="bg-white p-3 rounded-lg">
                        {/* ✅ Renders LaTeX as HTML */}
                        <p
                          className="text-sm whitespace-pre-wrap"
                          dangerouslySetInnerHTML={{ __html: section.trim() }}
                        />
                      </div>
                    ))}
                </div>
              ) : (
                <p className="text-blue-800 font-semibold">{msg.content}</p>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask a question..."
            className="flex-1 p-3 border border-blue-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
