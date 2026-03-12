"use client";

import { useState, useRef, useEffect } from "react";
import { Scale, Send, Loader2, AlertCircle, BookOpen, ChevronDown } from "lucide-react";

// ─── Types ──────────────────────────────────────────────────────────────────

interface ApplicableLaw {
  act: string;
  section: string;
  text: string;
  status: string;
}

interface Precedent {
  case: string;
  year: number;
  court: string;
  citation: string;
  relevance: string;
}

interface ProcedureStep {
  step: number;
  action: string;
  details: string;
  forms: string[];
  court: string | null;
}

interface QueryResponse {
  response_id: string;
  answer: string;
  applicable_law: ApplicableLaw[];
  precedents: Precedent[];
  procedure: ProcedureStep[];
  jurisdiction_notes: string | null;
  confidence: string;
  disclaimer: string;
  sources_verified: boolean;
  language: string;
  session_id: string;
  created_at: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  loading?: boolean;
  error?: string;
}

// ─── Main Page ──────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + "px";
  };

  // Toggle citation card expand/collapse
  const toggleCard = (id: string) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Send query to backend
  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      loading: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsLoading(true);

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    try {
      const res = await fetch("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          language: "en",
          detail_level: "detailed",
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data: QueryResponse = await res.json();

      // Save session ID for multi-turn conversation
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? { ...msg, content: data.answer, response: data, loading: false }
            : msg
        )
      );
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Something went wrong";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? { ...msg, content: "", loading: false, error: errorMsg }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key (Shift+Enter for new line)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* ─── Header ────────────────────────────────────────────────── */}
      <header className="bg-navy-900 text-white px-4 py-3 shadow-lg flex-shrink-0">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Scale className="w-7 h-7 text-saffron-500" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">NyayaMitra</h1>
              <p className="text-xs text-navy-300">AI Legal Assistant for India</p>
            </div>
          </div>
          <div className="text-xs text-navy-400 hidden sm:block">
            Legal information, not legal advice
          </div>
        </div>
      </header>

      {/* ─── Chat Area ─────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto chat-scroll">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {/* Welcome message */}
          {messages.length === 0 && (
            <div className="text-center py-20">
              <Scale className="w-16 h-16 text-navy-300 mx-auto mb-6" />
              <h2 className="text-2xl font-bold text-navy-900 mb-3">
                Welcome to NyayaMitra
              </h2>
              <p className="text-gray-600 max-w-lg mx-auto mb-8">
                Ask any legal question about Indian law. Get accurate answers
                with specific section citations, relevant case law, and
                step-by-step procedural guidance.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Can police arrest me without a warrant?",
                  "What are my rights as a tenant?",
                  "How to file a consumer complaint?",
                  "What is the process for filing an RTI?",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => {
                      setInput(suggestion);
                      inputRef.current?.focus();
                    }}
                    className="px-4 py-2 bg-white border border-gray-200 rounded-full text-sm text-gray-700 hover:bg-navy-50 hover:border-navy-300 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg) => (
            <div key={msg.id} className="mb-6">
              {/* User message */}
              {msg.role === "user" && (
                <div className="flex justify-end">
                  <div className="bg-navy-900 text-white px-4 py-3 rounded-2xl rounded-br-md max-w-[80%]">
                    {msg.content}
                  </div>
                </div>
              )}

              {/* Assistant message */}
              {msg.role === "assistant" && (
                <div className="flex justify-start">
                  <div className="max-w-[90%] w-full">
                    {/* Loading state */}
                    {msg.loading && (
                      <div className="flex items-center gap-2 text-gray-500 px-4 py-3">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm">Researching legal provisions...</span>
                      </div>
                    )}

                    {/* Error state */}
                    {msg.error && (
                      <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 flex items-start gap-2">
                        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-sm font-medium text-red-800">Failed to get response</p>
                          <p className="text-xs text-red-600 mt-1">{msg.error}</p>
                        </div>
                      </div>
                    )}

                    {/* Response */}
                    {msg.response && (
                      <div className="space-y-3">
                        {/* Answer */}
                        <div className="bg-white border border-gray-200 rounded-xl px-5 py-4 shadow-sm">
                          <p className="text-gray-800 leading-relaxed">{msg.response.answer}</p>

                          {/* Confidence badge */}
                          <div className="mt-3 flex items-center gap-2">
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                                msg.response.confidence === "high"
                                  ? "bg-green-100 text-green-700"
                                  : msg.response.confidence === "medium"
                                  ? "bg-yellow-100 text-yellow-700"
                                  : "bg-red-100 text-red-700"
                              }`}
                            >
                              {msg.response.confidence} confidence
                            </span>
                            {msg.response.sources_verified && (
                              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
                                citations verified
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Applicable Law */}
                        {msg.response.applicable_law.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                            <button
                              onClick={() => toggleCard(`law-${msg.id}`)}
                              className="w-full px-5 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                            >
                              <div className="flex items-center gap-2">
                                <BookOpen className="w-4 h-4 text-navy-600" />
                                <span className="text-sm font-semibold text-navy-800">
                                  Applicable Law ({msg.response.applicable_law.length})
                                </span>
                              </div>
                              <ChevronDown
                                className={`w-4 h-4 text-gray-400 transition-transform ${
                                  expandedCards[`law-${msg.id}`] ? "rotate-180" : ""
                                }`}
                              />
                            </button>
                            {expandedCards[`law-${msg.id}`] && (
                              <div className="px-5 pb-4 space-y-3">
                                {msg.response.applicable_law.map((law, i) => (
                                  <div key={i} className="bg-navy-50 rounded-lg p-3">
                                    <p className="text-sm font-semibold text-navy-900">
                                      {law.act} — Section {law.section}
                                    </p>
                                    <p className="text-sm text-gray-700 mt-1">{law.text}</p>
                                    <span
                                      className={`inline-block mt-2 text-xs px-2 py-0.5 rounded-full ${
                                        law.status === "active"
                                          ? "bg-green-100 text-green-700"
                                          : "bg-red-100 text-red-700"
                                      }`}
                                    >
                                      {law.status}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Precedents */}
                        {msg.response.precedents.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                            <button
                              onClick={() => toggleCard(`prec-${msg.id}`)}
                              className="w-full px-5 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                            >
                              <div className="flex items-center gap-2">
                                <Scale className="w-4 h-4 text-navy-600" />
                                <span className="text-sm font-semibold text-navy-800">
                                  Precedents ({msg.response.precedents.length})
                                </span>
                              </div>
                              <ChevronDown
                                className={`w-4 h-4 text-gray-400 transition-transform ${
                                  expandedCards[`prec-${msg.id}`] ? "rotate-180" : ""
                                }`}
                              />
                            </button>
                            {expandedCards[`prec-${msg.id}`] && (
                              <div className="px-5 pb-4 space-y-3">
                                {msg.response.precedents.map((prec, i) => (
                                  <div key={i} className="bg-navy-50 rounded-lg p-3">
                                    <p className="text-sm font-semibold text-navy-900">
                                      {prec.case} ({prec.year})
                                    </p>
                                    <p className="text-xs text-gray-500 mt-0.5">
                                      {prec.court} — {prec.citation}
                                    </p>
                                    <p className="text-sm text-gray-700 mt-1">{prec.relevance}</p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Procedure */}
                        {msg.response.procedure.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                            <button
                              onClick={() => toggleCard(`proc-${msg.id}`)}
                              className="w-full px-5 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                            >
                              <span className="text-sm font-semibold text-navy-800">
                                Procedure ({msg.response.procedure.length} steps)
                              </span>
                              <ChevronDown
                                className={`w-4 h-4 text-gray-400 transition-transform ${
                                  expandedCards[`proc-${msg.id}`] ? "rotate-180" : ""
                                }`}
                              />
                            </button>
                            {expandedCards[`proc-${msg.id}`] && (
                              <div className="px-5 pb-4 space-y-2">
                                {msg.response.procedure.map((step) => (
                                  <div key={step.step} className="flex gap-3">
                                    <div className="flex-shrink-0 w-6 h-6 bg-navy-900 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                      {step.step}
                                    </div>
                                    <div>
                                      <p className="text-sm font-medium text-gray-900">{step.action}</p>
                                      <p className="text-sm text-gray-600 mt-0.5">{step.details}</p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Jurisdiction notes */}
                        {msg.response.jurisdiction_notes && (
                          <p className="text-xs text-gray-500 px-2">
                            {msg.response.jurisdiction_notes}
                          </p>
                        )}

                        {/* Disclaimer */}
                        <p className="text-xs text-gray-400 italic px-2">
                          {msg.response.disclaimer}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* ─── Input Area ────────────────────────────────────────────── */}
      <footer className="bg-white border-t border-gray-200 px-4 py-3 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a legal question... (e.g., What are my rights if arrested?)"
              rows={1}
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-navy-500 focus:border-transparent placeholder:text-gray-400"
              disabled={isLoading}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className="bg-navy-900 text-white p-3 rounded-xl hover:bg-navy-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            NyayaMitra provides legal information, not legal advice. Always consult a qualified advocate for case-specific guidance.
          </p>
        </div>
      </footer>
    </div>
  );
}