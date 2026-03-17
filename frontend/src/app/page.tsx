"use client";

import { useState, useRef, useEffect, useCallback } from "react";

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

interface SSEMetadata {
  confidence: string;
  domain: string;
  query_type: string;
  jurisdiction_notes: string;
  session_id: string;
  sources_verified: boolean;
  verification_accuracy: number | null;
  router_confidence: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  laws: ApplicableLaw[];
  precedents: Precedent[];
  procedure: ProcedureStep[];
  metadata: SSEMetadata | null;
  loading: boolean;
  streaming: boolean;
  error: string;
}

// ─── Guided Flow Suggestions ────────────────────────────────────────────────

const SUGGESTIONS = [
  { label: "🚔 Arrest Rights", query: "What are my rights if I am arrested by the police?" },
  { label: "🏠 Tenant Eviction", query: "Can a landlord evict me without notice?" },
  { label: "🛒 Consumer Complaint", query: "How do I file a consumer complaint for a defective product?" },
  { label: "👨‍👩‍👧 Divorce Process", query: "How to file for mutual consent divorce?" },
  { label: "📜 FIR Filing", query: "What is the procedure to file an FIR?" },
  { label: "⚖️ Fundamental Rights", query: "What are the fundamental rights under the Indian Constitution?" },
];

// ─── API Base URL ───────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

// ─── Helper: New Empty Message ──────────────────────────────────────────────

function newAssistantMessage(): Message {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "",
    laws: [],
    precedents: [],
    procedure: [],
    metadata: null,
    loading: true,
    streaming: false,
    error: "",
  };
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
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + "px";
  };

  const toggleCard = (id: string) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // ─── Streaming SSE Query ────────────────────────────────────────────

  const sendQuery = useCallback(
    async (query: string) => {
      if (!query.trim() || isLoading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query.trim(),
        laws: [],
        precedents: [],
        procedure: [],
        metadata: null,
        loading: false,
        streaming: false,
        error: "",
      };

      const assistantMsg = newAssistantMessage();

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setIsLoading(true);

      if (inputRef.current) {
        inputRef.current.style.height = "auto";
      }

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/api/v1/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query.trim(),
            session_id: sessionId,
            language: "en",
          }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // Mark as streaming
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id ? { ...m, loading: false, streaming: true } : m
          )
        );

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);

              switch (event.type) {
                case "text":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, content: m.content + event.content }
                        : m
                    )
                  );
                  break;

                case "applicable_law":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, laws: event.content || [] }
                        : m
                    )
                  );
                  break;

                case "precedents":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, precedents: event.content || [] }
                        : m
                    )
                  );
                  break;

                case "procedure":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, procedure: event.content || [] }
                        : m
                    )
                  );
                  break;

                case "metadata":
                  if (event.content?.session_id && !sessionId) {
                    setSessionId(event.content.session_id);
                  }
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, metadata: event.content }
                        : m
                    )
                  );
                  break;

                case "done":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, streaming: false }
                        : m
                    )
                  );
                  break;

                case "error":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, error: event.content || "An error occurred", streaming: false, loading: false }
                        : m
                    )
                  );
                  break;
              }
            } catch {
              // skip malformed JSON
            }
          }
        }

        // Ensure streaming is marked done
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id ? { ...m, streaming: false, loading: false } : m
          )
        );
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          const errorMsg = err instanceof Error ? err.message : "Something went wrong";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsg.id
                ? { ...m, error: errorMsg, loading: false, streaming: false }
                : m
            )
          );
        }
      } finally {
        setIsLoading(false);
        abortRef.current = null;
      }
    },
    [isLoading, sessionId]
  );

  const handleSubmit = () => sendQuery(input);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleRetry = (query: string) => {
    sendQuery(query);
  };

  const handleNewChat = () => {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    setSessionId(null);
    setIsLoading(false);
    setExpandedCards({});
  };

  // ─── Render ─────────────────────────────────────────────────────────

  const showWelcome = messages.length === 0;

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-navy-900 border-b border-navy-700 px-4 py-3 flex items-center justify-between flex-shrink-0"
        style={{ backgroundColor: "#1a1a3e" }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: "#FF9933" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div>
            <h1 className="text-white font-bold text-lg leading-tight">NyayaMitra</h1>
            <p className="text-slate-400 text-xs">AI Legal Assistant for India</p>
          </div>
        </div>
        <button
          onClick={handleNewChat}
          className="text-slate-400 hover:text-white text-sm px-3 py-1.5 rounded-lg hover:bg-white/10 transition-colors"
        >
          New Chat
        </button>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {showWelcome && (
            <div className="text-center py-16">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl flex items-center justify-center"
                style={{ backgroundColor: "#FF993320" }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FF9933" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800 mb-2">
                Welcome to NyayaMitra
              </h2>
              <p className="text-slate-500 mb-1">
                Your AI-powered legal assistant for Indian law
              </p>
              <p className="text-slate-400 text-sm mb-8">
                Ask any legal question — get answers with real citations
              </p>

              {/* Suggestion Chips */}
              <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => sendQuery(s.query)}
                    className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm text-slate-600 hover:border-indigo-300 hover:text-indigo-600 hover:bg-indigo-50 transition-all shadow-sm"
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message List */}
          {messages.map((msg) => (
            <div key={msg.id} className={`mb-6 ${msg.role === "user" ? "flex justify-end" : ""}`}>
              {msg.role === "user" ? (
                <div className="bg-indigo-600 text-white px-4 py-3 rounded-2xl rounded-br-md max-w-[85%] md:max-w-[70%] shadow-sm">
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                </div>
              ) : (
                <div className="max-w-full">
                  {/* Loading Spinner */}
                  {msg.loading && (
                    <div className="flex items-center gap-2 text-slate-400 py-3">
                      <div className="w-5 h-5 border-2 border-slate-300 border-t-indigo-500 rounded-full animate-spin" />
                      <span className="text-sm">Searching legal database...</span>
                    </div>
                  )}

                  {/* Error State */}
                  {msg.error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-3">
                      <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                        <div>
                          <p className="text-red-700 text-sm">{msg.error}</p>
                          <button
                            onClick={() => {
                              const prevUserMsg = messages[messages.indexOf(msg) - 1];
                              if (prevUserMsg?.role === "user") handleRetry(prevUserMsg.content);
                            }}
                            className="mt-2 text-red-600 text-xs font-medium hover:underline"
                          >
                            Try again
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Answer Text */}
                  {msg.content && (
                    <div className="prose prose-sm prose-slate max-w-none mb-3">
                      <div className="whitespace-pre-wrap text-slate-700 text-sm leading-relaxed">
                        {msg.content}
                        {msg.streaming && (
                          <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-0.5 animate-pulse rounded-sm" />
                        )}
                      </div>
                    </div>
                  )}

                  {/* Verification Badge + Confidence */}
                  {msg.metadata && !msg.streaming && (
                    <div className="flex flex-wrap items-center gap-2 mb-3">
                      {msg.metadata.sources_verified ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-green-50 border border-green-200 rounded-full text-xs text-green-700">
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                          </svg>
                          Citations Verified
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-amber-50 border border-amber-200 rounded-full text-xs text-amber-700">
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                          </svg>
                          Unverified
                        </span>
                      )}
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                        msg.metadata.confidence === "high"
                          ? "bg-green-50 text-green-700 border border-green-200"
                          : msg.metadata.confidence === "medium"
                          ? "bg-amber-50 text-amber-700 border border-amber-200"
                          : "bg-red-50 text-red-700 border border-red-200"
                      }`}>
                        {msg.metadata.confidence.charAt(0).toUpperCase() + msg.metadata.confidence.slice(1)} Confidence
                      </span>
                      {msg.metadata.domain && (
                        <span className="px-2.5 py-1 bg-slate-100 border border-slate-200 rounded-full text-xs text-slate-600">
                          {msg.metadata.domain}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Applicable Law Card */}
                  {msg.laws.length > 0 && !msg.streaming && (
                    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm mb-2">
                      <button
                        onClick={() => toggleCard(`law-${msg.id}`)}
                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 transition-colors"
                      >
                        <span className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                          <span>📖</span> Applicable Law ({msg.laws.length})
                        </span>
                        <svg className={`w-4 h-4 text-slate-400 transition-transform ${expandedCards[`law-${msg.id}`] ? "rotate-180" : ""}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7"/>
                        </svg>
                      </button>
                      {expandedCards[`law-${msg.id}`] && (
                        <div className="px-4 pb-3 space-y-2">
                          {msg.laws.map((law, i) => (
                            <div key={i} className="p-3 bg-slate-50 rounded-lg">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-sm font-medium text-slate-800">
                                  Section {law.section} — {law.act}
                                </span>
                                <span className={`text-xs px-2 py-0.5 rounded-full ${
                                  law.status === "active"
                                    ? "bg-green-100 text-green-700"
                                    : "bg-red-100 text-red-700"
                                }`}>
                                  {law.status}
                                </span>
                              </div>
                              <p className="text-xs text-slate-500 line-clamp-2">{law.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Precedents Card */}
                  {msg.precedents.length > 0 && !msg.streaming && (
                    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm mb-2">
                      <button
                        onClick={() => toggleCard(`prec-${msg.id}`)}
                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 transition-colors"
                      >
                        <span className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                          <span>⚖️</span> Precedents ({msg.precedents.length})
                        </span>
                        <svg className={`w-4 h-4 text-slate-400 transition-transform ${expandedCards[`prec-${msg.id}`] ? "rotate-180" : ""}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7"/>
                        </svg>
                      </button>
                      {expandedCards[`prec-${msg.id}`] && (
                        <div className="px-4 pb-3 space-y-2">
                          {msg.precedents.map((prec, i) => (
                            <div key={i} className="p-3 bg-slate-50 rounded-lg">
                              <p className="text-sm font-medium text-slate-800">
                                {prec.case} ({prec.year})
                                {prec.relevance?.includes("OVERRULED") && (
                                  <span className="ml-2 text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full">OVERRULED</span>
                                )}
                              </p>
                              <p className="text-xs text-slate-500 mt-0.5">
                                {prec.court}{prec.citation ? ` — ${prec.citation}` : ""}
                              </p>
                              {prec.relevance && (
                                <p className="text-xs text-slate-500 mt-1 line-clamp-2">{prec.relevance}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Procedure Card */}
                  {msg.procedure.length > 0 && !msg.streaming && (
                    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm mb-2">
                      <button
                        onClick={() => toggleCard(`proc-${msg.id}`)}
                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 transition-colors"
                      >
                        <span className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                          <span>📋</span> What You Should Do ({msg.procedure.length} steps)
                        </span>
                        <svg className={`w-4 h-4 text-slate-400 transition-transform ${expandedCards[`proc-${msg.id}`] ? "rotate-180" : ""}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7"/>
                        </svg>
                      </button>
                      {expandedCards[`proc-${msg.id}`] && (
                        <div className="px-4 pb-3 space-y-2">
                          {msg.procedure.map((step, i) => (
                            <div key={i} className="flex gap-3 p-3 bg-slate-50 rounded-lg">
                              <div className="w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                                {step.step}
                              </div>
                              <div>
                                <p className="text-sm font-medium text-slate-800">{step.action}</p>
                                {step.details && (
                                  <p className="text-xs text-slate-500 mt-0.5">{step.details}</p>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Disclaimer */}
                  {msg.content && !msg.streaming && !msg.loading && !msg.error && (
                    <div className="mt-3 px-3 py-2 bg-amber-50 border border-amber-100 rounded-lg">
                      <p className="text-xs text-amber-700">
                        ⚖️ This is legal information, not legal advice. For case-specific advice, consult a qualified advocate.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-200 bg-white px-4 py-3 flex-shrink-0">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-2 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a legal question..."
              rows={1}
              disabled={isLoading}
              className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-400"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className="h-11 w-11 rounded-xl flex items-center justify-center transition-colors disabled:opacity-40"
              style={{ backgroundColor: input.trim() && !isLoading ? "#1a1a3e" : "#cbd5e1" }}
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              )}
            </button>
          </div>
          <p className="text-center text-xs text-slate-400 mt-2">
            NyayaMitra provides legal information, not legal advice. Always consult a lawyer for specific cases.
          </p>
        </div>
      </div>
    </div>
  );
}