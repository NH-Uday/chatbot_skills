'use client'

import { useEffect, useRef, useState } from 'react'
import axios from 'axios'

// 🔒 Extend window to support MathJax types
declare global {
  interface Window {
    MathJax?: { typeset: () => void }
  }
}

type Message = { role: 'user' | 'assistant'; content: string }
type BotKey = 'FB1' | 'FB2' | 'FB3'

export default function ChatPage() {
  // --- hooks/state ---
  const [mounted, setMounted] = useState(false)
  const [input, setInput] = useState<string>('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [bot, setBot] = useState<BotKey>('FB1')
  const [sessionId, setSessionId] = useState<string>('')

  const endRef = useRef<HTMLDivElement | null>(null)

  // API base
  const apiBaseFromEnv =
    typeof process !== 'undefined' ? (process as any).env?.NEXT_PUBLIC_API_BASE : undefined
  const apiBase = (apiBaseFromEnv || 'http://localhost:8000').replace(/\/$/, '')

  useEffect(() => setMounted(true), [])

  // Create session id on first mount
  useEffect(() => {
    if (!sessionId) {
      const id =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`
      setSessionId(id)
    }
  }, [sessionId])

  // Load MathJax once
  useEffect(() => {
    if (!window.MathJax) {
      const s = document.createElement('script')
      s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
      s.async = true
      document.head.appendChild(s)
    }
  }, [])

  // Re-typeset after message updates
  useEffect(() => {
    window.MathJax?.typeset?.()
  }, [messages])

  // Auto-scroll to bottom of the answers panel
  useEffect(() => {
    const t = setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }, 80)
    return () => clearTimeout(t)
  }, [messages, loading])

  if (!mounted) return null

  const ask = async () => {
    const question = input.trim()
    if (!question || loading) return

    setError(null)
    setLoading(true)
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setInput('')

    try {
      const url = `${apiBase}/chat`
      const res = await axios.post(url, { question, target: bot, session_id: sessionId })
      const answer: string = res?.data?.answer ?? ''
      setMessages(prev => [...prev, { role: 'assistant', content: answer || '(no answer returned)' }])
    } catch (e: any) {
      console.error(e)
      setError(e?.message || 'Request failed')
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Entschuldigung – beim Abrufen der Antwort ist ein Problem aufgetreten.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') ask()
  }

  const refreshSession = async () => {
    try {
      await axios.post(`${apiBase}/session/reset`, null, { params: { session_id: sessionId } })
    } catch {
      console.warn('Session reset failed; rotating client session anyway.')
    } finally {
      const newId =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`
      setSessionId(newId)
      setMessages([])
      setError(null)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="mx-auto max-w-3xl px-4 py-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold text-gray-800">SKiLLs Chatbot Assistant</h1>

          <div className="ml-auto flex items-center gap-2">
            <label htmlFor="bot" className="text-sm text-gray-600">model:</label>
            <select
              id="bot"
              value={bot}
              onChange={(e) => setBot(e.target.value as BotKey)}
              className="px-3 py-2 border rounded-lg bg-white text-sm"
              disabled={loading}
            >
              <option value="FB1">FB1</option>
              <option value="FB2">FB2</option>
              <option value="FB3">FB3</option>
            </select>

            <button
              onClick={refreshSession}
              disabled={loading}
              className="ml-2 px-3 py-2 rounded-lg border border-gray-300 bg-white text-sm hover:bg-gray-50 disabled:opacity-50"
              title="Start a new chat session"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Session badge */}
        <div className="mt-2 text-[11px] text-gray-500 select-all">
          session: <span className="font-mono">{sessionId || '—'}</span>
        </div>

        {/* ===== PAGE BODY: two rows (answers / input) ===== */}
        <div className="mt-4 h-[calc(100vh-170px)] grid grid-rows-[1fr_auto] gap-2">
          {/* Top: Answers panel (scrollable) */}
          <div className="overflow-y-auto rounded-xl border border-gray-200 bg-white p-4">
            {error && <div className="mb-4 text-sm text-red-600">{error}</div>}

            {messages.length === 0 && (
              <div className="text-sm text-gray-500">
                Fragen Sie mich alles aus dem Vorlesungsunterlagen.
              </div>
            )}

            <div className="space-y-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl shadow-sm ${
                    m.role === 'user'
                      ? 'bg-white border border-gray-200'
                      : 'bg-blue-50 border border-blue-100'
                  }`}
                >
                  <div className="text-xs uppercase tracking-wide mb-2 text-gray-500">
                    {m.role === 'user' ? 'You' : `Assistant (${bot})`}
                  </div>
                  <p className="text-sm whitespace-pre-wrap">{m.content?.trim?.() ?? ''}</p>
                </div>
              ))}
              {/* scroll target */}
              <div ref={endRef} />
            </div>
          </div>

          {/* Bottom: Input panel (stays at bottom, not overlay) */}
          <div className="rounded-xl border border-blue-200 bg-white p-3 shadow-sm">
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={`Eine Frage stellen an ${bot}…`}
                className="flex-1 p-3 border border-blue-200 rounded-lg bg-white text-[15px] leading-6 focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
              <button
                onClick={ask}
                disabled={loading || !input.trim()}
                className="px-5 py-3 rounded-lg bg-blue-600 text-white text-sm disabled:opacity-50"
              >
                {loading ? 'Denken...' : 'Fragen'}
              </button>
            </div>
          </div>
        </div>
        {/* ===== END BODY ===== */}
      </div>
    </div>
  )
}
