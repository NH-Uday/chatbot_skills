'use client'

import { useEffect, useState } from 'react'
import axios from 'axios'

// 🔒 Extend window to support MathJax types
declare global {
  interface Window {
    MathJax?: { typeset: () => void }
  }
}

type Message = {
  role: 'user' | 'assistant'
  content: string
}

type BotKey = 'FB1' | 'FB2' | 'FB3'

export default function ChatPage() {
  // --- All hooks must be declared in the same order on every render ---
  const [mounted, setMounted] = useState(false)

  // Chat/UI state
  const [input, setInput] = useState<string>('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Bot selector
  const [bot, setBot] = useState<BotKey>('FB1')

  // NEW: session id (regenerated on Refresh)
  const [sessionId, setSessionId] = useState<string>('') // NEW

  // API base (configurable via .env.local -> NEXT_PUBLIC_API_BASE)
  const apiBaseFromEnv =
    typeof process !== 'undefined'
      ? (process as any).env?.NEXT_PUBLIC_API_BASE
      : undefined
  const apiBase = (apiBaseFromEnv || 'http://localhost:8000').replace(/\/$/, '')

  // Mount gate toggled after first client render
  useEffect(() => setMounted(true), [])

  // NEW: create a fresh session id on first mount
  useEffect(() => {
    if (!sessionId) {
      const id =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`
      setSessionId(id)
    }
  }, [sessionId])

  // Load MathJax once on client
  useEffect(() => {
    if (!window.MathJax) {
      const s = document.createElement('script')
      s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
      s.async = true
      document.head.appendChild(s)
    }
  }, [])

  // Re-typeset after messages change
  useEffect(() => {
    window.MathJax?.typeset?.()
  }, [messages])

  if (!mounted) return null

  const ask = async () => {
    const question = input.trim()
    if (!question || loading) return

    setError(null)
    setLoading(true)

    const userMsg: Message = { role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])
    setInput('')

    try {
      // Use generic endpoint with target selector + session_id (NEW)
      const url = `${apiBase}/chat`
      const res = await axios.post(url, { question, target: bot, session_id: sessionId }) // NEW
      const answer: string = res?.data?.answer ?? ''
      const assistantMsg: Message = {
        role: 'assistant',
        content: answer || '(no answer returned)',
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (e: any) {
      console.error(e)
      setError(e?.message || 'Request failed')
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry — I ran into a problem fetching the answer.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') ask()
  }

  // NEW: Refresh button handler — clears memory on server and starts a new session
  const refreshSession = async () => {
    try {
      await axios.post(`${apiBase}/session/reset`, null, {
        params: { session_id: sessionId },
      })
    } catch (e) {
      // Even if reset fails, we still rotate the client session to avoid stale context
      console.warn('Session reset failed; rotating client session anyway.')
    } finally {
      const newId =
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`
      setSessionId(newId)
      setMessages([]) // optional: start with a clean chat view
      setError(null)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-3xl mx-auto px-4 py-10">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold text-gray-800">Chatbot Assistant</h1>

          {/* Bot selector */}
          <div className="ml-auto flex items-center gap-2">
            <label htmlFor="bot" className="text-sm text-gray-600">model:</label>
            <select
              id="bot"
              value={bot}
              onChange={(e) => setBot(e.target.value as BotKey)}
              className="px-3 py-2 border rounded-lg bg-white text-sm"
              disabled={loading}
              aria-label="Select document bot"
            >
              <option value="FB1">FB1</option>
              <option value="FB2">FB2</option>
              <option value="FB3">FB3</option>
            </select>

            {/* NEW: Refresh (clears server-side session memory) */}
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

        {/* NEW: tiny session badge (optional, helps debug) */}
        <div className="mt-2 text-[11px] text-gray-500 select-all">
          session: <span className="font-mono">{sessionId || '—'}</span>
        </div>

        <div className="mt-6 flex items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={`Ask a question to ${bot}…`}
            className="flex-1 p-3 border border-blue-300 rounded-xl shadow-sm bg-white text-[15px] leading-6 focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            onClick={ask}
            disabled={loading || !input.trim()}
            className="px-4 py-2 rounded-xl bg-blue-600 text-white disabled:opacity-50"
          >
            {loading ? 'I am Thinking…' : 'Ask'}
          </button>
        </div>

        {error && <div className="mt-3 text-sm text-red-600">{error}</div>}

        <div className="mt-8 space-y-6">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl shadow-sm ${
                m.role === 'user' ? 'bg-white border border-gray-200' : 'bg-blue-50 border border-blue-100'
              }`}
            >
              <div className="text-xs uppercase tracking-wide mb-2 text-gray-500">
                {m.role === 'user' ? 'You' : `Assistant (${bot})`}
              </div>
              {/* Render as plain text; MathJax still parses TeX from text nodes */}
              <p className="text-sm whitespace-pre-wrap">{m.content?.trim?.() ?? ''}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
