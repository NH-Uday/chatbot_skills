'use client'

import { useEffect, useMemo, useState } from 'react'

const ADMIN_PASS = '030697'

export default function AdminLogsPage() {
  const [mounted, setMounted] = useState(false)
  const [authenticated, setAuthenticated] = useState(false)
  const [enteredKey, setEnteredKey] = useState('')

  const [season, setSeason] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<any[] | null>(null)

  // NEW: delete flow UI
  const [deleting, setDeleting] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  // API base
  const apiBaseFromEnv =
    typeof process !== 'undefined'
      ? (process as any).env?.NEXT_PUBLIC_API_BASE
      : undefined
  const apiBase = (apiBaseFromEnv || 'http://localhost:8000').replace(/\/$/, '')

  const viewUrl = useMemo(
    () => `${apiBase}/admin/logs${season ? `?season=${encodeURIComponent(season)}` : ''}`,
    [apiBase, season]
  )
  const downloadUrl = useMemo(
    () => `${apiBase}/admin/logs/export${season ? `?season=${encodeURIComponent(season)}` : ''}`,
    [apiBase, season]
  )

  useEffect(() => setMounted(true), [])
  if (!mounted) return null

  // If not authenticated yet
  if (!authenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-6 rounded-xl shadow-md w-80 text-center">
          <h1 className="text-lg font-semibold text-gray-800 mb-4">Admin Access</h1>
          <input
            type="password"
            inputMode="numeric"
            maxLength={6}
            value={enteredKey}
            onChange={(e) => setEnteredKey(e.target.value)}
            placeholder="Enter 6-digit passkey"
            className="w-full p-2 border rounded mb-3 text-center tracking-widest font-mono"
          />
          <button
            onClick={() => {
              if (enteredKey === ADMIN_PASS) {
                setAuthenticated(true)
              } else {
                alert('Invalid passkey.')
              }
            }}
            className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Unlock
          </button>
        </div>
      </div>
    )
  }

  const search = async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await fetch(viewUrl, { cache: 'no-store' })
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = await res.json()
      setResults(Array.isArray(data?.logs) ? data.logs : [])
    } catch (e: any) {
      setError(e?.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  // NEW: delete action (season-only or ALL)
  const deleteLogs = async () => {
    const target = season || 'ALL'
    const ok = window.confirm(
      season
        ? `Delete logs for season “${season}”? This cannot be undone.`
        : `Delete ALL logs? This cannot be undone. Type OK to proceed.`
    )
    if (!ok) return

    try {
      setDeleting(true)
      setToast(null)
      setError(null)

      const url = season
        ? `${apiBase}/admin/logs?season=${encodeURIComponent(season)}`
        : `${apiBase}/admin/logs?confirm=true`

      const res = await fetch(url, { method: 'DELETE' })
      const data = await res.json()

      if (!res.ok || data?.ok === false) {
        throw new Error(data?.error || `Delete failed (${res.status})`)
      }

      setToast(
        data?.deleted === -1
          ? `Deleted logs for ${data?.season || target}.`
          : `Deleted ${data?.deleted ?? 0} log(s) for ${data?.season || target}.`
      )

      // Clear current view results since they may no longer exist
      setResults(null)
    } catch (e: any) {
      setToast(e?.message || 'Delete failed')
    } finally {
      setDeleting(false)
      // auto-hide toast
      setTimeout(() => setToast(null), 4000)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold text-gray-800">Admin • Chat Logs</h1>
          <button
            onClick={() => setAuthenticated(false)}
            className="ml-auto text-sm text-gray-500 hover:text-gray-700"
          >
            Lock
          </button>
        </div>

        <div className="mt-6 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="season" className="text-sm text-gray-600">Season:</label>
            <input
              id="season"
              type="text"
              placeholder="e.g. 2025-Q4"
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              className="px-3 py-2 border rounded-lg bg-white text-sm"
            />
            <a
              href={viewUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-2 rounded-lg border border-gray-300 bg-white text-sm hover:bg-gray-50"
            >
              View JSON
            </a>
            <a
              href={downloadUrl}
              className="px-3 py-2 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-700"
            >
              Download JSON
            </a>
            <button
              onClick={search}
              disabled={loading}
              className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm hover:bg-gray-800 disabled:opacity-50"
            >
              {loading ? 'Searching…' : 'Search'}
            </button>

            {/* NEW: Delete button */}
            <button
              onClick={deleteLogs}
              disabled={deleting}
              className="px-3 py-2 rounded-lg bg-red-600 text-white text-sm hover:bg-red-700 disabled:opacity-50"
              title={season ? `Delete ${season}` : 'Delete ALL logs'}
            >
              {deleting ? 'Deleting…' : (season ? `Delete ${season}` : 'Delete ALL')}
            </button>
          </div>

          {/* NEW: toast */}
          {toast && (
            <div className="mt-3 inline-block rounded-md bg-gray-900 text-white text-sm px-3 py-2">
              {toast}
            </div>
          )}

          <p className="mt-3 text-xs text-gray-500">
            Backend: <code>{apiBase}</code> — endpoints: <code>/admin/logs</code> (GET, DELETE), <code>/admin/logs/export</code> (GET).
          </p>
        </div>

        {error && <div className="text-sm text-red-600 mt-4">{error}</div>}

        {Array.isArray(results) && (
          <div className="mt-4">
            <div className="text-sm text-gray-700 mb-2">
              Found <span className="font-semibold">{results.length}</span> entr{results.length === 1 ? 'y' : 'ies'}
            </div>
            <pre className="text-xs bg-gray-50 border border-gray-200 rounded-lg p-3 overflow-auto max-h-[480px]">
{JSON.stringify(results, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
