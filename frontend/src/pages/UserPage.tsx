import { useState, useCallback } from "react"
import { Link } from "react-router-dom"
import { SearchBar } from "../components/SearchBar"
import { SearchResults } from "../components/SearchResults"
import { api } from "../api/client"
import type { FundSearchResult } from "../types"

export function UserPage() {
  const [results, setResults] = useState<FundSearchResult[] | null>(null)
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = useCallback(async (q: string) => {
    setQuery(q)
    if (!q.trim()) { setResults(null); return }
    setResults(null)        // ← clear old results immediately
    setLoading(true)
    setError(null)
    try {
      setResults(await api.search(q))
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed")
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div>
      <header className="page-header">
        <div className="logo">
          {/* same SVG as AdminPage */}
          <span>Private Exposure</span>
        </div>
        <Link to="/admin" className="admin-badge">Admin ↗</Link>
      </header>
      <main className="page-main">
        <SearchBar onSearch={handleSearch} loading={loading} />
        {error && <p className="status-msg status-err">{error}</p>}
        {results !== null && (
          <SearchResults results={results} query={query} />
        )}
      </main>
    </div>
  )
}