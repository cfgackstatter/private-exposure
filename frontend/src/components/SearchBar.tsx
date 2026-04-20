import { useState } from "react"

interface Props {
  onSearch: (q: string) => void
  loading: boolean
}

export function SearchBar({ onSearch, loading }: Props) {
  const [value, setValue] = useState("")

  return (
    <div className="ingest-panel">
      <h2>Search holdings</h2>
      <p className="search-hint">
        Use <code>AND</code> / <code>OR</code> to combine — e.g.{" "}
        <code>space AND launch</code> or <code>uranium OR thorium</code>
      </p>
      <div className="ingest-row" style={{ marginTop: 12 }}>
        <input
          className="input"
          style={{ flex: 1 }}
          placeholder="e.g. space, uranium AND mining, AI AND chips"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === "Enter" && onSearch(value)}
          disabled={loading}
        />
        <button
          className="btn btn-primary"
          onClick={() => onSearch(value)}
          disabled={loading || !value.trim()}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>
    </div>
  )
}