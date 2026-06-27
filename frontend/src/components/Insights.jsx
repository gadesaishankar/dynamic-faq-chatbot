import { useState, useEffect } from 'react'
import { getJson, postJson } from '../api'

function Metric({ label, value }) {
  return <div className="metric"><div className="mv">{value}</div><div className="ml">{label}</div></div>
}

function pct(x) { return x == null ? '—' : Math.round(x * 100) + '%' }

export default function Insights() {
  const [analytics, setAnalytics] = useState(null)
  const [categories, setCategories] = useState([])
  const [gaps, setGaps] = useState([])
  const [kbName, setKbName] = useState('')
  const [kbText, setKbText] = useState('')
  const [kbStatus, setKbStatus] = useState('')

  useEffect(() => {
    const load = async () => {
      try { setAnalytics(await getJson('/analytics')) } catch {}
      try { setCategories((await getJson('/categories')).categories || []) } catch {}
      try { setGaps((await getJson('/admin/content-gaps')).gaps || []) } catch {}
    }
    load()
  }, [])

  const addKb = async (e) => {
    e.preventDefault()
    if (!kbName.trim() || !kbText.trim()) { setKbStatus('filename and text required'); return }
    setKbStatus('adding…')
    try {
      const d = await postJson('/admin/kb', { filename: kbName.trim(), text: kbText.trim() })
      setKbStatus(`✓ indexed ${d.chunks_indexed} chunks`)
      setKbName(''); setKbText('')
    } catch { setKbStatus('failed') }
  }

  return (
    <main>
      <p className="hint">
        Product view: usage metrics, the questions people ask, and the{' '}
        <strong>content gaps</strong> — asked a lot but answered poorly — so you
        know what to document next.
      </p>

      {analytics && (
        <div className="metrics">
          <Metric label="questions" value={analytics.total_questions} />
          <Metric label="topics" value={analytics.total_clusters} />
          <Metric label="helpful rate" value={pct(analytics.helpful_rate)} />
          <Metric label="cache hits" value={pct(analytics.cache_hit_rate)} />
          <Metric label="low-confidence" value={pct(analytics.low_confidence_rate)} />
          <Metric label="feedback" value={analytics.feedback_count} />
        </div>
      )}

      <h3>Most asked by category</h3>
      {categories.length === 0 ? (
        <div className="empty">No questions yet.</div>
      ) : (
        categories.map(cat => (
          <div key={cat.category} className="cat">
            <div className="cat-head">
              {cat.category} <span className="count">{cat.total_asks} asks</span>
            </div>
            {cat.questions.map(q => (
              <div key={q.cluster_id} className="row">
                <span>{q.question}</span>
                <span className="count">{q.ask_count}</span>
              </div>
            ))}
          </div>
        ))
      )}

      <h3>Content gaps — write these next</h3>
      {gaps.length === 0 ? (
        <div className="empty">No gaps detected. 🎉</div>
      ) : (
        gaps.map(g => (
          <div key={g.cluster_id} className="gap">
            <div className="q">
              <span>{g.question}</span>
              <span className="count">{g.ask_count} asks</span>
            </div>
            <div className="why">
              {g.reason} · relevance {g.avg_relevance}
              {g.helpful_rate != null && ` · helpful ${Math.round(g.helpful_rate * 100)}%`}
            </div>
          </div>
        ))
      )}

      <h3>Add knowledge</h3>
      <form className="kb-form" onSubmit={addKb}>
        <input placeholder="filename, e.g. hostel.md" value={kbName} onChange={e => setKbName(e.target.value)} />
        <textarea rows={4} placeholder="Paste the answer/content to add to the knowledge base…" value={kbText} onChange={e => setKbText(e.target.value)} />
        <button type="submit">Add &amp; re-ingest</button>
        {kbStatus && <span className="kb-status">{kbStatus}</span>}
      </form>

      {analytics && (
        <div className="cols">
          <div>
            <h3>Top questions</h3>
            {analytics.top_questions.length === 0 ? <div className="empty">none yet</div> :
              analytics.top_questions.map((q, i) => (
                <div key={i} className="row"><span>{q.question}</span><span className="count">{q.ask_count}</span></div>
              ))}
          </div>
          <div>
            <h3>Unanswered (KB gaps)</h3>
            {analytics.unanswered.length === 0 ? <div className="empty">none yet</div> :
              analytics.unanswered.map((q, i) => (
                <div key={i} className="row"><span>{q.question}</span><span className="count">{q.ask_count}</span></div>
              ))}
          </div>
        </div>
      )}
    </main>
  )
}
