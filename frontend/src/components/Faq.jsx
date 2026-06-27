import { useState, useEffect } from 'react'
import { getJson } from '../api'

export default function Faq() {
  const [items, setItems] = useState(null)

  useEffect(() => {
    const load = async () => {
      try { setItems((await getJson('/faq')).items) } catch { setItems([]) }
    }
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  if (items === null) return <main><div className="empty">Loading…</div></main>

  return (
    <main>
      <p className="hint">
        Auto-generated from real questions, grouped <em>semantically</em> and
        ranked by frequency. Only questions asked a few times appear; titles are
        LLM-generated.
      </p>
      {items.length === 0 ? (
        <div className="empty">No FAQs yet — questions must be asked a few times first.</div>
      ) : (
        items.map((it, i) => (
          <div key={it.cluster_id} className="faq-item">
            <div className="q">
              <span className="rank">{i + 1}.</span>
              <span>{it.question}</span>
              <span className="count">{it.ask_count} asks</span>
            </div>
            {it.answer && <div className="a">{it.answer}</div>}
            {it.examples?.length > 0 && (
              <div className="ex">
                people also asked: {it.examples.map(e => `“${e}”`).join(' · ')}
              </div>
            )}
          </div>
        ))
      )}
    </main>
  )
}
