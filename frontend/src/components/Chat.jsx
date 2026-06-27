import { useState, useRef, useEffect, useCallback } from 'react'
import { postJson, getJson, streamChat } from '../api'

function Feedback({ logId }) {
  const [voted, setVoted] = useState(null)
  const send = async (v) => {
    setVoted(v)
    try { await postJson('/feedback', { log_id: logId, vote: v }) } catch {}
  }
  return (
    <div className="fb">
      Helpful?{' '}
      <button className={`fb-btn ${voted === 1 ? 'voted' : ''}`} disabled={voted !== null} onClick={() => send(1)}>👍</button>
      <button className={`fb-btn ${voted === -1 ? 'voted' : ''}`} disabled={voted !== null} onClick={() => send(-1)}>👎</button>
    </div>
  )
}

function BotMessage({ text, meta }) {
  if (!meta) return <div className="msg bot">{text || '…'}</div>
  const sources = meta.citations ? [...new Set(meta.citations.map(c => c.source))] : []
  return (
    <div className="msg bot">
      {text}
      {sources.length > 0 && <div className="cites">Sources: {sources.join(', ')}</div>}
      <div className="tags">
        {meta.cache_hit && <span className="badge cache">⚡ cached</span>}
        {meta.confidence === 'low' && <span className="badge warn">low confidence</span>}
        <span className="badge">{meta.new_cluster ? '🆕 new topic' : `↩ topic #${meta.cluster_id}`}</span>
      </div>
      <Feedback logId={meta.log_id} />
    </div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const [history, setHistory] = useState([])
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const scroll = () => bottomRef.current?.scrollIntoView({ behavior: 'smooth' })

  const loadSuggestions = useCallback(async () => {
    try {
      const d = await getJson('/faq?top_n=5')
      setSuggestions(d.items || [])
    } catch { setSuggestions([]) }
  }, [])

  useEffect(() => { loadSuggestions() }, [loadSuggestions])
  useEffect(scroll, [messages])

  const ask = async (question) => {
    if (!question.trim() || sending) return
    const q = question.trim()
    setInput('')
    setSending(true)
    setSuggestions([])

    const userMsg = { role: 'user', text: q }
    const botIdx = messages.length + 1
    setMessages(prev => [...prev, userMsg, { role: 'bot', text: '', meta: null }])

    try {
      let fullText = ''
      await streamChat(q, history,
        (token) => {
          fullText += token
          setMessages(prev => {
            const copy = [...prev]
            copy[botIdx] = { ...copy[botIdx], text: fullText }
            return copy
          })
        },
        (meta) => {
          setMessages(prev => {
            const copy = [...prev]
            copy[botIdx] = { ...copy[botIdx], meta }
            return copy
          })
          setHistory(prev => [...prev, { role: 'user', content: q }, { role: 'assistant', content: fullText }])
        }
      )
    } catch {
      try {
        const d = await postJson('/chat', { question: q, history })
        setMessages(prev => {
          const copy = [...prev]
          copy[botIdx] = { role: 'bot', text: d.answer, meta: d }
          return copy
        })
        setHistory(prev => [...prev, { role: 'user', content: q }, { role: 'assistant', content: d.answer }])
      } catch {
        setMessages(prev => {
          const copy = [...prev]
          copy[botIdx] = { role: 'bot', text: 'Something went wrong. Is the server running?', meta: null }
          return copy
        })
      }
    }
    setSending(false)
    inputRef.current?.focus()
  }

  const newChat = () => {
    setMessages([])
    setHistory([])
    loadSuggestions()
    inputRef.current?.focus()
  }

  return (
    <main>
      <div className="hint-row">
        <p className="hint">
          Answers stream in and stay grounded in the FAQ knowledge base.
          Rate them 👍/👎 — that feedback powers the Insights tab.
          History persists across tabs; refresh starts a new chat.
        </p>
        <button className="btn-secondary" onClick={newChat}>＋ New chat</button>
      </div>

      {suggestions.length > 0 && (
        <div>
          <div className="sug-label">Frequently asked</div>
          {suggestions.map((s, i) => (
            <button key={i} className="sug-chip" onClick={() => ask(s.question)}>
              {s.question}
            </button>
          ))}
        </div>
      )}

      <div className="messages">
        {messages.map((m, i) =>
          m.role === 'user'
            ? <div key={i} className="msg user">{m.text}</div>
            : <BotMessage key={i} text={m.text} meta={m.meta} />
        )}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={e => { e.preventDefault(); ask(input) }}>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="e.g. How do I register for courses?"
          autoComplete="off"
        />
        <button disabled={sending}>Send</button>
      </form>
    </main>
  )
}
