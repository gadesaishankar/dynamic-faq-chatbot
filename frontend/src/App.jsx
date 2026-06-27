import { useState } from 'react'
import Chat from './components/Chat'
import Faq from './components/Faq'
import Insights from './components/Insights'

const TABS = [
  { id: 'chat', label: 'Chat' },
  { id: 'faq', label: 'Dynamic FAQ' },
  { id: 'insights', label: 'Insights' },
]

export default function App() {
  const [tab, setTab] = useState(() => {
    return window.location.hash === '#faq' ? 'faq' : 'chat'
  })

  return (
    <>
      <header>
        <h1>🎓 Department FAQ Chatbot</h1>
        <nav>
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      {tab === 'chat' && <Chat />}
      {tab === 'faq' && <Faq />}
      {tab === 'insights' && <Insights />}
    </>
  )
}
