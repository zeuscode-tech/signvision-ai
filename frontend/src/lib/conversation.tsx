// Conversation state shared by the Live screen and the partner screen.
// A BroadcastChannel keeps a partner window on a second display in sync.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

type State = { tokens: string[]; spoken: string; reply: string; replyAt: number; lastAddedAt: number }
type Ctx = State & {
  addToken: (token: string) => void
  undo: () => void
  clear: () => void
  setSpoken: (text: string) => void
  sendReply: (text: string) => void
}

const ConversationContext = createContext<Ctx | null>(null)
const EMPTY: State = { tokens: [], spoken: '', reply: '', replyAt: 0, lastAddedAt: 0 }

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>(EMPTY)
  const channel = useRef<BroadcastChannel | null>(null)

  useEffect(() => {
    if (!('BroadcastChannel' in window)) return
    const bc = new BroadcastChannel('signbridge-conversation')
    bc.onmessage = event => {
      if (event.data?.type === 'state') setState(event.data.state)
      if (event.data?.type === 'hello') bc.postMessage({ type: 'state', state: stateRef.current })
    }
    bc.postMessage({ type: 'hello' })
    channel.current = bc
    return () => bc.close()
  }, [])

  const stateRef = useRef(state)
  const update = useCallback((fn: (s: State) => State) => {
    setState(prev => {
      const next = fn(prev)
      stateRef.current = next
      channel.current?.postMessage({ type: 'state', state: next })
      return next
    })
  }, [])

  const value = useMemo<Ctx>(() => ({
    ...state,
    addToken: token => update(s => ({ ...s, tokens: [...s.tokens, token].slice(-40), lastAddedAt: Date.now() })),
    undo: () => update(s => ({ ...s, tokens: s.tokens.slice(0, -1) })),
    clear: () => update(s => ({ ...s, tokens: [], spoken: '' })),
    setSpoken: text => update(s => ({ ...s, spoken: text })),
    sendReply: text => update(s => ({ ...s, reply: text, replyAt: Date.now() })),
  }), [state, update])

  return <ConversationContext.Provider value={value}>{children}</ConversationContext.Provider>
}

export function useConversation(): Ctx {
  const ctx = useContext(ConversationContext)
  if (!ctx) throw new Error('ConversationProvider missing')
  return ctx
}
