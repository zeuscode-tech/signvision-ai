// Per-device interface preferences (theme, text size, motion, voice). Stored locally only.
import { useEffect, useState } from 'react'

export type Prefs = {
  theme: 'light' | 'dark' | 'system'
  scale: number
  contrast: boolean
  reduceMotion: boolean
  autoAddMs: number
  voice: string | null
  rate: number
  flashOnReply: boolean
  subtitles: boolean
}

const DEFAULTS: Prefs = { theme: 'system', scale: 1, contrast: false, reduceMotion: false, autoAddMs: 1500, voice: null, rate: 1, flashOnReply: true, subtitles: true }
const KEY = 'signbridge-prefs'

function read(): Prefs {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') } } catch { return DEFAULTS }
}

let current = read()
const listeners = new Set<(p: Prefs) => void>()

export function setPrefs(patch: Partial<Prefs>) {
  current = { ...current, ...patch }
  try { localStorage.setItem(KEY, JSON.stringify(current)) } catch { /* private mode */ }
  listeners.forEach(fn => fn(current))
}

export function usePrefs(): Prefs {
  const [prefs, setState] = useState(current)
  useEffect(() => { listeners.add(setState); return () => { listeners.delete(setState) } }, [])
  return prefs
}

export function applyPrefs(prefs: Prefs) {
  const root = document.documentElement
  const dark = prefs.theme === 'dark' || (prefs.theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches)
  root.dataset.theme = dark ? 'dark' : 'light'
  root.dataset.contrast = prefs.contrast ? 'high' : 'normal'
  root.dataset.motion = prefs.reduceMotion ? 'reduce' : 'normal'
  root.style.setProperty('--scale', String(prefs.scale))
}
