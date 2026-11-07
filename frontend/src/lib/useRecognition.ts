// WebSocket client for /ws/recognition: sends JPEG frames, receives honest predictions.
import { useCallback, useEffect, useRef, useState } from 'react'

export type Candidate = { label: string; confidence: number }
export type WsPrediction = {
  type: 'prediction'
  state: 'waiting' | 'collecting' | 'processing' | 'recognized' | 'uncertain' | 'no_hands' | 'cooldown' | 'model_unavailable' | 'error'
  label: string | null
  confidence: number | null
  top3: Candidate[]
  accepted: boolean
  stable_count?: number
  reason?: string
  event_id?: number | null
  inference_ms?: number
  buffered?: number
  required?: number
  code?: string
  message?: string
  timestamp: string
}
export type Ready = { session_id: string; model_id: string; required_frames: number; max_fps: number; domain_pack: string | null; hand_detection: boolean; thresholds: { confidence: number; stability_count: number; cooldown_ms: number } }
export type SocketStatus = 'off' | 'connecting' | 'live' | 'reconnecting' | 'unavailable'

type Options = {
  video: HTMLVideoElement | null
  enabled: boolean
  modelId?: string | null
  domainPack?: string | null
  onPrediction?: (p: WsPrediction) => void
  onNotice?: (code: string, message: string) => void
}

const FRAME_WIDTH = 480
const MAX_BUFFERED = 256 * 1024

export function useRecognition({ video, enabled, modelId, domainPack, onPrediction, onNotice }: Options) {
  const [status, setStatus] = useState<SocketStatus>('off')
  const [ready, setReady] = useState<Ready | null>(null)
  const [last, setLast] = useState<WsPrediction | null>(null)
  const [unavailableReason, setUnavailableReason] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef(video)
  const handlers = useRef({ onPrediction, onNotice })
  videoRef.current = video
  handlers.current = { onPrediction, onNotice }

  const send = useCallback((message: Record<string, unknown>) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message))
  }, [])

  useEffect(() => {
    if (!enabled) { setStatus('off'); setReady(null); setLast(null); return }
    let closedByUs = false
    let retry: number | undefined
    let ticker: number | undefined
    const canvas = document.createElement('canvas')
    let capturing = false

    const startFrames = (fps: number) => {
      window.clearInterval(ticker)
      ticker = window.setInterval(() => {
        const ws = wsRef.current
        const source = videoRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN || !source || source.readyState < 2 || capturing) return
        if (ws.bufferedAmount > MAX_BUFFERED) return // backend is behind: drop this frame
        const width = Math.min(FRAME_WIDTH, source.videoWidth || FRAME_WIDTH)
        const height = Math.round(width * (source.videoHeight || 3) / (source.videoWidth || 4))
        canvas.width = width; canvas.height = height
        canvas.getContext('2d')?.drawImage(source, 0, 0, width, height) // raw, not mirrored
        capturing = true
        canvas.toBlob(blob => {
          capturing = false
          if (blob && ws.readyState === WebSocket.OPEN) blob.arrayBuffer().then(buffer => ws.send(buffer))
        }, 'image/jpeg', 0.75)
      }, 1000 / fps)
    }

    const connect = () => {
      setStatus(s => (s === 'live' || s === 'reconnecting' ? 'reconnecting' : 'connecting'))
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/recognition`)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws
      ws.onopen = () => {
        const start: Record<string, unknown> = { type: 'start', mode: 'isolated_sign', domain_pack: domainPack ?? null,
          client: { frame_width: FRAME_WIDTH, frame_height: 360 } }
        if (modelId) start.model_id = modelId
        ws.send(JSON.stringify(start))
      }
      ws.onmessage = event => {
        const message = JSON.parse(String(event.data))
        if (message.type === 'ready') {
          setReady(message); setStatus('live'); setUnavailableReason('')
          startFrames(Math.min(12, message.max_fps || 12))
        } else if (message.type === 'prediction') {
          if (message.state === 'model_unavailable') { setStatus('unavailable'); setUnavailableReason(message.message || '') }
          setLast(message)
          handlers.current.onPrediction?.(message)
        } else if (message.type === 'error') {
          handlers.current.onNotice?.(message.code, message.message)
        }
      }
      ws.onclose = event => {
        window.clearInterval(ticker)
        if (wsRef.current === ws) wsRef.current = null
        if (closedByUs) return
        // 1013 = model not ready: retry slower. Anything else: quick reconnect.
        const delay = event.code === 1013 ? 5000 : 1000
        if (event.code !== 1013) setStatus('reconnecting')
        retry = window.setTimeout(connect, delay)
      }
    }

    connect()
    return () => {
      closedByUs = true
      window.clearTimeout(retry)
      window.clearInterval(ticker)
      const ws = wsRef.current
      if (ws) {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' }))
        ws.close(1000)
      }
      wsRef.current = null
    }
  }, [enabled, modelId, domainPack])

  return { status, ready, last, send, unavailableReason }
}
