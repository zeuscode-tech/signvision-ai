// Camera with the SignBridge look: hand points, skeleton, and a glowing trail of the index fingertip.
// Landmarks are computed in the browser (MediaPipe) only for drawing and for Studio samples;
// recognition itself happens on the local backend.
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]]
const TRAIL = 22
const EDGE = 0.03

export type HandsInfo = { count: number; partlyOut: 'left' | 'right' | null; landmarks: number[] | null; fps: number }

let landmarkerPromise: Promise<HandLandmarker> | null = null
function getLandmarker(): Promise<HandLandmarker> {
  landmarkerPromise ??= FilesetResolver.forVisionTasks('/mediapipe').then(vision => HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate: 'GPU' },
    runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: .55, minHandPresenceConfidence: .55, minTrackingConfidence: .5,
  }))
  landmarkerPromise.catch(() => { landmarkerPromise = null })
  return landmarkerPromise
}

type Props = {
  active: boolean
  onActiveChange?: (active: boolean) => void
  onVideo?: (video: HTMLVideoElement | null) => void
  onHands?: (info: HandsInfo) => void
  showPoints?: boolean
  showTrail?: boolean
  children?: ReactNode
  className?: string
  label?: string
}

export function LiveCamera({ active, onActiveChange, onVideo, onHands, showPoints = true, showTrail = true, children, className = '', label = 'Камера' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const raf = useRef(0)
  const trail = useRef<Array<{ x: number; y: number }>>([])
  const times = useRef<number[]>([])
  const lastTime = useRef(-1)
  const opts = useRef({ showPoints, showTrail, onHands })
  opts.current = { showPoints, showTrail, onHands }
  const [state, setState] = useState<'off' | 'loading' | 'on' | 'error'>('off')
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    trail.current = []
    onVideo?.(null)
    setState('off')
  }, [onVideo])

  const draw = useCallback((landmarker: HandLandmarker) => {
    const video = videoRef.current, canvas = canvasRef.current
    if (!video || !canvas || !streamRef.current) return
    if (video.readyState >= 2 && video.currentTime !== lastTime.current) {
      lastTime.current = video.currentTime
      const result = landmarker.detectForVideo(video, performance.now())
      const w = canvas.width = canvas.clientWidth * devicePixelRatio
      const h = canvas.height = canvas.clientHeight * devicePixelRatio
      const ctx = canvas.getContext('2d')!
      ctx.clearRect(0, 0, w, h)
      // object-fit: cover mapping, mirrored like the preview
      const vw = video.videoWidth, vh = video.videoHeight
      const scale = Math.max(w / vw, h / vh)
      const ox = (w - vw * scale) / 2, oy = (h - vh * scale) / 2
      const map = (p: NormalizedLandmark) => ({ x: w - (ox + p.x * vw * scale), y: oy + p.y * vh * scale })
      const ordered = new Array(126).fill(0)
      let partlyOut: HandsInfo['partlyOut'] = null
      result.landmarks.forEach((hand, index) => {
        const side = result.handedness[index]?.[0]?.categoryName === 'Left' ? 'left' : 'right'
        const offset = side === 'left' ? 0 : 63
        hand.forEach((p, i) => { ordered[offset + i * 3] = p.x; ordered[offset + i * 3 + 1] = p.y; ordered[offset + i * 3 + 2] = p.z })
        if (hand.some(p => p.x < EDGE || p.x > 1 - EDGE || p.y < EDGE || p.y > 1 - EDGE)) partlyOut = side
        if (!opts.current.showPoints) return
        const pts = hand.map(map)
        const ember = index === 0
        ctx.lineWidth = 3.5 * devicePixelRatio; ctx.lineCap = 'round'
        ctx.strokeStyle = ember ? 'rgba(255,107,53,.8)' : 'rgba(243,238,228,.55)'
        for (const [a, b] of CONNECTIONS) { ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke() }
        ctx.fillStyle = ember ? '#FFD9C7' : '#F3EEE4'
        pts.forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x, p.y, ([4, 8, 12, 16, 20].includes(i) ? 6 : 4.5) * devicePixelRatio, 0, Math.PI * 2); ctx.fill() })
      })
      // trail of the first hand's index fingertip
      const tip = result.landmarks[0]?.[8]
      if (tip) trail.current.push(map(tip)); else trail.current.shift()
      if (trail.current.length > TRAIL) trail.current.shift()
      if (opts.current.showTrail && trail.current.length > 1) {
        const t = trail.current
        for (let i = 1; i < t.length; i++) {
          const k = i / t.length
          ctx.strokeStyle = `rgba(255,107,53,${k * 0.9})`
          ctx.lineWidth = (4 + k * 8) * devicePixelRatio
          ctx.shadowColor = 'rgba(255,107,53,.8)'; ctx.shadowBlur = 18 * devicePixelRatio
          ctx.beginPath(); ctx.moveTo(t[i - 1].x, t[i - 1].y); ctx.lineTo(t[i].x, t[i].y); ctx.stroke()
        }
        ctx.shadowBlur = 0
        const head = t[t.length - 1]
        if (tip) { ctx.fillStyle = '#FF6B35'; ctx.beginPath(); ctx.arc(head.x, head.y, 11 * devicePixelRatio, 0, Math.PI * 2); ctx.fill() }
      }
      const now = performance.now(); times.current.push(now); times.current = times.current.filter(x => now - x < 1000)
      opts.current.onHands?.({ count: result.landmarks.length, partlyOut, landmarks: result.landmarks.length ? ordered : null, fps: times.current.length })
    }
    raf.current = requestAnimationFrame(() => draw(landmarker))
  }, [])

  const start = useCallback(async () => {
    setState('loading'); setError('')
    try {
      const [landmarker, stream] = await Promise.all([
        getLandmarker(),
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }),
      ])
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      onVideo?.(video)
      setState('on')
      raf.current = requestAnimationFrame(() => draw(landmarker))
    } catch (reason) {
      const name = reason instanceof DOMException ? reason.name : ''
      setError(name === 'NotAllowedError' ? 'Доступ к камере запрещён. Разрешите его: System Settings → Privacy & Security → Camera.'
        : name === 'NotFoundError' ? 'Камера не найдена.' : 'Не удалось включить камеру.')
      setState('error')
      onActiveChange?.(false)
    }
  }, [draw, onActiveChange, onVideo])

  useEffect(() => {
    if (active && state === 'off') void start()
    if (!active && state !== 'off' && state !== 'error') stop()
  }, [active]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => stop, [stop])

  return <section aria-label={label} className={`cam dots-bg ${className}`}>
    <video ref={videoRef} muted playsInline aria-hidden="true" style={{ display: state === 'on' ? 'block' : 'none' }} />
    <canvas ref={canvasRef} aria-hidden="true" />
    <svg className="cam__frame" viewBox="0 0 800 600" preserveAspectRatio="none" aria-hidden="true">
      <path d="M40 80 v-40 h40 M760 80 v-40 h-40 M40 520 v40 h40 M760 520 v40 h-40" fill="none" stroke="#F3EEE4" strokeOpacity=".5" strokeWidth="3" strokeLinecap="round" vectorEffect="non-scaling-stroke"/>
    </svg>
    {state !== 'on' && <div className="cam__empty">
      {state === 'loading' ? <><span className="spin" style={{ width: 36, height: 36, borderRadius: '50%', border: '4px solid #FF6B35', borderTopColor: 'transparent' }} /><strong>Включаем камеру…</strong><span>Трекер рук загружается локально</span></>
        : <><strong>{state === 'error' ? 'Камера недоступна' : 'Камера выключена'}</strong><span>{error || 'Видео обрабатывается на этом ноутбуке и никуда не отправляется'}</span>
          <button type="button" className="btn btn--ember btn--lg" onClick={() => { onActiveChange?.(true); void start() }}>Включить камеру</button></>}
    </div>}
    {children}
  </section>
}
