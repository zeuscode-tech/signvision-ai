import { useCallback, useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { Camera, CameraOff, RefreshCw } from 'lucide-react'
import type { CameraFrame } from '../types'

const connections = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]
]

type Props = { onFrame?: (frame: CameraFrame) => void; compact?: boolean; includeImage?: boolean }

export function CameraView({ onFrame, compact = false, includeImage = false }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const requestRef = useRef<number>(0)
  const lastVideoTime = useRef(-1)
  const frameTimes = useRef<number[]>([])
  const imageFrameCounter = useRef(0)
  const captureCanvas = useRef<HTMLCanvasElement | null>(null)
  const [status, setStatus] = useState<'idle'|'loading'|'active'|'error'>('idle')
  const [error, setError] = useState('')
  const [stats, setStats] = useState({ fps: 0, latency: 0, hands: 0 })

  const stop = useCallback(() => {
    cancelAnimationFrame(requestRef.current)
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    setStatus('idle')
  }, [])

  const loop = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const landmarker = landmarkerRef.current
    if (!video || !canvas || !landmarker || !streamRef.current) return
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime.current) {
      lastVideoTime.current = video.currentTime
      const started = performance.now()
      const result = landmarker.detectForVideo(video, started)
      const latency = performance.now() - started
      const ctx = canvas.getContext('2d')!
      canvas.width = video.videoWidth; canvas.height = video.videoHeight
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1)
      for (const hand of result.landmarks) {
        ctx.strokeStyle = 'rgba(255,107,53,.8)'; ctx.lineWidth = 3
        connections.forEach(([a,b]) => {
          ctx.beginPath(); ctx.moveTo(hand[a].x*canvas.width, hand[a].y*canvas.height)
          ctx.lineTo(hand[b].x*canvas.width, hand[b].y*canvas.height); ctx.stroke()
        })
        ctx.fillStyle = '#FFD9C7'
        hand.forEach(p => { ctx.beginPath(); ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);ctx.fill() })
      }
      ctx.restore()
      const ordered = new Array(126).fill(0)
      result.landmarks.forEach((hand, index) => {
        const category = result.handedness[index]?.[0]?.categoryName
        // Mirrored preview does not change MediaPipe source coordinates; retain semantic handedness.
        const offset = category === 'Left' ? 0 : 63
        hand.forEach((p, i) => { ordered[offset+i*3]=p.x; ordered[offset+i*3+1]=p.y; ordered[offset+i*3+2]=p.z })
      })
      const now = performance.now(); frameTimes.current.push(now)
      frameTimes.current = frameTimes.current.filter(t => now-t < 1000)
      const nextStats = { fps: frameTimes.current.length, latency: Math.round(latency), hands: result.landmarks.length }
      let image: string | undefined
      imageFrameCounter.current += 1
      if (includeImage && imageFrameCounter.current % 2 === 0) {
        const capture = captureCanvas.current ?? document.createElement('canvas')
        captureCanvas.current = capture
        const width = Math.min(480, video.videoWidth)
        const height = Math.round(width * video.videoHeight / video.videoWidth)
        capture.width = width; capture.height = height
        capture.getContext('2d')!.drawImage(video, 0, 0, width, height)
        image = capture.toDataURL('image/jpeg', .74)
      }
      setStats(nextStats); onFrame?.({ landmarks: result.landmarks.length ? ordered : null, ...nextStats, image })
    }
    requestRef.current = requestAnimationFrame(loop)
  }, [includeImage, onFrame])

  const start = useCallback(async () => {
    try {
      setStatus('loading'); setError('')
      if (!landmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks('/mediapipe')
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate: 'GPU' },
          runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: .55,
          minHandPresenceConfidence: .55, minTrackingConfidence: .5,
        })
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
      setStatus('active'); requestRef.current = requestAnimationFrame(loop)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось открыть камеру'); setStatus('error')
    }
  }, [loop])

  useEffect(() => stop, [stop])

  return <div className={`camera ${compact ? 'camera--compact' : ''}`}>
    <div className="camera__viewport">
      <video ref={videoRef} muted playsInline />
      <canvas ref={canvasRef} />
      {status !== 'active' && <div className="camera__empty">
        {status === 'loading' ? <RefreshCw className="spin"/> : <Camera size={30}/>}<strong>{status === 'loading' ? 'Загружаем локальный трекер…' : 'Камера выключена'}</strong>
        <span>{error || 'Видео не сохраняется и не уходит в интернет'}</span>
        <button className="button button--primary" onClick={start} disabled={status === 'loading'}>Включить камеру</button>
      </div>}
      {status === 'active' && <div className="camera__stats"><span>{stats.hands}/2 рук</span><span>{stats.fps} FPS</span><span>{stats.latency} мс</span></div>}
    </div>
    {status === 'active' && <button className="button button--ghost camera__stop" onClick={stop}><CameraOff size={16}/> Выключить</button>}
  </div>
}

