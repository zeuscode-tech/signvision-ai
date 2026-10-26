import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, ScanLine } from 'lucide-react'
import { api } from '../api'
import type { CameraFrame, Prediction } from '../types'
import { CameraView } from './CameraView'

type Props = { target?: string; mode: 'practice'|'quiz'|'recognition' }

export function RecognitionPanel({ target, mode }: Props) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const lastRequest = useRef(0)
  const history = useRef<Prediction[]>([])
  const started = useRef(Date.now())

  useEffect(() => { api.health().then(x => setModelReady(x.model_ready)).catch(() => {}) }, [])
  const onFrame = useCallback(async (frame: CameraFrame) => {
    if (!frame.landmarks || !modelReady || busy || performance.now()-lastRequest.current < 240) return
    lastRequest.current = performance.now(); setBusy(true)
    try {
      const next = await api.predict(frame.landmarks)
      history.current = [...history.current.slice(-6), next]
      const accepted = history.current.filter(x => x.accepted && x.label === next.label)
      setPrediction(accepted.length >= 4 ? next : { ...next, accepted: false, label: null, message: 'Удерживайте жест стабильно' })
    } catch { setPrediction(null) } finally { setBusy(false) }
  }, [busy, modelReady])
  const correct = Boolean(target && prediction?.accepted && prediction.label === target)
  const save = async (success: boolean) => {
    if (!target) return
    await api.attempt({ gesture_name: target, mode, success, confidence: prediction?.confidence ?? null, duration_seconds: Math.round((Date.now()-started.current)/1000) })
    started.current = Date.now()
  }
  return <div className="recognition-grid">
    <CameraView onFrame={onFrame}/>
    <aside className="feedback card">
      <div className="eyebrow">Локальное распознавание</div>
      {target && <><span className="muted">Целевой класс</span><h2>{target}</h2></>}
      {!modelReady ? <div className="notice"><AlertCircle/><div><strong>Модель отсутствует</strong><p>Сначала соберите минимум два класса в AI Training Studio.</p></div></div>
      : !prediction ? <div className="recognition-wait"><ScanLine/><p>Покажите жест в центре кадра</p></div>
      : <div className={`result ${prediction.accepted ? (correct || !target ? 'result--ok' : 'result--warn') : ''}`}>
          {prediction.accepted ? <CheckCircle2/> : <ScanLine/>}
          <span>{prediction.accepted ? prediction.label : prediction.message}</span>
          <small>{Math.round(prediction.confidence*100)}% — вероятность модели, не оценка качества техники</small>
        </div>}
      {target && <div className="feedback__actions"><button className="button" onClick={() => save(false)}>Не получилось</button><button className="button button--primary" disabled={!correct} onClick={() => save(true)}>Засчитать</button></div>}
      <p className="fineprint">Система различает только обученные вами классы. Она не переводит свободную жестовую речь и не даёт советов по пальцам без вычислимого правила.</p>
    </aside>
  </div>
}

