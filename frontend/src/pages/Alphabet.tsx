import { useCallback, useEffect, useRef, useState } from 'react'
import { BookOpenCheck, Cpu, LoaderCircle, ShieldCheck } from 'lucide-react'
import { api } from '../api'
import { CameraView } from '../components/CameraView'
import type { CameraFrame } from '../types'

type Info = { ready:boolean; window_size:number; classes:string[]; reported_top1:number }
type Result = { label:string|null; candidate:string; confidence:number; accepted:boolean; is_no_event:boolean; top3:Array<{label:string;confidence:number}>; note:string }

export function Alphabet() {
  const [info,setInfo]=useState<Info|null>(null); const [result,setResult]=useState<Result|null>(null)
  const [frames,setFrames]=useState(0); const [error,setError]=useState('')
  const buffer=useRef<string[]>([]); const busy=useRef(false)
  useEffect(()=>{api.bukvaInfo().then(setInfo).catch(e=>setError(e instanceof Error?e.message:'Модель недоступна'))},[])
  const onFrame=useCallback((frame:CameraFrame)=>{
    if(!frame.image||!info?.ready)return
    buffer.current.push(frame.image)
    if(buffer.current.length>info.window_size)buffer.current.shift()
    setFrames(buffer.current.length)
    if(buffer.current.length===info.window_size&&!busy.current){
      busy.current=true;const batch=[...buffer.current];buffer.current=[];setFrames(0)
      api.bukvaPredict(batch).then(setResult).catch(e=>setError(e instanceof Error?e.message:'Ошибка распознавания')).finally(()=>{busy.current=false})
    }
  },[info])
  return <div className="page"><div className="page-title"><div><span className="eyebrow">Готовая модель · обучение не требуется</span><h1>Алфавит РЖЯ — Bukva</h1><p>Динамическая модель анализирует последовательность из восьми кадров и распознаёт 33 буквы и состояние «нет жеста».</p></div><span className="badge"><ShieldCheck/> локальный ONNX</span></div>
    <div className="recognition-grid"><CameraView onFrame={onFrame} includeImage/><aside className="feedback card"><div className="eyebrow">MobileNetV2-TSM</div>{!info?<div className="recognition-wait"><LoaderCircle className="spin"/><p>Загружаем модель…</p></div>:!info.ready?<div className="notice"><Cpu/><div><strong>Вес модели не найден</strong><p>Повторите setup.sh.</p></div></div>:<><div className={`alphabet-result ${result?.accepted&&!result.is_no_event?'alphabet-result--active':''}`}><span className="alphabet-letter">{result?.is_no_event?'—':result?.accepted?result.label:'?'}</span><strong>{result?.accepted?result.label:result?'Не удалось уверенно распознать':'Покажите букву'}</strong><small>{result?`${Math.round(result.confidence*100)}% · кандидат ${result.candidate}`:`Накоплено кадров: ${frames}/${info.window_size}`}</small></div>{result&&<div className="top-three">{result.top3.map(item=><div key={item.label}><span>{item.label}</span><div><i style={{width:`${item.confidence*100}%`}}/></div><small>{Math.round(item.confidence*100)}%</small></div>)}</div>}</>}{error&&<div className="status-line danger">{error}</div>}<div className="model-facts"><p><BookOpenCheck/> 33 буквы, включая динамические</p><p><Cpu/> Инференс на CPU этого Mac</p></div><p className="fineprint">83,6% — опубликованный Top‑1 результат авторов на тестовой части Bukva, а не измерение на вашей камере. При уверенности ниже 45% система отказывается от ответа. Кадры поступают только локальному серверу 127.0.0.1 и не сохраняются.</p></aside></div>
  </div>
}
