import { useCallback, useEffect, useState } from 'react'
import { BrainCircuit, Plus, Save, Trash2 } from 'lucide-react'
import { api } from '../api'
import type { CameraFrame, GestureClass } from '../types'
import { CameraView } from '../components/CameraView'

type Sample = { id: number; created_at: string }

export function Studio() {
  const [classes,setClasses]=useState<GestureClass[]>([]); const [selected,setSelected]=useState<number|''>('')
  const [session,setSession]=useState<number|null>(null); const [frame,setFrame]=useState<number[]|null>(null)
  const [samples,setSamples]=useState<Sample[]>([]); const [message,setMessage]=useState(''); const [metrics,setMetrics]=useState<Record<string, unknown>|null>(null)
  const refresh=()=>api.classes().then(rows=>{setClasses(rows);if(!selected&&rows.length)setSelected(rows[0].id)})
  useEffect(()=>{refresh()},[])
  const onFrame=useCallback((x:CameraFrame)=>setFrame(x.landmarks),[])
  const create=async(e:React.FormEvent<HTMLFormElement>)=>{e.preventDefault();const form=new FormData(e.currentTarget);await api.createClass({name:String(form.get('name')),description:String(form.get('description')||''),is_unknown:form.get('unknown')==='on'});e.currentTarget.reset();refresh()}
  const begin=async()=>{if(!selected)return;const x=await api.createSession(selected);setSession(x.id);setSamples([]);setMessage('Новая независимая сессия начата')}
  const capture=async()=>{if(!session||!frame)return;await api.addSample(session,frame);setSamples(await api.samples(session));setMessage('Landmarks сохранены — исходное видео не записывалось')}
  const removeSample=async(id:number)=>{if(!session)return;await api.deleteSample(id);setSamples(await api.samples(session))}
  const train=async()=>{try{setMessage('Обучение…');const result=await api.train();setMetrics(result);setMessage('Модель обучена и сохранена локально')}catch(e){setMessage(e instanceof Error?e.message:'Ошибка обучения')}}
  return <div className="page"><div className="page-title"><div><span className="eyebrow">AI Training Studio</span><h1>Научите систему своим жестам</h1><p>Записывайте разные сессии в разном освещении. Для честной проверки нужны минимум две независимые сессии каждого класса.</p></div></div>
    <div className="studio-grid"><div className="studio-main"><CameraView onFrame={onFrame}/><div className="capture-bar card"><div><strong>{samples.length}</strong><span>образцов в этой сессии</span></div><button className="button" onClick={begin} disabled={!selected}><Plus/> Новая сессия</button><button className="button button--primary" onClick={capture} disabled={!session||!frame}><Save/> Сохранить кадр</button></div>{samples.length>0&&<div className="sample-strip card">{samples.slice(0,12).map((sample,index)=><button title="Удалить этот образец" key={sample.id} onClick={()=>removeSample(sample.id)}><span>{samples.length-index}</span><Trash2/></button>)}</div>}{message&&<div className="status-line">{message}</div>}</div>
      <aside className="studio-side"><form className="card form" onSubmit={create}><span className="eyebrow">Новый класс</span><label>Название<input name="name" required placeholder="Например: мой жест 1"/></label><label>Описание<textarea name="description" placeholder="Источник и примечания"/></label><label className="check"><input type="checkbox" name="unknown"/> Это отрицательный класс «неизвестно»</label><button className="button button--primary" type="submit"><Plus/> Создать класс</button></form>
      <div className="card class-list"><div className="class-list__head"><span className="eyebrow">Датасет</span><button className="icon-button" onClick={refresh}>↻</button></div>{classes.map(item=><div className={`class-row ${selected===item.id?'active':''}`} key={item.id} onClick={()=>{setSelected(item.id);setSession(null);setSamples([])}}><div><strong>{item.name}</strong><small>{item.sample_count} образцов · {item.session_count} сессий</small></div><button className="icon-button danger" aria-label="Удалить" onClick={async e=>{e.stopPropagation();await api.deleteClass(item.id);refresh()}}><Trash2/></button></div>)}{!classes.length&&<p className="muted">Создайте обычный и отрицательный классы.</p>}<button className="button button--primary train-button" onClick={train}><BrainCircuit/> Обучить модель</button></div></aside></div>
    {metrics&&<Metrics data={metrics}/>}</div>
}

function Metrics({data}:{data:Record<string,unknown>}) {
  const m=data.metrics as Record<string,unknown>|null
  if(!m)return <section className="card metrics"><div className="section-head"><div><span className="eyebrow">Результат обучения</span><h2>Объективная оценка пока невозможна</h2></div></div><p>Модель обучена на всех данных, но для test split не хватает минимум двух независимых сессий каждого класса. Соберите новые сессии — не просто соседние кадры.</p></section>
  const labels=m.labels as string[], matrix=m.confusion_matrix as number[][]
  return <section className="card metrics"><div className="section-head"><div><span className="eyebrow">Результат обучения</span><h2>Проверка на независимых сессиях</h2></div></div><div className="metric-cards">{labels.map((label,i)=><div key={label}><strong>{label}</strong><span>Precision {Math.round((m.precision as number[])[i]*100)}%</span><span>Recall {Math.round((m.recall as number[])[i]*100)}%</span><span>F1 {Math.round((m.f1 as number[])[i]*100)}%</span></div>)}</div><div className="matrix-wrap"><h3>Confusion matrix</h3><table><thead><tr><th>Факт ↓ / прогноз →</th>{labels.map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>{matrix.map((row,i)=><tr key={labels[i]}><th>{labels[i]}</th>{row.map((value,j)=><td key={j}>{value}</td>)}</tr>)}</tbody></table></div><p className="fineprint">Стратифицированное разделение: целые capture sessions. Тестовых образцов: {String(m.test_samples)}.</p></section>
}
