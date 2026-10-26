import { useEffect, useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { api } from '../api'
import type { GestureClass } from '../types'
import { RecognitionPanel } from '../components/RecognitionPanel'

export function Practice({ quiz=false, recognition=false }: { quiz?: boolean; recognition?: boolean }) {
  const [classes,setClasses]=useState<GestureClass[]>([]); const [selected,setSelected]=useState('')
  useEffect(()=>{api.classes().then(rows=>{const usable=rows.filter(x=>!x.is_unknown);setClasses(usable);setSelected(usable[0]?.name||'')})},[])
  const target=useMemo(()=> quiz && classes.length ? classes[Math.floor(Math.random()*classes.length)].name : selected,[quiz,classes,selected])
  return <div className="page"><div className="page-title"><div><span className="eyebrow">{recognition?'Ограниченный словарь':quiz?'Проверка без подсказки':'Практика'}</span><h1>{recognition?'Свободное распознавание':quiz?'Quiz Mode':'Practice Mode'}</h1><p>{recognition?'Покажите любой класс из вашей локальной модели.':'Выберите созданный вами класс и повторите его перед камерой.'}</p></div>{!recognition && !quiz && <select value={selected} onChange={e=>setSelected(e.target.value)}>{classes.length?classes.map(x=><option key={x.id}>{x.name}</option>):<option>Нет классов</option>}</select>}{quiz&&<button className="button"><Shuffle/> Новое задание</button>}</div>
    {!recognition && !target ? <div className="empty card"><h2>Пока нечего практиковать</h2><p>Создайте классы и обучите модель в AI Training Studio.</p></div>:<RecognitionPanel target={recognition?undefined:target} mode={recognition?'recognition':quiz?'quiz':'practice'}/>}</div>
}

