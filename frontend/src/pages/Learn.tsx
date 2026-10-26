import { useEffect, useState } from 'react'
import { Check, LockKeyhole } from 'lucide-react'
import { api } from '../api'
import { CameraView } from '../components/CameraView'

export function Learn() {
  const [levels, setLevels] = useState<Array<Record<string, unknown>>>([])
  useEffect(()=>{api.lessons().then(setLevels)},[])
  return <div className="page"><div className="page-title"><div><span className="eyebrow">Учебная программа</span><h1>РЖЯ — от трекинга к практике</h1><p>Встроены только технически проверяемые уроки. Жестовые материалы ждут лицензированного источника и проверки экспертом.</p></div></div>
    <div className="learn-layout"><div className="lesson-list">{levels.map(level => <article className="lesson card" key={String(level.id)}><div className="lesson__number">{String(level.level).padStart(2,'0')}</div><div><span className="badge">{level.material_status==='ready'?<><Check/> готово</>:<><LockKeyhole/> требуется наполнение</>}</span><h3>{String(level.title)}</h3><p>{String(level.description)}</p></div></article>)}</div><aside className="learn-demo card"><span className="eyebrow">Диагностика</span><h2>Проверьте трекинг</h2><p>Покажите одну и затем две руки. Скелет должен оставаться устойчивым при небольших движениях.</p><CameraView compact/></aside></div>
  </div>
}

