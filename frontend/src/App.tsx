import { useEffect, useState } from 'react'
import { BarChart3, BookOpen, BrainCircuit, CircleGauge, GraduationCap, Languages, Menu, ScanLine, Settings as SettingsIcon, Sparkles, X } from 'lucide-react'
import type { NavKey } from './types'
import { Dashboard } from './pages/Dashboard'
import { Learn } from './pages/Learn'
import { Practice } from './pages/Practice'
import { Studio } from './pages/Studio'
import { Progress } from './pages/Progress'
import { Settings } from './pages/Settings'
import { Alphabet } from './pages/Alphabet'

const nav:[NavKey,string,React.ElementType][]=[['dashboard','Обзор',CircleGauge],['learn','Обучение',BookOpen],['alphabet','Алфавит',Languages],['practice','Практика',GraduationCap],['quiz','Квиз',Sparkles],['recognition','Распознавание',ScanLine],['studio','AI Studio',BrainCircuit],['progress','Прогресс',BarChart3],['settings','Настройки',SettingsIcon]]

export default function App(){const [page,setPage]=useState<NavKey>('dashboard');const [theme,setTheme]=useState(localStorage.getItem('sv-theme')||'dark');const [menu,setMenu]=useState(false);useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('sv-theme',theme)},[theme]);const go=(p:NavKey)=>{setPage(p);setMenu(false)};return <div className="app"><header className="mobile-head"><Brand/><button className="icon-button" onClick={()=>setMenu(!menu)}>{menu?<X/>:<Menu/>}</button></header><aside className={`sidebar ${menu?'sidebar--open':''}`}><Brand/><nav>{nav.map(([id,label,Icon])=><button key={id} className={page===id?'active':''} onClick={()=>go(id)}><Icon/><span>{label}</span></button>)}</nav><div className="sidebar__local"><span className="pulse"/><div><strong>Локальный режим</strong><small>Данные на этом Mac</small></div></div></aside><main>{page==='dashboard'&&<Dashboard navigate={go}/>} {page==='learn'&&<Learn/>}{page==='alphabet'&&<Alphabet/>}{page==='practice'&&<Practice/>}{page==='quiz'&&<Practice quiz/>}{page==='recognition'&&<Practice recognition/>}{page==='studio'&&<Studio/>}{page==='progress'&&<Progress/>}{page==='settings'&&<Settings theme={theme} toggleTheme={()=>setTheme(theme==='dark'?'light':'dark')}/>}</main></div>}
function Brand(){return <div className="brand"><div className="brand__mark">S</div><div><strong>SignVision</strong><span>AI Trainer</span></div></div>}

