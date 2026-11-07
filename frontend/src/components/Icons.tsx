// Stroke icons and the "bridge of dots" mark from the SignBridge design system.
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement> & { size?: number }
const base = ({ size = 24, ...rest }: P) => ({ width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true, ...rest })

export const Logo = ({ size = 46 }: { size?: number }) => (
  <svg width={size} height={size * 0.65} viewBox="0 0 40 26" role="img" aria-label="SignBridge">
    <circle cx="4" cy="21" r="3.6" fill="currentColor"/><circle cx="36" cy="21" r="3.6" fill="currentColor"/>
    <circle cx="7.4" cy="13" r="1.9" fill="currentColor"/><circle cx="12.6" cy="7.4" r="1.9" fill="currentColor"/>
    <circle cx="20" cy="4.6" r="3" fill="#FF5A1F"/><circle cx="27.4" cy="7.4" r="1.9" fill="currentColor"/>
    <circle cx="32.6" cy="13" r="1.9" fill="currentColor"/>
  </svg>
)
export const IconHome = (p: P) => <svg {...base(p)}><path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/></svg>
export const IconHand = (p: P) => <svg {...base(p)}><path d="M7 12V5.5a1.5 1.5 0 0 1 3 0V11M10 10.5V4a1.5 1.5 0 0 1 3 0v6.5M13 10.5V5a1.5 1.5 0 0 1 3 0v6M16 11V8a1.5 1.5 0 0 1 3 0v6a7 7 0 0 1-7 7h-.5a6.5 6.5 0 0 1-5.3-2.8L3.6 14.6a1.5 1.5 0 0 1 2.4-1.8L7 14"/></svg>
export const IconScreen = (p: P) => <svg {...base(p)}><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4M7 8.5h10M7 11.5h6"/></svg>
export const IconLearn = (p: P) => <svg {...base(p)}><path d="M2 8l10-4 10 4-10 4z"/><path d="M6 10v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5"/></svg>
export const IconStudio = (p: P) => <svg {...base(p)}><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.5" fill="currentColor"/></svg>
export const IconSliders = (p: P) => <svg {...base(p)}><path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/></svg>
export const IconSpeaker = (p: P) => <svg {...base(p)}><path d="M4 10v4h4l5 4V6L8 10z"/><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/></svg>
export const IconBackspace = (p: P) => <svg {...base(p)}><path d="M9 5h11v14H9l-6-7z"/><path d="M12 9l5 6M17 9l-5 6"/></svg>
export const IconClose = (p: P) => <svg {...base(p)}><path d="M6 6l12 12M18 6L6 18"/></svg>
export const IconSwap = (p: P) => <svg {...base(p)}><path d="M4 7h13l-3-3M20 17H7l3 3"/></svg>
export const IconArrow = (p: P) => <svg {...base(p)}><path d="M5 12h14M13 6l6 6-6 6"/></svg>
export const IconBack = (p: P) => <svg {...base(p)}><path d="M15 5l-7 7 7 7"/></svg>
export const IconRepeat = (p: P) => <svg {...base(p)}><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>
export const IconLock = (p: P) => <svg {...base(p)}><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>
export const IconPause = (p: P) => <svg {...base(p)} fill="currentColor" stroke="none"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
export const IconPlay = (p: P) => <svg {...base(p)} fill="currentColor" stroke="none"><path d="M7 5l12 7-12 7z"/></svg>
export const IconPoints = (p: P) => <svg {...base(p)}><circle cx="6" cy="18" r="2"/><circle cx="12" cy="6" r="2"/><circle cx="18" cy="14" r="2"/><path d="M7.2 16.4L11 7.8M13.4 7.4l3.4 5"/></svg>
export const IconCamera = (p: P) => <svg {...base(p)}><path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="3.5"/></svg>
export const IconUpRight = (p: P) => <svg {...base(p)}><path d="M7 17L17 7M17 7H9M17 7v8"/></svg>
export const IconCheck = (p: P) => <svg {...base(p)}><path d="M5 12l5 5L20 7"/></svg>
export const IconTrash = (p: P) => <svg {...base(p)}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>
export const IconExternal = (p: P) => <svg {...base(p)}><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>
