import React from 'react'
import ReactDOM from 'react-dom/client'
// Fonts are bundled so the app works fully offline.
import '@fontsource/unbounded/500.css'
import '@fontsource/unbounded/700.css'
import '@fontsource/unbounded/800.css'
import '@fontsource/unbounded/900.css'
import '@fontsource/onest/400.css'
import '@fontsource/onest/500.css'
import '@fontsource/onest/600.css'
import '@fontsource/onest/700.css'
import '@fontsource/jetbrains-mono/500.css'
import '@fontsource/jetbrains-mono/700.css'
import App from './App'
import { applyPrefs, usePrefs } from './lib/prefs'
import './styles.css'

function Root() { applyPrefs(usePrefs()); return <App /> }

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><Root /></React.StrictMode>)
