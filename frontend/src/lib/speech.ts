// Text-to-speech with the voices installed in macOS: works without internet.

export function russianVoices(): SpeechSynthesisVoice[] {
  if (!('speechSynthesis' in window)) return []
  return speechSynthesis.getVoices().filter(voice => voice.lang.toLowerCase().startsWith('ru'))
}

export function speak(text: string, options: { voice?: string | null; rate?: number } = {}): boolean {
  if (!('speechSynthesis' in window) || !text.trim()) return false
  speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = 'ru-RU'
  utterance.rate = options.rate ?? 1
  const voices = russianVoices()
  const voice = voices.find(v => v.name === options.voice) ?? voices.find(v => v.localService) ?? voices[0]
  if (voice) utterance.voice = voice
  speechSynthesis.speak(utterance)
  return true
}

export function phraseToText(tokens: string[]): string {
  if (!tokens.length) return ''
  const text = tokens.join(' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}
