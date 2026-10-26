import type { GestureClass, Prediction } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...init?.headers }, ...init })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

export const api = {
  health: () => request<{ status: string; model_ready: boolean; bukva_ready: boolean; trained_at: string | null }>('/api/health'),
  bukvaInfo: () => request<{ ready: boolean; window_size: number; classes: string[]; reported_top1: number }>('/api/bukva/info'),
  bukvaPredict: (frames: string[]) => request<{ label: string|null; candidate:string; confidence: number; accepted:boolean; is_no_event: boolean; top3: Array<{label:string;confidence:number}>; note:string }>('/api/bukva/predict', { method: 'POST', body: JSON.stringify({ frames }) }),
  lessons: () => request<Array<Record<string, unknown>>>('/api/lessons'),
  classes: () => request<GestureClass[]>('/api/classes'),
  createClass: (body: { name: string; description: string; is_unknown: boolean }) =>
    request<GestureClass>('/api/classes', { method: 'POST', body: JSON.stringify(body) }),
  deleteClass: (id: number) => request(`/api/classes/${id}`, { method: 'DELETE' }),
  createSession: (gesture_class_id: number) => request<{ id: number }>('/api/sessions', { method: 'POST', body: JSON.stringify({ gesture_class_id }) }),
  addSample: (session_id: number, landmarks: number[]) => request<{ id: number }>('/api/samples', { method: 'POST', body: JSON.stringify({ session_id, landmarks }) }),
  samples: (sessionId: number) => request<Array<{ id: number; created_at: string }>>(`/api/sessions/${sessionId}/samples`),
  deleteSample: (id: number) => request(`/api/samples/${id}`, { method: 'DELETE' }),
  train: () => request<Record<string, unknown>>('/api/train', { method: 'POST' }),
  model: () => request<Record<string, unknown>>('/api/model'),
  predict: (landmarks: number[]) => request<Prediction>('/api/predict', { method: 'POST', body: JSON.stringify({ landmarks }) }),
  attempt: (body: Record<string, unknown>) => request('/api/attempts', { method: 'POST', body: JSON.stringify(body) }),
  progress: () => request<Record<string, unknown>>('/api/progress'),
  exportData: () => request<Record<string, unknown>>('/api/export'),
  importData: (body: unknown) => request('/api/import', { method: 'POST', body: JSON.stringify(body) }),
  deleteData: () => request('/api/data', { method: 'DELETE' }),
}

