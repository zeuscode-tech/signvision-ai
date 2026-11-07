// Typed client for the local backend's /api/v1. Everything stays on 127.0.0.1.

export class ApiError extends Error {
  constructor(message: string, readonly code: string, readonly status: number) { super(message) }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new ApiError(data.error?.message || `HTTP ${response.status}`, data.error?.code || 'HTTP', response.status)
  }
  return response.json()
}
const post = <T>(path: string, body?: unknown) => call<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
const put = <T>(path: string, body: unknown) => call<T>(path, { method: 'PUT', body: JSON.stringify(body) })

export type Health = { status: 'ok' | 'degraded'; database: string; active_model: string | null; model_status: string; version: string }
export type SystemInfo = { backend_version: string; onnxruntime_version: string; active_model: string | null; model_status: string; hand_detector: string }
export type ModelEntry = { id: string; name: string; version: string; type: string; source: string; labels_count: number; status: string; status_message: string | null; runtime_status: string; is_active: boolean }
export type DomainPack = { id: string; name: string; version: string; description: string; enabled_labels: string[]; quick_responses: string[]; review_status: string; status?: 'valid' | 'invalid'; error?: string; is_active?: boolean; compatibility?: { model_id: string | null; checked: boolean; compatible: boolean | null; missing_labels: string[] } | null }
export type AppSettings = { theme: string; language: string; active_model_id: string | null; active_domain_pack_id: string | null; confidence_threshold: number | null; stability_count: number | null; cooldown_ms: number | null; speech_enabled: boolean; speech_rate: number; speech_voice: string | null }
export type Lesson = { id: string; level: number; title: string; description: string; material_status: string; exercises: Array<Record<string, unknown>> }
export type Progress = { attempts: number; successes: number; success_rate: number | null; practice_seconds: number; daily: Array<{ date: string; attempts: number; successes: number; seconds: number }>; gestures: Array<{ gesture_id: string; attempts: number; successes: number; success_rate: number }>; reviews_due: number; srs_items: number }
export type SrsItem = { gesture_id: string; repetitions: number; interval_days: number; due_date: string }
export type TrainingClass = { id: number; name: string; description: string; is_negative: boolean; session_count: number; sample_count: number; static_samples: number; created_at: string }
export type TrainingJob = { id: string; status: 'queued' | 'running' | 'completed' | 'failed'; model_id: string | null; metrics: Record<string, unknown> | null; error_message: string | null }
export type TrainingModel = { id: string; name: string; version: string; labels_count: number; metrics: Record<string, unknown> | null; is_active: boolean; created_at: string }
export type Prediction = { state: 'recognized' | 'uncertain' | 'unknown'; label: string | null; candidate: string; confidence: number; accepted: boolean; is_negative: boolean; top3: Array<{ label: string; confidence: number }> }

export const v1 = {
  health: () => call<Health>('/health'),
  system: () => call<SystemInfo>('/system/info'),
  settings: () => call<AppSettings>('/settings'),
  updateSettings: (body: Partial<AppSettings>) => put<AppSettings>('/settings', body),
  models: () => call<ModelEntry[]>('/models?type=isolated_sign'),
  activateModel: (id: string) => post<ModelEntry>(`/models/${id}/activate`),
  packs: () => call<DomainPack[]>('/domain-packs'),
  lessons: () => call<Lesson[]>('/lessons'),
  progress: () => call<Progress>('/progress'),
  dueReviews: () => call<SrsItem[]>('/repetition/next?limit=20'),
  classes: () => call<TrainingClass[]>('/training/classes'),
  createClass: (name: string, isNegative = false) => post<TrainingClass>('/training/classes', { name, is_negative: isNegative }),
  deleteClass: (id: number) => call(`/training/classes/${id}`, { method: 'DELETE' }),
  createSession: (classId: number) => post<{ id: number }>('/training/sessions', { class_id: classId }),
  addSample: (sessionId: number, landmarks: number[]) => post('/training/samples', { session_id: sessionId, kind: 'static', landmarks }),
  startJob: () => post<TrainingJob>('/training/jobs', { sample_kind: 'static', activate_on_success: true }),
  job: (id: string) => call<TrainingJob>(`/training/jobs/${id}`),
  trainingModels: () => call<TrainingModel[]>('/training/models'),
  predict: (landmarks: number[]) => post<Prediction>('/training/predict', { kind: 'static', landmarks }),
  exportUrl: '/api/v1/data/export',
  importData: (file: File) => { const form = new FormData(); form.append('file', file); return call<Record<string, unknown>>('/data/import', { method: 'POST', body: form }) },
  deleteAll: () => call('/data?confirm=DELETE_ALL_USER_DATA', { method: 'DELETE' }),
}
