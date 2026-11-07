export type Prediction = { label: string | null; candidate: string; confidence: number; accepted: boolean; message: string }
export type GestureClass = { id: number; name: string; description: string; is_unknown: boolean; session_count: number; sample_count: number }
export type CameraFrame = { landmarks: number[] | null; hands: number; fps: number; latency: number; image?: string }

