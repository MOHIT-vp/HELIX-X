const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api/v1'

export type Entity = { id: number; name: string; type: string }
export type Status = { service: string; status: 'starting' | 'ready' | 'degraded' | 'unavailable'; mode: 'production' | 'demo' | 'unavailable'; graph_source: 'hetionet' | 'synthetic_demo'; graph_label: string; model_loaded: boolean; checkpoint: string; embeddings_cached: boolean; inference_ready: boolean; node_count: number; edge_count: number; feature_dimension: number; device: string; api_integrations: string[]; startup_error?: string | null }
export type EvidenceNode = { id: string; index: number; name: string; type: string; path_positions: number[] }
export type EvidenceEdge = { id: string; source: string; target: string; relationship: string; path_index: number }
export type EvidencePath = { id: string; hops: number; metapath: string; node_ids: string[]; nodes: EvidenceNode[] }
export type EvidenceGraph = { nodes: EvidenceNode[]; edges: EvidenceEdge[]; paths: EvidencePath[] }
export type Prediction = { compound: Entity; disease: Entity; score: number; score_type: 'model_score'; evidence_path_count: number; evidence_graph: EvidenceGraph; scientific_notice: string }
export type Candidate = { rank: number; entity: Entity; score: number; score_type: 'model_score'; evidence_path_count: number }
export type CandidateResponse = { source: Entity; direction: 'diseases' | 'compounds'; candidates: Candidate[]; scientific_notice: string }
export type Methodology = { stages: { title: string; detail: string }[]; model: Record<string,string>; explanation: Record<string,string>; caveats: string[] }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) { const body = await response.json().catch(() => ({})); const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message; throw new Error(body.message || detail || `Request failed: ${response.status}`) }
  return response.json()
}

export const api = {
  status: () => request<Status>('/status'),
  compounds: () => request<Entity[]>('/entities/compounds'),
  diseases: () => request<Entity[]>('/entities/diseases'),
  predict: (compound_id: number, disease_id: number) => request<Prediction>('/predict', { method: 'POST', body: JSON.stringify({ compound_id, disease_id }) }),
  candidates: (direction: 'diseases' | 'compounds', entity_id: number, limit: number) => request<CandidateResponse>(`/candidates/${direction}`, { method: 'POST', body: JSON.stringify({ entity_id, limit }) }),
  evidence: (compound_id: number, disease_id: number) => request<EvidenceGraph>('/evidence/graph', { method: 'POST', body: JSON.stringify({ compound_id, disease_id, max_paths: 8 }) }),
  methodology: () => request<Methodology>('/methodology'),
}
