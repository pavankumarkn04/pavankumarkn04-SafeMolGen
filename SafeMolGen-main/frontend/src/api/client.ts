const API_BASE = '/api/v1'

export type GenerateResultItem = {
  smiles: string
  valid: boolean
  passed_filters: boolean
  logp: number | null
  mw: number | null
  hbd: number | null
  hba: number | null
  tpsa: number | null
  qed: number | null
}

export type GenerateResponse = {
  results: GenerateResultItem[]
  summary: { total: number; valid: number; passed_filters: number }
}

export type GenerateParams = {
  n?: number
  temperature?: number
  top_k?: number
  logp_min?: number
  logp_max?: number
  mw_min?: number
  mw_max?: number
  hbd_max?: number
  hba_max?: number
  tpsa_max?: number
  qed_min?: number
  /** Conditioning strength 0.5–0.75 (only if generator supports it). Higher = more drug-like bias. */
  target_phase?: number
}

export async function generateMolecules(params: GenerateParams): Promise<GenerateResponse> {
  const r = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    throw new Error(msg)
  }
  return r.json()
}

export function moleculeSvgUrl(smiles: string, width = 240, height = 180): string {
  const enc = encodeURIComponent(smiles)
  return `${API_BASE}/molecule/svg?smiles=${enc}&width=${width}&height=${height}`
}

export type HealthResponse = {
  status: string
  generator_loaded: boolean
  conditioning_available?: boolean
  models_loaded?: boolean
}

export async function getHealth(): Promise<HealthResponse> {
  const r = await fetch(`${API_BASE}/health`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export type OraclePrediction = {
  phase1_prob: number
  phase2_prob: number
  phase3_prob: number
  overall_prob: number
  admet_predictions: Record<string, number>
  risk_factors: Array<{ name?: string; category?: string; description?: string; impact?: number; source?: string }>
  structural_alerts: string[]
  recommendations: Array<{ type?: string; issue?: string; suggestion?: string; severity?: string; expected_improvement?: string }>
}

export type DesignResult = {
  final_smiles: string
  canonical_smiles?: string
  final_phase1: number
  final_phase2: number
  final_phase3: number
  final_overall: number
  target_achieved: boolean
  total_iterations: number
  history: Array<{
    iteration: number
    smiles: string
    phase1_prob: number
    phase2_prob: number
    phase3_prob: number
    overall_prob: number
    improvements: string[]
    structural_alerts: string[]
    passed_safety: boolean
    used_oracle_feedback: boolean
    recommendations?: Array<{ type?: string; issue?: string; suggestion?: string; severity?: string; expected_improvement?: string }>
  }>
  recommendations?: Array<{ type?: string; issue?: string; suggestion?: string; severity?: string; expected_improvement?: string }>
  _strategy_used?: string
}

export type DesignParams = {
  target_success?: number
  max_iterations?: number
  candidates_per_iteration?: number
  top_k?: number
  property_targets?: Record<string, number | [number, number]>
  seed_smiles?: string
  use_rl_model?: boolean
  design_mode?: 'single' | 'restarts' | 'evolutionary'
  n_restarts?: number
  population_size?: number
  generations?: number
  use_phase_aware_steering?: boolean
  first_iteration_temperature?: number
  use_improvement_pacing?: boolean
  max_step_per_iteration?: number
  safety_threshold?: number
  require_no_structural_alerts?: boolean
  use_oracle_feedback?: boolean
  selection_mode?: string
  diversity_tanimoto_max?: number
  exploration_fraction?: number
  max_rotatable_bonds?: number
}

export type ConfigResponse = {
  max_iterations_min: number
  max_iterations_max: number
  target_success_min: number
  target_success_max: number
  top_k_min?: number
  top_k_max?: number
  first_iteration_temperature_default?: number
  generator_early_available?: boolean
  default_property_targets?: Record<string, number | [number, number]>
  selection_modes?: string[]
  diversity_tanimoto_max_default?: number
  exploration_fraction_default?: number
  max_rotatable_bonds_default?: number
}

export async function getConfig(): Promise<ConfigResponse> {
  const r = await fetch(`${API_BASE}/config`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

function normalizeDesignResult(raw: any): DesignResult {
  if (raw && typeof raw.final_phase1 === 'number') {
    return raw as DesignResult
  }
  const pred = raw?.final_prediction ?? {}
  const histSrc = Array.isArray(raw?.iteration_history) ? raw.iteration_history : []
  const history = histSrc.map((h: any) => {
    if (typeof h?.phase1_prob === 'number') return h
    const hp = h?.prediction ?? {}
    return {
      iteration: h?.iteration ?? 0,
      smiles: h?.smiles ?? '',
      phase1_prob: hp?.phase1_prob ?? 0,
      phase2_prob: hp?.phase2_prob ?? 0,
      phase3_prob: hp?.phase3_prob ?? 0,
      overall_prob: hp?.overall_prob ?? 0,
      improvements: Array.isArray(h?.improvements) ? h.improvements : [],
      structural_alerts: Array.isArray(hp?.structural_alerts) ? hp.structural_alerts : [],
      passed_safety: !!h?.passed_safety,
      used_oracle_feedback: !!h?.used_oracle_feedback,
      recommendations: Array.isArray(hp?.recommendations) ? hp.recommendations : (Array.isArray(h?.recommendations) ? h.recommendations : []),
    }
  })
  return {
    final_smiles: raw?.final_smiles ?? '',
    canonical_smiles: raw?.canonical_smiles,
    final_phase1: pred?.phase1_prob ?? 0,
    final_phase2: pred?.phase2_prob ?? 0,
    final_phase3: pred?.phase3_prob ?? 0,
    final_overall: pred?.overall_prob ?? 0,
    target_achieved: !!raw?.target_achieved,
    total_iterations: raw?.total_iterations ?? history.length,
    history,
    recommendations: raw?.recommendations ?? pred?.recommendations ?? [],
    _strategy_used: raw?._strategy_used,
  }
}

export async function designSync(params: DesignParams): Promise<DesignResult> {
  const r = await fetch(`${API_BASE}/design`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    throw new Error(msg)
  }
  return normalizeDesignResult(await r.json())
}

export async function designStream(
  params: DesignParams,
  onIteration: (data: DesignResult) => void,
  onDone: (data: DesignResult) => void,
  onError: (err: string) => void,
  options?: { signal?: AbortSignal; onStarted?: () => void }
): Promise<void> {
  let r: Response
  try {
    r = await fetch(`${API_BASE}/design/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: options?.signal,
    })
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      onError('Cancelled')
      return
    }
    throw err
  }
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    onError(msg)
    return
  }
  const reader = r.body?.getReader()
  if (!reader) {
    onError('No response body')
    return
  }
  const decoder = new TextDecoder()
  let buf = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const events = buf.split('\n\n')
      buf = events.pop() ?? ''
      for (const block of events) {
        const line = block.split('\n').find((l) => l.startsWith('data: '))
        if (!line) continue
        const jsonStr = line.slice(6)
        try {
          const payload = JSON.parse(jsonStr) as { event: string; data: any }
          if (payload.event === 'started') options?.onStarted?.()
          else if (payload.event === 'iteration') onIteration(normalizeDesignResult(payload.data))
          else if (payload.event === 'done') onDone(normalizeDesignResult(payload.data))
          else if (payload.event === 'error') onError((payload.data as { detail?: string }).detail ?? 'Unknown error')
        } catch {}
      }
    }
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      onError('Cancelled')
      return
    }
    throw err
  } finally {
    reader.releaseLock()
  }
}

