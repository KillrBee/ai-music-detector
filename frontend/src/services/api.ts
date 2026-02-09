import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

// --- Types ---

export type AnalysisDepth = 'quick' | 'standard' | 'deep'

export interface AIArtifact {
  name: string
  detected: boolean
  severity: 'none' | 'low' | 'medium' | 'high'
  value: number | null
  description: string
  probability: number
  domain: string
  weight: number
  tier: 1 | 2 | 3 | 4 // 1=definitive, 2=strong, 3=moderate, 4=weak
}

export interface DomainResult {
  domain: string
  display_name: string
  score: number
  active: boolean
  weight: number
  artifacts: AIArtifact[]
}

export interface AnalysisResult {
  // Scan metadata
  scan_id: string
  analyzed_at: string
  tool_version: string

  // Audio properties
  filename: string
  duration_seconds: number
  sample_rate: number
  channels: number
  peak_db: number
  rms_db: number

  // AI detection scoring
  overall_score: number
  confidence: 'low' | 'medium' | 'high'
  confidence_value: number
  depth_used: string

  // Domain-grouped results
  domain_results: DomainResult[]

  // Flat artifact list
  ai_artifacts: AIArtifact[]
  overall_ai_likelihood: 'unknown' | 'unlikely' | 'possible' | 'likely'

  // Legacy quick-access fields
  high_freq_cutoff_hz: number | null
  stereo_correlation: number | null
}

// --- API Methods ---

export async function analyzeAudio(
  file: File,
  depth: AnalysisDepth = 'standard'
): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await api.post<AnalysisResult>(
    `/analyze?depth=${depth}`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 min timeout for deep analysis
    }
  )
  return data
}

export async function healthCheck(): Promise<{ status: string }> {
  const { data } = await api.get('/health')
  return data
}
