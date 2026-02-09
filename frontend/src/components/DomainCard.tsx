import { useState } from 'react'
import { ChevronDown, ChevronRight, Activity, Radio, Sliders, Clock, Layers, Mic, Fingerprint } from 'lucide-react'
import type { DomainResult, AIArtifact } from '../services/api'

interface DomainCardProps {
  domain: DomainResult
}

const DOMAIN_ICONS: Record<string, typeof Activity> = {
  spectral: Activity,
  spatial: Radio,
  production: Sliders,
  temporal: Clock,
  structural: Layers,
  vocal: Mic,
  watermark: Fingerprint,
}

const DOMAIN_DESCRIPTIONS: Record<string, string> = {
  spectral: 'Frequency-domain analysis of neural synthesis artifacts',
  spatial: 'Stereo imaging and phase coherence analysis',
  production: 'Mixing and mastering quality metrics',
  temporal: 'Transient sharpness and rhythmic integrity',
  structural: 'Musical form and compositional coherence',
  vocal: 'Vocal synthesis artifact detection',
  watermark: 'AI provenance watermark detection',
}

const TIER_LABELS: Record<number, string> = {
  1: 'Definitive',
  2: 'Strong',
  3: 'Moderate',
  4: 'Weak',
}

const TIER_STYLES: Record<number, string> = {
  1: 'bg-purple-500/20 text-purple-400 border-purple-500/40',
  2: 'bg-orange-500/20 text-orange-400 border-orange-500/40',
  3: 'bg-gray-500/15 text-gray-400 border-gray-600/40',
  4: 'bg-gray-500/10 text-gray-600 border-gray-700/40',
}

function getScoreColor(score: number) {
  if (score >= 0.65) return 'bg-red-500'
  if (score >= 0.35) return 'bg-yellow-500'
  return 'bg-green-500'
}

function getScoreTextColor(score: number) {
  if (score >= 0.65) return 'text-red-400'
  if (score >= 0.35) return 'text-yellow-400'
  return 'text-green-400'
}

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, string> = {
    high: 'bg-red-500/20 text-red-400 border-red-500/40',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
    none: 'bg-gray-500/15 text-gray-500 border-gray-600/40',
  }
  return (
    <span className={`text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded border ${styles[severity] || styles.none}`}>
      {severity}
    </span>
  )
}

function TierBadge({ tier }: { tier: number }) {
  const label = TIER_LABELS[tier] || 'Unknown'
  const style = TIER_STYLES[tier] || TIER_STYLES[3]
  return (
    <span className={`text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded border ${style}`}>
      {label}
    </span>
  )
}

function ProbabilityBar({ probability }: { probability: number }) {
  const pct = Math.round(probability * 100)
  const color = probability >= 0.65 ? 'bg-red-500' : probability >= 0.35 ? 'bg-yellow-500' : 'bg-green-500'

  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-1.5 bg-gray-700/60 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%`, transition: 'width 0.6s ease' }}
        />
      </div>
      <span className="text-[11px] text-gray-500 font-mono w-8 text-right">{pct}%</span>
    </div>
  )
}

function ArtifactRow({ artifact }: { artifact: AIArtifact }) {
  const formatName = (name: string) =>
    name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className={`px-4 py-3 border-b border-gray-800/60 last:border-b-0 ${artifact.detected ? 'bg-gray-800/20' : ''}`}>
      <div className="flex items-center justify-between gap-3 mb-1">
        <div className="flex items-center gap-2 min-w-0">
          <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${artifact.detected ? 'bg-red-400' : 'bg-gray-600'}`} />
          <span className="text-sm font-medium text-gray-200 truncate">{formatName(artifact.name)}</span>
          <TierBadge tier={artifact.tier} />
          <SeverityBadge severity={artifact.severity} />
        </div>
        <ProbabilityBar probability={artifact.probability} />
      </div>
      <p className="text-xs text-gray-500 ml-4 leading-relaxed">{artifact.description}</p>
      {artifact.value !== null && artifact.value !== undefined && (
        <div className="text-[11px] text-gray-600 ml-4 mt-1 font-mono">
          Measured: {typeof artifact.value === 'number' ? artifact.value.toLocaleString() : artifact.value}
        </div>
      )}
    </div>
  )
}

export default function DomainCard({ domain }: DomainCardProps) {
  const [expanded, setExpanded] = useState(domain.active && domain.score > 0.2)

  const Icon = DOMAIN_ICONS[domain.domain] || Activity
  const description = DOMAIN_DESCRIPTIONS[domain.domain] || ''
  const scorePct = Math.round(domain.score * 100)
  const detectedCount = domain.artifacts.filter(a => a.detected).length

  if (!domain.active) {
    return (
      <div className="bg-gray-800/30 rounded-xl border border-gray-700/30 p-4 opacity-60">
        <div className="flex items-center gap-3">
          <Icon className="w-5 h-5 text-gray-600" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-gray-500">{domain.display_name}</h3>
            <p className="text-xs text-gray-600 mt-0.5">Inactive — requires deeper analysis or not applicable</p>
          </div>
          <span className="text-xs text-gray-600 bg-gray-800/50 px-2 py-1 rounded">Skipped</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800/40 rounded-xl border border-gray-700/40 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3.5 flex items-center gap-3 hover:bg-gray-800/60 transition-colors"
      >
        <Icon className={`w-5 h-5 ${getScoreTextColor(domain.score)}`} />

        <div className="flex-1 text-left">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-200">{domain.display_name}</h3>
            {detectedCount > 0 && (
              <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded-full">
                {detectedCount} detected
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        </div>

        {/* Score bar */}
        <div className="flex items-center gap-3 mr-2">
          <div className="w-24 h-2 bg-gray-700/50 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${getScoreColor(domain.score)}`}
              style={{ width: `${scorePct}%`, transition: 'width 0.8s ease' }}
            />
          </div>
          <span className={`text-sm font-bold min-w-[36px] text-right ${getScoreTextColor(domain.score)}`}>
            {scorePct}%
          </span>
        </div>

        {expanded ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </button>

      {/* Artifact rows */}
      {expanded && domain.artifacts.length > 0 && (
        <div className="border-t border-gray-700/40">
          {domain.artifacts.map((artifact) => (
            <ArtifactRow key={artifact.name} artifact={artifact} />
          ))}
        </div>
      )}
    </div>
  )
}
