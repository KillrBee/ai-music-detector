import { Zap, Settings, Search } from 'lucide-react'
import type { AnalysisDepth } from '../services/api'

interface DepthSelectorProps {
  value: AnalysisDepth
  onChange: (depth: AnalysisDepth) => void
  disabled?: boolean
}

const DEPTH_OPTIONS: {
  value: AnalysisDepth
  label: string
  time: string
  description: string
  Icon: typeof Zap
}[] = [
  {
    value: 'quick',
    label: 'Quick',
    time: '~5s',
    description: 'Spectral, spatial, and basic production checks',
    Icon: Zap,
  },
  {
    value: 'standard',
    label: 'Standard',
    time: '~15s',
    description: 'Adds temporal analysis and reverb tail detection',
    Icon: Settings,
  },
  {
    value: 'deep',
    label: 'Deep',
    time: '~45s',
    description: 'Full analysis including structure, vocals, and watermark',
    Icon: Search,
  },
]

export default function DepthSelector({ value, onChange, disabled }: DepthSelectorProps) {
  return (
    <div className="flex gap-2">
      {DEPTH_OPTIONS.map((opt) => {
        const isSelected = value === opt.value
        const Icon = opt.Icon
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            disabled={disabled}
            title={opt.description}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium
              transition-all duration-200
              ${
                isSelected
                  ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                  : 'bg-gray-800/40 border-gray-700/40 text-gray-400 hover:bg-gray-800/60 hover:text-gray-300'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <Icon className="w-4 h-4" />
            <span>{opt.label}</span>
            <span className={`text-xs ${isSelected ? 'text-purple-400/70' : 'text-gray-600'}`}>
              {opt.time}
            </span>
          </button>
        )
      })}
    </div>
  )
}
