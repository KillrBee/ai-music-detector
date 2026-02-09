import { Clock, Radio, Waves, Volume2 } from 'lucide-react'
import type { AnalysisResult } from '../services/api'

interface AudioInfoProps {
  result: AnalysisResult
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatSampleRate(sr: number): string {
  return `${(sr / 1000).toFixed(1)} kHz`
}

export default function AudioInfo({ result }: AudioInfoProps) {
  const items = [
    { Icon: Clock, label: 'Duration', value: formatDuration(result.duration_seconds) },
    { Icon: Waves, label: 'Sample Rate', value: formatSampleRate(result.sample_rate) },
    { Icon: Radio, label: 'Channels', value: result.channels === 2 ? 'Stereo' : 'Mono' },
    { Icon: Volume2, label: 'Peak / RMS', value: `${result.peak_db} / ${result.rms_db} dB` },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {items.map(({ Icon, label, value }) => (
        <div key={label} className="bg-gray-800/40 rounded-lg border border-gray-700/30 px-3 py-2.5">
          <div className="flex items-center gap-2 mb-1">
            <Icon className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-[11px] text-gray-500 uppercase tracking-wider">{label}</span>
          </div>
          <p className="text-sm font-semibold text-gray-200">{value}</p>
        </div>
      ))}
    </div>
  )
}
