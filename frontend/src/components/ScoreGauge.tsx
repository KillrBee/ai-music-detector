import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion } from 'lucide-react'

interface ScoreGaugeProps {
  score: number           // 0-100
  confidence: string      // "low" | "medium" | "high"
  likelihood: string      // "unlikely" | "possible" | "likely" | "unknown"
}

export default function ScoreGauge({ score, confidence, likelihood }: ScoreGaugeProps) {
  const radius = 80
  const stroke = 10
  const normalizedRadius = radius - stroke / 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (score / 100) * circumference

  // Color based on score
  const getScoreColor = () => {
    if (score >= 65) return { ring: '#ef4444', text: 'text-red-400', bg: 'from-red-500/20 to-red-600/5' }
    if (score >= 35) return { ring: '#eab308', text: 'text-yellow-400', bg: 'from-yellow-500/20 to-yellow-600/5' }
    return { ring: '#22c55e', text: 'text-green-400', bg: 'from-green-500/20 to-green-600/5' }
  }

  const colors = getScoreColor()

  const getLikelihoodInfo = () => {
    switch (likelihood) {
      case 'likely':
        return {
          label: 'Likely AI-Generated',
          Icon: ShieldAlert,
          color: 'text-red-400',
          bg: 'bg-red-500/10 border-red-500/30',
        }
      case 'possible':
        return {
          label: 'Possibly AI-Generated',
          Icon: Shield,
          color: 'text-yellow-400',
          bg: 'bg-yellow-500/10 border-yellow-500/30',
        }
      case 'unlikely':
        return {
          label: 'Unlikely AI-Generated',
          Icon: ShieldCheck,
          color: 'text-green-400',
          bg: 'bg-green-500/10 border-green-500/30',
        }
      default:
        return {
          label: 'Unable to Determine',
          Icon: ShieldQuestion,
          color: 'text-gray-400',
          bg: 'bg-gray-500/10 border-gray-500/30',
        }
    }
  }

  const info = getLikelihoodInfo()
  const Icon = info.Icon

  const confidenceBadge = () => {
    const styles: Record<string, string> = {
      high: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      low: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    }
    return styles[confidence] || styles.low
  }

  return (
    <div className={`bg-gradient-to-br ${colors.bg} rounded-2xl border border-gray-700/50 p-8 flex flex-col items-center gap-6`}>
      {/* SVG Gauge */}
      <div className="relative">
        <svg height={radius * 2} width={radius * 2} className="-rotate-90">
          {/* Background ring */}
          <circle
            stroke="rgba(75, 85, 99, 0.3)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          {/* Score ring */}
          <circle
            stroke={colors.ring}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            style={{ strokeDashoffset, transition: 'stroke-dashoffset 1s ease-in-out' }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        {/* Score text centered in the ring */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-bold ${colors.text}`}>
            {score.toFixed(0)}%
          </span>
          <span className="text-xs text-gray-500 uppercase tracking-wider mt-1">AI Score</span>
        </div>
      </div>

      {/* Likelihood badge */}
      <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${info.bg}`}>
        <Icon className={`w-5 h-5 ${info.color}`} />
        <span className={`font-semibold ${info.color}`}>{info.label}</span>
      </div>

      {/* Confidence badge */}
      <div className={`text-xs px-3 py-1 rounded-full border ${confidenceBadge()}`}>
        {confidence.charAt(0).toUpperCase() + confidence.slice(1)} Confidence
      </div>
    </div>
  )
}
