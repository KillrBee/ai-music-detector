import { FileAudio, RotateCcw, Download } from 'lucide-react'
import type { AnalysisResult } from '../services/api'
import ScoreGauge from './ScoreGauge'
import DomainCard from './DomainCard'
import AudioInfo from './AudioInfo'

interface AnalysisResultsProps {
  result: AnalysisResult
  onReset: () => void
}

export default function AnalysisResults({ result, onReset }: AnalysisResultsProps) {
  const activeDomains = result.domain_results.filter(d => d.active)
  const inactiveDomains = result.domain_results.filter(d => !d.active)

  // Sort active domains by score descending so most suspicious appear first
  const sortedActive = [...activeDomains].sort((a, b) => b.score - a.score)

  const handleExportJSON = () => {
    const exportData = {
      ...result,
      export_metadata: {
        exported_at: new Date().toISOString(),
        export_format_version: '1.0',
      },
    }

    const jsonString = JSON.stringify(exportData, null, 2)
    const blob = new Blob([jsonString], { type: 'application/json' })
    const url = URL.createObjectURL(blob)

    // Generate filename: originalname_analysis_timestamp.json
    const baseName = result.filename.replace(/\.[^/.]+$/, '')
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    const exportFilename = `${baseName}_analysis_${timestamp}.json`

    const link = document.createElement('a')
    link.href = url
    link.download = exportFilename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      {/* Header with filename, export, and reset */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-lg bg-purple-500/15 flex items-center justify-center flex-shrink-0">
            <FileAudio className="w-5 h-5 text-purple-400" />
          </div>
          <div className="min-w-0">
            <h2 className="text-lg font-semibold text-gray-100 truncate">{result.filename}</h2>
            <p className="text-xs text-gray-500">
              Analyzed at <span className="text-gray-400 font-medium">{result.depth_used}</span> depth
              {result.scan_id && (
                <span className="ml-2 text-gray-600" title={`Scan ID: ${result.scan_id}`}>
                  {result.scan_id.slice(0, 8)}
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800/50 border border-gray-700/40 text-gray-400 hover:text-gray-200 hover:bg-gray-800/80 transition text-sm"
            title="Export scan results as JSON"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800/50 border border-gray-700/40 text-gray-400 hover:text-gray-200 hover:bg-gray-800/80 transition text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Analyze Another
          </button>
        </div>
      </div>

      {/* Audio properties */}
      <AudioInfo result={result} />

      {/* Score gauge + domain cards layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Score gauge - left column */}
        <div className="lg:col-span-1">
          <ScoreGauge
            score={result.overall_score}
            confidence={result.confidence}
            likelihood={result.overall_ai_likelihood}
          />

          {/* Quick stats under gauge */}
          <div className="mt-4 bg-gray-800/30 rounded-xl border border-gray-700/30 p-4 space-y-3">
            <h4 className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Detection Summary</h4>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Domains analyzed</span>
              <span className="text-gray-200 font-medium">{activeDomains.length} / {result.domain_results.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Artifacts detected</span>
              <span className="text-gray-200 font-medium">
                {result.ai_artifacts.filter(a => a.detected).length} / {result.ai_artifacts.length}
              </span>
            </div>
            {result.high_freq_cutoff_hz && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">HF cutoff</span>
                <span className="text-gray-200 font-medium">
                  {(result.high_freq_cutoff_hz / 1000).toFixed(1)} kHz
                </span>
              </div>
            )}
            {result.stereo_correlation !== null && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Stereo correlation</span>
                <span className="text-gray-200 font-medium">{result.stereo_correlation?.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>

        {/* Domain cards - right two columns */}
        <div className="lg:col-span-2 space-y-3">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Domain Analysis</h3>
          {sortedActive.map((domain) => (
            <DomainCard key={domain.domain} domain={domain} />
          ))}
          {inactiveDomains.length > 0 && (
            <>
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mt-4">Inactive Domains</h3>
              {inactiveDomains.map((domain) => (
                <DomainCard key={domain.domain} domain={domain} />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
