import { useState } from 'react'
import { AudioLines, Loader2, AlertCircle } from 'lucide-react'
import FileUploader from './components/FileUploader'
import DepthSelector from './components/DepthSelector'
import AnalysisResults from './components/AnalysisResults'
import { analyzeAudio, type AnalysisDepth, type AnalysisResult } from './services/api'

type AppState = 'idle' | 'analyzing' | 'results' | 'error'

export default function App() {
  const [state, setState] = useState<AppState>('idle')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [depth, setDepth] = useState<AnalysisDepth>('standard')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string>('')
  const [progress, setProgress] = useState<string>('')

  const handleFileSelected = (file: File) => {
    setSelectedFile(file)
    setError('')
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setState('analyzing')
    setError('')
    setResult(null)

    // Set progress message based on depth
    const depthMessages: Record<AnalysisDepth, string> = {
      quick: 'Running spectral, spatial, and production checks...',
      standard: 'Running full analysis including temporal checks...',
      deep: 'Running deep analysis including structural, vocal, and watermark detection...',
    }
    setProgress(depthMessages[depth])

    try {
      const analysisResult = await analyzeAudio(selectedFile, depth)
      setResult(analysisResult)
      setState('results')
    } catch (err: unknown) {
      let message = 'An unexpected error occurred'

      if (err instanceof Error) {
        if (err.message.includes('Network Error')) {
          message = 'Cannot connect to the backend server. Make sure it is running on port 8000.'
        } else if (err.message.includes('timeout')) {
          message = 'Analysis timed out. Try a shorter audio file or a quicker analysis depth.'
        } else {
          message = err.message
        }
      }

      // Check for Axios error response
      if (typeof err === 'object' && err !== null && 'response' in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } }
        if (axiosErr.response?.data?.detail) {
          message = axiosErr.response.data.detail
        }
      }

      setError(message)
      setState('error')
    }
  }

  const handleReset = () => {
    setState('idle')
    setSelectedFile(null)
    setResult(null)
    setError('')
    setProgress('')
  }

  return (
    <div className="min-h-screen bg-grid-pattern">
      {/* Header */}
      <header className="border-b border-gray-800/60 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <AudioLines className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-100">AI Audio Detector</h1>
            <p className="text-[11px] text-gray-500 -mt-0.5">Signal-level AI generation analysis</p>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Results view */}
        {state === 'results' && result && (
          <AnalysisResults result={result} onReset={handleReset} />
        )}

        {/* Analyzing view */}
        {state === 'analyzing' && (
          <div className="flex flex-col items-center justify-center py-24 gap-6">
            <div className="relative">
              <div className="w-20 h-20 rounded-full border-2 border-purple-500/30 flex items-center justify-center">
                <Loader2 className="w-10 h-10 text-purple-400 animate-spin" />
              </div>
              <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-purple-400/20 animate-spin-slow" />
            </div>
            <div className="text-center">
              <h2 className="text-xl font-semibold text-gray-200 mb-2">Analyzing Audio</h2>
              <p className="text-sm text-gray-500 max-w-md">{progress}</p>
              <p className="text-xs text-gray-600 mt-3">
                {depth === 'quick' ? '~5 seconds' : depth === 'standard' ? '~15 seconds' : '~45 seconds'}
              </p>
            </div>
          </div>
        )}

        {/* Idle / Error view — upload form */}
        {(state === 'idle' || state === 'error') && (
          <div className="space-y-6">
            {/* Intro text */}
            <div className="text-center py-4">
              <h2 className="text-2xl font-bold text-gray-100 mb-2">
                Detect AI-Generated Audio
              </h2>
              <p className="text-sm text-gray-500 max-w-xl mx-auto">
                Upload an audio file to analyze it for AI generation artifacts across 7 detection
                domains: spectral, spatial, temporal, structural, production, vocal, and watermark.
              </p>
            </div>

            {/* File uploader */}
            <FileUploader
              onFileSelected={handleFileSelected}
              selectedFile={selectedFile}
              onClear={handleClearFile}
              disabled={false}
            />

            {/* Depth selector + Analyze button */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <label className="block text-xs text-gray-500 uppercase tracking-wider font-semibold mb-2">
                  Analysis Depth
                </label>
                <DepthSelector value={depth} onChange={setDepth} />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!selectedFile}
                className={`
                  px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200
                  flex items-center gap-2
                  ${selectedFile
                    ? 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/20 hover:shadow-purple-500/30'
                    : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }
                `}
              >
                <AudioLines className="w-4 h-4" />
                Analyze Audio
              </button>
            </div>

            {/* Error display */}
            {state === 'error' && error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-red-400">Analysis Failed</h3>
                  <p className="text-sm text-red-300/70 mt-1">{error}</p>
                  <button
                    onClick={() => { setState('idle'); setError('') }}
                    className="text-xs text-red-400 hover:text-red-300 mt-2 underline underline-offset-2"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* Domain info */}
            <div className="mt-8 pt-6 border-t border-gray-800/50">
              <h3 className="text-xs text-gray-600 uppercase tracking-wider font-semibold mb-4 text-center">
                Detection Domains
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                {[
                  { name: 'Spectral', desc: 'Frequency artifacts', depth: 'quick' },
                  { name: 'Spatial', desc: 'Stereo imaging', depth: 'quick' },
                  { name: 'Production', desc: 'Dynamics & reverb', depth: 'quick' },
                  { name: 'Temporal', desc: 'Transients & rhythm', depth: 'standard' },
                  { name: 'Structural', desc: 'Form & entropy', depth: 'deep' },
                  { name: 'Vocal', desc: 'Formants & breath', depth: 'deep' },
                  { name: 'Watermark', desc: 'AudioSeal check', depth: 'deep' },
                ].map((d) => {
                  const isActive =
                    depth === 'deep' ||
                    (depth === 'standard' && d.depth !== 'deep') ||
                    (depth === 'quick' && d.depth === 'quick')
                  return (
                    <div
                      key={d.name}
                      className={`rounded-lg px-3 py-2.5 text-center border transition-all duration-300 ${
                        isActive
                          ? 'bg-gray-800/50 border-gray-700/50 text-gray-300'
                          : 'bg-gray-900/30 border-gray-800/30 text-gray-700'
                      }`}
                    >
                      <p className={`text-xs font-semibold ${isActive ? 'text-gray-200' : 'text-gray-600'}`}>
                        {d.name}
                      </p>
                      <p className="text-[10px] mt-0.5">{d.desc}</p>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800/40 mt-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between text-xs text-gray-700">
          <span>AI Audio Detector v0.1.0</span>
          <span>17 checks &middot; 7 domains &middot; Signal-level analysis</span>
        </div>
      </footer>
    </div>
  )
}
