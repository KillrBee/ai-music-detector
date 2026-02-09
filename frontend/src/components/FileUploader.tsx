import { useState, useCallback, useRef } from 'react'
import { Upload, FileAudio, X } from 'lucide-react'

interface FileUploaderProps {
  onFileSelected: (file: File) => void
  selectedFile: File | null
  onClear: () => void
  disabled?: boolean
}

const SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
const ACCEPT_STRING = SUPPORTED_FORMATS.map(ext => `audio/${ext.slice(1)}`).join(',') + ',.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma'

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileUploader({ onFileSelected, selectedFile, onClear, disabled }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const validateFile = useCallback((file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    return SUPPORTED_FORMATS.includes(ext)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      if (disabled) return
      const file = e.dataTransfer.files[0]
      if (file && validateFile(file)) {
        onFileSelected(file)
      }
    },
    [disabled, onFileSelected, validateFile]
  )

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file && validateFile(file)) {
        onFileSelected(file)
      }
      // Reset so the same file can be selected again
      e.target.value = ''
    },
    [onFileSelected, validateFile]
  )

  if (selectedFile) {
    return (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-4 flex items-center gap-4">
        <div className="w-12 h-12 rounded-lg bg-purple-500/15 flex items-center justify-center">
          <FileAudio className="w-6 h-6 text-purple-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-200 truncate">{selectedFile.name}</p>
          <p className="text-xs text-gray-500">{formatFileSize(selectedFile.size)}</p>
        </div>
        {!disabled && (
          <button
            onClick={onClear}
            className="p-1.5 rounded-lg hover:bg-gray-700/50 text-gray-500 hover:text-gray-300 transition"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`
        rounded-xl border-2 border-dashed p-10 text-center cursor-pointer
        transition-all duration-200
        ${isDragging
          ? 'border-purple-400 bg-purple-500/10'
          : 'border-gray-700/50 hover:border-gray-600 hover:bg-gray-800/30'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_STRING}
        onChange={handleChange}
        className="hidden"
      />
      <Upload className={`w-10 h-10 mx-auto mb-3 ${isDragging ? 'text-purple-400' : 'text-gray-600'}`} />
      <p className="text-sm text-gray-300 font-medium">
        {isDragging ? 'Drop your audio file here' : 'Drag & drop an audio file, or click to browse'}
      </p>
      <p className="text-xs text-gray-600 mt-2">
        Supports {SUPPORTED_FORMATS.map(f => f.slice(1).toUpperCase()).join(', ')}
      </p>
    </div>
  )
}
