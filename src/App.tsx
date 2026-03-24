import { useState, useRef, useCallback } from 'react'
import './App.css'

type Status = 'idle' | 'dragging' | 'loading' | 'done' | 'error'

const API = '/predict'

export default function App() {
  const [status, setStatus]       = useState<Status>('idle')
  const [original, setOriginal]   = useState<string | null>(null)
  const [result, setResult]       = useState<string | null>(null)
  const [errorMsg, setErrorMsg]   = useState('')
  const inputRef                  = useRef<HTMLInputElement>(null)

  const process = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload an image file.')
      setStatus('error')
      return
    }

    setOriginal(URL.createObjectURL(file))
    setResult(null)
    setStatus('loading')

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(API, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const blob = await res.blob()
      setResult(URL.createObjectURL(blob))
      setStatus('done')
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error')
      setStatus('error')
    }
  }, [])

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) process(file)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setStatus('idle')
    const file = e.dataTransfer.files[0]
    if (file) process(file)
  }

  const reset = () => {
    setStatus('idle')
    setOriginal(null)
    setResult(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <main>
      <header>
        <h1>Satellite Segmentation</h1>
        <p>Upload a satellite image to generate a segmentation mask</p>
      </header>

      {/* Drop zone — shown when idle or after error */}
      {(status === 'idle' || status === 'dragging' || status === 'error') && (
        <div
          className={`dropzone${status === 'dragging' ? ' dragging' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setStatus('dragging') }}
          onDragLeave={() => setStatus('idle')}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={onFile}
          />
          <UploadIcon />
          <span>Drop an image here, or <u>browse</u></span>
          <small>PNG, JPG, TIF supported</small>
          {status === 'error' && <p className="error-msg">{errorMsg}</p>}
        </div>
      )}

      {/* Loading */}
      {status === 'loading' && (
        <div className="state-box">
          <Spinner />
          <p>Running segmentation…</p>
        </div>
      )}

      {/* Results */}
      {(status === 'done' || (status === 'loading' && original)) && (
        <div className="results">
          <div className="panel">
            <label>Original</label>
            {original && <img src={original} alt="Original satellite image" />}
          </div>
          <div className="divider" />
          <div className="panel">
            <label>Segmentation mask</label>
            {status === 'loading'
              ? <div className="img-placeholder"><Spinner /></div>
              : result && <img src={result} alt="Segmentation result" />
            }
          </div>
        </div>
      )}

      {status === 'done' && (
        <div className="actions">
          <a className="btn-secondary" href={result!} download="mask.png">
            Download mask
          </a>
          <button className="btn-primary" onClick={reset}>
            Upload another
          </button>
        </div>
      )}
    </main>
  )
}

function UploadIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  )
}

function Spinner() {
  return <div className="spinner" aria-label="Loading" />
}
