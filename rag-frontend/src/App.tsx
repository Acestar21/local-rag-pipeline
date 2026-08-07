import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

// ============================================================================
// Types & Interfaces
// ============================================================================
type MessageRole = 'user' | 'assistant'

interface Message {
  id: string
  role: MessageRole
  content: string
  isStreaming?: boolean
  isThinking?: boolean
  isError?: boolean
}

interface DocumentStatus {
  loaded: boolean
  fileName: string | null
  chunks: number
}

// FastAPI Backend URL
const API_BASE = 'http://localhost:8000'

function App() {
  // --------------------------------------------------------------------------
  // State Management
  // --------------------------------------------------------------------------
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [documentStatus, setDocumentStatus] = useState<DocumentStatus>({
    loaded: false,
    fileName: null,
    chunks: 0,
  })
  const [isUploading, setIsUploading] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // DOM Refs for file selection and auto-scroll target
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  // --------------------------------------------------------------------------
  // Auto-scroll to bottom on message changes
  // --------------------------------------------------------------------------
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  // Helper to update specific message in state by ID
  const updateMessage = (id: string, updater: (msg: Message) => Message) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? updater(msg) : msg))
    )
  }

  // --------------------------------------------------------------------------
  // Stream Reader for FastAPI /query SSE Response
  // --------------------------------------------------------------------------
  const readStreamingResponse = async (
    response: Response,
    assistantId: string
  ) => {
    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('ReadableStream is not supported or available.')
    }

    const decoder = new TextDecoder()
    let buffer = ''
    let hasReceivedFirstToken = false

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data:')) continue

        const token = line.slice(5)
        if (!token) continue

        // On first token received, turn off the "Thinking..." indicator
        if (!hasReceivedFirstToken) {
          hasReceivedFirstToken = true
          updateMessage(assistantId, (msg) => ({
            ...msg,
            isThinking: false,
            content: token,
          }))
        } else {
          updateMessage(assistantId, (msg) => ({
            ...msg,
            content: msg.content + token,
          }))
        }
      }
    }

    // Flush any remaining buffer text starting with "data:"
    if (buffer.startsWith('data:')) {
      const token = buffer.slice(5)
      if (token) {
        if (!hasReceivedFirstToken) {
          updateMessage(assistantId, (msg) => ({
            ...msg,
            isThinking: false,
            content: token,
          }))
        } else {
          updateMessage(assistantId, (msg) => ({
            ...msg,
            content: msg.content + token,
          }))
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // PDF Document Ingestion Handler
  // --------------------------------------------------------------------------
  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    // Reset file input value to allow re-uploading the same file if needed
    event.target.value = ''

    if (!file) return

    // Validation: Verify PDF format
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setErrorMessage('Please select a valid PDF document (.pdf).')
      return
    }

    setErrorMessage(null)
    setIsUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `Ingestion failed with status code ${response.status}.`)
      }

      const data: { message: string; chunks: number } = await response.json()
      setDocumentStatus({
        loaded: true,
        fileName: file.name,
        chunks: data.chunks,
      })
    } catch (error) {
      setDocumentStatus({ loaded: false, fileName: null, chunks: 0 })
      let msg = 'PDF upload failed.'
      if (error instanceof TypeError && error.message.includes('fetch')) {
        msg = 'Cannot connect to backend server at http://localhost:8000. Ensure FastAPI server and Ollama are running.'
      } else if (error instanceof Error) {
        msg = error.message
      }
      setErrorMessage(msg)
    } finally {
      setIsUploading(false)
    }
  }

  // --------------------------------------------------------------------------
  // Question Submission & Query Handler
  // --------------------------------------------------------------------------
  const handleQuestionSubmit = async (
    event?: React.FormEvent<HTMLFormElement>,
    overrideQuestion?: string
  ) => {
    if (event) event.preventDefault()

    const queryText = (overrideQuestion ?? question).trim()
    if (!queryText || isSending || isUploading) return

    setErrorMessage(null)

    const userMsgId = crypto.randomUUID()
    const assistantMsgId = crypto.randomUUID()

    // Add User question and Assistant message initialized with isThinking=true
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: queryText },
      {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        isThinking: true,
      },
    ])

    setQuestion('')
    setIsSending(true)

    try {
      const formData = new FormData()
      formData.append('question', queryText)

      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorDetail = await response.text()
        throw new Error(errorDetail || `Query failed with status code ${response.status}.`)
      }

      await readStreamingResponse(response, assistantMsgId)

      // Mark streaming complete
      updateMessage(assistantMsgId, (msg) => ({
        ...msg,
        isStreaming: false,
        isThinking: false,
      }))
    } catch (error) {
      let msgError = 'Query request failed.'
      if (error instanceof TypeError && error.message.includes('fetch')) {
        msgError = 'Backend unreachable (http://localhost:8000). Verify FastAPI & Ollama services are active.'
      } else if (error instanceof Error) {
        msgError = error.message
      }

      updateMessage(assistantMsgId, (msg) => ({
        ...msg,
        content: msg.content || msgError,
        isStreaming: false,
        isThinking: false,
        isError: true,
      }))
      setErrorMessage(msgError)
    } finally {
      setIsSending(false)
    }
  }

  // Quick suggestion click handler
  const handleChipClick = (suggestion: string) => {
    handleQuestionSubmit(undefined, suggestion)
  }

  // --------------------------------------------------------------------------
  // Render App UI
  // --------------------------------------------------------------------------
  return (
    <div className="app-shell">
      {/* App Header */}
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-icon-wrapper">
            <svg
              className="brand-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <div>
            <p className="brand-subtitle">Document Intelligence</p>
            <h1 className="brand-title">RAG Chat Assistant</h1>
          </div>
        </div>

        <div className="header-actions">
          {/* Document Status Pill */}
          <div
            className={`status-badge ${
              isUploading
                ? 'uploading'
                : documentStatus.loaded
                ? 'loaded'
                : ''
            }`}
          >
            <span className="status-dot" />
            {isUploading ? (
              <span>Ingesting PDF...</span>
            ) : documentStatus.loaded ? (
              <>
                <span className="doc-filename" title={documentStatus.fileName ?? undefined}>
                  📄 {documentStatus.fileName}
                </span>
                <span>• {documentStatus.chunks} chunks</span>
              </>
            ) : (
              <span>No document loaded</span>
            )}
          </div>

          {/* Upload PDF Button */}
          <button
            type="button"
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || isSending}
          >
            {isUploading ? (
              <>
                <svg className="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
                Processing...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Upload PDF
              </>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
        </div>
      </header>

      {/* Error Alert Banner */}
      {errorMessage && (
        <div className="error-banner" role="alert">
          <div className="error-banner-content">
            <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            className="error-dismiss"
            onClick={() => setErrorMessage(null)}
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Chat Panel */}
      <main className="chat-panel" aria-label="RAG chat conversation">
        {messages.length === 0 ? (
          /* Empty State View */
          <div className="empty-state-card">
            <div className="empty-icon-ring">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <circle cx="11.5" cy="14.5" r="2.5" />
                <path d="M13.25 16.25L15 18" />
              </svg>
            </div>
            <h2 className="empty-title">
              {documentStatus.loaded ? 'Document Ready!' : 'Upload a PDF to Start'}
            </h2>
            <p className="empty-description">
              {documentStatus.loaded
                ? `Ask any question below about "${documentStatus.fileName}" to stream AI-generated contextual answers.`
                : 'Upload a document (e.g. research papers, manuals, reports) to perform local RAG question answering.'}
            </p>

            {!documentStatus.loaded ? (
              /* Step-by-Step Guidance */
              <>
                <div className="step-cards">
                  <div className="step-card">
                    <span className="step-num">1</span>
                    <p className="step-text">Upload PDF file</p>
                  </div>
                  <div className="step-card">
                    <span className="step-num">2</span>
                    <p className="step-text">Vector embedding</p>
                  </div>
                  <div className="step-card">
                    <span className="step-num">3</span>
                    <p className="step-text">Stream AI response</p>
                  </div>
                </div>
                <button
                  type="button"
                  className="upload-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                >
                  Choose PDF File
                </button>
              </>
            ) : (
              /* Starter Question Suggestions */
              <div className="suggestions-block">
                <p className="suggestions-title">Suggested questions:</p>
                <div className="suggestion-chips">
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handleChipClick('Summarize the main topics discussed in this document.')}
                    disabled={isSending}
                  >
                    💡 Summarize main topics
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handleChipClick('What are the key takeaways and conclusions?')}
                    disabled={isSending}
                  >
                    📌 Key takeaways
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handleChipClick('Explain the primary concepts in simple terms.')}
                    disabled={isSending}
                  >
                    🔍 Explain core concepts
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Active Chat Message List */
          <div className="chat-content">
            {messages.map((msg) => (
              <article key={msg.id} className={`message-row ${msg.role}`}>
                <div className={`avatar-badge ${msg.role}`}>
                  {msg.role === 'user' ? 'You' : 'AI'}
                </div>
                <div
                  className={`message-bubble ${msg.role} ${
                    msg.isError ? 'error' : ''
                  }`}
                >
                  {msg.role === 'user' ? (
                    <span>{msg.content}</span>
                  ) : msg.isThinking ? (
                    /* AI Thinking State Indicator */
                    <div className="thinking-container">
                      <div className="pulse-dots">
                        <span className="pulse-dot" />
                        <span className="pulse-dot" />
                        <span className="pulse-dot" />
                      </div>
                      <span>Analyzing document context...</span>
                    </div>
                  ) : (
                    /* AI Markdown Response */
                    <div className="markdown-body">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  )}

                  {/* Streaming cursor token highlight */}
                  {msg.isStreaming && !msg.isThinking && (
                    <span className="streaming-cursor" aria-hidden="true" />
                  )}
                </div>
              </article>
            ))}
            {/* Scroll anchor target */}
            <div ref={chatEndRef} />
          </div>
        )}
      </main>

      {/* Composer Input Bar */}
      <footer className="composer-area">
        <form className="composer-form" onSubmit={handleQuestionSubmit}>
          <input
            className="composer-input"
            type="text"
            placeholder={
              isUploading
                ? 'PDF ingestion in progress...'
                : documentStatus.loaded
                ? `Ask a question about ${documentStatus.fileName}...`
                : 'Upload a PDF to ask questions...'
            }
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={isSending || isUploading}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={isSending || isUploading || !question.trim()}
          >
            {isSending ? (
              <>
                <svg className="spin-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
                Asking
              </>
            ) : (
              <>
                <span>Send</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </>
            )}
          </button>
        </form>
      </footer>
    </div>
  )
}

export default App
