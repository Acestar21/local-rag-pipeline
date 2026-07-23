import { useEffect, useRef, useState } from 'react'
import './App.css'

type MessageRole = 'user' | 'assistant'

type Message = {
  id: string
  role: MessageRole
  content: string
  isStreaming?: boolean
  isError?: boolean
}

type DocumentStatus = {
  loaded: boolean
  chunks: number
}

const API_BASE = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [documentStatus, setDocumentStatus] = useState<DocumentStatus>({
    loaded: false,
    chunks: 0,
  })
  const [isUploading, setIsUploading] = useState(false)
  const [isSending, setIsSending] = useState(false)

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const updateMessage = (id: string, updater: (message: Message) => Message) => {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.id === id ? updater(message) : message,
      ),
    )
  }

  const readStreamingResponse = async (response: Response, assistantId: string) => {
    const reader = response.body?.getReader()

    if (!reader) {
      throw new Error('The response stream is unavailable.')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data:')) {
          continue
        }

        const token = line.slice(5)
        if (token.length === 0) {
          continue
        }

        updateMessage(assistantId, (message) => ({
          ...message,
          content: message.content + token,
        }))
      }
    }

    if (buffer.startsWith('data:')) {
      const token = buffer.slice(5)
      if (token.length > 0) {
        updateMessage(assistantId, (message) => ({
          ...message,
          content: message.content + token,
        }))
      }
    }
  }

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''

    if (!file) {
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setIsUploading(true)

    try {
      const response = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Failed to ingest PDF.')
      }

      const data: { message: string; chunks: number } = await response.json()
      setDocumentStatus({ loaded: true, chunks: data.chunks })
    } catch (error) {
      setDocumentStatus({ loaded: false, chunks: 0 })
      const message = error instanceof Error ? error.message : 'Upload failed.'
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: message,
          isError: true,
        },
      ])
    } finally {
      setIsUploading(false)
    }
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isSending) {
      return
    }

    const userMessageId = crypto.randomUUID()
    const assistantMessageId = crypto.randomUUID()

    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: userMessageId,
        role: 'user',
        content: trimmedQuestion,
      },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      },
    ])
    setQuestion('')
    setIsSending(true)

    try {
      const formData = new FormData()
      formData.append('question', trimmedQuestion)

      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || 'Query request failed.')
      }

      await readStreamingResponse(response, assistantMessageId)
      updateMessage(assistantMessageId, (message) => ({
        ...message,
        isStreaming: false,
      }))
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Query failed.'
      updateMessage(assistantMessageId, (currentMessage) => ({
        ...currentMessage,
        content: currentMessage.content || message,
        isStreaming: false,
        isError: true,
      }))
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <p className="eyebrow">RAG chat</p>
          <h1>Quiet Expert</h1>
        </div>

        <div className="header-actions">
          <p className={`status-pill ${documentStatus.loaded ? 'loaded' : ''}`}>
            {documentStatus.loaded
              ? `document loaded (${documentStatus.chunks} chunks)`
              : 'no document loaded'}
          </p>

          <button
            type="button"
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload PDF'}
          </button>
          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
          />
        </div>
      </header>

      <main className="chat-panel" aria-label="RAG chat messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p className="empty-title">Ask a question about your PDF.</p>
            <p className="empty-copy">
              Upload a document, then send a question to stream an answer in real time.
            </p>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <article
                key={message.id}
                className={`message-row ${message.role}`}
                aria-live={message.isStreaming ? 'polite' : undefined}
              >
                <div className={`message-bubble ${message.role} ${message.isError ? 'error' : ''}`}>
                  {message.content || (message.isStreaming ? ' ' : '')}
                  {message.isStreaming ? <span className="cursor" aria-hidden="true" /> : null}
                </div>
              </article>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}
      </main>

      <footer className="composer-bar">
        <form className="composer" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="question">
            Ask a question
          </label>
          <input
            id="question"
            className="composer-input"
            type="text"
            placeholder="Ask about the loaded document..."
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={isSending}
          />
          <button type="submit" className="send-button" disabled={isSending}>
            {isSending ? 'Sending...' : 'Send'}
          </button>
        </form>
      </footer>
    </div>
  )
}

export default App
