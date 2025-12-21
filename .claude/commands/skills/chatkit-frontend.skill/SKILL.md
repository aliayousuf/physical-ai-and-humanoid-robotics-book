

skill:
  name: "ChatKit Frontend Setup"
  metadata:
    id: chatkit-frontend
    category: setup
    tags:
      - chatkit
      - frontend
      - react
      - typescript
    description: >
      Creates a ChatKit React frontend using TypeScript with a dark theme
      and conversation persistence. The template uses Vite but works with
      any React-based framework (Next.js, CRA, Docusaurus, etc.).

inputs:
  theme:
    type: string
    required: false
    default: dark
    description: "Color scheme: dark, light"
  greeting:
    type: string
    required: false
    default: "Welcome!"
    description: "Start screen greeting"
  backendUrl:
    type: string
    required: false
    default: "http://localhost:8000/chatkit"
    description: "Backend API URL"
  port:
    type: number
    required: false
    default: 3000
    description: "Dev server port"
  layout:
    type: string
    required: false
    default: fullpage
    description: "Layout mode: fullpage, popup-right, popup-left"

outputs:
  files:
    - path: frontend/index.html
      description: "HTML entry with ChatKit CDN script"
    - path: frontend/package.json
      description: "Frontend dependencies"
    - path: frontend/tsconfig.json
      description: "TypeScript configuration"
    - path: frontend/vite.config.ts
      description: "Vite configuration"
    - path: frontend/src/main.tsx
      description: "React entry point"
    - path: frontend/src/App.tsx
      description: "ChatKit React component"

prerequisites:
  - Node.js 18+
  - npm or yarn

execution_steps:
  - step: "Create directory structure"
    structure: |
      frontend/
      ├── index.html
      ├── package.json
      ├── tsconfig.json
      ├── vite.config.ts
      └── src/
          ├── main.tsx
          └── App.tsx

  - step: "Generate package.json"
    file: frontend/package.json
    contents: |
      {
        "name": "chatkit-frontend",
        "private": true,
        "version": "1.0.0",
        "type": "module",
        "scripts": {
          "dev": "vite",
          "build": "tsc && vite build",
          "preview": "vite preview"
        },
        "dependencies": {
          "@openai/chatkit-react": "^1.3.0",
          "react": "^18.3.1",
          "react-dom": "^18.3.1"
        },
        "devDependencies": {
          "@types/react": "^18.3.12",
          "@types/react-dom": "^18.3.1",
          "@vitejs/plugin-react": "^4.3.4",
          "typescript": "^5.6.3",
          "vite": "^6.0.3"
        }
      }

  - step: "Generate index.html"
    notes:
      - "CRITICAL: ChatKit CDN script is required"
    file: frontend/index.html
    contents: |
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>ChatKit App</title>
          <script src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" async></script>
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            html, body, #root { height: 100%; width: 100%; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            .h-full { height: 100%; }
            .w-full { width: 100%; }
          </style>
        </head>
        <body>
          <div id="root"></div>
          <script type="module" src="/src/main.tsx"></script>
        </body>
      </html>

  - step: "Generate App.tsx"
    critical_configuration:
      - "domainKey: 'localhost' is REQUIRED for local dev"
      - "Use label instead of name in prompts"
      - "Do NOT use icon property"
    file: frontend/src/App.tsx
    contents: |
      import { ChatKit, useChatKit } from '@openai/chatkit-react'
      import { useState, useEffect } from 'react'

      function App() {
        const [initialThread, setInitialThread] = useState<string | null>(null)
        const [isReady, setIsReady] = useState(false)

        useEffect(() => {
          const savedThread = localStorage.getItem('chatkit-thread-id')
          setInitialThread(savedThread)
          setIsReady(true)
        }, [])

        const { control } = useChatKit({
          api: {
            url: '{{BACKEND_URL}}',
            domainKey: 'localhost',
          },
          initialThread,
          theme: {
            colorScheme: '{{THEME}}',
            color: {
              grayscale: { hue: 220, tint: 6, shade: -1 },
              accent: { primary: '#4cc9f0', level: 1 },
            },
            radius: 'round',
          },
          startScreen: {
            greeting: '{{GREETING}}',
            prompts: [
              { label: 'Hello', prompt: 'Say hello and introduce yourself' },
              { label: 'Help', prompt: 'What can you help me with?' },
            ],
          },
          composer: { placeholder: 'Type a message...' },
          onThreadChange: ({ threadId }) => {
            if (threadId) localStorage.setItem('chatkit-thread-id', threadId)
          },
          onError: ({ error }) => console.error('ChatKit error:', error),
        })

        if (!isReady) return <div>Loading...</div>

        return (
          <div style={{ height: '100vh', width: '100vw' }}>
            <ChatKit control={control} className="h-full w-full" />
          </div>
        )
      }

      export default App

  - step: "Generate vite.config.ts"
    file: frontend/vite.config.ts
    contents: |
      import { defineConfig } from 'vite'
      import react from '@vitejs/plugin-react'

      export default defineConfig({
        plugins: [react()],
        server: { port: {{PORT}} },
      })

  - step: "Generate tsconfig.json"
    file: frontend/tsconfig.json
    contents: |
      {
        "compilerOptions": {
          "target": "ES2020",
          "useDefineForClassFields": true,
          "lib": ["ES2020", "DOM", "DOM.Iterable"],
          "module": "ESNext",
          "skipLibCheck": true,
          "moduleResolution": "bundler",
          "allowImportingTsExtensions": true,
          "resolveJsonModule": true,
          "isolatedModules": true,
          "noEmit": true,
          "jsx": "react-jsx",
          "strict": true
        },
        "include": ["src"]
      }

  - step: "Generate main.tsx"
    file: frontend/src/main.tsx
    contents: |
      import React from 'react'
      import ReactDOM from 'react-dom/client'
      import App from './App'

      ReactDOM.createRoot(document.getElementById('root')!).render(
        <React.StrictMode>
          <App />
        </React.StrictMode>,
      )

validation:
  checklist:
    - "npm install completes without errors"
    - "npm run dev starts dev server"
    - "Chat UI renders (not blank)"
    - "Messages can be sent"
    - "Thread persists on page refresh"

common_errors:
  - error: "Blank screen"
    fix: "Add ChatKit CDN script to index.html"
  - error: "FatalAppError: Invalid input at api"
    fix: "Add domainKey: 'localhost'"
  - error: "Unrecognized key 'name'"
    fix: "Use label instead of name"
  - error: "Unrecognized key 'icon'"
    fix: "Remove icon property"

layout_variants:
  popup_chat:
    description: "Floating popup chat (bottom-right recommended)"
    features:
      - "Floating button (60x60px, circular)"
      - "Popup window (420x600px)"
      - "Backdrop click to close"
      - "Header with New Chat and Close buttons"
      - "Smooth scale + fade animation"
    positioning:
      bottom_right: "right: '2rem'"
      bottom_left: "left: '2rem'"
    notes:
      - "Change both button and popup positions together"

related_skills:
  - chatkit-backend
  - chatkit-store
