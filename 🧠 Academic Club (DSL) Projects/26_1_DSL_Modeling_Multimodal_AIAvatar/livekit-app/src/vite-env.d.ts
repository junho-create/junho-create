/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_LIVEKIT_URL: string
  readonly VITE_LIVEKIT_TOKEN: string
  readonly VITE_LIVEKIT_SANDBOX_ID: string
  readonly VITE_OPENROUTER_API_KEY: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
