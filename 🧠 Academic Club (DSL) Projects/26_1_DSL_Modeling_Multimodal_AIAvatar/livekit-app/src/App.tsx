import { useState, useCallback, useEffect, useRef } from 'react'
import {
  LiveKitRoom,
  RoomAudioRenderer,
  VideoTrack,
  useTracks,
  useRoomContext,
} from '@livekit/components-react'
import '@livekit/components-styles'
import { Track, RoomEvent } from 'livekit-client'
import './App.css'

// ===== Constants =====
const SANDBOX_ID = 'kl-avatar-13tdv0'
const TOKEN_SERVER_URL = 'https://cloud-api.livekit.io/api/sandbox/connection-details'
const OPENROUTER_API_KEY = import.meta.env.VITE_OPENROUTER_API_KEY
const DEBATE_MAX_SEC = 180

const TOPICS = [
  '천친반이 더 강하다 vs 크리링이 더 강하다',
  '물렁복숭아 vs 딱딱복숭아',
  '인어 vs 어인, 어느 것으로 살 것인가?',
  '유비 vs 조조, 누가 더 좋은 직장상사인가?',
  '여름 vs 겨울, 평생 한 계절로 산다면?',
  '사자 vs 호랑이, 백수의 왕은 누구인가?',
  '야채 호빵 vs 단팥호빵, 무엇이 진리인가?',
  '용의 꼬리 vs 뱀의 머리, 무엇으로 사는 것이 좋은가?',
  '가위 vs 바위 vs 보, 무엇이 유리한가?',
]

// ===== Types =====
type Page = 'start' | 'topic' | 'debate' | 'result'

interface TranscriptEntry {
  speaker: '침착맨' | '나'
  text: string
}

interface ConnectionDetails {
  serverUrl: string
  participantToken: string
}

// ===== Helpers =====
function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = (seconds % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

async function analyzeWinner(topic: string, transcript: TranscriptEntry[]): Promise<string> {
  if (transcript.length === 0) {
    return '대화 내용이 없어서 승자를 판단하기 어렵습니다. 다음엔 활발하게 토론해보세요!'
  }
  const conversation = transcript.map(t => `${t.speaker}: ${t.text}`).join('\n')
  try {
    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'openai/gpt-4o-mini',
        messages: [
          {
            role: 'user',
            content: `다음은 "${topic}" 주제의 침멀토론 내용입니다.\n\n${conversation}\n\n토론을 분석해서 누가 더 논리적이고 설득력 있었는지 판단해주세요. 반드시 "승자: 침착맨" 또는 "승자: 나" 중 하나로 시작하고, 이유를 2~3문장으로 한국어로 설명해주세요.`,
          },
        ],
      }),
    })
    const data = await res.json()
    return data.choices?.[0]?.message?.content ?? '분석 실패'
  } catch {
    return '네트워크 오류로 분석에 실패했습니다.'
  }
}

// ===== Inner Room Components (must be inside LiveKitRoom) =====

function AvatarVideo() {
  const tracks = useTracks(
    [{ source: Track.Source.Camera, withPlaceholder: false }],
    { onlySubscribed: true }
  )
  const avatarTrack = tracks.find(t => t.participant.identity === 'anam-avatar-agent')

  return (
    <div className="video-panel">
      <div className="video-label">침착맨</div>
      {avatarTrack ? (
        <VideoTrack trackRef={avatarTrack} className="debate-video" />
      ) : (
        <div className="video-placeholder">
          <img src="/chim.png" alt="침착맨" />
          <p>침착맨 연결 중...</p>
        </div>
      )}
    </div>
  )
}

function UserVideo() {
  const tracks = useTracks(
    [{ source: Track.Source.Camera, withPlaceholder: true }],
    { onlySubscribed: false }
  )
  const localTrack = tracks.find(t => t.participant.isLocal)

  return (
    <div className="video-panel">
      <div className="video-label">나</div>
      {localTrack ? (
        <VideoTrack trackRef={localTrack} className="debate-video mirror" />
      ) : (
        <div className="video-placeholder">
          <p>카메라 연결 중...</p>
        </div>
      )}
    </div>
  )
}

function TranscriptCollector({
  onTranscript,
}: {
  onTranscript: (entry: TranscriptEntry) => void
}) {
  const room = useRoomContext()

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handle = (segments: any[], participant: any) => {
      for (const seg of segments) {
        if (seg.final && seg.text?.trim()) {
          onTranscript({
            speaker: participant?.identity?.includes('agent') ? '침착맨' : '나',
            text: seg.text.trim(),
          })
        }
      }
    }
    room.on(RoomEvent.TranscriptionReceived, handle)
    return () => {
      room.off(RoomEvent.TranscriptionReceived, handle)
    }
  }, [room, onTranscript])

  return null
}

function AvatarConnectWatcher({ onConnected }: { onConnected: () => void }) {
  const room = useRoomContext()

  useEffect(() => {
    // Check if already connected (agent joined before this component mounted)
    for (const p of room.remoteParticipants.values()) {
      if (p.identity.includes('agent')) {
        onConnected()
        return
      }
    }
    const handle = (participant: any) => {
      if (participant?.identity?.includes('agent')) {
        onConnected()
      }
    }
    room.on(RoomEvent.ParticipantConnected, handle)
    return () => {
      room.off(RoomEvent.ParticipantConnected, handle)
    }
  }, [room, onConnected])

  return null
}

interface DebateRoomContentProps {
  topic: string
  seconds: number
  onEnd: (force?: boolean) => void
  onTranscript: (entry: TranscriptEntry) => void
  onAvatarConnected: () => void
}

function DebateRoomContent({ topic, seconds, onEnd, onTranscript, onAvatarConnected }: DebateRoomContentProps) {
  return (
    <>
      <AvatarConnectWatcher onConnected={onAvatarConnected} />
      <TranscriptCollector onTranscript={onTranscript} />
      <RoomAudioRenderer />

      <div className="debate-header">
        <div className="debate-timer">{formatTime(seconds)}</div>
        <button className="end-debate-btn" onClick={() => onEnd(true)}>
          토론 종료
        </button>
      </div>

      <div className="debate-arena">
        <AvatarVideo />
        <div className="vs-divider">VS.</div>
        <UserVideo />
      </div>

      <div className="debate-topic-bar">
        <span className="topic-badge">침멀토론</span>
        <span className="topic-text">{topic}</span>
      </div>
    </>
  )
}

// ===== Page Components =====

function StartPage({ onStart }: { onStart: () => void }) {
  return (
    <div className="start-page">
      <div className="chim-face">
        <img src="/chim.png" alt="침착맨" />
      </div>
      <h1 className="start-title">침멀토론</h1>
      <p className="start-subtitle">침착맨과 1:1 토론을 시작해보세요!</p>
      <button className="start-btn" onClick={onStart}>
        토론하기
      </button>
    </div>
  )
}

function TopicPage({ onSelect }: { onSelect: (idx: number) => void }) {
  return (
    <div className="topic-page">
      <h1 className="topic-title">토론 주제 선택</h1>
      <p className="topic-subtitle">오늘의 토론 주제를 선택해주세요</p>
      <div className="topic-grid">
        {TOPICS.map((topic, idx) => (
          <button key={idx} className="topic-card" onClick={() => onSelect(idx)}>
            <span className="topic-num">{idx + 1}</span>
            <span className="topic-card-text">{topic}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

interface DebatePageProps {
  topicIndex: number
  onEnd: (transcript: TranscriptEntry[]) => void
}

function DebatePage({ topicIndex, onEnd }: DebatePageProps) {
  const [conn, setConn] = useState<ConnectionDetails | null>(null)
  const [seconds, setSeconds] = useState(0)
  const [avatarConnected, setAvatarConnected] = useState(false)
  const transcriptRef = useRef<TranscriptEntry[]>([])
  const endedRef = useRef(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const topic = TOPICS[topicIndex]
  const roomSuffix = useRef(Date.now())

  const handleEnd = useCallback((force = false) => {
    if (endedRef.current) return
    // 아바타가 아직 안 들어왔으면 강제 종료(force)가 아닌 이상 결과 화면으로 가지 않음
    if (!force && !avatarConnected) return
    endedRef.current = true
    if (timerRef.current) clearInterval(timerRef.current)
    onEnd([...transcriptRef.current])
  }, [onEnd, avatarConnected])

  // Connect
  useEffect(() => {
    fetch(TOKEN_SERVER_URL, {
      method: 'POST',
      headers: { 'X-Sandbox-ID': SANDBOX_ID, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        room_name: `chimchakman-debate-${topicIndex}-${roomSuffix.current}`,
        participant_name: 'User',
      }),
    })
      .then(r => r.json())
      .then(d => setConn({ serverUrl: d.serverUrl, participantToken: d.participantToken }))
  }, [topicIndex])

  // Timer — only starts after avatar connects
  useEffect(() => {
    if (!avatarConnected) return
    timerRef.current = setInterval(() => {
      setSeconds(s => {
        const next = s + 1
        if (next >= DEBATE_MAX_SEC) {
          handleEnd(true)
          return DEBATE_MAX_SEC
        }
        return next
      })
    }, 1000)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [avatarConnected, handleEnd])

  const addTranscript = useCallback((entry: TranscriptEntry) => {
    transcriptRef.current = [...transcriptRef.current, entry]
  }, [])

  const handleAvatarConnected = useCallback(() => {
    setAvatarConnected(true)
  }, [])

  if (!conn) {
    return (
      <div className="loading-page">
        <div className="spinner" />
        <p>토론방 준비 중...</p>
      </div>
    )
  }

  return (
    <div className="debate-page">
      <LiveKitRoom
        video={true}
        audio={true}
        token={conn.participantToken}
        serverUrl={conn.serverUrl}
        data-lk-theme="default"
        onDisconnected={handleEnd}
      >
        <DebateRoomContent
          topic={topic}
          seconds={seconds}
          onEnd={handleEnd}
          onTranscript={addTranscript}
          onAvatarConnected={handleAvatarConnected}
        />
      </LiveKitRoom>
    </div>
  )
}

interface ResultPageProps {
  topic: string
  transcript: TranscriptEntry[]
  onRetry: () => void
  onHome: () => void
}

function ResultPage({ topic, transcript, onRetry, onHome }: ResultPageProps) {
  const [result, setResult] = useState<string | null>(null)

  useEffect(() => {
    analyzeWinner(topic, transcript).then(setResult)
  }, [topic, transcript])

  const winnerMatch = result?.match(/승자\s*[:：]\s*(침착맨|나)/)
  const isChimWinner = winnerMatch?.[1] === '침착맨'

  return (
    <div className={`result-page ${result ? (isChimWinner ? 'result-lose' : 'result-win') : ''}`}>
      {!result ? (
        <div className="result-analyzing">
          <div className="spinner" />
          <p>승자 분석 중...</p>
        </div>
      ) : (
        <>
          <div className="result-verdict">{isChimWinner ? 'LOSE' : 'WIN'}</div>

          <div className="result-chim-wrap">
            <img
              src={isChimWinner ? '/chim_win.png' : '/chim_lose.png'}
              alt="침착맨"
              className="result-chim-img"
            />
            <div className="result-speech-bubble">
              {isChimWinner ? '이마 한번 탁 치세요' : '이마 한번 탁 쳤습니다'}
            </div>
          </div>

          <p className="result-analysis">{result}</p>
        </>
      )}

      <div className="result-buttons">
        <button className="retry-btn" onClick={onRetry}>
          재토론하기
        </button>
        <button className="home-btn" onClick={onHome}>
          끝내기
        </button>
      </div>
    </div>
  )
}

// ===== Main App =====

export default function App() {
  const [page, setPage] = useState<Page>('start')
  const [topicIndex, setTopicIndex] = useState(0)
  const [debateResult, setDebateResult] = useState<{
    transcript: TranscriptEntry[]
    topic: string
  } | null>(null)

  const handleDebateEnd = useCallback(
    (transcript: TranscriptEntry[]) => {
      setDebateResult({ transcript, topic: TOPICS[topicIndex] })
      setPage('result')
    },
    [topicIndex]
  )

  if (page === 'start') return <StartPage onStart={() => setPage('topic')} />

  if (page === 'topic')
    return (
      <TopicPage
        onSelect={idx => {
          setTopicIndex(idx)
          setPage('debate')
        }}
      />
    )

  if (page === 'debate')
    return <DebatePage topicIndex={topicIndex} onEnd={handleDebateEnd} />

  if (page === 'result' && debateResult)
    return (
      <ResultPage
        topic={debateResult.topic}
        transcript={debateResult.transcript}
        onRetry={() => setPage('topic')}
        onHome={() => setPage('start')}
      />
    )

  return null
}
