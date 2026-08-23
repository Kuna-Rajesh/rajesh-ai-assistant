import './index.css';
import CallScreen from './components/CallScreen';
import { useLiveKitRoom } from './hooks/useLiveKitRoom';

function App() {
  const {
    connectionState,
    isMuted,
    agentSpeaking,
    transcript,
    errorMessage,
    toggleMute,
    disconnect,
    sendTextMessage,
  } = useLiveKitRoom();

  return (
    <CallScreen
      connectionState={connectionState}
      isMuted={isMuted}
      agentSpeaking={agentSpeaking}
      transcript={transcript}
      errorMessage={errorMessage}
      onToggleMute={toggleMute}
      onHangup={disconnect}
      onSendText={sendTextMessage}
    />
  );
}

export default App;
