import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import LiveView from "./components/LiveView";
import EventReview from "./components/EventReview";
import CameraSetup from "./components/CameraSetup";
import DataLoop from "./components/DataLoop";
import ArchitectureView from "./components/ArchitectureView";
import Icon from "./components/Icon";
import { events } from "./data/demoData";

const decisionCopy = {
  attention: "已標記為「需關注」，並記錄於人工回饋。",
  dismissed: "已標記為「誤報」，將進入數據質量審核。",
  unclear: "已標記為「無法判斷」，不會用作直接訓練標籤。",
};

export default function App() {
  const [active, setActive] = useState("live");
  const [selectedCameraId, setSelectedCameraId] = useState("aisle");
  const [selectedEventId, setSelectedEventId] = useState(events[0].id);
  const [evidenceStep, setEvidenceStep] = useState("missing");
  const [reviewStatus, setReviewStatus] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [comment, setComment] = useState("");
  const [toast, setToast] = useState("");
  const toastTimerRef = useRef(null);

  const showToast = useCallback((message) => {
    setToast(message);
    window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToast(""), 3200);
  }, []);

  useEffect(
    () => () => window.clearTimeout(toastTimerRef.current),
    [],
  );

  const handleDecision = useCallback(
    (status) => {
      setReviewStatus(status);
      showToast(decisionCopy[status]);
    },
    [showToast],
  );

  const handleReview = useCallback((eventId) => {
    setSelectedEventId(eventId);
    setActive("review");
    setEvidenceStep("missing");
  }, []);

  const surface = useMemo(() => {
    if (active === "review") {
      return (
        <EventReview
          comment={comment}
          eventId={selectedEventId}
          evidenceStep={evidenceStep}
          isPlaying={isPlaying}
          onBack={() => setActive("live")}
          onComment={setComment}
          onDecision={handleDecision}
          onEvidenceStep={setEvidenceStep}
          onExport={() => showToast("Demo：已建立帶時間戳的證據匯出工作。")}
          onPlayToggle={() => setIsPlaying((playing) => !playing)}
          reviewStatus={reviewStatus}
        />
      );
    }

    if (active === "cameras") return <CameraSetup />;
    if (active === "data") return <DataLoop />;
    if (active === "architecture") return <ArchitectureView />;

    return (
      <LiveView
        onCameraSelect={setSelectedCameraId}
        onDecision={handleDecision}
        onEventSelect={setSelectedEventId}
        onReview={handleReview}
        selectedCameraId={selectedCameraId}
        selectedEventId={selectedEventId}
      />
    );
  }, [
    active,
    comment,
    evidenceStep,
    handleDecision,
    handleReview,
    isPlaying,
    reviewStatus,
    selectedCameraId,
    selectedEventId,
    showToast,
  ]);

  return (
    <div className="app-shell">
      <Sidebar active={active} onSelect={setActive} />
      <div className="app-main">
        <TopBar active={active} />
        <main>{surface}</main>
      </div>
      {toast ? (
        <div aria-live="polite" className="toast" role="status">
          <Icon name="info" size={17} />
          {toast}
        </div>
      ) : null}
    </div>
  );
}
