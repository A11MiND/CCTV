import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import LiveView from "./components/LiveView";
import EventReview from "./components/EventReview";
import CameraSetup from "./components/CameraSetup";
import DataLoop from "./components/DataLoop";
import ArchitectureView from "./components/ArchitectureView";
import Icon from "./components/Icon";
import { datasetCameraSets, events, primaryCamera } from "./data/demoData";

const decisionCopy = {
  attention: "已標記為「需關注」，並記錄於人工回饋。",
  dismissed: "已標記為「誤報」，將進入數據質量審核。",
  unclear: "已標記為「無法判斷」，不會用作直接訓練標籤。",
};

export default function App() {
  const [active, setActive] = useState("live");
  const [cameraSetId, setCameraSetId] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get("datasetSet");
    return datasetCameraSets.some((item) => item.id === requested)
      ? requested
      : datasetCameraSets[0].id;
  });
  const [selectedCameraId, setSelectedCameraId] = useState("aisle");
  const [selectedEventId, setSelectedEventId] = useState(events[0].id);
  const [evidenceStep, setEvidenceStep] = useState("missing");
  const [reviewStatus, setReviewStatus] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [comment, setComment] = useState("");
  const [toast, setToast] = useState("");
  const toastTimerRef = useRef(null);
  const cameraSet =
    datasetCameraSets.find((item) => item.id === cameraSetId) ??
    datasetCameraSets[0];
  const liveCameras = useMemo(
    () => [primaryCamera, ...cameraSet.cameras],
    [cameraSet],
  );

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

  const handleCameraSet = useCallback((nextSetId) => {
    setCameraSetId(nextSetId);
    setSelectedCameraId("aisle");
    const url = new URL(window.location.href);
    url.searchParams.set("datasetSet", nextSetId);
    window.history.replaceState({}, "", url);
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
        cameraSetId={cameraSetId}
        cameraSetOptions={datasetCameraSets}
        cameras={liveCameras}
        onCameraSet={handleCameraSet}
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
    cameraSetId,
    comment,
    evidenceStep,
    handleDecision,
    handleReview,
    handleCameraSet,
    isPlaying,
    liveCameras,
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
