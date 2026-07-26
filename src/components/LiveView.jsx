import { cameras, events } from "../data/demoData";
import CameraFeed from "./CameraFeed";
import EventRail from "./EventRail";
import Icon from "./Icon";

export default function LiveView({
  selectedCameraId,
  selectedEventId,
  onCameraSelect,
  onEventSelect,
  onReview,
  onDecision,
}) {
  const selectedCamera =
    cameras.find((camera) => camera.id === selectedCameraId) ?? cameras[0];
  const secondaryCameras = cameras.filter(
    (camera) => camera.id !== selectedCamera.id,
  );

  return (
    <div className="live-layout">
      <section className="video-workspace">
        <div className="workspace-heading">
          <div>
            <span className="section-kicker">門店現場</span>
            <h1>4 路影像，1 宗事件待覆核</h1>
          </div>
          <span className="latency-indicator">
            <i />
            邊緣分析延遲 1.8 秒
          </span>
        </div>

        <div className="primary-feed-wrap">
          <CameraFeed
            camera={selectedCamera}
            isSelected
            onSelect={onCameraSelect}
            showOverlay={selectedCamera.id === "aisle" && !selectedCamera.video}
          />
          <div className="playback-bar">
            <button aria-label="暫停" className="playback-icon" type="button">
              <Icon name="pause" size={17} />
            </button>
            <span>1×</span>
            <span>14:32:12</span>
            <div className="scrub-track">
              <span className="scrub-range" />
              <i />
            </div>
            <span className="scrub-label">12 秒證據窗口</span>
            <span>14:33:30</span>
          </div>
        </div>

        <div className="secondary-feeds">
          {secondaryCameras.map((camera) => (
            <CameraFeed
              camera={camera}
              compact
              isSelected={false}
              key={camera.id}
              onSelect={onCameraSelect}
            />
          ))}
        </div>
      </section>
      <EventRail
        events={events}
        onDecision={onDecision}
        onReview={onReview}
        onSelect={onEventSelect}
        selectedId={selectedEventId}
      />
    </div>
  );
}
