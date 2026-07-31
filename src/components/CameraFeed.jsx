import Icon from "./Icon";
import RiskOverlay from "./RiskOverlay";

export default function CameraFeed({
  camera,
  isSelected,
  onSelect,
  compact = false,
  showOverlay = false,
}) {
  return (
    <button
      aria-pressed={isSelected}
      className={`camera-feed ${compact ? "is-compact" : ""} ${isSelected ? "is-selected" : ""}`}
      onClick={() => onSelect?.(camera.id)}
      type="button"
    >
      <div className="feed-title">
        <span>{camera.shortLabel}</span>
        <span className="feed-actions">
          <i className="live-dot" />
          <span className="live-text">直播中</span>
          <Icon name="expand" size={16} />
        </span>
      </div>
      <div className="feed-visual">
        {camera.video ? (
          <video
            aria-label={`${camera.label}影片`}
            autoPlay
            loop
            muted
            playsInline
            poster={camera.image}
            src={camera.video}
          />
        ) : (
          <img alt={`${camera.label}模擬監控畫面`} src={camera.image} />
        )}
        {showOverlay ? <RiskOverlay compact={compact} /> : null}
        {camera.video && !compact ? (
          <span className="real-video-badge">
            <i />
            {camera.performanceLabel ?? "真實影片"}
          </span>
        ) : null}
        {camera.datasetLabel ? (
          <span className="dataset-video-badge">{camera.datasetLabel}</span>
        ) : null}
        {!compact ? (
          <span className="feed-timecode">14:32:18</span>
        ) : null}
      </div>
    </button>
  );
}
