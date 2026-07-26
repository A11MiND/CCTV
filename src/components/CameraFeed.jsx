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
        {camera.video && !compact ? (
          <video
            aria-label={`${camera.label} YOLO26 實跑影片`}
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
            RTX 4060 實跑
          </span>
        ) : null}
        {!compact ? (
          <span className="feed-timecode">14:32:18</span>
        ) : null}
      </div>
    </button>
  );
}
