import { useEffect, useRef, useState } from "react";
import { cameras, evidenceSteps, events } from "../data/demoData";
import Icon from "./Icon";
import RiskOverlay from "./RiskOverlay";

const evidenceSeconds = {
  take: 1,
  pause: 4,
  overlap: 7,
  missing: 9.4,
};

function formatTime(seconds) {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `00:${String(Math.floor(safeSeconds)).padStart(2, "0")}`;
}

const signalsByStep = {
  take: [
    ["貨架互動", "0.88"],
    ["手部靠近商品", "0.9 秒"],
    ["人物仍在高風險區", "是"],
  ],
  pause: [
    ["貨架區停留", "36 秒"],
    ["反覆查看周邊", "2 次"],
    ["商品仍可見", "是"],
  ],
  overlap: [
    ["手部／衣袋重疊", "1.4 秒"],
    ["商品與手部關聯", "0.81"],
    ["遮擋程度", "中"],
  ],
  missing: [
    ["貨架互動", "0.88"],
    ["手部／衣袋重疊", "1.4 秒"],
    ["商品未偵測到放回", "是"],
  ],
};

export default function EventReview({
  eventId,
  evidenceStep,
  isPlaying,
  reviewStatus,
  comment,
  onBack,
  onEvidenceStep,
  onPlayToggle,
  onDecision,
  onComment,
  onExport,
}) {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const event = events.find((item) => item.id === eventId) ?? events[0];
  const camera = cameras.find((item) => item.id === event.cameraId) ?? cameras[0];
  const signals = signalsByStep[evidenceStep];
  const duration = 11.37;
  const progress = Math.min(100, (currentTime / duration) * 100);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    const seekToEvidence = () => {
      video.currentTime = evidenceSeconds[evidenceStep] ?? 0;
      setCurrentTime(video.currentTime);
    };
    if (video.readyState >= 1) {
      seekToEvidence();
      return undefined;
    }
    video.addEventListener("loadedmetadata", seekToEvidence, { once: true });
    return () => video.removeEventListener("loadedmetadata", seekToEvidence);
  }, [camera.video, evidenceStep]);

  const handleEvidenceStep = (stepId) => {
    onEvidenceStep(stepId);
    const video = videoRef.current;
    if (video) {
      video.currentTime = evidenceSeconds[stepId] ?? 0;
      setCurrentTime(video.currentTime);
    }
  };

  return (
    <div className="review-layout">
      <section className="review-main">
        <div className="review-title-row">
          <button className="back-button" onClick={onBack} type="button">
            <Icon name="arrowRight" size={16} />
            返回即時態勢
          </button>
          <div>
            <span className="section-kicker">人工覆核</span>
            <h1>事件 {event.id} · {camera.label}</h1>
          </div>
          <span className="review-live">
            <i />
            證據片段
          </span>
        </div>

        <div className="review-player">
          {camera.video ? (
            <video
              aria-label="YOLO26 事件證據實跑影片"
              loop
              muted
              onTimeUpdate={(videoEvent) => setCurrentTime(videoEvent.currentTarget.currentTime)}
              playsInline
              poster={camera.image}
              preload="auto"
              ref={videoRef}
              src={camera.video}
            />
          ) : (
            <>
              <img alt="事件證據模擬畫面" src={camera.image} />
              <RiskOverlay review />
            </>
          )}
          {camera.video ? (
            <span className="review-source-badge">
              真實素材僅示範人員偵測／追蹤 · 事件訊號為模擬
            </span>
          ) : null}
          <div className="review-player-controls">
            <button aria-label={isPlaying ? "暫停" : "播放"} onClick={onPlayToggle} type="button">
              <Icon name={isPlaying ? "pause" : "play"} size={16} />
            </button>
            <span>{formatTime(currentTime)} / 00:11</span>
            <div className="review-progress">
              <span style={{ width: `${progress}%` }} />
              <i style={{ left: `${progress}%` }} />
            </div>
            <span>1×</span>
            <Icon name="expand" size={17} />
          </div>
        </div>

        <section className="evidence-section">
          <div className="evidence-header">
            <div>
              <span className="section-kicker">12 秒上下文</span>
              <h2>證據鏈（依時間順序）</h2>
            </div>
            <span>選擇一步查看命中訊號</span>
          </div>
          <div className="evidence-steps">
            {evidenceSteps.map((step, index) => (
              <div className="evidence-step-wrap" key={step.id}>
                <button
                  aria-pressed={evidenceStep === step.id}
                  className={`evidence-step ${evidenceStep === step.id ? "is-active" : ""}`}
                  onClick={() => handleEvidenceStep(step.id)}
                  type="button"
                >
                  <Icon name={step.icon} size={24} />
                  <span>
                    <small>{step.time}</small>
                    <strong>{step.label}</strong>
                  </span>
                </button>
                {index < evidenceSteps.length - 1 ? (
                  <Icon className="step-arrow" name="arrowRight" size={18} />
                ) : null}
              </div>
            ))}
          </div>
        </section>

        <section className="signal-panel">
          <div className="signal-list">
            <div className="signal-title">
              <span>
                <Icon name="alert" size={17} />
                命中訊號
              </span>
              <small>非單一影格判斷</small>
            </div>
            {signals.map(([name, value]) => (
              <div className="signal-row" key={name}>
                <span>{name}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
          <div className="signal-explanation">
            <span className="section-kicker">系統解釋</span>
            <p>
              該步驟後持續 3.2 秒未偵測到商品被放回貨架，並同時命中手部與衣袋區域重疊；系統因此提高覆核優先級。
            </p>
            <button type="button">查看偵測細節 <Icon name="arrowRight" size={14} /></button>
          </div>
        </section>
      </section>

      <aside className="review-inspector">
        <div className="inspector-heading">
          <span className="section-kicker">可追溯判斷</span>
          <h2>事件摘要</h2>
        </div>
        <div className="risk-summary">
          <span className="risk-icon"><Icon name="alert" size={27} /></span>
          <div>
            <strong>風險摘要</strong>
            <span>模型判斷：需人工覆核</span>
          </div>
          <b>{event.score}%</b>
        </div>

        <dl className="metadata-list">
          <div><dt>攝影機</dt><dd>{camera.code} · 1080p</dd></div>
          <div><dt>模型與版本</dt><dd>YOLO26s · action-v0.3 · rule-v12</dd></div>
          <div><dt>事件時間</dt><dd>2026-07-26 {event.time}</dd></div>
          <div><dt>保留期限</dt><dd>2026-08-25</dd></div>
        </dl>

        <div className="decision-block">
          <span className="field-label">處理動作</span>
          <button
            aria-pressed={reviewStatus === "attention"}
            className={`decision-primary ${reviewStatus === "attention" ? "is-active" : ""}`}
            onClick={() => onDecision("attention")}
            type="button"
          >
            <Icon name="flag" size={17} />
            標記為需關注
          </button>
          <div className="decision-secondary">
            <button
              aria-pressed={reviewStatus === "dismissed"}
              className={reviewStatus === "dismissed" ? "is-active" : ""}
              onClick={() => onDecision("dismissed")}
              type="button"
            >
              <Icon name="dismiss" size={17} />
              誤報
            </button>
            <button
              aria-pressed={reviewStatus === "unclear"}
              className={reviewStatus === "unclear" ? "is-active" : ""}
              onClick={() => onDecision("unclear")}
              type="button"
            >
              <Icon name="question" size={17} />
              無法判斷
            </button>
          </div>
        </div>

        <label className="comment-field">
          <span className="field-label">處理備註</span>
          <textarea
            maxLength={200}
            onChange={(eventChange) => onComment(eventChange.target.value)}
            placeholder="輸入備註，例如：請店長留意出口動線…"
            value={comment}
          />
          <small>{comment.length} / 200</small>
        </label>

        <button className="export-button" onClick={onExport} type="button">
          <Icon name="download" size={18} />
          匯出帶時間戳證據
        </button>
        <p className="legal-note">
          <Icon name="info" size={15} />
          系統僅描述可觀察行為，不判斷身份或犯罪意圖。
        </p>
      </aside>
    </div>
  );
}
