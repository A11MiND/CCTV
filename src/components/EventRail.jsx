import Icon from "./Icon";
import { cameras } from "../data/demoData";

export default function EventRail({
  events,
  selectedId,
  onSelect,
  onReview,
  onDecision,
}) {
  return (
    <aside className="event-rail">
      <div className="rail-heading">
        <div>
          <span className="section-kicker">即時佇列</span>
          <h2>事件時間軸</h2>
        </div>
        <button className="icon-button labelled" type="button">
          <Icon name="filter" size={16} />
          篩選
        </button>
      </div>

      <div className="event-list">
        {events.map((event) => {
          const camera = cameras.find((item) => item.id === event.cameraId);
          const selected = event.id === selectedId;
          return (
            <article
              className={`event-item ${selected ? "is-selected" : ""} severity-${event.severity}`}
              key={event.id}
            >
              <button className="event-select" onClick={() => onSelect(event.id)} type="button">
                <span className="event-time">{event.time}</span>
                <span className="event-main">
                  <strong>{event.type}</strong>
                  <small>{camera?.shortLabel}</small>
                </span>
                <span className="event-score">{event.score}%</span>
              </button>
              {selected ? (
                <div className="event-expanded">
                  <p>{event.evidence}</p>
                  <button className="evidence-link" onClick={() => onReview(event.id)} type="button">
                    <span className="play-disc">
                      <Icon name="play" size={11} />
                    </span>
                    查看 {event.clipLength} 秒證據
                    <Icon name="arrowRight" size={15} />
                  </button>
                  <div className="event-decisions">
                    <button className="button button-attention" onClick={() => onDecision("attention")} type="button">
                      <Icon name="flag" size={16} />
                      標記為需關注
                    </button>
                    <button className="button button-neutral" onClick={() => onDecision("dismissed")} type="button">
                      <Icon name="dismiss" size={16} />
                      誤報
                    </button>
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>

      <div className="rail-timeline">
        <div className="timeline-caption">
          <span>貨架通道 03</span>
          <strong>證據片段 · 12 秒</strong>
        </div>
        <div className="time-track">
          <span className="time-window" />
          <i className="playhead" />
        </div>
        <div className="time-labels">
          <span>14:31:48</span>
          <span>14:32:06</span>
          <span>14:32:18</span>
          <span>14:32:36</span>
        </div>
      </div>
    </aside>
  );
}
