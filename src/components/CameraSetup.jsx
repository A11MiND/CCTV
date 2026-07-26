import { cameras } from "../data/demoData";
import Icon from "./Icon";

export default function CameraSetup() {
  return (
    <div className="support-view">
      <header className="support-header">
        <div>
          <span className="section-kicker">旁掛式接入</span>
          <h1>保留現有 NVR，新增 AI 邊緣主機</h1>
          <p>
            分析用子碼流，事件證據使用主碼流或 NVR 錄影；不改變原有全天錄影職責。
          </p>
        </div>
        <span className="verified-callout">
          <Icon name="server" size={20} />
          NVR4216-16P-4KS3 · RTSP / ONVIF / CGI / SDK
        </span>
      </header>

      <div className="integration-flow">
        {[
          ["camera", "現有攝影機", "H.264 / H.265"],
          ["server", "大華 NVR", "16 路 PoE · 繼續錄影"],
          ["refresh", "AI 邊緣主機", "解碼 · YOLO26 · 時序模型"],
          ["monitor", "店員介面", "告警 · 證據 · 人工回饋"],
        ].map(([icon, title, detail], index, all) => (
          <div className="flow-node-wrap" key={title}>
            <div className="flow-node">
              <Icon name={icon} size={23} />
              <strong>{title}</strong>
              <small>{detail}</small>
            </div>
            {index < all.length - 1 ? <Icon name="arrowRight" size={20} /> : null}
          </div>
        ))}
      </div>

      <section className="device-table-section">
        <div className="table-heading">
          <div>
            <span className="section-kicker">模擬接入狀態</span>
            <h2>4 路攝影機健康度</h2>
          </div>
          <button className="button button-neutral" type="button">
            <Icon name="refresh" size={16} />
            重新測試串流
          </button>
        </div>
        <div className="device-table">
          <div className="device-row device-head">
            <span>通道</span><span>型號／代碼</span><span>分析串流</span><span>碼率</span><span>狀態</span>
          </div>
          {cameras.map((camera, index) => (
            <div className="device-row" key={`${camera.id}-${index}`}>
              <span><strong>{camera.label}</strong><small>CH {String(index + 1).padStart(2, "0")}</small></span>
              <span><strong>{camera.model}</strong><small>{camera.code}</small></span>
              <span>{camera.stream}</span>
              <span>{camera.bitrate}</span>
              <span className="status-ok"><i /> 在線</span>
            </div>
          ))}
        </div>
      </section>

      <div className="verification-grid">
        <section>
          <span className="section-kicker">已從官方資料確認</span>
          <h3>原則上可接入</h3>
          <p>錄影主機支援 RTSP、ONVIF Profile S/G/T、CGI 及 SDK；CM532 型號亦支援 RTSP 與 ONVIF。</p>
        </section>
        <section className="needs-check">
          <span className="section-kicker">現場必須驗證</span>
          <h3>PoE 私有網段與完整型號</h3>
          <p>AI 主機能否直連每個通道、第三款攝影機是否為 IPC-HDW1239V-A，以及實際 FPS／碼率均需現場核對。</p>
        </section>
      </div>
    </div>
  );
}
