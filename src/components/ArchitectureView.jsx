import Icon from "./Icon";

function Node({ icon, title, detail, tone = "default" }) {
  return (
    <div className={`architecture-node tone-${tone}`}>
      <Icon name={icon} size={21} />
      <strong>{title}</strong>
      <small>{detail}</small>
    </div>
  );
}

export default function ArchitectureView() {
  return (
    <div className="support-view architecture-view">
      <header className="support-header">
        <div>
          <span className="section-kicker">技術原則</span>
          <h1>YOLO26 是感知底座，不是「盜竊判斷器」</h1>
          <p>
            生產級告警需要檢測、跟蹤、時序動作、人與物關聯、區域規則及人工覆核共同完成。
          </p>
        </div>
        <span className="verified-callout">
          <Icon name="shield" size={20} />
          邊緣即時決策 · 雲端配置與模型治理
        </span>
      </header>

      <section className="architecture-panel">
        <div className="table-heading">
          <div>
            <span className="section-kicker">模型網絡架構</span>
            <h2>從單一影格到可解釋事件</h2>
          </div>
          <span className="panel-note">YOLO26s · ByteTrack · Action v0.3 · Rule v12</span>
        </div>
        <div className="architecture-line">
          <Node icon="camera" title="RTSP 影格" detail="子碼流 2–5 FPS" />
          <Icon name="arrowRight" size={19} />
          <Node icon="focus" title="YOLO26" detail="人／袋／商品／姿態" tone="teal" />
          <Icon name="arrowRight" size={19} />
          <Node icon="refresh" title="多目標跟蹤" detail="每路獨立 Track ID" />
          <Icon name="arrowRight" size={19} />
          <Node icon="hand" title="時序互動" detail="人－手－物－袋" tone="amber" />
          <Icon name="arrowRight" size={19} />
          <Node icon="clipboard" title="風險狀態機" detail="去重、冷卻、證據鏈" />
          <Icon name="arrowRight" size={19} />
          <Node icon="user" title="人工覆核" detail="行為描述，不作指控" tone="red" />
        </div>
      </section>

      <div className="architecture-split">
        <section className="architecture-panel">
          <div className="table-heading">
            <div>
              <span className="section-kicker">YOLO26 內部</span>
              <h2>雙檢測頭與多尺度特徵</h2>
            </div>
          </div>
          <div className="yolo-stack">
            <div><span>輸入</span><strong>640×640／ROI 裁剪</strong></div>
            <div><span>Backbone</span><strong>Conv + C3k2 · P3/P4/P5</strong></div>
            <div><span>Neck</span><strong>SPPF + C2PSA · FPN/PAN</strong></div>
            <div className="dual-head">
              <span>Detection Head</span>
              <strong>One-to-One（無 NMS）</strong>
              <strong>One-to-Many（需 NMS）</strong>
            </div>
          </div>
        </section>

        <section className="architecture-panel">
          <div className="table-heading">
            <div>
              <span className="section-kicker">端－邊－雲</span>
              <h2>旁掛式系統架構</h2>
            </div>
          </div>
          <div className="system-columns">
            <div>
              <span>門店視頻網</span>
              <Node icon="camera" title="現有攝影機" detail="主／子碼流" />
              <Node icon="server" title="現有 NVR" detail="全天錄影與取證" />
            </div>
            <div>
              <span>AI 邊緣主機</span>
              <Node icon="refresh" title="推理與規則" detail="斷網仍可運作" tone="teal" />
              <Node icon="server" title="環形緩衝" detail="事件前後短片" />
            </div>
            <div>
              <span>雲端控制面</span>
              <Node icon="shield" title="配置與 RBAC" detail="出站 TLS" />
              <Node icon="book" title="模型治理" detail="灰度／回滾／審計" tone="amber" />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
