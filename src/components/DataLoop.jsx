import { datasets } from "../data/demoData";
import Icon from "./Icon";

export default function DataLoop() {
  return (
    <div className="support-view">
      <header className="support-header">
        <div>
          <span className="section-kicker">由公開基準到店內驗收</span>
          <h1>數據不是一次性交付，而是持續閉環</h1>
          <p>
            公開數據用來建立底座；店內正常負樣本、演練正例及店員回饋，才決定系統能否達到可接受的誤報水平。
          </p>
        </div>
        <span className="verified-callout amber">
          <Icon name="alert" size={20} />
          最重要指標：每門店每小時誤報數
        </span>
      </header>

      <section className="dataset-section">
        <div className="table-heading">
          <div>
            <span className="section-kicker">建議組合</span>
            <h2>與店內盜損最相關的數據來源</h2>
          </div>
        </div>
        <div className="dataset-table">
          <div className="dataset-row dataset-head">
            <span>數據集</span><span>規模／形式</span><span>在本項目的角色</span><span>適配</span><span>主要限制</span>
          </div>
          {datasets.map((dataset) => (
            <div className="dataset-row" key={dataset.name}>
              <strong>{dataset.name}</strong>
              <span>{dataset.scope}</span>
              <span>{dataset.role}</span>
              <b>{dataset.fit}</b>
              <span>{dataset.caveat}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="feedback-loop">
        <div>
          <span className="section-kicker">可審計流程</span>
          <h2>每個人工判斷都成為下一輪證據</h2>
        </div>
        <div className="loop-track">
          {[
            ["camera", "事件片段", "連同模型、規則版本"],
            ["clipboard", "人工標註", "需關注／誤報／無法判斷"],
            ["search", "質量審核", "去除偏差與敏感資料"],
            ["refresh", "再訓練", "按日期／鏡頭隔離"],
            ["shield", "影子部署", "先觀察，不通知店員"],
            ["monitor", "小流量啟用", "達標後逐步放量"],
          ].map(([icon, title, detail], index, list) => (
            <div className="loop-node-wrap" key={title}>
              <div className="loop-node">
                <Icon name={icon} size={21} />
                <strong>{title}</strong>
                <small>{detail}</small>
              </div>
              {index < list.length - 1 ? <Icon name="arrowRight" size={18} /> : null}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
