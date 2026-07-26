import Icon from "./Icon";
import { navItems } from "../data/demoData";

export default function Sidebar({ active, onSelect }) {
  return (
    <aside className="sidebar">
      <button className="brand" onClick={() => onSelect("live")} type="button">
        <span className="brand-mark">
          <Icon name="shield" size={27} />
        </span>
        <span className="brand-copy">
          <strong>守望</strong>
          <small>零售防損</small>
        </span>
      </button>

      <nav aria-label="主要功能" className="primary-nav">
        {navItems.map((item) => (
          <button
            aria-current={active === item.id ? "page" : undefined}
            className={`nav-item ${active === item.id ? "is-active" : ""}`}
            key={item.id}
            onClick={() => onSelect(item.id)}
            type="button"
          >
            <Icon name={item.icon} size={21} />
            <span>{item.label}</span>
            {item.id === "live" ? <i className="nav-live-dot" /> : null}
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="human-review-note">
          <Icon name="info" size={17} />
          <span>AI 只提示風險，須由店員確認</span>
        </div>
        <p className="prototype-credit">Prototype · Edcosys</p>
      </div>
    </aside>
  );
}
