import Icon from "./Icon";

const titleMap = {
  live: "即時態勢",
  review: "事件研判",
  cameras: "攝影機接入",
  data: "數據閉環",
  architecture: "架構說明",
};

export default function TopBar({ active }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="store-control" type="button">
          <Icon name="store" size={18} />
          <span>旺角店</span>
          <Icon name="chevronDown" size={15} />
        </button>
        <span className="stream-status">
          <i />
          4 / 4 路在線
        </span>
        <span className="current-surface">{titleMap[active]}</span>
      </div>
      <div className="topbar-right">
        <span className="date-time">
          <Icon name="clock" size={18} />
          2026-07-26&nbsp;&nbsp;14:32:25
        </span>
        <button className="user-control" type="button">
          <Icon name="user" size={18} />
          <span>店員 A</span>
          <Icon name="chevronDown" size={14} />
        </button>
      </div>
    </header>
  );
}
