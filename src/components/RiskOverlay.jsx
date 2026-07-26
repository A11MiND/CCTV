export default function RiskOverlay({ compact = false, review = false }) {
  return (
    <svg
      aria-label="人物姿態及風險訊號示意"
      className={`risk-overlay ${compact ? "is-compact" : ""}`}
      viewBox="0 0 1000 600"
    >
      {!compact ? (
        <path
          className="track-line"
          d="M250 515 C320 485 360 500 425 456 S510 428 555 380"
        />
      ) : null}
      {!compact
        ? [
            [250, 515],
            [320, 486],
            [390, 478],
            [455, 445],
            [512, 417],
          ].map(([x, y], index) => (
            <circle className="track-dot" cx={x} cy={y} key={`${x}-${y}`} r={index === 4 ? 7 : 4} />
          ))
        : null}
      <g className="pose">
        <circle cx="618" cy="185" r="13" />
        <path d="M618 199 603 270 620 342M603 270 565 228M603 270 657 252M620 342 574 424M620 342 662 424" />
        <circle cx="565" cy="228" r="5" />
        <circle cx="657" cy="252" r="5" />
        <circle cx="574" cy="424" r="5" />
        <circle cx="662" cy="424" r="5" />
        <path className="torso" d="M618 200 579 224 603 270 620 342 660 273 650 218Z" />
      </g>
      {review ? (
        <>
          <rect className="hand-zone" height="58" rx="5" width="68" x="635" y="224" />
          <rect className="pocket-zone" height="70" rx="5" width="70" x="584" y="306" />
          <text className="overlay-label label-hand" x="710" y="245">手部區域</text>
          <text className="overlay-label label-pocket" x="660" y="335">衣袋區域</text>
        </>
      ) : null}
    </svg>
  );
}
