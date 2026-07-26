const common = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export default function Icon({ name, size = 20, className = "" }) {
  const paths = {
    shield: (
      <>
        <path {...common} d="M12 2.8 20 6v5.8c0 5-3.4 8.1-8 9.4-4.6-1.3-8-4.4-8-9.4V6l8-3.2Z" />
        <path {...common} d="M12 7.2v9.2M8.8 11.8H15" />
      </>
    ),
    monitor: (
      <>
        <rect {...common} x="3" y="4" width="18" height="14" rx="2" />
        <path {...common} d="m6.8 14.2 3-3 2.3 2 4.2-5 1.9 2.1M9 21h6" />
      </>
    ),
    clipboard: (
      <>
        <rect {...common} x="5" y="4.5" width="14" height="17" rx="2" />
        <path {...common} d="M9 4.5V3h6v1.5M8.5 10h7M8.5 14h4.2m-4.2 4h6" />
        <path {...common} d="m14.7 15.8 1.3 1.3 2.6-3" />
      </>
    ),
    camera: (
      <>
        <path {...common} d="M4 7.5h10.5a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Z" />
        <path {...common} d="m16.5 11 5-2.8v9.6l-5-2.8Z" />
      </>
    ),
    refresh: (
      <>
        <path {...common} d="M20 7.5A8.5 8.5 0 0 0 5.3 5L3 7.5" />
        <path {...common} d="M3 3.5v4h4M4 16.5A8.5 8.5 0 0 0 18.7 19l2.3-2.5" />
        <path {...common} d="M21 20.5v-4h-4" />
      </>
    ),
    book: (
      <>
        <path {...common} d="M3.5 5.2A3.2 3.2 0 0 1 6.7 2H11v18H6.7a3.2 3.2 0 0 0-3.2 2V5.2ZM20.5 5.2A3.2 3.2 0 0 0 17.3 2H13v18h4.3a3.2 3.2 0 0 1 3.2 2V5.2Z" />
      </>
    ),
    store: (
      <>
        <path {...common} d="M4 9h16l-1.4-5H5.4L4 9Z" />
        <path {...common} d="M5 9v11h14V9M9 20v-6h6v6" />
        <path {...common} d="M4 9a3 3 0 0 0 5 2.2A3.1 3.1 0 0 0 12 12a3.1 3.1 0 0 0 3-.8A3 3 0 0 0 20 9" />
      </>
    ),
    clock: (
      <>
        <circle {...common} cx="12" cy="12" r="9" />
        <path {...common} d="M12 7v5l3.4 2" />
      </>
    ),
    user: (
      <>
        <circle {...common} cx="12" cy="8" r="3.3" />
        <path {...common} d="M5.5 21c.7-4.2 3-6.4 6.5-6.4s5.8 2.2 6.5 6.4" />
      </>
    ),
    chevronDown: <path {...common} d="m7 9.5 5 5 5-5" />,
    expand: (
      <>
        <path {...common} d="M8.5 3.5h-5v5M15.5 3.5h5v5M8.5 20.5h-5v-5M15.5 20.5h5v-5" />
      </>
    ),
    filter: (
      <>
        <path {...common} d="M3 5h18l-7 8v6l-4 2v-8L3 5Z" />
      </>
    ),
    flag: (
      <>
        <path {...common} d="M5 21V4m0 1h11l-1.6 3L16 11H5" />
      </>
    ),
    dismiss: (
      <>
        <circle {...common} cx="12" cy="12" r="9" />
        <path {...common} d="m8.5 8.5 7 7m0-7-7 7" />
      </>
    ),
    question: (
      <>
        <circle {...common} cx="12" cy="12" r="9" />
        <path {...common} d="M9.7 9.1a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1.2.9-1.2 1.8M12 17h.01" />
      </>
    ),
    play: <path d="m8 5.5 10 6.5-10 6.5v-13Z" fill="currentColor" />,
    pause: (
      <>
        <path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor" />
      </>
    ),
    basket: (
      <>
        <path {...common} d="m4 9 2 11h12l2-11H4Z" />
        <path {...common} d="m8 9 4-6 4 6M8.5 13v3m7-3v3" />
      </>
    ),
    hand: (
      <>
        <path {...common} d="M7.8 11V5.8a1.3 1.3 0 0 1 2.6 0v4.4-6a1.3 1.3 0 0 1 2.6 0v5.7-4.8a1.3 1.3 0 0 1 2.6 0v5.2-3.5a1.3 1.3 0 0 1 2.6 0v7c0 4.8-2.6 7.2-6.6 7.2-3.7 0-5.7-2.3-7.5-6.6-.5-1.2.1-2.5 1.2-2.8 1-.2 1.8.3 2.5 1.4Z" />
      </>
    ),
    focus: (
      <>
        <path {...common} d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
        <rect {...common} x="8" y="8" width="8" height="8" rx="1.5" strokeDasharray="2 2" />
      </>
    ),
    alert: (
      <>
        <path {...common} d="M12 3 22 20H2L12 3Z" />
        <path {...common} d="M12 9v5m0 3h.01" />
      </>
    ),
    download: (
      <>
        <path {...common} d="M12 3v12m-4-4 4 4 4-4M4 19v2h16v-2" />
      </>
    ),
    info: (
      <>
        <circle {...common} cx="12" cy="12" r="9" />
        <path {...common} d="M12 10v7m0-10h.01" />
      </>
    ),
    search: (
      <>
        <circle {...common} cx="10.5" cy="10.5" r="6.5" />
        <path {...common} d="m15.5 15.5 5 5" />
      </>
    ),
    server: (
      <>
        <rect {...common} x="3" y="3.5" width="18" height="7" rx="1.5" />
        <rect {...common} x="3" y="13.5" width="18" height="7" rx="1.5" />
        <path {...common} d="M7 7h.01M7 17h.01M11 7h6M11 17h6" />
      </>
    ),
    arrowRight: <path {...common} d="M4 12h15m-5-5 5 5-5 5" />,
  };

  return (
    <svg
      aria-hidden="true"
      className={className}
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {paths[name] ?? paths.info}
    </svg>
  );
}
