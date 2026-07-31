import { writeFile } from "node:fs/promises";
import { chromium } from "file:///C:/Users/Pumpkin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";

const baseUrl = process.env.DEMO_URL ?? "http://127.0.0.1:4173";
const outputRoot = "P:/CCTV/design";
const errors = [];
const assertions = [];

function assert(name, condition, detail = "") {
  assertions.push({ name, passed: Boolean(condition), detail });
  if (!condition) {
    throw new Error(`${name}${detail ? `: ${detail}` : ""}`);
  }
}

const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
});

try {
  const desktop = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });
  desktop.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  desktop.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

  const response = await desktop.goto(baseUrl, {
    waitUntil: "networkidle",
    timeout: 30_000,
  });
  assert("HTTP 200", response?.status() === 200, String(response?.status()));
  assert(
    "頁面標題",
    (await desktop.title()).includes("Edcosys"),
    await desktop.title(),
  );
  assert("即時態勢可見", await desktop.getByText("4 路影像，1 宗事件待覆核").isVisible());
  await desktop.locator("video").first().waitFor({ state: "visible" });
  await desktop.waitForTimeout(1_200);
  const videoState = await desktop.locator("video").first().evaluate((video) => ({
    readyState: video.readyState,
    width: video.videoWidth,
    height: video.videoHeight,
  }));
  assert(
    "YOLO26 H.264 影片已載入",
    videoState.readyState >= 2 && videoState.width === 1280 && videoState.height === 720,
    JSON.stringify(videoState),
  );

  async function assertDatasetSet(expectedLabels) {
    for (const label of expectedLabels) {
      assert(
        `${label} 可見`,
        await desktop
          .locator(".dataset-video-badge", { hasText: label })
          .isVisible(),
      );
    }
    const states = await desktop.locator(".secondary-feeds video").evaluateAll(
      (videos) =>
        videos.map((video) => ({
          readyState: video.readyState,
          width: video.videoWidth,
          height: video.videoHeight,
        })),
    );
    assert(
      "三支 training dataset 影片已載入",
      states.length === 3 &&
        states.every(
          (state) =>
            state.readyState >= 2 && state.width === 640 && state.height === 480,
        ),
      JSON.stringify(states),
    );
  }

  await assertDatasetSet([
    "TRAIN · SHOPLIFTING 08",
    "TRAIN · NORMAL 06",
    "TRAIN · SHOPLIFTING 01",
  ]);
  await desktop.screenshot({
    path: `${outputRoot}/prototype-live.png`,
    fullPage: false,
  });

  await desktop.getByRole("button", { name: "B · Positive" }).click();
  await desktop.waitForTimeout(700);
  assert("URL 記錄 Positive set", (await desktop.url()).includes("datasetSet=positive"));
  await assertDatasetSet([
    "TRAIN · SHOPLIFTING 10",
    "TRAIN · SHOPLIFTING 14",
    "TRAIN · SHOPLIFTING 16",
  ]);
  await desktop.screenshot({
    path: `${outputRoot}/dataset-camera-set-b-positive.png`,
    fullPage: false,
  });

  await desktop.getByRole("button", { name: "C · Normal" }).click();
  await desktop.waitForTimeout(700);
  assert("URL 記錄 Normal set", (await desktop.url()).includes("datasetSet=normal"));
  await assertDatasetSet([
    "TRAIN · NORMAL 02",
    "TRAIN · NORMAL 07",
    "TRAIN · NORMAL 11",
  ]);
  await desktop.screenshot({
    path: `${outputRoot}/dataset-camera-set-c-normal.png`,
    fullPage: false,
  });

  await desktop.getByRole("button", { name: "A · Mixed" }).click();
  await desktop.waitForTimeout(500);

  await desktop.getByRole("button", { name: /查看 12 秒證據/ }).click();
  await desktop.locator(".review-layout").waitFor({ state: "visible" });
  assert("事件研判畫面可見", await desktop.getByText("證據鏈（依時間順序）").isVisible());
  await desktop.screenshot({
    path: `${outputRoot}/prototype-review.png`,
    fullPage: false,
  });

  await desktop.getByRole("button", { name: /手部靠近衣袋/ }).click();
  const seekTime = await desktop.locator(".review-player video").evaluate((video) => video.currentTime);
  assert("證據節點可跳轉影片", seekTime >= 6.8 && seekTime <= 7.3, String(seekTime));
  await desktop.getByRole("button", { name: "播放" }).click();
  await desktop.waitForTimeout(350);
  const playing = await desktop.locator(".review-player video").evaluate((video) => !video.paused);
  assert("影片播放按鈕有效", playing);

  await desktop.getByRole("button", { name: "標記為需關注" }).last().click();
  assert(
    "人工處理 Toast 可見",
    await desktop.getByRole("status").getByText(/已標記為「需關注」/).isVisible(),
  );

  await desktop.getByRole("button", { name: /架構說明/ }).click();
  await desktop.getByText("YOLO26 是感知底座，不是「盜竊判斷器」").waitFor();
  assert("架構畫面可切換", await desktop.getByText("雙檢測頭與多尺度特徵").isVisible());
  await desktop.screenshot({
    path: `${outputRoot}/prototype-architecture.png`,
    fullPage: false,
  });

  const mobile = await browser.newPage({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 1,
  });
  mobile.on("console", (message) => {
    if (message.type() === "error") errors.push(`mobile console: ${message.text()}`);
  });
  mobile.on("pageerror", (error) => errors.push(`mobile pageerror: ${error.message}`));
  await mobile.goto(baseUrl, { waitUntil: "networkidle", timeout: 30_000 });
  await mobile.locator("video").first().waitFor({ state: "visible" });
  const overflow = await mobile.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  assert("手機版沒有全頁橫向溢出", overflow <= 1, String(overflow));
  await mobile.screenshot({
    path: `${outputRoot}/prototype-mobile.png`,
    fullPage: true,
  });
  await mobile.close();

  assert("無瀏覽器 console／page error", errors.length === 0, errors.join(" | "));

  await writeFile(
    "P:/CCTV/docs/frontend-qa-results.json",
    JSON.stringify(
      {
        tested_at: new Date().toISOString(),
        url: baseUrl,
        assertions,
        browser_errors: errors,
        screenshots: [
          "design/prototype-live.png",
          "design/dataset-camera-set-b-positive.png",
          "design/dataset-camera-set-c-normal.png",
          "design/prototype-review.png",
          "design/prototype-architecture.png",
          "design/prototype-mobile.png",
        ],
      },
      null,
      2,
    ),
    "utf8",
  );
  console.log(JSON.stringify({ passed: assertions.length, errors }, null, 2));
} finally {
  await browser.close();
}
