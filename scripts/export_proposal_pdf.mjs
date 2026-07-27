import { pathToFileURL } from "node:url";
import { chromium } from "file:///C:/Users/Pumpkin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";

const htmlPath = "P:/CCTV/output/proposal-en.html";
const pdfPath = "P:/CCTV/Edcosys_CCTV_Retail_Loss_Prevention_Proposal_EN.pdf";

const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
});

try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: pdfPath,
    format: "Letter",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate:
      '<div style="width:100%;padding:0 0.62in;font:8px Calibri,Arial;color:#7a8790;">EDCOSYS &nbsp;|&nbsp; CCTV RETAIL LOSS PREVENTION</div>',
    footerTemplate:
      '<div style="width:100%;padding:0 0.62in;font:8px Calibri,Arial;color:#7a8790;display:flex;justify-content:space-between;"><span>Technical and Product Proposal</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>',
  });
  console.log(pdfPath);
} finally {
  await browser.close();
}
