// Render the standalone measured-results report without network requests.
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const fs = require('node:fs');
const packages = process.env.OCTO_NODE_MODULES ||
  '/Users/sudar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { chromium } = require(path.join(packages, 'playwright'));

(async () => {
  const root = path.resolve(process.argv[2] || 'artifacts/runs/e2e_train_1');
  const state = JSON.parse(fs.readFileSync(path.join(root, 'status.json')));
  const final = state.phase === 'complete';
  const browser = await chromium.launch({ headless: true, executablePath: process.env.OCTO_CHROMIUM_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 1120, height: 1000 }, deviceScaleFactor: 2 });
    await page.route('https://**/*', route => route.abort());
    await page.route('http://**/*', route => route.abort());
    await page.goto(pathToFileURL(path.join(root, 'report.html')).href);
    await page.evaluate(() => document.fonts.ready);
    const box = await page.locator('.share-note').boundingBox();
    const output = path.join(root, final ? 'report-social.png' : 'report-preview.png');
    await page.screenshot({ path: output, clip: { x: 0, y: 0, width: 1120, height: Math.ceil(box.y + box.height + 12) } });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    if (overflow) throw new Error('Report has horizontal overflow');
    await page.setViewportSize({ width: 390, height: 844 });
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth))
      throw new Error('Mobile report has horizontal overflow');
    console.log(JSON.stringify({ output, state: state.phase, desktopAndMobileOverflow: false }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
