import "./evaluate.css";

export type EvaluationRow = Record<string, unknown>;

const scorePattern = /^(recall|hit|ndcg|rr|mrr|map|precision|f1|accuracy|context_recall|context_precision|faithfulness|answer_relevance)(_|$)/i;
const recallPattern = /^recall_at_(\d+)$/i;
const esc = (value: unknown) =>
  String(value ?? "").replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
      char
    ]!,
  );

function columns(rows: EvaluationRow[]): string[] {
  return Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
}

function raw(row: EvaluationRow, key: string): string {
  const value = row[key];
  return value == null ? "" : String(value);
}

function numberValue(row: EvaluationRow, key: string): number | null {
  const value = raw(row, key).trim();
  if (!value) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function scoreColumns(rows: EvaluationRow[]): string[] {
  return columns(rows).filter(
    (key) => scorePattern.test(key) && rows.some((row) => numberValue(row, key) !== null),
  );
}

function timingColumns(rows: EvaluationRow[]): string[] {
  const order = ["query_ms_per_question", "query_ms", "index_seconds"];
  return order.filter(
    (key) =>
      columns(rows).some((column) => column.toLowerCase() === key) &&
      rows.some((row) => numberValue(row, key) !== null),
  );
}

function methodName(row: EvaluationRow, index: number): string {
  return raw(row, "method") || `แถว ${index + 1}`;
}

function shortMethod(row: EvaluationRow, index: number): string {
  const method = methodName(row, index);
  const match = method.match(/\bR\s*(\d+)\b/i);
  return match ? `R${match[1]}` : method;
}

function metricLabel(key: string): string {
  const known: Record<string, string> = {
    recall_at_1: "Recall@1",
    recall_at_5: "Recall@5",
    recall_at_10: "Recall@10",
    hit_at_5: "Hit@5",
    ndcg_at_5: "NDCG@5",
    ndcg_at_10: "NDCG@10",
    rr_at_10: "RR@10",
    mrr_at_10: "MRR@10",
  };
  return known[key.toLowerCase()] || key.replaceAll("_", " ");
}

function scorePercent(value: number): number | null {
  // Retrieval summaries in this project store every score in the 0–1 range.
  return value >= 0 && value <= 1 ? value * 100 : null;
}

function displayPercent(value: number): string {
  return `${new Intl.NumberFormat("th-TH", { maximumFractionDigits: 1 }).format(value)}%`;
}

function scoreBars(rows: EvaluationRow[], metrics: string[], selected: string): string {
  const ticks = [0, 25, 50, 75, 100];
  const bars = rows
    .map((row, index) => {
      const number = numberValue(row, selected);
      if (number === null) {
        return `<button class="eval-score-row is-missing" type="button" data-eval-row="${index}" aria-label="${esc(methodName(row, index))}, ไม่มีค่า ${esc(metricLabel(selected))}"><span class="eval-score-name">${esc(shortMethod(row, index))}</span><span class="eval-score-track"><span class="eval-score-fill" style="width:0%"></span></span><b class="eval-score-value">—</b></button>`;
      }
      const percent = scorePercent(number);
      if (percent === null) {
        return `<button class="eval-score-row is-invalid" type="button" data-eval-row="${index}" aria-label="เปิดค่าจริงของ ${esc(methodName(row, index))} · ${esc(metricLabel(selected))} ${esc(raw(row, selected))} อยู่นอกช่วง 0 ถึง 1"><span class="eval-score-name" title="${esc(methodName(row, index))}">${esc(methodName(row, index))}</span><span class="eval-score-track"><span class="eval-score-fill" style="width:0%"></span></span><b class="eval-score-value">ช่วงผิด</b></button>`;
      }
      return `<button class="eval-score-row" type="button" data-eval-row="${index}" aria-label="เปิดค่าจริงของ ${esc(methodName(row, index))} · ${esc(metricLabel(selected))} ${esc(raw(row, selected))}"><span class="eval-score-name" title="${esc(methodName(row, index))}">${esc(methodName(row, index))}</span><span class="eval-score-track"><span class="eval-score-fill" style="width:${percent}%"></span></span><b class="eval-score-value">${displayPercent(percent)}</b></button>`;
    })
    .join("");
  return `<section class="eval-panel eval-score-panel"><div class="eval-panel-head"><div><span class="eval-overline">SCORE COMPARISON</span><h2>คะแนนราย Retriever</h2></div><span class="eval-range">0–100%</span></div><div class="eval-metric-tabs" role="group" aria-label="เลือก metric">${metrics
    .map(
      (key) =>
        `<button type="button" class="eval-metric-tab ${key === selected ? "is-active" : ""}" data-eval-metric="${esc(key)}" aria-pressed="${key === selected}">${esc(metricLabel(key))}</button>`,
    )
    .join("")}</div><div class="eval-score-axis" aria-hidden="true"><span></span><div>${ticks.map((tick) => `<span>${tick}</span>`).join("")}</div><span></span></div><div class="eval-score-list">${bars || '<p class="eval-muted">ไม่มีคะแนนให้แสดง</p>'}</div></section>`;
}

function recallLineChart(rows: EvaluationRow[], keys: string[]): string {
  const sorted = keys
    .map((key) => ({ key, cutoff: Number(key.match(recallPattern)?.[1] || 0) }))
    .sort((a, b) => a.cutoff - b.cutoff);
  const width = 760;
  const height = 246;
  const left = 44;
  const right = 20;
  const top = 15;
  const bottom = 38;
  const x = (index: number) =>
    left + (sorted.length < 2 ? 0 : (index * (width - left - right)) / (sorted.length - 1));
  const y = (value: number) => top + ((100 - value) / 100) * (height - top - bottom);
  const grid = [0, 25, 50, 75, 100]
    .map((tick) => {
      const yy = y(tick);
      return `<line class="eval-line-grid" x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}"/><text class="eval-line-tick" x="${left - 9}" y="${yy + 4}" text-anchor="end">${tick}%</text>`;
    })
    .join("");
  const xLabels = sorted
    .map(({ cutoff }, index) => `<text class="eval-line-xlabel" x="${x(index)}" y="${height - 10}" text-anchor="middle">Top-${cutoff}</text>`)
    .join("");
  const series = rows
    .map((row, rowIndex) => {
      const values = sorted.map(({ key }) => {
        const number = numberValue(row, key);
        return number === null ? null : scorePercent(number);
      });
      const valid = values.flatMap((value, index) =>
        value === null ? [] : [{ x: x(index), y: y(value), value, key: sorted[index].key }],
      );
      if (!valid.length) return "";
      const points = valid.map((point) => `${point.x},${point.y}`).join(" ");
      const circles = valid
        .map(
          (point) =>
            `<circle class="eval-line-point" cx="${point.x}" cy="${point.y}" r="5" data-eval-row="${rowIndex}" tabindex="0" role="button" aria-label="เปิดค่าจริง ${esc(methodName(row, rowIndex))} ${esc(point.key)} ${esc(raw(row, point.key))}"><title>${esc(methodName(row, rowIndex))} · ${esc(metricLabel(point.key))}: ${esc(raw(row, point.key))}</title></circle>`,
        )
        .join("");
      return `<g class="eval-line-series" style="--series-index:${rowIndex}"><polyline class="eval-line-path" points="${points}"/>${circles}</g>`;
    })
    .join("");
  const legend = rows
    .map(
      (row, index) =>
        `<button type="button" class="eval-legend-item" data-eval-row="${index}" title="${esc(methodName(row, index))}"><i class="eval-legend-dot" style="--series-index:${index}"></i>${esc(methodName(row, index))}</button>`,
    )
    .join("");
  return `<section class="eval-panel eval-line-panel"><div class="eval-panel-head"><div><span class="eval-overline">RECALL BY CUTOFF</span><h2>Recall ตามจำนวนผลค้น</h2></div><span class="eval-range">0–100%</span></div><div class="eval-chart-scroll"><svg class="eval-line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="กราฟ Recall แยกตาม Retriever และจำนวนผลค้น">${grid}${xLabels}${series}</svg></div><div class="eval-legend">${legend}</div></section>`;
}

function heatmap(rows: EvaluationRow[], metrics: string[]): string {
  const head = `<div class="eval-heatmap-row eval-heatmap-head"><span>Retriever</span>${metrics.map((key) => `<span>${esc(metricLabel(key))}</span>`).join("")}</div>`;
  const body = rows
    .map(
      (row, index) =>
        `<div class="eval-heatmap-row"><button type="button" class="eval-heatmap-method" data-eval-row="${index}">${esc(shortMethod(row, index))}</button>${metrics
          .map((key) => {
            const value = numberValue(row, key);
            if (value === null)
              return `<button type="button" class="eval-heatmap-cell is-missing" data-eval-row="${index}" aria-label="${esc(methodName(row, index))} ${esc(metricLabel(key))}: ไม่มีค่า">—</button>`;
            const percent = scorePercent(value);
            if (percent === null)
              return `<button type="button" class="eval-heatmap-cell is-invalid" data-eval-row="${index}" aria-label="เปิดค่าจริง ${esc(methodName(row, index))} ${esc(metricLabel(key))}: อยู่นอกช่วง 0 ถึง 1">ช่วงผิด</button>`;
            const alpha = (0.07 + (percent / 100) * 0.32).toFixed(3);
            return `<button type="button" class="eval-heatmap-cell" data-eval-row="${index}" style="--heat:${alpha}" aria-label="เปิดค่าจริง ${esc(methodName(row, index))} ${esc(metricLabel(key))}: ${esc(raw(row, key))}">${displayPercent(percent)}</button>`;
          })
          .join("")}</div>`,
    )
    .join("");
  return `<section class="eval-panel eval-heatmap-panel"><div class="eval-panel-head"><div><span class="eval-overline">METRIC MATRIX</span><h2>Heatmap คะแนนทั้งหมด</h2></div><span class="eval-subtitle">สีเข้ม = คะแนนสูง</span></div><div class="eval-heatmap-wrap"><div class="eval-heatmap" style="--metric-count:${metrics.length};--matrix-width:${78 + metrics.length * 72}px">${head}${body}</div></div></section>`;
}

function timingPanel(rows: EvaluationRow[], keys: string[]): string {
  const chart = keys
    .map((key) => {
      const values = rows.map((row) => numberValue(row, key));
      const max = Math.max(0, ...values.filter((value): value is number => value !== null));
      const title = key === "query_ms_per_question" ? "เวลา query ต่อคำถาม" : key === "query_ms" ? "เวลา query" : "เวลาเตรียม index";
      const unit = key === "index_seconds" ? "s" : "ms";
      const bars = rows
        .map((row, index) => {
          const value = values[index];
          const width = value === null || max <= 0 ? 0 : (value / max) * 100;
          const shown = value === null ? "—" : `${new Intl.NumberFormat("th-TH", { maximumFractionDigits: 2 }).format(value)} ${unit}`;
          return `<button type="button" class="eval-time-row ${value === null ? "is-missing" : ""}" data-eval-row="${index}" aria-label="เปิดค่าจริง ${esc(methodName(row, index))} ${esc(key)} ${esc(raw(row, key))}"><span class="eval-time-name" title="${esc(methodName(row, index))}">${esc(methodName(row, index))}</span><span class="eval-time-track"><span style="width:${width}%"></span></span><b>${esc(shown)}</b></button>`;
        })
        .join("");
      return `<div class="eval-time-chart"><h3>${title}</h3>${bars}</div>`;
    })
    .join("");
  return `<section class="eval-panel eval-timing-panel"><div class="eval-panel-head"><div><span class="eval-overline">RUNTIME</span><h2>เวลา index และ query</h2></div><span class="eval-subtitle">แสดงค่าตามหน่วยต้นทาง</span></div><div class="eval-timing-grid">${chart}</div></section>`;
}

function rowDetails(rows: EvaluationRow[], keys: string[], selected: number): string {
  const row = rows[selected];
  if (!row) return `<section class="eval-panel eval-detail-panel"><div class="eval-muted">เลือกแถวจากกราฟเพื่อดูค่าต้นทาง</div></section>`;
  return `<section class="eval-panel eval-detail-panel" aria-live="polite"><div class="eval-panel-head"><div><span class="eval-overline">SOURCE VALUES</span><h2>ค่าจริง · ${esc(methodName(row, selected))}</h2></div><span class="eval-exact-tag">ตาม CSV</span></div><dl class="eval-detail-grid">${keys.map((key) => `<div><dt>${esc(key)}</dt><dd>${row[key] == null || raw(row, key) === "" ? '<span class="eval-missing">—</span>' : esc(raw(row, key))}</dd></div>`).join("")}</dl></section>`;
}

function rawTable(rows: EvaluationRow[], keys: string[]): string {
  return `<details class="eval-raw-details"><summary><span><b>ข้อมูลดิบทั้งหมด</b><small>แสดงทุกคอลัมน์และค่าจาก API โดยตรง</small></span><span class="eval-raw-count">${rows.length} แถว · ${keys.length} คอลัมน์</span></summary><div class="eval-raw-scroll"><table><thead><tr>${keys.map((key) => `<th>${esc(key)}</th>`).join("")}</tr></thead><tbody>${rows
    .map(
      (row, index) =>
        `<tr data-eval-row="${index}" tabindex="0" aria-label="เลือกแถว ${index + 1}">${keys.map((key) => `<td>${row[key] == null || raw(row, key) === "" ? '<span class="eval-missing">—</span>' : esc(raw(row, key))}</td>`).join("")}</tr>`,
    )
    .join("")}</tbody></table></div></details>`;
}

export function renderEvaluation(rows: EvaluationRow[]): string {
  const keys = columns(rows);
  const metrics = scoreColumns(rows);
  const recalls = metrics.filter((key) => recallPattern.test(key));
  const timings = timingColumns(rows);
  const selectedMetric = metrics.includes("recall_at_5") ? "recall_at_5" : metrics[0];
  const parts = [
    `<div class="eval-dashboard" data-eval-selected="0" data-eval-metric-selected="${esc(selectedMetric || "")}">`,
    `<div class="eval-overview"><div><span class="eval-overline">EXPERIMENT OVERVIEW</span><h2>${rows.length} Retriever ${rows.length === 1 ? "result" : "results"}</h2><p>เลือกกราฟหรือแถวเพื่อเปิดค่าจริงจากไฟล์ผลประเมิน</p></div><span class="eval-method-count">${metrics.length} metrics</span></div>`,
  ];
  if (metrics.length) parts.push(scoreBars(rows, metrics, selectedMetric));
  if (recalls.length >= 2) parts.push(recallLineChart(rows, recalls));
  if (metrics.length) parts.push(heatmap(rows, metrics));
  if (timings.length) parts.push(timingPanel(rows, timings));
  parts.push(rowDetails(rows, keys, 0));
  parts.push(rawTable(rows, keys));
  parts.push("</div>");
  return parts.join("");
}

export function wireEvaluation(root: HTMLElement): void {
  const dashboard = root.querySelector<HTMLElement>(".eval-dashboard");
  if (!dashboard) return;
  const selectRow = (index: number) => {
    const rows = dashboard.querySelectorAll<HTMLElement>("[data-eval-row]");
    const rowIndex = Number.isInteger(index) ? index : -1;
    if (rowIndex < 0) return;
    dashboard.dataset.evalSelected = String(rowIndex);
    const details = dashboard.querySelector<HTMLElement>(".eval-detail-panel");
    const source = root.dataset.evalRows;
    if (details && source) {
      const parsed = JSON.parse(source) as EvaluationRow[];
      details.outerHTML = rowDetails(parsed, columns(parsed), rowIndex);
    }
    rows.forEach((item) => {
      item.classList.toggle("is-selected", Number(item.dataset.evalRow) === rowIndex);
    });
  };
  const selectMetric = (key: string) => {
    const source = root.dataset.evalRows;
    if (!source) return;
    const rows = JSON.parse(source) as EvaluationRow[];
    const scorePanel = dashboard.querySelector<HTMLElement>(".eval-score-panel");
    if (!scorePanel) return;
    const wrapper = document.createElement("div");
    wrapper.innerHTML = scoreBars(rows, scoreColumns(rows), key);
    scorePanel.replaceWith(wrapper.firstElementChild || scorePanel);
    dashboard.dataset.evalMetricSelected = key;
    selectRow(Number(dashboard.dataset.evalSelected || 0));
  };
  root.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const metric = target.closest<HTMLElement>("[data-eval-metric]");
    if (metric?.dataset.evalMetric) {
      selectMetric(metric.dataset.evalMetric);
      return;
    }
    const row = target.closest<HTMLElement>("[data-eval-row]");
    if (row?.dataset.evalRow != null) selectRow(Number(row.dataset.evalRow));
  });
  root.addEventListener("keydown", (event) => {
    const keyEvent = event as KeyboardEvent;
    if (keyEvent.key !== "Enter" && keyEvent.key !== " ") return;
    const target = keyEvent.target as HTMLElement;
    const row = target.closest<HTMLElement>("[data-eval-row]");
    if (row?.dataset.evalRow != null) {
      keyEvent.preventDefault();
      selectRow(Number(row.dataset.evalRow));
    }
  });
  const raw = root.dataset.evalRows;
  if (raw) selectRow(0);
}

export function setEvaluationRows(root: HTMLElement, rows: EvaluationRow[]): void {
  root.dataset.evalRows = JSON.stringify(rows);
  root.innerHTML = renderEvaluation(rows);
  wireEvaluation(root);
}
