from __future__ import annotations

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>CV Metrics</title>
    <style>
      body { font-family: 'Inter', sans-serif; padding: 16px; background: #0c101b; color: #e8e8e8; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
      .card { background: #131a2a; padding: 14px 16px; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
      h1 { margin-top: 0; letter-spacing: 0.5px; }
      .label { color: #9bb0d0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; }
      .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
      .muted { color: #8a94a6; font-size: 12px; margin-top: 6px; }
    </style>
  </head>
  <body>
    <h1>Metrics</h1>
    <div class="card" style="overflow-x:auto;">
      <table id="metrics" style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th class="label">Metric</th>
            <th class="label">Value</th>
            <th class="label">Avg/Note</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="card" id="summary" style="margin-top:12px;"></div>
    <div class="muted" id="updated"></div>
    <script>
      async function refresh() {
        try {
          const resp = await fetch('/metrics?t=' + Date.now(), { cache: 'no-store' });
          const data = await resp.json();
          const fmt = (v, digits=2) => (Number.isFinite(v) ? v.toFixed(digits) : '-');
      const entries = [
            ['FPS', fmt(data.fps), `window avg fps (${fmt(data.fps)})`],
            ['Avg Latency (ms)', fmt(data.avg_latency_ms), `window avg (${fmt(data.avg_latency_ms)})`],
            ['p50 (ms)', fmt(data.p50), `window p50 (${fmt(data.p50)})`],
            ['p90 (ms)', fmt(data.p90), `window p90 (${fmt(data.p90)})`],
            ['p95 (ms)', fmt(data.p95), `window p95 (${fmt(data.p95)})`],
            ['Backend', data.backend ?? '-', 'backend'],
            ['Detections', data.detections_total ?? '-', 'total sent'],
            ['GPU', data.gpu ? JSON.stringify(data.gpu) : 'N/A', data.gpu ? 'util/mem' : 'N/A'],
      ];
      const rows = entries.map(
        ([k,v,n]) => `<tr><td>${k}</td><td>${v ?? '-'}</td><td class="muted">${n ?? '-'}</td></tr>`
      );
          document.getElementById('rows').innerHTML = rows.join('');
          const summary = `
            <div class="label">Summary</div>
            <div class="value">
              FPS: ${data.fps?.toFixed(2)} |
              Avg: ${data.avg_latency_ms?.toFixed(2)} ms |
              p50/p90/p95: ${data.p50?.toFixed(1)} / ${data.p90?.toFixed(1)} / ${data.p95?.toFixed(1)} ms
            </div>
          `;
          document.getElementById('summary').innerHTML = summary;
          document.getElementById('updated').innerText = 'Last updated: ' + new Date().toLocaleTimeString();
        } catch (e) {
          document.getElementById('metrics').innerHTML = '<div class="card"><div class="label">error</div><div class="value">'+e+'</div></div>';
        }
      }
      refresh();
      setInterval(refresh, 2000);
    </script>
  </body>
</html>
"""

