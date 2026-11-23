/* Shared chart helpers for HealthGuard (Chart.js 3.7). */
(function () {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  const C = {
    s1: v('--series-1') || '#2a78d6', s2: v('--series-2') || '#eb6834',
    good: v('--good') || '#0ca30c', warning: v('--warning') || '#fab219', critical: v('--critical') || '#d03b3b',
    text: v('--text-secondary') || '#52514e', muted: v('--text-muted') || '#8a8984', grid: '#eceef1', surface: '#fcfcfb',
  };
  C.risk = { High: C.critical, Medium: C.warning, Low: C.good };

  if (window.Chart) {
    Chart.defaults.font.family = '"Open Sans", system-ui, sans-serif';
    Chart.defaults.font.size = 12;
    Chart.defaults.color = C.text;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.boxHeight = 10;
    Chart.defaults.plugins.tooltip.backgroundColor = '#13233a';
    Chart.defaults.plugins.tooltip.padding = 8;
    Chart.defaults.interaction.intersect = false;
  }

  const fmt = {
    int: (n) => Number(n || 0).toLocaleString(),
    money: (n) => {
      n = Number(n || 0);
      if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
      if (Math.abs(n) >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
      return '$' + n.toFixed(0);
    },
    pct: (n, d = 1) => (Number(n || 0)).toFixed(d) + '%',
    prob: (p) => (100 * Number(p || 0)).toFixed(1) + '%',
  };

  const axis = (title, extra = {}) => Object.assign({
    grid: { color: C.grid, drawBorder: false }, ticks: { color: C.muted },
    title: title ? { display: true, text: title, color: C.text } : undefined,
  }, extra);

  const registry = {};
  function draw(id, config) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (registry[id]) { registry[id].destroy(); }
    registry[id] = new Chart(el.getContext('2d'), config);
    return registry[id];
  }

  function bar(id, labels, datasets, opts = {}) {
    return draw(id, {
      type: 'bar',
      data: {
        labels, datasets: datasets.map((d) => Object.assign({
          borderRadius: 4, borderSkipped: 'start', maxBarThickness: 34,
          borderColor: C.surface, borderWidth: { top: 0, right: 2, bottom: 0, left: 0 },
        }, d)),
      },
      options: Object.assign({
        indexAxis: opts.horizontal ? 'y' : 'x',
        plugins: { legend: { display: datasets.length > 1 || !!opts.legend }, tooltip: { callbacks: opts.tooltip || {} } },
        scales: {
          x: axis(opts.xTitle, opts.horizontal ? { beginAtZero: true } : { grid: { display: false } }),
          y: axis(opts.yTitle, opts.horizontal ? { grid: { display: false } } : { beginAtZero: true }),
        },
      }, opts.options || {}),
    });
  }

  function line(id, labels, datasets, opts = {}) {
    return draw(id, {
      type: 'line',
      data: { labels, datasets: datasets.map((d) => Object.assign({ borderWidth: 2, pointRadius: 3, pointHoverRadius: 5, tension: 0.25, fill: false }, d)) },
      options: Object.assign({
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: datasets.length > 1 }, tooltip: { callbacks: opts.tooltip || {} } },
        scales: { x: axis(opts.xTitle, { grid: { display: false } }), y: axis(opts.yTitle, opts.y || { beginAtZero: true }) },
      }, opts.options || {}),
    });
  }

  function scatter(id, datasets, opts = {}) {
    return draw(id, {
      type: 'scatter',
      data: { datasets: datasets.map((d) => Object.assign({ pointRadius: 4, pointHoverRadius: 6, borderColor: C.surface, borderWidth: 1 }, d)) },
      options: Object.assign({
        interaction: { mode: 'nearest', intersect: true },
        plugins: { legend: { display: true }, tooltip: { callbacks: opts.tooltip || {} } },
        scales: { x: axis(opts.xTitle, opts.x || {}), y: axis(opts.yTitle, opts.y || {}) },
      }, opts.options || {}),
    });
  }

  /* Accessible table view for a chart: <button data-table-for="chartId"> toggles a generated table. */
  function tableView(chartId, headers, rows) {
    const holder = document.getElementById(chartId + '-table');
    if (!holder) return;
    const head = '<tr>' + headers.map((h, i) => `<th class="${i ? 'num' : ''}">${h}</th>`).join('') + '</tr>';
    const body = rows.map((r) => '<tr>' + r.map((c, i) => `<td class="${i ? 'num' : ''}">${c}</td>`).join('') + '</tr>').join('');
    holder.innerHTML = `<div class="table-responsive mt-2"><table class="table table-sm"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
  }
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-table-for]');
    if (!btn) return;
    const holder = document.getElementById(btn.dataset.tableFor + '-table');
    if (holder) {
      holder.classList.toggle('d-none');
      btn.textContent = holder.classList.contains('d-none') ? 'Show table' : 'Hide table';
    }
  });

  function csrf() {
    const m = document.cookie.match(/csrftoken=([^;]+)/);
    return m ? m[1] : '';
  }

  window.HG = { C, fmt, bar, line, scatter, tableView, csrf, axis };
})();
