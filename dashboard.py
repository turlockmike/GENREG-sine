#!/usr/bin/env python3
"""
Live Experiment Dashboard

Serves a web dashboard showing all running experiments in results/live/.

Usage:
    uv run python dashboard.py
    # Open http://localhost:8050

Any CSV in results/live/ with columns (gen, test_accuracy, ...) will be displayed.
Optional .json metadata files add config details.
"""

import http.server
import socketserver
import json
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8050
RESULTS_DIR = Path(__file__).parent / "results" / "live"

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>GENREG Experiment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 24px;
        }
        .subtitle {
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .chart-box {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #0f3460;
        }
        .chart-title { color: #00d4ff; margin-bottom: 10px; font-size: 16px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }
        .stat-card {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #0f3460;
        }
        .stat-card h3 {
            color: #00d4ff;
            font-size: 14px;
            margin-bottom: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 13px;
        }
        .stat-label { color: #888; }
        .stat-value { color: #fff; font-weight: 500; }
        .stat-value.accuracy { color: #00ff88; font-size: 18px; }
        .status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .status.running { background: #00d4ff33; color: #00d4ff; }
        .status.complete { background: #00ff8833; color: #00ff88; }
        .no-data {
            text-align: center;
            padding: 60px;
            color: #666;
        }
        .no-data code {
            background: #0f3460;
            padding: 2px 8px;
            border-radius: 4px;
            color: #00d4ff;
        }
        .refresh-info {
            text-align: right;
            color: #666;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px 15px;
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
        }
        .controls label {
            color: #888;
            font-size: 13px;
            margin-right: 10px;
        }
        .controls select {
            background: #0f3460;
            color: #fff;
            border: 1px solid #00d4ff44;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 13px;
            cursor: pointer;
        }
        .controls select:hover {
            border-color: #00d4ff;
        }
        .experiment-count {
            color: #00d4ff;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GENREG Experiment Dashboard</h1>
        <p class="subtitle">Live monitoring of experiments in <code>results/live/</code></p>
        <p class="refresh-info">Auto-refreshes every 3 seconds</p>

        <div class="controls">
            <div>
                <label for="sort-select">Sort by:</label>
                <select id="sort-select" onchange="refresh()">
                    <option value="accuracy-desc">Accuracy (High → Low)</option>
                    <option value="accuracy-asc">Accuracy (Low → High)</option>
                    <option value="gen-desc">Generation (High → Low)</option>
                    <option value="gen-asc">Generation (Low → High)</option>
                    <option value="fitness-desc">Fitness (High → Low)</option>
                    <option value="fitness-asc">Fitness (Low → High)</option>
                    <option value="name-asc">Name (A → Z)</option>
                    <option value="name-desc">Name (Z → A)</option>
                    <option value="elapsed-desc">Time (Longest)</option>
                    <option value="elapsed-asc">Time (Shortest)</option>
                </select>
            </div>
            <span class="experiment-count" id="exp-count"></span>
        </div>

        <div class="charts">
            <div class="chart-box">
                <div class="chart-title">Test Accuracy vs Generation</div>
                <div id="accuracy-chart"></div>
            </div>
            <div class="chart-box">
                <div class="chart-title">Fitness vs Generation</div>
                <div id="fitness-chart"></div>
            </div>
        </div>

        <div id="stats-container" class="stats-grid"></div>
    </div>

    <script>
        const COLORS = [
            '#00d4ff', '#00ff88', '#ff6b6b', '#ffd93d', '#6bcb77',
            '#9b59b6', '#e74c3c', '#3498db', '#1abc9c', '#f39c12'
        ];

        async function fetchData() {
            try {
                const response = await fetch('/api/experiments');
                return await response.json();
            } catch (e) {
                console.error('Failed to fetch:', e);
                return { experiments: [] };
            }
        }

        function sortExperiments(experiments) {
            const sortBy = document.getElementById('sort-select').value;
            const [field, direction] = sortBy.split('-');
            const mult = direction === 'desc' ? -1 : 1;

            return [...experiments].sort((a, b) => {
                const latestA = a.data[a.data.length - 1] || {};
                const latestB = b.data[b.data.length - 1] || {};

                let valA, valB;
                switch (field) {
                    case 'accuracy':
                        valA = latestA.test_accuracy || 0;
                        valB = latestB.test_accuracy || 0;
                        break;
                    case 'gen':
                        valA = latestA.gen || 0;
                        valB = latestB.gen || 0;
                        break;
                    case 'fitness':
                        valA = latestA.best_fitness || -Infinity;
                        valB = latestB.best_fitness || -Infinity;
                        break;
                    case 'name':
                        return mult * a.name.localeCompare(b.name);
                    case 'elapsed':
                        valA = latestA.elapsed_s || 0;
                        valB = latestB.elapsed_s || 0;
                        break;
                    default:
                        valA = 0; valB = 0;
                }
                return mult * (valA - valB);
            });
        }

        function updateCharts(experiments) {
            document.getElementById('exp-count').textContent = `${experiments.length} experiments`;

            if (experiments.length === 0) {
                document.getElementById('accuracy-chart').innerHTML = '<div class="no-data">No experiments running.<br><br>Add CSVs to <code>results/live/</code></div>';
                document.getElementById('fitness-chart').innerHTML = '';
                document.getElementById('stats-container').innerHTML = '';
                return;
            }

            // Sort experiments for stat cards
            const sortedExperiments = sortExperiments(experiments);

            // Accuracy chart
            const accTraces = experiments.map((exp, i) => ({
                x: exp.data.map(d => d.gen),
                y: exp.data.map(d => d.test_accuracy * 100),
                type: 'scatter',
                mode: 'lines',
                name: exp.name,
                line: { color: COLORS[i % COLORS.length], width: 2 }
            }));

            Plotly.react('accuracy-chart', accTraces, {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
                margin: { t: 20, r: 20, b: 40, l: 50 },
                xaxis: { title: 'Generation', gridcolor: '#333', zerolinecolor: '#333' },
                yaxis: { title: 'Accuracy (%)', gridcolor: '#333', zerolinecolor: '#333' },
                legend: { orientation: 'h', y: -0.2 },
                height: 300
            }, { responsive: true });

            // Fitness chart
            const fitTraces = experiments.map((exp, i) => ({
                x: exp.data.map(d => d.gen),
                y: exp.data.map(d => d.best_fitness),
                type: 'scatter',
                mode: 'lines',
                name: exp.name,
                line: { color: COLORS[i % COLORS.length], width: 2 }
            }));

            Plotly.react('fitness-chart', fitTraces, {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
                margin: { t: 20, r: 20, b: 40, l: 50 },
                xaxis: { title: 'Generation', gridcolor: '#333', zerolinecolor: '#333' },
                yaxis: { title: 'Fitness', gridcolor: '#333', zerolinecolor: '#333' },
                legend: { orientation: 'h', y: -0.2 },
                height: 300
            }, { responsive: true });

            // Stats cards (use sorted order)
            const statsHtml = sortedExperiments.map((exp, i) => {
                const latest = exp.data[exp.data.length - 1] || {};
                const acc = (latest.test_accuracy * 100).toFixed(1);
                const elapsed = latest.elapsed_s ? formatTime(latest.elapsed_s) : '-';
                const eta = estimateETA(exp);

                return `
                    <div class="stat-card" style="border-left: 3px solid ${COLORS[i % COLORS.length]}">
                        <h3>${exp.name}</h3>
                        <div class="stat-row">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value accuracy">${acc}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Generation</span>
                            <span class="stat-value">${latest.gen || 0}${exp.config?.generations ? ' / ' + exp.config.generations : ''}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Fitness</span>
                            <span class="stat-value">${latest.best_fitness?.toFixed(4) || '-'}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Elapsed</span>
                            <span class="stat-value">${elapsed}</span>
                        </div>
                        ${eta ? `<div class="stat-row"><span class="stat-label">ETA</span><span class="stat-value">${eta}</span></div>` : ''}
                        ${exp.config ? `
                            ${exp.config.description ? `
                                <div class="stat-row">
                                    <span class="stat-label">Description</span>
                                    <span class="stat-value" style="font-size: 11px; color: #aaa;">${exp.config.description}</span>
                                </div>
                            ` : ''}
                            ${exp.config.arch ? `
                                <div class="stat-row">
                                    <span class="stat-label">Architecture</span>
                                    <span class="stat-value">${exp.config.arch}</span>
                                </div>
                            ` : ''}
                            <div class="stat-row">
                                <span class="stat-label">Config</span>
                                <span class="stat-value">${formatConfig(exp.config)}</span>
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('');

            document.getElementById('stats-container').innerHTML = statsHtml;
        }

        function formatTime(seconds) {
            if (seconds < 60) return `${seconds.toFixed(0)}s`;
            if (seconds < 3600) return `${(seconds/60).toFixed(1)}m`;
            return `${(seconds/3600).toFixed(1)}h`;
        }

        function estimateETA(exp) {
            if (!exp.config?.generations || exp.data.length < 2) return null;
            const latest = exp.data[exp.data.length - 1];
            const remaining = exp.config.generations - latest.gen;
            const rate = latest.elapsed_s / latest.gen;
            const etaSeconds = remaining * rate;
            return formatTime(etaSeconds);
        }

        function formatConfig(config) {
            // Build config string from available fields
            const parts = [];

            // Architecture params (in order of importance)
            if (config.H !== undefined) parts.push(`H=${config.H}`);
            if (config.K !== undefined) parts.push(`K=${config.K}`);
            if (config.L !== undefined) parts.push(`L=${config.L}`);
            if (config.S !== undefined) parts.push(`S=${config.S}`);

            // Training params
            if (config.lambda !== undefined && config.lambda > 0) parts.push(`λ=${config.lambda}`);
            if (config.flip_rate !== undefined) parts.push(`flip=${config.flip_rate}`);
            if (config.index_swap_rate !== undefined) parts.push(`idx=${config.index_swap_rate}`);
            if (config.pop_size !== undefined) parts.push(`pop=${config.pop_size}`);

            // Fallback to old format if new fields not found
            if (parts.length === 0) {
                if (config.hidden) parts.push(`H=${config.hidden}`);
                if (config.k) parts.push(`K=${config.k}`);
            }

            return parts.length > 0 ? parts.join(', ') : '-';
        }

        async function refresh() {
            const data = await fetchData();
            updateCharts(data.experiments);
        }

        // Initial load and auto-refresh
        refresh();
        setInterval(refresh, 3000);
    </script>
</body>
</html>
"""


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif parsed.path == '/api/experiments':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            experiments = []

            if RESULTS_DIR.exists():
                for csv_file in sorted(RESULTS_DIR.glob("*.csv")):
                    exp = self._load_experiment(csv_file)
                    if exp:
                        experiments.append(exp)

            self.wfile.write(json.dumps({"experiments": experiments}).encode())

        else:
            super().do_GET()

    def _load_experiment(self, csv_file: Path) -> dict:
        """Load experiment data from CSV and optional JSON config."""
        try:
            data = []
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    return None

                header = lines[0].strip().split(',')
                for line in lines[1:]:
                    if line.strip():
                        values = line.strip().split(',')
                        row = {}
                        for i, col in enumerate(header):
                            try:
                                row[col] = float(values[i])
                            except (ValueError, IndexError):
                                row[col] = values[i] if i < len(values) else None
                        data.append(row)

            # Load optional config
            config = None
            json_file = csv_file.with_suffix('.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    config = json.load(f)

            return {
                "name": csv_file.stem,
                "data": data,
                "config": config
            }
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None

    def log_message(self, format, *args):
        # Suppress request logging
        pass


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"\n{'='*50}")
        print(f"GENREG Experiment Dashboard")
        print(f"{'='*50}")
        print(f"\n  Open: http://localhost:{PORT}")
        print(f"  Watching: {RESULTS_DIR}/")
        print(f"\n  Add CSVs to results/live/ to see them here.")
        print(f"  Press Ctrl+C to stop.\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
