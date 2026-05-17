import React, { useState, useEffect, useRef } from 'react';
import {
  LayoutDashboard, Video, Bell, Settings, Car, Eye, ShieldCheck,
  AlertTriangle, WifiOff, Activity, CheckCircle2, XCircle,
  Info, Trash2, TrendingDown
} from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import './index.css';

const API = 'http://localhost:5000';
const DROWSY_THRESHOLD = 10;
const MAX_HISTORY = 60; // keep last 60 data points in chart

/* ─────────────────────────────────────────
   Custom Recharts Tooltip
───────────────────────────────────────── */
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </p>
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────
   Shared Backend Hook
───────────────────────────────────────── */
function useBackend(settings) {
  const [data, setData] = useState({
    status: 'offline', ear_avg: 0, counter: 0,
    eyes_found: 0, closed_eyes: 0, alarm_on: false
  });
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState([]); // chart history
  const [alerts, setAlerts] = useState([
    { id: 0, type: 'info', title: 'System Initialised', detail: 'Waiting for backend connection.', time: new Date(), ear: null }
  ]);
  const alertIdRef = useRef(1);
  const lastStatusRef = useRef('offline');
  const tickRef = useRef(0);

  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`${API}/status`, { signal: AbortSignal.timeout(800) });
        const d = await res.json();
        setConnected(true);
        setData(d);

        // Append history point
        tickRef.current++;
        setHistory(prev => {
          const pt = {
            t: tickRef.current,
            ear: parseFloat(d.ear_avg.toFixed(4)),
            counter: d.counter,
            threshold: 0.20,
          };
          const next = [...prev, pt];
          return next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next;
        });

        // Alert on status transition
        const prev = lastStatusRef.current;
        const curr = d.status;
        if (curr !== prev && settings.alertsEnabled) {
          let entry = null;
          if (curr === 'danger') entry = { type: 'danger', title: 'CRITICAL: Drowsiness Detected', detail: `Eyes closed. EAR=${d.ear_avg.toFixed(3)}.` };
          else if (curr === 'warning') entry = { type: 'warning', title: 'Warning: Fatigue Signs', detail: `EAR declining. Counter at ${d.counter}/${DROWSY_THRESHOLD}.` };
          else if (curr === 'safe' && (prev === 'danger' || prev === 'warning')) entry = { type: 'safe', title: 'Driver Recovered', detail: 'Eye activity returned to normal.' };
          if (entry) {
            setAlerts(p => [{ id: alertIdRef.current++, ...entry, time: new Date(), ear: d.ear_avg }, ...p.slice(0, 99)]);
          }
        }
        lastStatusRef.current = curr;
      } catch {
        setConnected(false);
        lastStatusRef.current = 'offline';
        setData(p => ({ ...p, status: 'offline' }));
      }
    }, 250);
    return () => clearInterval(iv);
  }, [settings.alertsEnabled]);

  return { data, connected, history, alerts, setAlerts };
}

/* ─────────────────────────────────────────
   Dashboard Page
───────────────────────────────────────── */
function DashboardPage({ data, connected, alerts, history }) {
  const pct = Math.min(100, Math.round((data.counter / DROWSY_THRESHOLD) * 100));
  const barColor = data.status === 'danger' ? 'var(--danger)' : data.status === 'warning' ? 'var(--warning)' : 'var(--safe)';

  const statusColor = {
    safe: 'var(--safe)', warning: 'var(--warning)', danger: 'var(--danger)', offline: 'var(--text-muted)'
  }[connected ? data.status : 'offline'];

  const statusLabel = { safe: 'Driver Alert', warning: 'Fatigue Detected', danger: 'DROWSINESS!', offline: 'Offline' };
  const statusDesc = {
    safe: 'Eyes open. Monitoring active.',
    warning: 'EAR dropping — stay alert.',
    danger: 'Alarm triggered! Eyes closed.',
    offline: 'Start server.py to begin.',
  };

  const dangerAlerts = alerts.filter(a => a.type === 'danger').length;
  const warnAlerts = alerts.filter(a => a.type === 'warning').length;

  return (
    <>
      {!connected && (
        <div className="offline-banner">
          <WifiOff size={15} />
          Backend not connected — run <code style={{ fontFamily: 'monospace', margin: '0 6px', fontSize: '0.82rem' }}>python server.py</code> in your terminal.
        </div>
      )}

      {/* ── Stat row ── */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-icon blue"><Eye size={20} /></div>
          <div className="stat-info">
            <div className="stat-label">Avg EAR</div>
            <div className="stat-value">{connected ? data.ear_avg.toFixed(3) : '–'}</div>
            <div className="stat-sub">Eye Aspect Ratio</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon amber"><Activity size={20} /></div>
          <div className="stat-info">
            <div className="stat-label">Drowsy Frames</div>
            <div className="stat-value">{connected ? data.counter : '–'}<span style={{ fontSize: '0.85rem', fontWeight: 400 }}>/10</span></div>
            <div className="stat-sub">Consecutive frames</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon red"><AlertTriangle size={20} /></div>
          <div className="stat-info">
            <div className="stat-label">Critical Alerts</div>
            <div className="stat-value">{dangerAlerts}</div>
            <div className="stat-sub">{warnAlerts} warnings this session</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon green"><CheckCircle2 size={20} /></div>
          <div className="stat-info">
            <div className="stat-label">Eyes Detected</div>
            <div className="stat-value">{connected ? data.eyes_found : '–'}</div>
            <div className="stat-sub">of 2 expected</div>
          </div>
        </div>
      </div>

      {/* ── Main grid: Video + Status ── */}
      <div className="dashboard-main-grid">
        {/* Video */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <span className="card-title"><Video size={15} />Live Camera Feed</span>
            {connected && <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>MediaPipe Active</span>}
          </div>
          <div style={{ flex: 1, padding: '12px', minHeight: 0 }}>
            <div className="video-box">
              {connected ? (
                <>
                  <img src={`${API}/video_feed`} alt="Live Feed" />
                  <div className="video-overlays">
                    <div className="vid-badge"><div className="live-dot" />LIVE</div>
                  </div>
                </>
              ) : (
                <div className="no-cam">
                  <Video size={44} />
                  <p>Camera offline</p>
                  <code>python server.py</code>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Driver Status Panel */}
        <div className="card driver-status-card">
          <div className="card-header">
            <span className="card-title"><ShieldCheck size={15} />Driver Status</span>
          </div>

          {/* Status ring */}
          <div className="status-hero">
            <div className={`status-ring ${connected ? data.status : 'offline'}`}>
              {!connected && <WifiOff size={38} />}
              {connected && data.status === 'safe' && <ShieldCheck size={38} />}
              {connected && data.status === 'warning' && <AlertTriangle size={38} />}
              {connected && data.status === 'danger' && <AlertTriangle size={38} />}
            </div>
            <h3 className="status-title" style={{ color: statusColor }}>
              {connected ? statusLabel[data.status] : 'OFFLINE'}
            </h3>
            <p className="status-desc">{connected ? statusDesc[data.status] : 'Backend server not running.'}</p>
          </div>

          {/* Drowsiness bar */}
          <div className="drowsy-section">
            <div className="drowsy-bar-label">
              <span>Drowsiness Level</span>
              <span style={{ fontFamily: 'monospace', color: statusColor }}>{pct}%</span>
            </div>
            <div className="drowsy-bar-bg">
              <div className="drowsy-bar-fill" style={{ width: `${pct}%`, background: barColor }} />
            </div>
          </div>

          {/* 4 metric chips */}
          <div className="metric-chips">
            <div className="metric-chip">
              <div className="metric-chip-label">EAR Value</div>
              <div className="metric-chip-value">{connected ? data.ear_avg.toFixed(3) : '–'}</div>
            </div>
            <div className="metric-chip">
              <div className="metric-chip-label">Counter</div>
              <div className="metric-chip-value">{connected ? `${data.counter}/10` : '–'}</div>
            </div>
            <div className="metric-chip">
              <div className="metric-chip-label">Eyes Found</div>
              <div className="metric-chip-value">{connected ? data.eyes_found : '–'}</div>
            </div>
            <div className="metric-chip">
              <div className="metric-chip-label">Alarm</div>
              <div className="metric-chip-value" style={{ color: data.alarm_on ? 'var(--danger)' : 'var(--safe)' }}>
                {connected ? (data.alarm_on ? '🔴 ON' : '🟢 OFF') : '–'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Charts row ── */}
      <div className="charts-grid">
        {/* EAR Line Chart */}
        <div className="card">
          <div className="card-header">
            <span className="card-title"><TrendingDown size={15} />EAR History (last 60 readings)</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Threshold: 0.20</span>
          </div>
          <div className="card-body" style={{ height: 200 }}>
            {history.length < 2 ? (
              <div className="chart-empty">Collecting data...</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="earGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3FB950" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#3FB950" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, 0.5]} tick={{ fontSize: 10, fill: '#8B949E' }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={0.20} stroke="#F85149" strokeDasharray="4 3" strokeWidth={1.5} label={{ value: 'EAR 0.20', fill: '#F85149', fontSize: 10 }} />
                  <Area type="monotone" dataKey="ear" name="EAR" stroke="#3FB950" fill="url(#earGrad)" strokeWidth={2} dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Drowsy counter bar chart */}
        <div className="card">
          <div className="card-header">
            <span className="card-title"><Activity size={15} />Drowsy Frame Counter</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Alarm at 10</span>
          </div>
          <div className="card-body" style={{ height: 200 }}>
            {history.length < 2 ? (
              <div className="chart-empty">Collecting data...</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={history.slice(-20)} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, DROWSY_THRESHOLD + 2]} tick={{ fontSize: 10, fill: '#8B949E' }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={DROWSY_THRESHOLD} stroke="#F85149" strokeDasharray="4 3" strokeWidth={1.5} />
                  <Bar dataKey="counter" name="Counter" fill="#D29922" radius={[3, 3, 0, 0]} maxBarSize={20} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

/* ─────────────────────────────────────────
   Camera Feed Page
───────────────────────────────────────── */
function CameraPage({ data, connected }) {
  return (
    <div className="cam-page-grid">
      <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="card-header">
          <span className="card-title"><Video size={15} />MediaPipe Annotated Feed</span>
          {connected && <div className="vid-badge" style={{ position: 'static', background: 'var(--danger-bg)', border: '1px solid var(--danger-border)' }}><div className="live-dot" />LIVE</div>}
        </div>
        <div style={{ flex: 1, padding: '12px', minHeight: 0 }}>
          <div className="cam-feed-full">
            {connected
              ? <img src={`${API}/video_feed`} alt="Live Feed" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              : <div className="no-cam"><Video size={56} /><p>Start the Flask backend to view feed</p><code>python server.py</code></div>
            }
          </div>
        </div>
      </div>

      <div className="cam-side">
        <div className="card">
          <div className="card-header"><span className="card-title"><Eye size={15} />Eye Metrics</span></div>
          <div className="card-body">
            <div className="info-row">
              {[
                ['Avg EAR', connected ? data.ear_avg.toFixed(4) : '—'],
                ['EAR Threshold', '0.2000'],
                ['Eyes Detected', connected ? `${data.eyes_found} / 2` : '—'],
                ['Closed Eyes', connected ? data.closed_eyes : '—'],
                ['Drowsy Counter', connected ? `${data.counter} / ${DROWSY_THRESHOLD}` : '—'],
                ['Alarm Active', connected ? (data.alarm_on ? '🔴 YES' : '🟢 NO') : '—'],
                ['Detection State', connected ? data.status.toUpperCase() : 'OFFLINE'],
              ].map(([label, value]) => (
                <div key={label} className="info-block">
                  <span className="info-block-label">{label}</span>
                  <span className="info-block-value">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title"><Info size={15} />Detection Pipeline</span></div>
          <div className="card-body">
            {[
              ['MediaPipe Face Mesh', '468 facial landmarks detected per frame.'],
              ['EAR Algorithm', 'Measures eye openness from 6 key landmark points.'],
              ['Keras Model', 'Custom TF model classifies 24×24 eye patch.'],
              ['Alarm Trigger', 'If EAR < 0.20 for 10+ frames → alarm fires.'],
            ].map(([h, p]) => (
              <div key={h} style={{ marginBottom: '14px' }}>
                <p style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.82rem', marginBottom: '3px' }}>{h}</p>
                <p style={{ fontSize: '0.78rem', lineHeight: 1.5, color: 'var(--text-secondary)' }}>{p}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────
   Alerts Page
───────────────────────────────────────── */
function AlertsPage({ alerts, setAlerts }) {
  const [filter, setFilter] = useState('all');
  const filtered = filter === 'all' ? alerts : alerts.filter(a => a.type === filter);
  const iconMap = {
    danger: <XCircle size={15} />, warning: <AlertTriangle size={15} />,
    safe: <CheckCircle2 size={15} />, info: <Info size={15} />
  };

  return (
    <>
      <div className="alerts-toolbar">
        {['all', 'danger', 'warning', 'safe', 'info'].map(t => (
          <button key={t} className={`filter-btn ${filter === t ? 'active' : ''}`} onClick={() => setFilter(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
        <span className="alerts-count">{filtered.length} event{filtered.length !== 1 ? 's' : ''}</span>
        <button className="filter-btn" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '5px' }} onClick={() => setAlerts([])}>
          <Trash2 size={13} /> Clear All
        </button>
      </div>
      <div className="alert-list">
        {filtered.length === 0
          ? <div className="no-alerts"><Bell size={44} /><p>No {filter !== 'all' ? filter : ''} events recorded yet.</p></div>
          : filtered.map(a => (
            <div key={a.id} className={`alert-row ${a.type}`}>
              <div className={`alert-icon ${a.type}`}>{iconMap[a.type]}</div>
              <div className="alert-body"><h4>{a.title}</h4><p>{a.detail}</p></div>
              <div className="alert-meta">
                <span className="alert-time">{a.time.toLocaleTimeString()}</span>
                {a.ear != null && <span className="alert-ear">EAR: {a.ear.toFixed(3)}</span>}
              </div>
            </div>
          ))}
      </div>
    </>
  );
}

/* ─────────────────────────────────────────
   Settings Page
───────────────────────────────────────── */
const DEFAULT_SETTINGS = { earThreshold: 0.20, drowsyThreshold: 10, alertsEnabled: true, alarmEnabled: true, warningStage: true };

function SettingsPage({ settings, setSettings }) {
  const [local, setLocal] = useState(settings);
  const [saved, setSaved] = useState(false);

  function save() {
    setSettings(local);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="settings-grid">
      <div className="card settings-section">
        <div className="card-header"><span className="card-title"><Eye size={15} />Detection Parameters</span></div>
        <div className="setting-row">
          <div className="setting-info"><h4>EAR Threshold</h4><p>Below this value, eye is flagged as closed.</p></div>
          <div className="setting-control range-wrap">
            <input type="range" min={0.10} max={0.35} step={0.01} value={local.earThreshold} onChange={e => setLocal(p => ({ ...p, earThreshold: parseFloat(e.target.value) }))} />
            <span className="range-val">{local.earThreshold.toFixed(2)}</span>
          </div>
        </div>
        <div className="setting-row">
          <div className="setting-info"><h4>Drowsy Frame Threshold</h4><p>Consecutive closed-eye frames before alarm.</p></div>
          <div className="setting-control range-wrap">
            <input type="range" min={5} max={30} step={1} value={local.drowsyThreshold} onChange={e => setLocal(p => ({ ...p, drowsyThreshold: parseInt(e.target.value) }))} />
            <span className="range-val">{local.drowsyThreshold}f</span>
          </div>
        </div>
      </div>

      <div className="card settings-section">
        <div className="card-header"><span className="card-title"><Bell size={15} />Alerts & Audio</span></div>
        {[
          ['alertsEnabled', 'Enable Alert Logging', 'Log status-change events to the Alerts page.'],
          ['alarmEnabled', 'Audio Alarm', 'Play alarm.wav on drowsiness detection.'],
          ['warningStage', 'Warning Stage', 'Show amber warning before full danger state.'],
        ].map(([key, label, desc]) => (
          <div key={key} className="setting-row">
            <div className="setting-info"><h4>{label}</h4><p>{desc}</p></div>
            <label className="toggle">
              <input type="checkbox" checked={local[key]} onChange={e => setLocal(p => ({ ...p, [key]: e.target.checked }))} />
              <span className="toggle-slider" />
            </label>
          </div>
        ))}
      </div>

      <div className="card settings-section" style={{ gridColumn: '1 / -1' }}>
        <div className="card-header"><span className="card-title"><Info size={15} />System Info</span></div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
          {[
            ['Model', 'drowsiness_model.keras'],
            ['Landmarks', 'face_landmarker.task'],
            ['Backend', 'http://localhost:5000'],
            ['Frontend', 'http://localhost:5173'],
            ['Python', '3.11 required'],
            ['Framework', 'MediaPipe + TensorFlow'],
          ].map(([l, v]) => (
            <div key={l} className="setting-row">
              <div className="setting-info"><h4>{l}</h4></div>
              <code style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: 'var(--accent)' }}>{v}</code>
            </div>
          ))}
        </div>
      </div>

      <div style={{ gridColumn: '1/-1', display: 'flex', gap: '12px', alignItems: 'center' }}>
        <button className="settings-save-btn" onClick={save}>{saved ? '✓ Saved!' : 'Save Settings'}</button>
        <button className="filter-btn" onClick={() => setLocal(DEFAULT_SETTINGS)}>Reset Defaults</button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────
   Root App
───────────────────────────────────────── */
const PAGES = [
  { id: 'dashboard', label: 'Dashboard',   icon: <LayoutDashboard size={17} /> },
  { id: 'camera',    label: 'Camera Feed',  icon: <Video size={17} /> },
  { id: 'alerts',    label: 'Alerts',       icon: <Bell size={17} /> },
  { id: 'settings',  label: 'Settings',     icon: <Settings size={17} /> },
];

const PAGE_META = {
  dashboard: { title: 'Live Dashboard',   sub: 'Real-time driver monitoring overview' },
  camera:    { title: 'Camera Feed',      sub: 'Full-resolution MediaPipe annotated stream' },
  alerts:    { title: 'Alert History',    sub: 'Logged drowsiness and warning events' },
  settings:  { title: 'Settings',         sub: 'Configure detection parameters and preferences' },
};

export default function App() {
  const [page, setPage]       = useState('dashboard');
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const { data, connected, history, alerts, setAlerts } = useBackend(settings);

  const meta        = PAGE_META[page];
  const dangerCount = alerts.filter(a => a.type === 'danger').length;

  const statusCls = connected ? data.status : 'offline';

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon"><Car size={19} /></div>
          <div className="sidebar-brand-text">
            <h1>DriveGuard</h1>
            <p>Drowsiness Detection</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">Navigation</div>
          {PAGES.map(p => (
            <button key={p.id} className={`nav-btn ${page === p.id ? 'active' : ''}`} onClick={() => setPage(p.id)}>
              {p.icon} {p.label}
              {p.id === 'alerts' && dangerCount > 0 && <span className="nav-badge">{dangerCount}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="conn-status">
            <div className={`conn-dot ${connected ? 'connected' : 'disconnected'}`} />
            <div className="conn-info">
              <p>{connected ? 'Backend Online' : 'Backend Offline'}</p>
              <span>localhost:5000</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left"><h2>{meta.title}</h2><p>{meta.sub}</p></div>
          <div className="topbar-right">
            <div className={`status-pill ${statusCls}`}>
              <div className="status-dot" />
              {statusCls.toUpperCase()}
            </div>
          </div>
        </header>

        <div className="page-content">
          {page === 'dashboard' && <DashboardPage data={data} connected={connected} alerts={alerts} history={history} />}
          {page === 'camera'    && <CameraPage    data={data} connected={connected} />}
          {page === 'alerts'    && <AlertsPage    alerts={alerts} setAlerts={setAlerts} />}
          {page === 'settings'  && <SettingsPage  settings={settings} setSettings={setSettings} />}
        </div>
      </div>
    </div>
  );
}
