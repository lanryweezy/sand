import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
  const [telemetry, setTelemetry] = useState([]);
  const [metrics, setMetrics] = useState({
    area: '0 um²',
    power: '0 mW',
    timing: '0 ps',
    confidence: '0%'
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/src/data/telemetry.json');
        if (response.ok) {
          const data = await response.json();
          setTelemetry(data.feed || []);
          setMetrics(data.metrics || {});
        }
      } catch (e) {
        console.log("Waiting for telemetry data...");
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="glass-panel">
        <div className="logo">SILICON INTELLIGENCE AUTHORITY</div>
        <div className="user-profile" style={{ textAlign: 'right', fontSize: '0.8rem' }}>
          <div style={{ fontWeight: 'bold', color: 'var(--accent-cyan)' }}>SULAIMAN ADEBAYO</div>
          <div style={{ color: 'var(--text-secondary)' }}>CEO, Street Heart Technologies</div>
        </div>
      </header>

      {/* Sidebar - Analytics */}
      <aside className="sidebar glass-panel">
        <h3 style={{ fontSize: '0.9rem', marginBottom: '10px', color: 'var(--accent-magenta)' }}>DESIGN ANALYTICS</h3>

        <div className="metric-card glass-panel">
          <div className="metric-label">Estimated Area</div>
          <div className="metric-value">{metrics.area}</div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-label">Dynamic Power</div>
          <div className="metric-value">{metrics.power}</div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-label">Worst Slack (WNS)</div>
          <div className="metric-value" style={{ color: '#00ff00' }}>{metrics.timing}</div>
        </div>

        <h3 style={{ fontSize: '0.8rem', marginTop: '20px', color: 'var(--text-secondary)' }}>SIGN-OFF STATUS</h3>
        <div className="milestone-grid">
          <div className="milestone-item"><div className="led led-active"></div> <span>GNN TOPOLOGICAL</span></div>
          <div className="milestone-item"><div className="led led-active"></div> <span>AST REFACTORING</span></div>
          <div className="milestone-item"><div className="led led-active"></div> <span>PDK SIGN-OFF</span></div>
          <div className="milestone-item"><div className="led led-active"></div> <span>RL POLICY CONV.</span></div>
          <div className="milestone-item"><div className="led led-warning"></div> <span>OPENROAD_LVL_3</span></div>
        </div>

        <div style={{ marginTop: 'auto', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
          PDK: SkyWater 130nm (sky130A)
        </div>
      </aside>

      {/* Main - Floorplan */}
      <main className="glass-panel">
        <div style={{ padding: '15px', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 'bold' }}>PHYSICAL FLOORPLAN INFRASTRUCTURE</span>
          <span className="poker-chip">AUTONOMOUS MODE</span>
        </div>

        <div className="floorplan-container">
          <svg viewBox="0 0 800 600" className="floorplan-svg">
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Chip Boundary */}
            <rect x="50" y="50" width="700" height="500" fill="none" stroke="rgba(0, 242, 255, 0.2)" strokeWidth="2" />

            {/* Core Blocks */}
            <rect className="block-rect" x="100" y="100" width="200" height="200" rx="4" />
            <text x="120" y="130" fill="var(--accent-cyan)" fontSize="12" fontWeight="bold">ALU_CORE_0</text>

            <rect className="block-rect pipelined-block" x="320" y="100" width="150" height="150" rx="4" />
            <text x="340" y="130" fill="var(--accent-magenta)" fontSize="12" fontWeight="bold">MAC_UNIT_PRO</text>

            <rect className="block-rect gated-block" x="490" y="100" width="200" height="100" rx="4" />
            <text x="510" y="130" fill="#ffcc00" fontSize="12" fontWeight="bold">CLOCK_GATE_BANK</text>

            <rect className="block-rect" x="100" y="320" width="370" height="180" rx="4" />
            <text x="120" y="350" fill="var(--accent-cyan)" fontSize="12" fontWeight="bold">CACHE_SUBSYSTEM</text>

            <rect className="block-rect" x="490" y="220" width="200" height="280" rx="4" />
            <text x="510" y="250" fill="var(--accent-cyan)" fontSize="12" fontWeight="bold">IO_CONTROLLER</text>

            {/* Routing Hints with Animation */}
            <path className="signal-path" d="M 300 150 L 320 150" stroke="var(--accent-cyan)" strokeWidth="2" filter="url(#glow)" />
            <path className="signal-path" d="M 470 150 L 490 150" stroke="#ffcc00" strokeWidth="2" filter="url(#glow)" />
            <path className="signal-path" d="M 400 250 L 400 320" stroke="var(--accent-magenta)" strokeWidth="1" strokeDasharray="2" />
          </svg>
        </div>
      </main>

      {/* Right - Intelligence Feed */}
      <section className="feeds glass-panel">
        <h3 style={{ padding: '15px', fontSize: '0.9rem', borderBottom: '1px solid var(--glass-border)', color: 'var(--accent-cyan)' }}>INTELLIGENCE FEED</h3>
        <div className="cyber-feed">
          {telemetry.map(item => (
            <div key={item.id} className="feed-item" style={{ borderLeftColor: item.title.includes('FACTORY') ? '#00ff00' : item.title.includes('AST') ? 'var(--accent-magenta)' : 'var(--accent-cyan)' }}>
              <div className="feed-time">{item.time}</div>
              <div className="feed-title">{item.title}</div>
              <div className="feed-desc" style={{ color: item.title === 'SYSTEM_READY' ? '#00ff00' : 'var(--text-secondary)' }}>{item.desc}</div>
            </div>
          ))}
          <div style={{ marginTop: 'auto', textAlign: 'center' }}>
            <button style={{
              background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta))',
              border: 'none',
              color: 'white',
              padding: '10px 25px',
              borderRadius: '25px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: 'bold',
              boxShadow: '0 0 15px rgba(255, 0, 255, 0.4)'
            }}>DEPLOY TO INNOVUS</button>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
