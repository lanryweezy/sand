# User Experience Guide: Silicon Intelligence Authority

The Silicon Intelligence platform is designed for two distinct user personas: the **Silicon Pro (Engineer)** and the **Sovereign Executive (CEO)**.

---

## 🛠️ The "Pro" Workflow (Industrial Engineer)

The "Pro" interacts with the platform's core engines to automate the RTL-to-GDSII flow and optimize PPA.

### 1. Intent-Driven Architecture
- **Tool**: `AutoArchitect`
- **Action**: Define an `IntentSpec` in Python or a JSON config.
- **Result**: The system generates optimized, pipelined Verilog RTL (ML cores, RISC-V stages, etc.).

### 2. Autonomous Optimization
- **Tool**: `AutonomousOptimizer` + `CUDASiliconAccelerator`
- **Action**: Feed existing RTL and netlists into the GNN-RL loop.
- **Result**: High-speed, GPU-accelerated placement and routing configurations that minimize power and area.

### 3. Physical Sign-off
- **Tool**: `OpenROADFlowIntegration`
- **Action**: Execute the Dockerized industrial flow.
- **Result**: Real GDSII files, timing reports, and SPICE-level parasitic models for tape-out.

---

## 📊 The "Executive" Workflow (CEO/Stakeholders)

The "Executive" uses the platform for high-level visibility, command, and control.

### 1. The Command Center (Web Portal)
- **URL**: `http://localhost:5173/`
- **Utility**: Real-time telemetry of the design agents. Watch the "Brain" iterate on placement in the 2D visualizer.
- **Milestones**: Track the "Alpha-to-Omega" status from floorplan to final DRC check.

### 2. PPA Telemetry
- **Visuals**: Vibrant HSL-colored dashboards showing Power (mW), Area (um2), and Timing (ns) trends.
- **Decision Support**: Clear "Go/No-Go" signals based on confidence scores (e.g., "GAAFET Physics Confirmed: 99.1%").

### 3. Strategic Roadmap
- **Utility**: View the prioritized roadmap for 3nm/2nm scaling, 3D Chiplets, and the "Silicon Brain" ASIC project.

---

## 🚀 Deployment Modes
- **Single-Core (Desktop)**: Running `ultimate_signoff.py` for local design exploration.
- **Cluster/Headless**: Running the Dockerized flow on GPU server racks for massive parallel architecture searches.
- **Web-Only**: Viewing the progress from a remote "War Room" using the Vite/React portal.

**STREET HEART TECHNOLOGIES: POWERING THE ARCHITECTS OF THE FUTURE.**
