"""Manual control GUI — human-in-the-loop plant control via tkinter.

Launches a desktop window with 5 sliders (pump1, pump2, heater1-3), a reset
button, a real-time level/temp plot, and a live KPI readout. The user drives
the plant through IA2 in real time.

Requires the IA2 chain up (run_mode.sh gui handles the boot). tkinter + matplotlib
are rendered natively on Windows 11 via WSLg (no X server setup).

Usage:
    python3 controllers/manual_gui.py --steps 200
    # or via run_mode.sh gui (boots IA2 automatically)

Note on edge backend: If using `--backend edge:<name>`, be aware that each
step requires an SSH round-trip proxied through the dev server (~6 handshakes
per 0.5 s step). For edge deployments, increase `--control-dt` to accommodate
the network latency.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv  # noqa: E402
from controllers.aiogym_register import register_threetank  # noqa: E402
register_threetank()
from controllers.threetank_model import ThreeTankModel  # noqa: E402
from aiogym.scoring import KPIScorer  # noqa: E402

LOG = logging.getLogger("manual_gui")
matplotlib = None  # lazy import inside _build_plot

CFG = json.load(open(ROOT / "ia2_config.json"))
H_SP = [CFG["control"]["setpoints_m"]["tank1_level"],
        CFG["control"]["setpoints_m"]["tank2_level"],
        CFG["control"]["setpoints_m"]["tank3_level"]]
T_SP = list(CFG["control"]["setpoints_c"].values())
T_COLD = float(CFG["process"]["t_supply_c"])
T_AMB = float(CFG["process"]["t_ambient_c"])

# Actuator register names for Manual mode's write targets.
MANUAL_VARS = ["manual_p1", "manual_p2", "manual_h1", "manual_h2", "manual_h3"]


class ManualGUI:
    """tkinter window: sliders → PLC, real-time plot ← IA2 snapshot."""

    def __init__(self, env: CascadeBridgeEnv, steps: int = 200):
        self.env = env
        self.b = env.backend
        self.model = ThreeTankModel()
        self.scorer = KPIScorer(self.model)
        self.max_steps = steps
        self.k = 0
        self.running = True
        self.steps_data = []

        # --- window ---
        self.root = tk.Tk()
        self.root.title("Cascade Control Sandbox — Manual Control")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- left: controls ---
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Mode: Manual", font=("-weight", "bold")).pack(anchor=tk.W, pady=(0, 10))

        self.sliders = {}
        self.slider_vars = {}
        labels = ["Pump 1", "Pump 2", "Heater 1", "Heater 2", "Heater 3"]
        for i, label in enumerate(labels):
            row = ttk.Frame(left)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=9).pack(side=tk.LEFT)
            
            # Shared variable for both the slider and the spinbox
            var = tk.IntVar(value=50)
            
            # The slider
            s = ttk.Scale(row, from_=0, to=100, orient=tk.HORIZONTAL, variable=var)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            # The spinbox (allows typing exact values)
            sp = ttk.Spinbox(row, from_=0, to=100, textvariable=var, width=4, justify=tk.RIGHT)
            sp.pack(side=tk.RIGHT)
            
            self.sliders[i] = s
            self.slider_vars[i] = var
            
            # Trace writes to the variable so typing or dragging both trigger the callback
            var.trace_add("write", lambda *args, idx=i: self._on_slider(idx))

        ttk.Button(left, text="Reset Episode", command=self._on_reset).pack(fill=tk.X, pady=10)

        # Plant state readout (levels + temps for all 3 tanks)
        self.state_frame = ttk.LabelFrame(left, text="Plant State", padding=5)
        self.state_frame.pack(fill=tk.X, pady=5)
        self.state_labels = {}
        for tank in range(1, 4):
            row = ttk.Frame(self.state_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"Tank {tank}:", width=8).pack(side=tk.LEFT)
            self.state_labels[f"h{tank}"] = ttk.Label(row, text="-- m", width=10)
            self.state_labels[f"h{tank}"].pack(side=tk.LEFT)
            self.state_labels[f"T{tank}"] = ttk.Label(row, text="-- °C")
            self.state_labels[f"T{tank}"].pack(side=tk.LEFT)

        # KPI readout
        self.kpi_frame = ttk.LabelFrame(left, text="KPI", padding=5)
        self.kpi_frame.pack(fill=tk.X, pady=10)
        self.kpi_labels = {}
        for key in ["score", "temp_err", "level_err", "step"]:
            self.kpi_labels[key] = ttk.Label(self.kpi_frame, text=f"{key}: --")
            self.kpi_labels[key].pack(anchor=tk.W)

        # --- right: plot ---
        self._build_plot()

        # --- init: reset + push initial slider states ---
        self.env.reset()
        self.scorer.reset()
        # Write initial slider values to the manual_* registers
        for i in range(len(self.sliders)):
            self._on_slider(i)

        # --- start the update loop ---
        self.root.after(500, self._tick)

    def _build_plot(self):
        """Embed a matplotlib figure in the tkinter window."""
        global matplotlib
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        self.ax_h, self.ax_t = axes
        self.h_lines = [self.ax_h.plot([], [], label=f"h{i+1}")[0] for i in range(3)]
        for i, sp in enumerate(H_SP):
            self.ax_h.axhline(sp, color=f"C{i}", ls="--", alpha=0.3)
        self.ax_h.set_ylabel("Level (m)")
        self.ax_h.set_ylim(-0.02, 0.65)
        self.ax_h.legend(fontsize=7, loc="upper right")
        self.ax_h.set_title("Cascade Control — Manual", fontsize=10)

        self.t_lines = [self.ax_t.plot([], [], label=f"T{i+1}")[0] for i in range(3)]
        self.ax_t.axhline(T_SP[0], color="gray", ls="--", alpha=0.3, label="SP")
        self.ax_t.set_ylabel("Temp (°C)")
        self.ax_t.set_ylim(15, 85)
        self.ax_t.legend(fontsize=7, loc="upper right")
        self.ax_t.set_xlabel("Step")

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.hist_h = [[] for _ in range(3)]
        self.hist_t = [[] for _ in range(3)]
        self.hist_x = []

    # --- callbacks ---
    def _on_slider(self, idx):
        """Write the shared variable value (0-100) to the manual_* register (raw 0-10000)."""
        try:
            val = self.slider_vars[idx].get()
        except tk.TclError:
            # Happens if the user temporarily types a non-integer or clears the box
            return
        
        # Clamp to safe limits just in case
        val = max(0, min(100, int(val)))
        name = MANUAL_VARS[idx]
        self.b.write_register(name, int(val * 100))

    def _on_reset(self):
        self._reset_nonce = getattr(self, "_reset_nonce", 0) % 65535 + 1  # C2 fix
        self.b.write_register("reset_cmd", self._reset_nonce)
        self.root.after(300, lambda: self.b.write_register("reset_cmd", 0))
        self.scorer.reset()
        self.k = 0
        self.hist_h = [[] for _ in range(3)]
        self.hist_t = [[] for _ in range(3)]
        self.hist_x = []
        LOG.info("episode reset")

    def _on_close(self):
        self.running = False
        self.root.destroy()

    # --- main tick (every control_dt) ---
    def _tick(self):
        if not self.running or self.k >= self.max_steps:
            self._finish()
            return

        # In Manual mode, sliders are already written on drag.
        # Read the plant state
        time.sleep(self.env.control_dt)
        raw = self.b.read_raw()
        obs = self.env._decode_obs(raw)
        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]

        # Read the actual actuator values (post-L5-shield) from the snapshot
        act_raw = [raw.get(n, 0) for n in ["actuator1", "actuator2",
                                            "heater1", "heater2", "heater3"]]
        action = [r * 1e-4 for r in act_raw]  # raw → fraction

        # Update the scorer
        act_dict = {"pumps": action[:2], "valves": [], "heaters": action[2:]}
        env_dict = {"t_cold": T_COLD, "t_amb": T_AMB, "extra_outflow": 0.0}
        heat_w = self.model.heater_power(act_dict)
        ideal_w = self.model.ideal_power(levels, temps, T_SP, env_dict, act_dict)
        self.scorer.step_penalty(levels, temps, H_SP, T_SP,
                                  heat_w, ideal_w, False, self.env.control_dt)

        # Collect data
        self.steps_data.append({
            "step": self.k, "levels": levels, "temps": temps,
            "action": action, "reward": 0.0})
        for i in range(3):
            self.hist_h[i].append(levels[i])
            self.hist_t[i].append(temps[i])
        self.hist_x.append(self.k)

        # Update plot
        for i in range(3):
            self.h_lines[i].set_data(self.hist_x, self.hist_h[i])
            self.t_lines[i].set_data(self.hist_x, self.hist_t[i])
        for ax in (self.ax_h, self.ax_t):
            ax.set_xlim(max(0, self.k - 60), max(self.k, 60))
        self.canvas.draw_idle()

        # Update KPI readout
        rep = self.scorer.report()
        self.kpi_labels["score"].config(text=f"score:    {rep['score']:.1f}")
        self.kpi_labels["temp_err"].config(text=f"temp_err: {rep['avg_temp_err']:.1f} °C")
        self.kpi_labels["level_err"].config(text=f"lvl_err:  {rep['avg_level_err_cm']:.1f} cm")
        self.kpi_labels["step"].config(text=f"step:     {self.k}/{self.max_steps}")

        # Update plant state readout
        for i in range(3):
            self.state_labels[f"h{i + 1}"].config(text=f"{levels[i]:.3f} m")
            self.state_labels[f"T{i + 1}"].config(text=f"{temps[i]:.1f} °C")

        self.k += 1
        self.root.after(int(self.env.control_dt * 1000), self._tick)

    def _finish(self):
        """Called when the rollout ends or the window closes."""
        if self.steps_data:
            from controllers.rollout_report import report
            report(self.steps_data, tag="gui_manual")
        LOG.info("GUI session ended")
        if self.running:
            self.running = False
            self.root.quit()

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Manual control GUI.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--backend", default="ia2",
                    help="Communication backend: auto | ia2 | modbus | edge:<name> (default: ia2)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    # Env constructor sets mode to 'manual' on the PLC. The GUI does not write mode.
    # R4b fix: manual mode writes PLC vars (manual_p*, mode, *_sp) which don't exist
    # in the cabinet register space — require the IA2 backend (not modbus).
    if args.backend == "modbus":
        print("ERROR: manual GUI requires the IA2 backend (writes PLC variables like manual_p1, mode, *_sp). "
              "Run `./run_mode.sh gui` (boots IA2) or use --backend ia2.")
        sys.exit(1)
    env = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt, mode="manual")
    gui = ManualGUI(env, steps=args.steps)
    gui.run()
    env.close()


if __name__ == "__main__":
    main()