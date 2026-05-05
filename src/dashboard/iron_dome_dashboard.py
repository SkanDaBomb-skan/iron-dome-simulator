
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import time
import threading
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath('')), 'src'))
from simulation.rocket3d import Rocket3D, ThreatGenerator3D
from simulation.interceptor3d import Interceptor3D
from ml.predictor3d import Radar3D, RocketLSTM3D


class IronDomeDashboard:
    

    # ── Colors ──
    C_BG = '#0b0e17'
    C_PANEL = '#131829'
    C_BORDER = '#1e2a4a'
    C_TEXT = '#c8d0e0'
    C_ACCENT = '#3b82f6'
    C_DANGER = '#ef4444'
    C_SUCCESS = '#22c55e'
    C_WARNING = '#f59e0b'
    C_MUTED = '#64748b'
    C_ROCKET = '#ef4444'
    C_INTERCEPT = '#06b6d4'
    C_ZONE = '#22c55e'
    C_IMPACT = '#f59e0b'
    C_HIT = '#4ade80'

    def __init__(self):
        self.zone = {'x_min': -2000, 'x_max': 2000,
                     'z_min': -2000, 'z_max': 2000}
        self.base_pos = (0, 0, 0)
        self.missile_speed = 800
        self.kill_radius = 30

        # Multi-rocket state
        self.rockets = []       # list of dicts
        self.running = False
        self.lock = threading.Lock()
        self.rocket_counter = 0
        self.intercept_counter = 0
        self.threat_counter = 0
        self.destroyed_counter = 0

        # LSTM Model & Radar
        self.radar = Radar3D(sigma=30, frequence=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RocketLSTM3D(input_size=6, hidden_size=128, num_layers=2).to(self.device)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml', 'rocket_lstm3d.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.observation_points = 30 # 3 seconds

        self._build()

    # ═══════════════════════════════════════════════
    #  BUILD UI
    # ═══════════════════════════════════════════════
    def _build(self):
        # ── 3D Figure ──
        self.fig = go.FigureWidget()

        # Trace 0: Protected zone
        zx = [self.zone['x_min'], self.zone['x_max'],
              self.zone['x_max'], self.zone['x_min'],
              self.zone['x_min']]
        zz = [self.zone['z_min'], self.zone['z_min'],
              self.zone['z_max'], self.zone['z_max'],
              self.zone['z_min']]
        self.fig.add_trace(go.Scatter3d(
            x=zx, y=zz, z=[0]*5, mode='lines',
            line=dict(color=self.C_ZONE, width=4),
            name='PROTECTED ZONE', showlegend=True
        ))

        # Trace 1: Base
        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            marker=dict(color=self.C_ACCENT, size=6, symbol='diamond'),
            name='LAUNCHER', showlegend=True
        ))

        self.fig.update_layout(
            title=dict(
                text='IRON DOME — COMMAND & CONTROL',
                font=dict(size=16, color=self.C_TEXT, family='Consolas, monospace'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(title='X (m)', color=self.C_MUTED,
                           gridcolor='#1a2040', backgroundcolor=self.C_BG),
                yaxis=dict(title='Z (m)', color=self.C_MUTED,
                           gridcolor='#1a2040', backgroundcolor=self.C_BG),
                zaxis=dict(title='ALT (m)', color=self.C_MUTED,
                           gridcolor='#1a2040', backgroundcolor=self.C_BG),
                aspectmode='data',
                bgcolor=self.C_BG
            ),
            height=700,
            paper_bgcolor=self.C_PANEL,
            plot_bgcolor=self.C_BG,
            font=dict(color=self.C_TEXT, family='Consolas, monospace'),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False
        )

        # ── Widgets ──
        panel_layout = widgets.Layout(width='340px')
        btn_w = widgets.Layout(width='300px', height='40px')

        self.btn_launch = widgets.Button(
            description='LAUNCH THREAT',
            layout=btn_w,
            style=dict(button_color='#1e3a5f', font_weight='bold'))
        self.btn_salvo = widgets.Button(
            description='LAUNCH SALVO (5)',
            layout=btn_w,
            style=dict(button_color='#1e3a5f', font_weight='bold'))
        self.btn_fire = widgets.Button(
            description='FIRE ALL INTERCEPTORS',
            layout=btn_w,
            disabled=True,
            style=dict(button_color=self.C_DANGER, font_weight='bold'))
        self.btn_reset = widgets.Button(
            description='RESET',
            layout=btn_w,
            style=dict(button_color='#374151', font_weight='bold'))

        self.btn_launch.on_click(self._on_launch_one)
        self.btn_salvo.on_click(self._on_launch_salvo)
        self.btn_fire.on_click(self._on_fire_all)
        
        self.btn_auto = widgets.ToggleButton(
            value=False,
            description='AUTO-DEFENSE: OFF',
            icon='shield',
            layout=btn_w,
            style=dict(font_weight='bold')
        )
        self.btn_auto.observe(self._on_auto_toggle, names='value')
        
        self.btn_reset.on_click(self._on_reset)

        hdr = f"""<div style='font-family:Consolas,monospace;color:{self.C_TEXT};
                   font-size:13px;padding:2px 0;border-bottom:1px solid {self.C_BORDER};
                   margin-bottom:6px;font-weight:bold;letter-spacing:1px'>
                   SITUATION REPORT</div>"""

        self.lbl_header = widgets.HTML(hdr)
        self.lbl_total = widgets.HTML(self._stat_html('TRACKS', '0'))
        self.lbl_threats = widgets.HTML(self._stat_html('THREATS', '0'))
        self.lbl_safe = widgets.HTML(self._stat_html('NON-THREAT', '0'))
        self.lbl_fired = widgets.HTML(self._stat_html('INTERCEPTORS', '0'))
        self.lbl_destroyed = widgets.HTML(self._stat_html('DESTROYED', '0'))
        self.lbl_status = widgets.HTML(self._status_html('STANDBY', self.C_MUTED))
        self.lbl_log = widgets.HTML(self._log_html('System ready.'))

        sep = widgets.HTML(f"<hr style='border:none;border-top:1px solid {self.C_BORDER};margin:6px 0'>")
        sep2 = widgets.HTML(f"<hr style='border:none;border-top:1px solid {self.C_BORDER};margin:6px 0'>")
        sep3 = widgets.HTML(f"<hr style='border:none;border-top:1px solid {self.C_BORDER};margin:6px 0'>")

        panel = widgets.VBox([
            self.lbl_header,
            self.lbl_total, self.lbl_threats, self.lbl_safe,
            self.lbl_fired, self.lbl_destroyed,
            sep,
            self.lbl_status,
            sep2,
            self.btn_launch, self.btn_salvo, self.btn_fire, self.btn_auto, self.btn_reset,
            sep3,
            self.lbl_log
        ], layout=widgets.Layout(
            width='340px', padding='12px',
            border=f'1px solid {self.C_BORDER}',
            overflow_y='auto'
        ))

        self.dashboard = widgets.HBox([self.fig, panel])

        # ── Lower Widgets (Table & Score) ──
        self.score_widget = widgets.HTML('')
        self.table_widget = widgets.HTML('')
        
        lower_panel = widgets.VBox([
            self.score_widget,
            self.table_widget
        ], layout=widgets.Layout(margin='10px 0 0 0'))

        self.app_layout = widgets.VBox([self.dashboard, lower_panel])

    # ── HTML helpers ──
    def _stat_html(self, label, value, color=None):
        c = color or self.C_TEXT
        return (f"<div style='font-family:Consolas,monospace;font-size:12px;"
                f"color:{self.C_MUTED};padding:1px 0'>"
                f"{label}: <span style='color:{c};font-weight:bold'>"
                f"{value}</span></div>")

    def _status_html(self, text, color):
        return (f"<div style='font-family:Consolas,monospace;font-size:14px;"
                f"font-weight:bold;color:{color};text-align:center;"
                f"padding:6px;letter-spacing:1px'>{text}</div>")

    def _log_html(self, text):
        return (f"<div style='font-family:Consolas,monospace;font-size:10px;"
                f"color:{self.C_MUTED};max-height:120px;overflow-y:auto;"
                f"padding:2px'>{text}</div>")

    # ═══════════════════════════════════════════════
    #  DISPLAY
    # ═══════════════════════════════════════════════
    def show(self):
        display(self.app_layout)

    # ═══════════════════════════════════════════════
    #  STATS UPDATE
    # ═══════════════════════════════════════════════
    def _update_threat_table(self):
        if not self.rockets:
            self.table_widget.value = ""
            return
            
        html = f"""<table style='width:100%; border-collapse: collapse; font-family: Consolas, monospace; font-size: 13px; color: {self.C_TEXT}; background-color: {self.C_PANEL}; border: 1px solid {self.C_BORDER};'>
            <tr style='background-color: #1a2040; color: #94a3b8; text-align: left;'>
                <th style='padding: 8px; border-bottom: 1px solid {self.C_BORDER};'>TARGET ID</th>
                <th style='padding: 8px; border-bottom: 1px solid {self.C_BORDER};'>STATUS</th>
                <th style='padding: 8px; border-bottom: 1px solid {self.C_BORDER};'>ETA (s)</th>
                <th style='padding: 8px; border-bottom: 1px solid {self.C_BORDER};'>LSTM PREDICTED IMPACT (X, Z)</th>
            </tr>"""
            
        for r in self.rockets:
            rid = f"TGT-{r['id']:02d}"
            
            # Status
            if r['done']:
                if r.get('interceptor') and r['interceptor'].get('intercepte'):
                    status = f"<span style='color: {self.C_ACCENT}'>DESTROYED</span>"
                elif r['threat']:
                    status = f"<span style='color: {self.C_DANGER}'>IMPACTED</span>"
                else:
                    status = f"<span style='color: {self.C_SUCCESS}'>SAFE</span>"
            elif r.get('assessed'):
                if r['threat']:
                    status = f"<span style='color: {self.C_DANGER}'>THREAT</span>"
                else:
                    status = f"<span style='color: {self.C_SUCCESS}'>NON-THREAT</span>"
            else:
                status = f"<span style='color: {self.C_WARNING}'>TRACKING...</span>"
                
            # ETA
            tc = r['traj'][0]
            remaining_steps = len(tc) - r['step']
            eta = max(0, remaining_steps * 0.1)
            eta_str = f"{eta:.1f}s" if not r['done'] else "-"
            
            # Impact
            if r.get('predicted_impact'):
                imp_x, imp_z = r['predicted_impact']
                impact_str = f"({imp_x:.0f}m, {imp_z:.0f}m)"
            else:
                impact_str = "CALCULATING..."
                
            html += f"""<tr>
                <td style='padding: 6px 8px; border-bottom: 1px solid #1a2040;'>{rid}</td>
                <td style='padding: 6px 8px; border-bottom: 1px solid #1a2040;'>{status}</td>
                <td style='padding: 6px 8px; border-bottom: 1px solid #1a2040;'>{eta_str}</td>
                <td style='padding: 6px 8px; border-bottom: 1px solid #1a2040;'>{impact_str}</td>
            </tr>"""
            
        html += "</table>"
        self.table_widget.value = html

    def _show_scoring(self):
        if not self.rockets:
            return
            
        total_threats = sum(1 for r in self.rockets if r.get('assessed') and r['threat'])
        ignored = sum(1 for r in self.rockets if r.get('assessed') and not r['threat'])
        missiles_used = sum(1 for r in self.rockets if r.get('interceptor'))
        destroyed = self.destroyed_counter
        
        # Collect altitudes and distances
        altitudes = []
        distances = []
        for r in self.rockets:
            if r.get('interceptor') and r['interceptor'].get('intercepte'):
                int_data = r['interceptor']
                altitudes.append(int_data['missile_y'][-1])
                dist = np.sqrt(int_data['missile_x'][-1]**2 + int_data['missile_y'][-1]**2 + int_data['missile_z'][-1]**2)
                distances.append(dist)
                
        taux_interception = (destroyed / total_threats * 100) if total_threats > 0 else (100.0 if destroyed == 0 else 0.0)
        efficacite = (destroyed / max(missiles_used, 1)) * 100
        alt_moy = np.mean(altitudes) if altitudes else 0
        dist_moy = np.mean(distances) if distances else 0
        
        s = 0
        s += min(taux_interception, 100) * 0.4
        s += efficacite * 0.2
        s += min(alt_moy / 1000, 1) * 100 * 0.2
        s += min(dist_moy / 5000, 1) * 100 * 0.2  # Reward for far interception (max at 5km)
        score_global = round(s, 1)
        
        color = self.C_SUCCESS if score_global >= 80 else (self.C_WARNING if score_global >= 50 else self.C_DANGER)
        
        html = f"""<div style='background-color: {self.C_PANEL}; border: 1px solid {self.C_BORDER}; padding: 15px; margin-bottom: 15px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; font-family: Consolas, monospace;'>
            <div>
                <div style='color: {self.C_MUTED}; font-size: 12px;'>MISSION REPORT (END OF SALVO)</div>
                <div style='color: {self.C_TEXT}; font-size: 13px; margin-top: 5px;'>
                    Threats: <span style='color: {self.C_DANGER};'>{total_threats}</span> | 
                    Ignored: <span style='color: {self.C_SUCCESS};'>{ignored}</span> | 
                    Missiles Fired: <span style='color: {self.C_ACCENT};'>{missiles_used}</span> | 
                    Destroyed: <span style='color: {self.C_SUCCESS}; font-weight:bold;'>{destroyed}</span><br>
                    <span style='color: {self.C_MUTED}; font-size: 11px;'>Avg Alt: {alt_moy:.0f}m | Avg Dist: {dist_moy:.0f}m | Efficiency: {efficacite:.0f}%</span>
                </div>
            </div>
            <div style='text-align: right;'>
                <div style='color: {self.C_MUTED}; font-size: 12px;'>GLOBAL SCORE</div>
                <div style='color: {color}; font-size: 28px; font-weight: bold;'>{score_global}/100</div>
            </div>
        </div>"""
        self.score_widget.value = html

    def _update_stats(self):
        total = len(self.rockets)
        threats = sum(1 for r in self.rockets if r['threat'])
        safe = total - threats
        self.lbl_total.value = self._stat_html('TRACKS', str(total), self.C_ACCENT)
        self.lbl_threats.value = self._stat_html('THREATS', str(threats), self.C_DANGER)
        self.lbl_safe.value = self._stat_html('NON-THREAT', str(safe), self.C_SUCCESS)
        self.lbl_fired.value = self._stat_html('INTERCEPTORS', str(self.intercept_counter), self.C_INTERCEPT)
        self.lbl_destroyed.value = self._stat_html('DESTROYED', str(self.destroyed_counter), self.C_HIT)

        if threats > 0 and not any(r.get('interceptor') for r in self.rockets if r['threat']):
            self.btn_fire.disabled = False
            self.lbl_status.value = self._status_html('THREAT DETECTED', self.C_DANGER)
        elif self.destroyed_counter > 0:
            self.lbl_status.value = self._status_html('ENGAGEMENT COMPLETE', self.C_SUCCESS)

    def _append_log(self, msg):
        ts = time.strftime('%H:%M:%S')
        current = self.lbl_log.value
        line = f"<div style='color:{self.C_MUTED};font-size:10px;font-family:Consolas,monospace'>[{ts}] {msg}</div>"
        self.lbl_log.value = line + current

    # ═══════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════
    def _on_launch_one(self, _b):
        self._spawn_rocket()

    def _on_launch_salvo(self, _b):
        for _ in range(5):
            self._spawn_rocket()
            time.sleep(0.05)

    def _on_fire_all(self, _b):
        self.btn_fire.disabled = True
        with self.lock:
            for r in self.rockets:
                if r['threat'] and not r.get('interceptor'):
                    self._fire_at(r)
                    
    def _on_auto_toggle(self, change):
        if change['new']:
            self.btn_auto.description = 'AUTO-DEFENSE: ON'
            self.btn_auto.style.button_color = self.C_HIT
        else:
            self.btn_auto.description = 'AUTO-DEFENSE: OFF'
            self.btn_auto.style.button_color = None

    def _on_reset(self, _b):
        self.running = False
        time.sleep(0.3)
        with self.lock:
            self.rockets.clear()
        self.rocket_counter = 0
        self.intercept_counter = 0
        self.threat_counter = 0
        self.destroyed_counter = 0

        # Remove all dynamic traces (keep 0=zone, 1=base)
        while len(self.fig.data) > 2:
            self.fig.data = self.fig.data[:2]

        self._update_stats()
        self.lbl_status.value = self._status_html('STANDBY', self.C_MUTED)
        self.lbl_log.value = self._log_html('System reset.')
        self.score_widget.value = ''
        self.table_widget.value = ''
        self.btn_fire.disabled = True

    # ═══════════════════════════════════════════════
    #  SPAWN ROCKET
    # ═══════════════════════════════════════════════
    def _spawn_rocket(self):
        self.rocket_counter += 1
        rid = self.rocket_counter

        gen = ThreatGenerator3D(seed=int(time.time() * 1000 + rid) % 1000000)
        rocket = gen.generer_une_menace()
        tc, xc, yc, zc = rocket.trajectoire_rk4(dt=0.1)

        # Generate noisy radar measurements for the whole trajectory
        radar_mesures = self.radar.observer(tc, xc, yc, zc, duree=tc[-1])

        # Threat status is UNKNOWN initially
        is_threat = False

        # Add traces: rocket trajectory + impact marker
        color = self.C_WARNING # Initial color until LSTM assessment
        label = f'TGT-{rid:02d}'

        with self.fig.batch_update():
            # Rocket trace
            self.fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[], mode='lines+markers',
                marker=dict(color=color, size=2),
                line=dict(color=color, width=2),
                name=label, showlegend=False
            ))
            rocket_trace_idx = len(self.fig.data) - 1

            # Impact marker
            self.fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[], mode='markers',
                marker=dict(color=self.C_IMPACT, size=6, symbol='x'),
                name=f'{label} IMP', showlegend=False
            ))
            impact_trace_idx = len(self.fig.data) - 1

        entry = {
            'id': rid,
            'rocket': rocket,
            'traj': (tc, xc, yc, zc),
            'radar': radar_mesures,
            'threat': is_threat,
            'assessed': False,
            'step': 0,
            'trace_idx': rocket_trace_idx,
            'impact_idx': impact_trace_idx,
            'interceptor': None,
            'int_trace_idx': None,
            'hit_trace_idx': None,
            'int_step': 0,
            'done': False,
            'result': None
        }

        with self.lock:
            self.rockets.append(entry)

        self._append_log(f'{label} detected — TRACKING (need {self.observation_points} pts)')
        self._update_stats()

        # Start animation if not already running
        if not self.running:
            self.running = True
            t = threading.Thread(target=self._animation_loop, daemon=True)
            t.start()

    # ═══════════════════════════════════════════════
    #  FIRE AT SPECIFIC ROCKET
    # ═══════════════════════════════════════════════
    def _fire_at(self, entry):
        tc, xc, yc, zc = entry['traj']
        idx = min(entry['step'], len(tc) - 2)

        missile = Interceptor3D(
            *self.base_pos,
            vitesse=self.missile_speed,
            kill_radius=self.kill_radius
        )
        t_rem = tc[idx:] - tc[idx]
        result = missile.proportional_navigation_3d(xc[idx:], yc[idx:], zc[idx:], t_rem)
        entry['interceptor'] = result
        entry['fire_step'] = idx  # rocket step when interceptor was launched

        # Compute rocket index at interception time for sync
        if result['intercepte'] and result['temps_interception'] is not None:
            t_hit = tc[idx] + result['temps_interception']
            entry['interception_rocket_idx'] = min(int(np.searchsorted(tc, t_hit)), len(tc) - 1)
        else:
            entry['interception_rocket_idx'] = len(tc) - 1

        self.intercept_counter += 1

        # Add interceptor trace + hit marker
        with self.fig.batch_update():
            self.fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[], mode='lines+markers',
                marker=dict(color=self.C_INTERCEPT, size=2),
                line=dict(color=self.C_INTERCEPT, width=2),
                name=f'INT-{entry["id"]:02d}', showlegend=False
            ))
            entry['int_trace_idx'] = len(self.fig.data) - 1

            self.fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[], mode='markers',
                marker=dict(color=self.C_HIT, size=10, symbol='diamond'),
                name=f'HIT-{entry["id"]:02d}', showlegend=False
            ))
            entry['hit_trace_idx'] = len(self.fig.data) - 1

        self._append_log(f'Interceptor launched at TGT-{entry["id"]:02d}')
        self._update_stats()

    # ═══════════════════════════════════════════════
    #  ANIMATION LOOP
    # ═══════════════════════════════════════════════
    def _animation_loop(self):
        while self.running:
            all_done = True

            with self.lock:
                rockets_copy = list(self.rockets)

            for entry in rockets_copy:
                if entry['done']:
                    continue
                all_done = False

                tc, xc, yc, zc = entry['traj']
                n = len(tc)
                i = entry['step']

                if i < n:
                    # Update rocket trace
                    tidx = entry['trace_idx']
                    with self.fig.batch_update():
                        self.fig.data[tidx].x = tuple(xc[:i+1])
                        self.fig.data[tidx].y = tuple(zc[:i+1])
                        self.fig.data[tidx].z = tuple(yc[:i+1])

                        # LSTM Threat assessment at 'observation_points'
                        if i == self.observation_points and not entry['done'] and not entry.get('assessed'):
                            entry['assessed'] = True
                            
                            t_obs = entry['radar']['t'][:self.observation_points]
                            x_obs = entry['radar']['x'][:self.observation_points]
                            y_obs = entry['radar']['y'][:self.observation_points]
                            z_obs = entry['radar']['z'][:self.observation_points]
                            
                            dt_r = t_obs[1] - t_obs[0]
                            vx_obs = np.gradient(x_obs, dt_r)
                            vy_obs = np.gradient(y_obs, dt_r)
                            vz_obs = np.gradient(z_obs, dt_r)
                            
                            seq = np.column_stack((x_obs/1000, y_obs/1000, z_obs/1000, vx_obs/1000, vy_obs/1000, vz_obs/1000))
                            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                pred = self.model(seq_t).cpu().numpy()[0]
                                
                            pred_x, pred_z = pred[0]*1000, pred[1]*1000
                            entry['predicted_impact'] = (pred_x, pred_z)
                            
                            is_threat = (self.zone['x_min'] <= pred_x <= self.zone['x_max'] and
                                         self.zone['z_min'] <= pred_z <= self.zone['z_max'])
                            entry['threat'] = is_threat
                            
                            # Show predicted impact
                            iidx = entry['impact_idx']
                            self.fig.data[iidx].x = (pred_x,)
                            self.fig.data[iidx].y = (pred_z,)
                            self.fig.data[iidx].z = (0.0,)
                            
                            if is_threat:
                                self.threat_counter += 1
                                self.fig.data[tidx].line.color = self.C_ROCKET
                                self.fig.data[tidx].marker.color = self.C_ROCKET
                                self._append_log(f'TGT-{entry["id"]:02d} THREAT DETECTED (LSTM Pred: {pred_x:.0f}, {pred_z:.0f})')
                                self.btn_fire.disabled = False
                                self.lbl_status.value = self._status_html('THREAT DETECTED', self.C_DANGER)
                                
                                # Auto-Fire Logic
                                if getattr(self, 'btn_auto', None) and self.btn_auto.value:
                                    self._fire_at(entry)
                            else:
                                self._append_log(f'TGT-{entry["id"]:02d} NON-THREAT (LSTM Pred: {pred_x:.0f}, {pred_z:.0f})')
                            
                            self._update_stats()

                        # Animate interceptor if launched (synchronized with rocket)
                        if entry['interceptor'] and entry['int_trace_idx'] is not None:
                            r = entry['interceptor']
                            mx, my, mz = r['missile_x'], r['missile_y'], r['missile_z']
                            max_int = len(mx)

                            # Sync: compute interceptor frame from rocket progress
                            fire_step = entry['fire_step']
                            target_idx = entry['interception_rocket_idx']
                            total_rocket_frames = max(target_idx - fire_step, 1)
                            rocket_frames_elapsed = i - fire_step
                            progress = min(rocket_frames_elapsed / total_rocket_frames, 1.0)
                            ii = min(int(progress * (max_int - 1)), max_int - 1)

                            self.fig.data[entry['int_trace_idx']].x = tuple(mx[:ii+1])
                            self.fig.data[entry['int_trace_idx']].y = tuple(mz[:ii+1])
                            self.fig.data[entry['int_trace_idx']].z = tuple(my[:ii+1])

                            if progress >= 1.0 and not entry['done']:
                                entry['done'] = True
                                # Also show rocket up to interception point
                                self.fig.data[tidx].x = tuple(xc[:target_idx+1])
                                self.fig.data[tidx].y = tuple(zc[:target_idx+1])
                                self.fig.data[tidx].z = tuple(yc[:target_idx+1])
                                if r['intercepte']:
                                    self.destroyed_counter += 1
                                    self.fig.data[entry['hit_trace_idx']].x = (float(mx[-1]),)
                                    self.fig.data[entry['hit_trace_idx']].y = (float(mz[-1]),)
                                    self.fig.data[entry['hit_trace_idx']].z = (float(my[-1]),)
                                    self._append_log(
                                        f'TGT-{entry["id"]:02d} DESTROYED — '
                                        f'd={r["distance_min"]:.1f}m  '
                                        f'alt={float(my[-1]):.0f}m')
                                else:
                                    self._append_log(
                                        f'TGT-{entry["id"]:02d} MISSED — '
                                        f'd_min={r["distance_min"]:.1f}m')
                                self._update_stats()

                    entry['step'] += 1
                else:
                    if not entry.get('interceptor'):
                        entry['done'] = True

            self._update_threat_table()

            early_complete = True
            if not rockets_copy:
                early_complete = False
            for r in rockets_copy:
                if not r.get('assessed'):
                    early_complete = False
                    break
                if r.get('threat') and not r.get('done'):
                    early_complete = False
                    break
                if r.get('interceptor') and not r.get('done'):
                    early_complete = False
                    break

            if all_done or early_complete:
                self.btn_fire.disabled = True
                self._append_log("THREATS NEUTRALIZED. SALVO COMPLETE.")
                self._show_scoring()
                self.running = False
                break

            time.sleep(0.12)

        self.running = False


# ── Launch ──
if __name__ == '__main__':
    dashboard = IronDomeDashboard()
    dashboard.show()
