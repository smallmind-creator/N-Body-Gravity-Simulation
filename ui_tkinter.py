import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation # For MP4 export
import csv
import os
import time
import json 

from physics_engine import SimBody, SimulationEngine

class NBodyApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("3D N-Body Gravitational Simulator")
        self.root.geometry("1400x900") # Increased size for more UI elements

        self.sim_engine = SimulationEngine()
        self.initial_body_config_dicts = [] 

        self.is_running = False
        self.simulation_mode = tk.StringVar(value="pre_defined") 
        self.total_sim_time_var = tk.DoubleVar(value=30 * 24 * 3600.0) 
        
        self.precalculated_frames_body_dicts = [] 
        self.precalculated_frame_times = [] 
        self.animation_frame_index = 0 
        self.animation_timer_id = None 

        self.camera_mode = tk.StringVar(value="free") 
        self.camera_target_body_id = tk.IntVar(value=-1) 
        self.camera_com_target_body_ids_str = tk.StringVar(value="") 

        self.auto_rotate_3d = tk.BooleanVar(value=True) 
        self.current_3d_azim = -60 
        self.current_3d_elev = 30  
        self.rotation_timer_id = None 

        self.plot_range_current = 4e8 
        self.plot_center_current = np.zeros(3) 

        self.energy_time_data = []
        self.total_energy_data = []
        self.initial_total_energy = None

        # New Variables for Enhancements
        self.time_scale_multiplier = tk.DoubleVar(value=1.0)
        self.projection_mode = tk.StringVar(value="3d") # '3d', 'xy', 'xz', 'yz'
        self.minimal_ui_mode = tk.BooleanVar(value=False)
        self.app_state = tk.StringVar(value="main_menu") # 'main_menu', 'simulating', 'node_editor_placeholder'

        # UI Frames
        self.main_menu_frame = None
        self.simulation_frame = None # To hold left_panel and right_panel (and potentially energy plot)
        self.left_panel = None
        self.right_panel = None 
        self.bottom_bar_frame = None # For new bottom controls

        # Energy Plot
        self.energy_fig = None
        self.energy_ax = None
        self.energy_canvas = None
        
        self._setup_ui()
        # _load_default_scenario() is called by _set_app_state('simulating') via main menu button
        # _start_3d_rotation_loop() also managed by state changes

        self._set_app_state("main_menu") # Start with main menu

    def _setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Main Menu Frame ---
        self.main_menu_frame = ttk.Frame(self.root, padding="50")
        # main_menu_frame will be gridded/ungridded by _set_app_state
        ttk.Label(self.main_menu_frame, text="Quantum Strata Dynamics", font=("Rajdhani", 32, "bold"), foreground="cyan").pack(pady=20)
        ttk.Label(self.main_menu_frame, text="N-Body Gravitational Simulator", font=("Rajdhani", 18)).pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Initialize Universe", command=lambda: self._set_app_state("simulating"), style="Accent.TButton").pack(pady=30, ipady=10, ipadx=20)
        self.root.style = ttk.Style()
        self.root.style.configure("Accent.TButton", font=("Rajdhani", 14, "bold"), padding=10)


        # --- Simulation Frame (holds left panel, right panel, energy plot) ---
        self.simulation_frame = ttk.Frame(self.root, padding="5")
        # simulation_frame will be gridded/ungridded by _set_app_state
        self.simulation_frame.columnconfigure(0, weight=1) # Left panel + Energy plot column
        self.simulation_frame.columnconfigure(1, weight=3) # Right panel (3D viz)
        self.simulation_frame.rowconfigure(0, weight=2) # Left/Right panels take most height
        self.simulation_frame.rowconfigure(1, weight=1) # Energy plot row

        # --- Left Panel (Controls) ---
        self.left_panel = ttk.Labelframe(self.simulation_frame, text="Controls", padding="10")
        # self.left_panel will be gridded inside simulation_frame
        # Grid configure for left_panel content
        for i in range(20): self.left_panel.rowconfigure(i, weight=0) # Default no weight

        row_idx = 0
        # Sim Mode Switch
        sim_mode_frame = ttk.Frame(self.left_panel)
        sim_mode_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=2)
        ttk.Label(sim_mode_frame, text="Sim Mode:").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sim_mode_frame, text="Pre-defined", variable=self.simulation_mode, value="pre_defined", command=self._on_mode_change_requested).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sim_mode_frame, text="Real-time", variable=self.simulation_mode, value="real_time", command=self._on_mode_change_requested).pack(side=tk.LEFT, padx=2)
        row_idx += 1
        
        self.total_sim_time_label = ttk.Label(self.left_panel, text="Total Sim Time (s):")
        self.total_sim_time_label.grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        self.total_sim_time_entry = ttk.Entry(self.left_panel, textvariable=self.total_sim_time_var, width=12)
        self.total_sim_time_entry.grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW); row_idx+=1
        
        # Separator
        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=5); row_idx+=1

        # Camera Controls
        ttk.Label(self.left_panel, text="Camera & View:", font=("Rajdhani", 10, "bold")).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(5,2)); row_idx+=1
        
        cam_mode_frame = ttk.Frame(self.left_panel)
        cam_mode_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=2)
        ttk.Radiobutton(cam_mode_frame, text="Free", variable=self.camera_mode, value="free", command=self._update_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(cam_mode_frame, text="Auto-Rot (3D)", variable=self.auto_rotate_3d, command=self._toggle_3d_rotation).pack(side=tk.LEFT, padx=10)
        row_idx += 1

        follow_body_frame = ttk.Frame(self.left_panel)
        follow_body_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=2)
        ttk.Radiobutton(follow_body_frame, text="Follow Body ID:", variable=self.camera_mode, value="follow_body", command=self._update_visualization).pack(side=tk.LEFT)
        self.camera_target_body_id_entry = ttk.Entry(follow_body_frame, textvariable=self.camera_target_body_id, width=5)
        self.camera_target_body_id_entry.pack(side=tk.LEFT, padx=5)
        row_idx += 1
        
        follow_com_frame = ttk.Frame(self.left_panel)
        follow_com_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=2)
        ttk.Radiobutton(follow_com_frame, text="Follow CoM IDs:", variable=self.camera_mode, value="follow_com", command=self.handle_com_camera_selection_change).pack(side=tk.LEFT)
        self.camera_com_ids_entry = ttk.Entry(follow_com_frame, textvariable=self.camera_com_target_body_ids_str, width=10)
        self.camera_com_ids_entry.pack(side=tk.LEFT, padx=5)
        row_idx += 1

        projection_frame = ttk.Frame(self.left_panel)
        projection_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=2)
        ttk.Label(projection_frame, text="Projection:").pack(side=tk.LEFT)
        ttk.Radiobutton(projection_frame, text="3D", variable=self.projection_mode, value="3d", command=self._on_projection_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(projection_frame, text="XY", variable=self.projection_mode, value="xy", command=self._on_projection_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(projection_frame, text="XZ", variable=self.projection_mode, value="xz", command=self._on_projection_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(projection_frame, text="YZ", variable=self.projection_mode, value="yz", command=self._on_projection_change).pack(side=tk.LEFT, padx=2)
        row_idx += 1

        # Separator
        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=5); row_idx+=1

        # Pod-like Buttons
        ttk.Label(self.left_panel, text="Configuration:", font=("Rajdhani", 10, "bold")).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(5,2)); row_idx+=1
        ttk.Button(self.left_panel, text="Add Body", command=self.open_add_body_pod).grid(row=row_idx, column=0, columnspan=3, pady=3, sticky=tk.EW); row_idx+=1
        ttk.Button(self.left_panel, text="System Config", command=self.open_system_config_pod).grid(row=row_idx, column=0, columnspan=3, pady=3, sticky=tk.EW); row_idx+=1
        ttk.Button(self.left_panel, text="Object Inspector", command=self.open_object_inspector_pod).grid(row=row_idx, column=0, columnspan=3, pady=3, sticky=tk.EW); row_idx+=1
        ttk.Button(self.left_panel, text="Scenarios", command=self.open_scenario_pod).grid(row=row_idx, column=0, columnspan=3, pady=3, sticky=tk.EW); row_idx+=1
        ttk.Button(self.left_panel, text="Node Editor (Placeholder)", command=self.open_node_editor_placeholder).grid(row=row_idx, column=0, columnspan=3, pady=3, sticky=tk.EW); row_idx+=1
        
        # Separator
        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW, pady=5); row_idx+=1
        
        # Export Buttons
        ttk.Label(self.left_panel, text="Export:", font=("Rajdhani", 10, "bold")).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(5,2)); row_idx+=1
        export_buttons_frame = ttk.Frame(self.left_panel)
        export_buttons_frame.grid(row=row_idx, column=0, columnspan=3, sticky=tk.EW)
        self.export_csv_button = ttk.Button(export_buttons_frame, text="Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.export_csv_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.export_video_button = ttk.Button(export_buttons_frame, text="Export MP4", command=self.export_mp4_video, state=tk.DISABLED)
        self.export_video_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        row_idx+=1


        # --- Right Panel (3D Visualization) ---
        self.right_panel = ttk.Labelframe(self.simulation_frame, text="3D Visualization", padding="10")
        # self.right_panel will be gridded inside simulation_frame

        self.fig = Figure(figsize=(7,6), dpi=100) # Adjusted size
        self._create_plot_axes() # Create initial axes based on projection_mode
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel) 
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Energy Plot Panel (Below Left Panel) ---
        energy_plot_frame = ttk.Labelframe(self.simulation_frame, text="System Energy", padding="5")
        # energy_plot_frame will be gridded inside simulation_frame

        self.energy_fig = Figure(figsize=(5,2.5), dpi=80)
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_ax.set_xlabel("Time (days)")
        self.energy_ax.set_ylabel("Total Energy (J)")
        self.energy_fig.tight_layout()
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=energy_plot_frame)
        self.energy_canvas_widget = self.energy_canvas.get_tk_widget()
        self.energy_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        # --- Bottom Bar (Play/Pause, Time Scale, View Mode) ---
        self.bottom_bar_frame = ttk.Frame(self.root, padding="5")
        # self.bottom_bar_frame will be gridded/ungridded by _set_app_state

        # Play/Pause/Reset Buttons
        controls_frame = ttk.Frame(self.bottom_bar_frame)
        controls_frame.pack(side=tk.LEFT, padx=10)
        self.play_button = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_simulation)
        self.play_button.pack(side=tk.LEFT, padx=2)
        self.pause_button = ttk.Button(controls_frame, text="❚❚ Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        self.reset_button = ttk.Button(controls_frame, text="↺ Reset", command=self.reset_simulation_from_button)
        self.reset_button.pack(side=tk.LEFT, padx=2)

        # Time Scale Slider
        time_scale_frame = ttk.Frame(self.bottom_bar_frame)
        time_scale_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(time_scale_frame, text="Anim Speed:").pack(side=tk.LEFT)
        self.time_scale_slider = ttk.Scale(time_scale_frame, from_=0.1, to=10.0, variable=self.time_scale_multiplier, orient=tk.HORIZONTAL, length=150, command=self._update_time_scale_label)
        self.time_scale_slider.pack(side=tk.LEFT, padx=5)
        self.time_scale_label = ttk.Label(time_scale_frame, text=f"{self.time_scale_multiplier.get():.1f}x")
        self.time_scale_label.pack(side=tk.LEFT)

        # View Mode Toggle
        view_mode_frame = ttk.Frame(self.bottom_bar_frame)
        view_mode_frame.pack(side=tk.RIGHT, padx=10)
        self.minimal_mode_button = ttk.Checkbutton(view_mode_frame, text="Minimal UI", variable=self.minimal_ui_mode, command=self._toggle_minimal_mode)
        self.minimal_mode_button.pack(side=tk.LEFT)
        ttk.Button(view_mode_frame, text="Back to Menu", command=lambda: self._set_app_state("main_menu")).pack(side=tk.LEFT, padx=5)


        # Status Bar (at the very bottom of the root window)
        self.status_var = tk.StringVar(value="State: IDLE | Mode: Pre-defined")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        # self.status_bar will be gridded/ungridded by _set_app_state
        
        self._apply_current_mode_ui_state()

    def _create_plot_axes(self):
        """Recreates the main plot axes based on current projection mode."""
        if hasattr(self, 'ax') and self.ax:
            self.fig.delaxes(self.ax)

        proj = self.projection_mode.get()
        if proj == "3d":
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)"); self.ax.set_zlabel("Z (m)")
            self.auto_rotate_3d.set(True) # Enable auto-rotate for 3D
            self._toggle_3d_rotation()
        else:
            self.ax = self.fig.add_subplot(111)
            if proj == "xy": self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
            elif proj == "xz": self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Z (m)")
            elif proj == "yz": self.ax.set_xlabel("Y (m)"); self.ax.set_ylabel("Z (m)")
            self.ax.set_aspect('equal', adjustable='box')
            self.auto_rotate_3d.set(False) # Disable auto-rotate for 2D
            self._toggle_3d_rotation()
        
        if hasattr(self, 'canvas'): # Redraw if canvas exists
            self.canvas.draw_idle()

    def _on_projection_change(self):
        self._create_plot_axes()
        self._update_visualization()


    def _set_app_state(self, new_state):
        self.app_state.set(new_state)
        
        # Hide all main frames first
        self.main_menu_frame.grid_forget()
        self.simulation_frame.grid_forget()
        self.bottom_bar_frame.grid_forget()
        self.status_bar.grid_forget()

        if new_state == "main_menu":
            self.main_menu_frame.grid(row=0, column=0, sticky="nsew")
            if self.is_running: self.pause_simulation() # Pause if returning to menu while running
        
        elif new_state == "simulating":
            self.simulation_frame.grid(row=0, column=0, sticky="nsew")
            
            # Grid the internal components of simulation_frame
            self.left_panel.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)
            self.right_panel.grid(row=0, column=1, rowspan=2, sticky="nswe", padx=5, pady=5) # Span 2 rows for 3D viz
            
            energy_plot_frame = self.energy_canvas_widget.master # Get the Labelframe
            energy_plot_frame.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)

            self.bottom_bar_frame.grid(row=1, column=0, sticky="ew")
            self.status_bar.grid(row=2, column=0, sticky="ew")
            
            if not self.initial_body_config_dicts: # Load default only if nothing is configured
                self._load_default_scenario()
            else: # If returning to sim view, ensure visualization is current
                self.reset_simulation_to_initial_config() # Resets and redraws

            self._apply_current_mode_ui_state()
            self._start_3d_rotation_loop()
        
        elif new_state == "node_editor_placeholder":
            # For now, just show a message or a very simple Toplevel
            # This state would typically overlay or replace parts of the 'simulating' state
            # Re-grid simulating state so user can return
            self.simulation_frame.grid(row=0, column=0, sticky="nsew")
            self.left_panel.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)
            self.right_panel.grid(row=0, column=1, rowspan=2, sticky="nswe", padx=5, pady=5)
            energy_plot_frame = self.energy_canvas_widget.master
            energy_plot_frame.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)
            self.bottom_bar_frame.grid(row=1, column=0, sticky="ew")
            self.status_bar.grid(row=2, column=0, sticky="ew")
            self.open_node_editor_placeholder() # Open the placeholder dialog

    def _update_time_scale_label(self, event=None):
        self.time_scale_label.config(text=f"{self.time_scale_multiplier.get():.1f}x")

    def _toggle_minimal_mode(self):
        if self.minimal_ui_mode.get():
            self.left_panel.grid_remove()
            self.energy_canvas_widget.master.grid_remove() # Remove energy plot's labelframe
            # Optionally make bottom bar less prominent or hide parts of it
        else:
            self.left_panel.grid()
            self.energy_canvas_widget.master.grid()
        # The main 3D viz (right_panel) and bottom_bar_frame remain.
        # Status bar also remains.

    def _load_default_scenario(self):
        self.sim_engine.clear_bodies()
        self.initial_body_config_dicts = []
        m_earth = 5.972e24; m_moon = 7.348e22; dist_em = 3.844e8
        G_const = self.sim_engine.G
        if G_const <= 0: messagebox.showerror("Config Error", "G must be positive."); return
        M_total = m_earth + m_moon
        r_earth_from_com = (m_moon / M_total) * dist_em
        r_moon_from_com = (m_earth / M_total) * dist_em
        pos_earth = np.array([-r_earth_from_com, 0.0, 0.0]); pos_moon = np.array([r_moon_from_com, 0.0, 0.0])
        try: omega = np.sqrt(G_const * M_total / (dist_em**3))
        except ZeroDivisionError: messagebox.showerror("Math Error", "dist_em cannot be zero."); return
        v_earth_mag = omega * r_earth_from_com; v_moon_mag = omega * r_moon_from_com
        vel_earth = np.array([0.0, v_earth_mag, 0.0]); vel_moon = np.array([0.0, -v_moon_mag, 0.0])
        try:
            b1_dict = SimBody(0, "Earth", m_earth, pos_earth, vel_earth, 6.371e6, 'deepskyblue').to_dict()
            b2_dict = SimBody(1, "Moon", m_moon, pos_moon, vel_moon, 1.737e6, 'lightgrey').to_dict()
        except (ValueError, TypeError) as e: messagebox.showerror("Body Error", f"{e}"); return
        self.initial_body_config_dicts = [b1_dict, b2_dict]
        self.sim_engine.dt = 3600.0 * 1 
        self.total_sim_time_var.set(60 * 24 * 3600.0)
        self.plot_range_current = dist_em * 1.5 
        self.reset_simulation_to_initial_config()

    def _apply_current_mode_ui_state(self):
        mode = self.simulation_mode.get()
        if mode == "pre_defined":
            self.total_sim_time_label.grid()
            self.total_sim_time_entry.grid()
            can_export = bool(self.precalculated_frames_body_dicts) and not self.is_running
            self.export_csv_button.config(state=tk.NORMAL if can_export else tk.DISABLED)
            self.export_video_button.config(state=tk.NORMAL if can_export else tk.DISABLED)
        else: 
            self.total_sim_time_label.grid_remove()
            self.total_sim_time_entry.grid_remove()
            self.export_csv_button.config(state=tk.DISABLED)
            self.export_video_button.config(state=tk.DISABLED)

        if self.is_running: 
            self.play_button.config(text="❚❚ Pause") 
            self.pause_button.config(state=tk.NORMAL) 
            self.reset_button.config(state=tk.DISABLED) 
            status_prefix = "RUNNING" if mode == "real_time" else "ANIMATING"
            self.status_var.set(f"State: {status_prefix} ({mode.replace('_',' ').title()}) | Time: {self.sim_engine.time_elapsed / (24*3600):.1f} days")
        else: 
            play_text = "▶ Play"
            if mode == "pre_defined" and self.precalculated_frames_body_dicts and self.animation_frame_index < len(self.precalculated_frames_body_dicts) and self.animation_frame_index > 0 :
                play_text = "▶ Resume Anim"
            self.play_button.config(text=play_text)
            self.pause_button.config(state=tk.DISABLED) 
            self.reset_button.config(state=tk.NORMAL) 
            status_prefix = "IDLE"
            if mode == "pre_defined" and self.precalculated_frames_body_dicts and self.animation_frame_index >= len(self.precalculated_frames_body_dicts):
                status_prefix = "FINISHED (Anim)" 
            elif mode == "pre_defined" and self.precalculated_frames_body_dicts and self.animation_frame_index > 0:
                 status_prefix = "PAUSED (Anim)" 
            self.status_var.set(f"State: {status_prefix} | Mode: {mode.replace('_',' ').title()}")
        
        # Update energy plot too
        self._update_energy_plot()


    def _on_mode_change_requested(self):
        if self.is_running:
            self.is_running = False 
            if self.animation_timer_id:
                self.root.after_cancel(self.animation_timer_id)
                self.animation_timer_id = None
        
        self.reset_simulation_to_initial_config() 
        self._apply_current_mode_ui_state()      
        self._start_3d_rotation_loop()     

    def reset_simulation_to_initial_config(self):
        self.is_running = False
        if self.animation_timer_id: 
            self.root.after_cancel(self.animation_timer_id)
            self.animation_timer_id = None
        
        self.sim_engine.clear_bodies() 
        for b_dict in self.initial_body_config_dicts:
            try:
                self.sim_engine.add_body_instance(SimBody.from_dict(b_dict))
            except (ValueError, TypeError) as e:
                messagebox.showerror("Config Error", f"Failed to load body: {e}"); continue 

        self.sim_engine.reset_time_and_trails() 
        
        self.precalculated_frames_body_dicts = []
        self.precalculated_frame_times = []
        self.animation_frame_index = 0 
        
        self.energy_time_data = []
        self.total_energy_data = []
        self.initial_total_energy = None
        
        if self.sim_engine.bodies:
             _,_, self.initial_total_energy = self.sim_engine.get_system_energy()
             if self.initial_total_energy is not None:
                self.energy_time_data.append(0.0) 
                self.total_energy_data.append(self.initial_total_energy)
        
        self._update_visualization() 
        self._apply_current_mode_ui_state() 

    def reset_simulation_from_button(self):
        self.reset_simulation_to_initial_config()
        messagebox.showinfo("Reset", "Simulation reset to initial conditions.")
        self._start_3d_rotation_loop() 

    def handle_com_camera_selection_change(self):
        if self.camera_mode.get() == "follow_com":
            self._update_visualization()


    def _update_visualization(self, bodies_for_vis=None, time_for_vis=None):
        current_bodies_raw = bodies_for_vis if bodies_for_vis is not None else self.sim_engine.bodies
        # Filter out merged bodies for visualization if using live engine bodies
        current_bodies = [b for b in current_bodies_raw if not (isinstance(b, SimBody) and b.merged)]
        
        current_t = time_for_vis if time_for_vis is not None else self.sim_engine.time_elapsed

        self.ax.cla() 
        proj_mode = self.projection_mode.get()

        if proj_mode == "3d":
            self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)"); self.ax.set_zlabel("Z (m)")
            self.ax.set_facecolor((0.05, 0.05, 0.1))
        else:
            if proj_mode == "xy": self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
            elif proj_mode == "xz": self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Z (m)")
            elif proj_mode == "yz": self.ax.set_xlabel("Y (m)"); self.ax.set_ylabel("Z (m)")
            self.ax.set_facecolor((0.1, 0.1, 0.15)) # Slightly different for 2D
            self.ax.grid(True, linestyle=':', alpha=0.5)
        
        self.ax.set_title(f"Time: {current_t / (24*3600):.2f} days", loc='center', pad=15 if proj_mode == "3d" else 5)

        if not current_bodies: 
            lim = self.plot_range_current
            if proj_mode == "3d":
                self.ax.set_xlim([-lim, lim]); self.ax.set_ylim([-lim, lim]); self.ax.set_zlim([-lim, lim])
                self.ax.view_init(elev=self.current_3d_elev, azim=self.current_3d_azim)
            else:
                self.ax.set_xlim([-lim, lim]); self.ax.set_ylim([-lim, lim])
                self.ax.set_aspect('equal', adjustable='box')
            self.canvas.draw_idle()
            return

        view_center_3d = np.zeros(3)
        current_view_range = self.plot_range_current 

        cam_mode = self.camera_mode.get()
        if cam_mode == "follow_body":
            target_id = self.camera_target_body_id.get()
            target_b_state = next((b for b in current_bodies if (b.id if isinstance(b, SimBody) else b.get('id')) == target_id), None)
            if target_b_state:
                view_center_3d = target_b_state.pos if isinstance(target_b_state, SimBody) else np.array(target_b_state['pos'])
                all_radii = [b.radius for b in self.sim_engine.bodies if not b.merged] # Use live non-merged bodies
                current_view_range = max(all_radii) * 100 if all_radii else self.plot_range_current * 0.1
                current_view_range = max(current_view_range, 1e7) 
            else: 
                view_center_3d, _ = self.sim_engine.get_center_of_mass() # Fallback to CoM of live bodies
                if target_id != -1: messagebox.showwarning("Camera Target", f"Body ID {target_id} not found.")
                self.camera_mode.set("free") 
        
        elif cam_mode == "follow_com":
            try:
                ids_str = self.camera_com_target_body_ids_str.get()
                target_ids_list = [int(s.strip()) for s in ids_str.split(',') if s.strip()] if ids_str else []
                
                # For CoM calculation, always use the live engine's non-merged bodies if possible
                com_source_bodies = [b for b in self.sim_engine.bodies if not b.merged]
                
                com_bodies_to_consider = []
                if target_ids_list:
                    com_bodies_to_consider = [b for b in com_source_bodies if b.id in target_ids_list]
                else: # CoM of all currently active bodies
                    com_bodies_to_consider = com_source_bodies
                
                if com_bodies_to_consider:
                    total_m = sum(b.mass for b in com_bodies_to_consider)
                    if abs(total_m) > 1e-18:
                        view_center_3d = sum(b.mass * b.pos for b in com_bodies_to_consider) / total_m
                    else: view_center_3d = np.zeros(3)
                else: view_center_3d = np.zeros(3)
                
                if com_bodies_to_consider:
                    all_pos = np.array([b.pos for b in com_bodies_to_consider])
                    if all_pos.size > 0:
                        max_span = np.max(np.max(all_pos, axis=0) - np.min(all_pos, axis=0)) if len(all_pos) > 1 else 1e7
                        current_view_range = max(max_span * 0.8, 1e7) 
            except ValueError:
                messagebox.showwarning("CoM Error", "Invalid Body IDs for CoM.")
                view_center_3d, _ = self.sim_engine.get_center_of_mass()
                self.camera_mode.set("free")
        
        else: # "free" camera
            all_coords = []
            for body_obj in current_bodies: # current_bodies already filtered for merged
                pos_to_use = body_obj.pos if isinstance(body_obj, SimBody) else np.array(body_obj['pos'])
                all_coords.append(pos_to_use)
                trail_to_use = body_obj.trail if isinstance(body_obj, SimBody) else \
                               [np.array(p) for p in body_obj.get('trail',[])]
                if trail_to_use: all_coords.extend(trail_to_use)
            
            if all_coords:
                all_coords_arr = np.array(all_coords)
                min_c = np.min(all_coords_arr, axis=0); max_c = np.max(all_coords_arr, axis=0)
                view_center_3d = (min_c + max_c) / 2
                max_span = np.max(max_c - min_c)
                current_view_range = max(max_span * 0.6, self.plot_range_current * 0.1, 1e6) # Min range
            else:
                 view_center_3d = self.plot_center_current
                 current_view_range = self.plot_range_current

        # Set plot limits based on projection
        if proj_mode == "3d":
            self.ax.set_xlim(view_center_3d[0] - current_view_range, view_center_3d[0] + current_view_range)
            self.ax.set_ylim(view_center_3d[1] - current_view_range, view_center_3d[1] + current_view_range)
            self.ax.set_zlim(view_center_3d[2] - current_view_range, view_center_3d[2] + current_view_range)
            self.ax.set_aspect('equal', adjustable='box') 
            self.ax.view_init(elev=self.current_3d_elev, azim=self.current_3d_azim)
        else: # 2D
            idx_map = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}
            x_idx, y_idx = idx_map[proj_mode]
            self.ax.set_xlim(view_center_3d[x_idx] - current_view_range, view_center_3d[x_idx] + current_view_range)
            self.ax.set_ylim(view_center_3d[y_idx] - current_view_range, view_center_3d[y_idx] + current_view_range)
            self.ax.set_aspect('equal', adjustable='box')

        # --- Draw Bodies and Trails ---
        masses = [bd.mass if isinstance(bd, SimBody) else bd.get('mass', 1.0) for bd in current_bodies]
        min_mass = min((m for m in masses if m > 0), default=1e-30) 
        max_mass = max(masses, default=1.0)
        log_min_mass = np.log10(min_mass) if min_mass > 0 else -30
        log_max_mass = np.log10(max_mass) if max_mass > 0 else 1
        if log_max_mass <= log_min_mass : log_max_mass = log_min_mass + 1 
        
        for body_data in current_bodies: # Already filtered
            pos = body_data.pos if isinstance(body_data, SimBody) else np.array(body_data.get('pos', [0,0,0]))
            color = body_data.color if isinstance(body_data, SimBody) else body_data.get('color', 'gray')
            mass_val = body_data.mass if isinstance(body_data, SimBody) else body_data.get('mass', 1.0)
            trail_points = body_data.trail if isinstance(body_data, SimBody) else \
                           [np.array(p) for p in body_data.get('trail',[])]

            if trail_points:
                trail_arr = np.array(trail_points)
                if trail_arr.ndim == 2 and trail_arr.shape[1] == 3: 
                    if proj_mode == "3d":
                        self.ax.plot(trail_arr[:,0], trail_arr[:,1], trail_arr[:,2], '-', color=color, alpha=0.5, linewidth=0.8, zorder=1)
                    else:
                        self.ax.plot(trail_arr[:,x_idx], trail_arr[:,y_idx], '-', color=color, alpha=0.5, linewidth=0.8, zorder=1)
            
            s_size = 10 
            if mass_val > 0 : 
                log_mass = np.log10(mass_val)
                s_size = 10 + 290 * (log_mass - log_min_mass) / (log_max_mass - log_min_mass)
            s_size = max(10, min(s_size, 300)) 

            if proj_mode == "3d":
                self.ax.scatter(pos[0], pos[1], pos[2], color=color, s=s_size, edgecolors='darkgrey', linewidth=0.3, zorder=10, depthshade=True)
            else:
                self.ax.scatter(pos[x_idx], pos[y_idx], color=color, s=s_size, edgecolors='darkgrey', linewidth=0.3, zorder=10)
            
        self.canvas.draw_idle() 
        self._apply_current_mode_ui_state() # Update status bar with time

    def _update_energy_plot(self):
        if not self.energy_ax or not hasattr(self.energy_canvas_widget, 'winfo_exists') or not self.energy_canvas_widget.winfo_exists():
             return # Plot not ready or destroyed

        self.energy_ax.cla()
        if self.energy_time_data and self.total_energy_data:
            time_days = np.array(self.energy_time_data) / (24 * 3600)
            self.energy_ax.plot(time_days, self.total_energy_data, marker='.', linestyle='-', markersize=2, linewidth=1, color='cyan')
            if self.initial_total_energy is not None and len(self.total_energy_data) > 1:
                # Show energy conservation (percentage change from initial)
                # Only if there's more than one data point and initial energy is non-zero
                if abs(self.initial_total_energy) > 1e-9: # Avoid division by zero or near-zero
                    perc_change = ((self.total_energy_data[-1] - self.initial_total_energy) / self.initial_total_energy) * 100
                    self.energy_ax.set_title(f"Total Energy (ΔE: {perc_change:.3e}%)", fontsize=9)
                else:
                    self.energy_ax.set_title("Total Energy", fontsize=9)

        self.energy_ax.set_xlabel("Time (days)", fontsize=8)
        self.energy_ax.set_ylabel("Energy (J)", fontsize=8)
        self.energy_ax.tick_params(axis='both', which='major', labelsize=7)
        self.energy_ax.grid(True, linestyle=':', alpha=0.6)
        self.energy_fig.tight_layout(pad=0.5)
        self.energy_canvas.draw_idle()

    def _toggle_3d_rotation(self):
        if self.projection_mode.get() != "3d": # No rotation in 2D
            self.auto_rotate_3d.set(False) 
        if self.auto_rotate_3d.get():
            self._start_3d_rotation_loop()
        else:
            if self.rotation_timer_id:
                self.root.after_cancel(self.rotation_timer_id)
                self.rotation_timer_id = None

    def _start_3d_rotation_loop(self):
        if self.rotation_timer_id: 
            self.root.after_cancel(self.rotation_timer_id)
            self.rotation_timer_id = None
        
        if self.auto_rotate_3d.get() and self.projection_mode.get() == "3d" and self.app_state.get() == "simulating": 
            self._rotate_3d_view_step() 

    def _rotate_3d_view_step(self):
        if not self.auto_rotate_3d.get() or self.projection_mode.get() != "3d" or self.app_state.get() != "simulating":
            self.rotation_timer_id = None
            return 
        
        self.current_3d_azim = (self.current_3d_azim + 0.25) % 360 
        if not self.is_running: self._update_visualization() 
        
        self.rotation_timer_id = self.root.after(40, self._rotate_3d_view_step) 

    def _simulation_loop_realtime(self):
        if not self.is_running or self.app_state.get() != "simulating": return 

        start_time_step = time.perf_counter() 
        self.sim_engine.simulation_step()
        _, _, total_e = self.sim_engine.get_system_energy()
        self.energy_time_data.append(self.sim_engine.time_elapsed)
        self.total_energy_data.append(total_e)
        self._update_visualization()
        self._update_energy_plot() # Update energy plot
        
        elapsed_step_time = time.perf_counter() - start_time_step
        target_frame_time_ms = (1000 / 30) / self.time_scale_multiplier.get()
        delay_ms = max(1, int(target_frame_time_ms - (elapsed_step_time * 1000)))
        self.animation_timer_id = self.root.after(delay_ms, self._simulation_loop_realtime)

    def _precalculate_simulation(self):
        if self.app_state.get() != "simulating": return
        self.status_var.set("State: CALCULATING (Pre-defined)... Please wait.")
        self.root.update_idletasks() 

        self.reset_simulation_to_initial_config() 
        self.precalculated_frames_body_dicts = [] 
        self.precalculated_frame_times = []
        
        try:
            total_sim_duration = self.total_sim_time_var.get()
            if total_sim_duration <= 0 or self.sim_engine.dt <= 0: raise ValueError("Sim time/dt must be > 0.")
            total_steps = int(total_sim_duration / self.sim_engine.dt)
        except ValueError as e:
            messagebox.showerror("Input Error", f"{e}"); self._apply_current_mode_ui_state(); return

        if total_steps == 0:
            messagebox.showinfo("Info", "Sim time too short for dt."); self._apply_current_mode_ui_state(); return

        if self.sim_engine.bodies: self.sim_engine._calculate_accelerations()

        for step in range(total_steps):
            self.sim_engine.simulation_step() 
            frame_body_states = [b.to_dict() for b in self.sim_engine.bodies if not b.merged]
            # Store trails as they are *at this point in time*
            for i, b_engine_state in enumerate(self.sim_engine.bodies):
                if b_engine_state.merged: continue
                corresponding_dict = next((d for d in frame_body_states if d['id'] == b_engine_state.id), None)
                if corresponding_dict:
                    corresponding_dict['trail'] = [p.tolist() for p in b_engine_state.trail]
            
            self.precalculated_frames_body_dicts.append(frame_body_states)
            self.precalculated_frame_times.append(self.sim_engine.time_elapsed)
            _, _, total_e = self.sim_engine.get_system_energy()
            self.energy_time_data.append(self.sim_engine.time_elapsed)
            self.total_energy_data.append(total_e)

            if step % (max(1, total_steps // 20)) == 0: 
                self.status_var.set(f"State: CALCULATING ({step*100/total_steps:.0f}%)")
                self.root.update_idletasks() 

        self._apply_current_mode_ui_state() 
        messagebox.showinfo("Pre-calculation Complete", "Data calculated.")
        self._start_3d_rotation_loop() 

    def _animate_precalculated_data(self):
        if not self.precalculated_frames_body_dicts or not self.is_running or self.app_state.get() != "simulating":
            self.is_running = False; self._apply_current_mode_ui_state()
            if self.animation_timer_id: self.root.after_cancel(self.animation_timer_id); self.animation_timer_id = None
            self._start_3d_rotation_loop(); return

        if self.animation_frame_index >= len(self.precalculated_frames_body_dicts): self.animation_frame_index = 0 

        frame_body_dicts = self.precalculated_frames_body_dicts[self.animation_frame_index]
        frame_time = self.precalculated_frame_times[self.animation_frame_index]
        temp_bodies_for_vis = []
        for b_data in frame_body_dicts:
            body_for_vis = SimBody.from_dict(b_data) 
            body_for_vis.trail = [np.array(p) for p in b_data.get('trail',[])] 
            temp_bodies_for_vis.append(body_for_vis)
        self._update_visualization(bodies_for_vis=temp_bodies_for_vis, time_for_vis=frame_time)
        # Energy plot uses the full dataset, not per-frame, so it's updated by _apply_current_mode_ui_state
        
        self.animation_frame_index += 1
        delay = int((1000 / 30) / self.time_scale_multiplier.get()) # ~30 FPS adjusted by time scale
        self.animation_timer_id = self.root.after(max(1, delay), self._animate_precalculated_data) 

    def toggle_simulation(self):
        if self.app_state.get() != "simulating": return # Only allow play if in sim state

        if self.is_running: self.pause_simulation()
        else: 
            self.is_running = True
            if self.rotation_timer_id: self.root.after_cancel(self.rotation_timer_id); self.rotation_timer_id = None
            self._apply_current_mode_ui_state(); self.reset_button.config(state=tk.DISABLED) 
            mode = self.simulation_mode.get()
            if mode == "real_time":
                if not self.sim_engine.bodies:
                    messagebox.showwarning("No Bodies", "Add bodies first."); self.is_running = False; self._apply_current_mode_ui_state(); self._start_3d_rotation_loop(); return
                if not any(not b.merged for b in self.sim_engine.bodies): # Check if all bodies are merged
                    messagebox.showwarning("No Active Bodies", "All bodies merged or none active."); self.is_running = False; self._apply_current_mode_ui_state(); self._start_3d_rotation_loop(); return
                self.sim_engine._calculate_accelerations(); self._simulation_loop_realtime() 
            elif mode == "pre_defined":
                if not self.initial_body_config_dicts:
                    messagebox.showwarning("No Config", "Configure bodies first."); self.is_running = False; self._apply_current_mode_ui_state(); self._start_3d_rotation_loop(); return
                if not self.precalculated_frames_body_dicts: self._precalculate_simulation() 
                if self.precalculated_frames_body_dicts: 
                    if self.animation_frame_index >= len(self.precalculated_frames_body_dicts): self.animation_frame_index = 0
                    self._animate_precalculated_data() 
    
    def pause_simulation(self):
        if self.is_running:
            self.is_running = False
            if self.animation_timer_id: self.root.after_cancel(self.animation_timer_id); self.animation_timer_id = None
            self._apply_current_mode_ui_state(); self._start_3d_rotation_loop() 

    def export_csv(self):
        if self.simulation_mode.get() != "pre_defined" or not self.precalculated_frames_body_dicts:
            messagebox.showerror("Export Error", "CSV export is for completed pre-defined simulations."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not filepath: return 
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Parameter", "Value"])
                writer.writerow(["G", self.sim_engine.G]); writer.writerow(["TimeStep", self.sim_engine.dt])
                writer.writerow(["TotalSimTime_Predefined", self.total_sim_time_var.get()])
                writer.writerow(["Integrator", self.sim_engine.integrator_type])
                writer.writerow(["CollisionModel", self.sim_engine.collision_model])
                writer.writerow([])
                for b_dict in self.initial_body_config_dicts:
                    name = b_dict.get('name', f"Body{b_dict['id']}")
                    writer.writerow([f"{name}_ID", b_dict['id']])
                    writer.writerow([f"{name}_Mass", b_dict['mass']])
                    writer.writerow([f"{name}_InitialPosX", b_dict['pos'][0]])
                    writer.writerow([f"{name}_InitialPosY", b_dict['pos'][1]])
                    writer.writerow([f"{name}_InitialPosZ", b_dict['pos'][2]])
                    writer.writerow([f"{name}_InitialVelX", b_dict['vel'][0]])
                    writer.writerow([f"{name}_InitialVelY", b_dict['vel'][1]])
                    writer.writerow([f"{name}_InitialVelZ", b_dict['vel'][2]])
                    writer.writerow([f"{name}_Radius", b_dict['radius']])
                writer.writerow([])
                
                header = ["Time"]
                # Use names from initial_body_config_dicts as precalc might have merged bodies
                # This assumes IDs in initial_config match those used in precalculated_frames (before merges)
                # or that users are interested in tracking based on initial setup.
                # A more robust CSV would handle merged bodies explicitly, perhaps by not including them
                # or by adding columns for merged bodies as they appear. For simplicity, we use initial bodies.
                
                # Let's use the bodies present in the first frame of precalculated data for headers
                # if precalculated_frames_body_dicts is not empty.
                
                body_names_in_first_frame = {} # id: name
                if self.precalculated_frames_body_dicts and self.precalculated_frames_body_dicts[0]:
                    for b_dict_frame0 in self.precalculated_frames_body_dicts[0]:
                        body_names_in_first_frame[b_dict_frame0['id']] = b_dict_frame0.get('name', f"Body{b_dict_frame0['id']}")
                        header.extend([f"{body_names_in_first_frame[b_dict_frame0['id']]}_Px", f"{body_names_in_first_frame[b_dict_frame0['id']]}_Py", f"{body_names_in_first_frame[b_dict_frame0['id']]}_Pz",
                                       f"{body_names_in_first_frame[b_dict_frame0['id']]}_Vx", f"{body_names_in_first_frame[b_dict_frame0['id']]}_Vy", f"{body_names_in_first_frame[b_dict_frame0['id']]}_Vz"])
                writer.writerow(header)

                for i, frame_time in enumerate(self.precalculated_frame_times):
                    row = [frame_time]
                    frame_data_map = {bd['id']: bd for bd in self.precalculated_frames_body_dicts[i]}
                    for body_id in body_names_in_first_frame.keys(): # Iterate in order of first frame appearance
                        b_dict = frame_data_map.get(body_id)
                        if b_dict: # If body still exists in this frame
                            row.extend(b_dict['pos'])
                            row.extend(b_dict['vel'])
                        else: # Body might have merged or disappeared
                            row.extend(["N/A"] * 6) # Fill with N/A for missing data
                    writer.writerow(row)
            messagebox.showinfo("Export Successful", f"Data exported to {filepath}")
        except Exception as e: messagebox.showerror("Export Error", f"{e}")
            
    def export_mp4_video(self):
        if self.simulation_mode.get() != "pre_defined" or not self.precalculated_frames_body_dicts:
            messagebox.showerror("Export Error", "MP4 export for completed pre-defined simulations."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if not filepath: return

        self.status_var.set("State: EXPORTING MP4... Please wait.")
        self.root.update_idletasks()
        
        original_azim = self.current_3d_azim
        original_elev = self.current_3d_elev
        original_auto_rotate = self.auto_rotate_3d.get()
        self.auto_rotate_3d.set(False) # Disable live auto-rotate during export

        fig_export = self.fig # Use the existing figure
        
        def update_frame_for_video(frame_idx):
            self.ax.cla() # Clear current axes content for redrawing
            frame_body_dicts = self.precalculated_frames_body_dicts[frame_idx]
            frame_time = self.precalculated_frame_times[frame_idx]
            temp_bodies = [SimBody.from_dict(b_data) for b_data in frame_body_dicts]
            for i, b_obj in enumerate(temp_bodies): # Restore trails for this frame
                 b_obj.trail = [np.array(p) for p in frame_body_dicts[i].get('trail',[])]

            # Simplified visualization call for export (no UI interactions needed for camera here)
            # For consistent video, use fixed camera or a pre-programmed camera path.
            # Here, we'll use the camera settings active at the start of export.
            # Or, to be safe, force free camera with auto-range for export.
            # For this implementation, we use the current camera view settings.
            # You might want to add options for fixed camera during export later.
            self._update_visualization(bodies_for_vis=temp_bodies, time_for_vis=frame_time)
            self.status_var.set(f"State: EXPORTING MP4... Frame {frame_idx+1}/{len(self.precalculated_frame_times)}")
            self.root.update_idletasks() # Keep UI responsive for status
            return self.ax, # FuncAnimation expects a tuple of artists

        try:
            # Framerate for video, e.g., 30 fps
            fps = 30 
            # Interval calculation for FuncAnimation (not directly used by FFMpegWriter's save)
            # interval = 1000 / fps 
            
            # Create the animation object
            ani = animation.FuncAnimation(fig_export, update_frame_for_video, 
                                          frames=len(self.precalculated_frame_times), 
                                          blit=False) # Blit=False is safer for complex plots

            # Save the animation
            # Ensure FFmpeg is installed and in PATH, or specify path to ffmpeg.exe
            # matplotlib.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\ffmpeg.exe'
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='NBodyApp'), bitrate=1800)
            ani.save(filepath, writer=writer, dpi=150) # Adjust DPI as needed for quality

            messagebox.showinfo("Export Successful", f"MP4 video saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export MP4: {e}\nEnsure FFmpeg is installed and in PATH.")
        finally:
            self.current_3d_azim = original_azim # Restore camera view
            self.current_3d_elev = original_elev
            self.auto_rotate_3d.set(original_auto_rotate)
            self._apply_current_mode_ui_state() # Reset status
            self._start_3d_rotation_loop()


    def open_add_body_pod(self):
        pod = tk.Toplevel(self.root); pod.title("Add New Body"); pod.transient(self.root); pod.grab_set()
        fields = ["Name", "Mass (kg)", "Pos X", "Pos Y", "Pos Z", "Vel X", "Vel Y", "Vel Z", "Radius (m)"]
        entries = {}
        next_id_for_config = max((b['id'] for b in self.initial_body_config_dicts), default=-1) + 1 if self.initial_body_config_dicts else 0
        
        for i, field in enumerate(fields):
            ttk.Label(pod, text=field + ":").grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            entry = ttk.Entry(pod, width=25); entry.grid(row=i, column=1, padx=5, pady=2, sticky=tk.EW)
            entries[field] = entry
        
        entries["Name"].insert(0, f"Body{next_id_for_config}")
        entries["Mass (kg)"].insert(0, "1.0e20")
        entries["Pos X"].insert(0, f"{np.random.uniform(-1e8, 1e8):.2e}")
        # ... (rest of default values)
        entries["Radius (m)"].insert(0, "1.0e6")
        chosen_color = tk.StringVar(value="#808080") 
        def pick_color():
            cc = colorchooser.askcolor(title="Choose body color")
            if cc and cc[1]: chosen_color.set(cc[1]); preview.config(bg=chosen_color.get())
        ttk.Label(pod, text="Color:").grid(row=len(fields), column=0)
        ttk.Button(pod, text="Pick", command=pick_color).grid(row=len(fields), column=1, sticky=tk.W)
        preview = tk.Label(pod, text="  ", bg=chosen_color.get(), relief=tk.SUNKEN); preview.grid(row=len(fields), column=1, sticky=tk.E)

        def submit():
            try:
                p = [float(entries["Pos X"].get()), float(entries["Pos Y"].get()), float(entries["Pos Z"].get())]
                v = [float(entries["Vel X"].get()), float(entries["Vel Y"].get()), float(entries["Vel Z"].get())]
                nbd = SimBody(next_id_for_config, entries["Name"].get(), float(entries["Mass (kg)"].get()), 
                              p, v, float(entries["Radius (m)"].get()), chosen_color.get()).to_dict()
                self.initial_body_config_dicts.append(nbd)
                if not self.is_running: self.reset_simulation_to_initial_config()
                else: messagebox.showinfo("Info", "New body added to config. Reset simulation to include it.")
                pod.destroy()
            except Exception as e: messagebox.showerror("Invalid Input", f"{e}")
        ttk.Button(pod, text="Add Body", command=submit).grid(row=len(fields)+1, columnspan=2, pady=10)
        self.root.wait_window(pod)

    def open_system_config_pod(self):
        pod = tk.Toplevel(self.root); pod.title("System Configuration"); pod.transient(self.root); pod.grab_set()
        g = tk.DoubleVar(value=self.sim_engine.G); dt = tk.DoubleVar(value=self.sim_engine.dt)
        integrator = tk.StringVar(value=self.sim_engine.integrator_type)
        collision = tk.StringVar(value=self.sim_engine.collision_model)
        ttk.Label(pod, text="G:").grid(row=0, column=0, sticky=tk.W); ttk.Entry(pod, textvariable=g).grid(row=0, column=1)
        ttk.Label(pod, text="dt (s):").grid(row=1, column=0, sticky=tk.W); ttk.Entry(pod, textvariable=dt).grid(row=1, column=1)
        ttk.Label(pod, text="Integrator:").grid(row=2, column=0, sticky=tk.W)
        ttk.Combobox(pod, textvariable=integrator, values=['rk4', 'verlet'], state='readonly').grid(row=2, column=1)
        ttk.Label(pod, text="Collision Model:").grid(row=3, column=0, sticky=tk.W)
        ttk.Combobox(pod, textvariable=collision, values=['ignore', 'elastic', 'merge'], state='readonly').grid(row=3, column=1) # Added 'merge'
        def apply():
            try:
                if g.get() <= 0 or dt.get() <= 0: raise ValueError("G and dt must be positive.")
                self.sim_engine.G = g.get(); self.sim_engine.dt = dt.get()
                self.sim_engine.integrator_type = integrator.get(); self.sim_engine.collision_model = collision.get()
                messagebox.showinfo("Settings", "Parameters updated. Reset simulation to apply."); pod.destroy()
            except Exception as e: messagebox.showerror("Invalid Input", f"{e}")
        ttk.Button(pod, text="Apply", command=apply).grid(row=4, columnspan=2, pady=10)
        self.root.wait_window(pod)

    def open_object_inspector_pod(self):
        active_bodies = [b for b in self.sim_engine.bodies if not b.merged]
        if not active_bodies: messagebox.showinfo("Inspector", "No active bodies."); return
        
        pod = tk.Toplevel(self.root); pod.title("Object Inspector"); pod.transient(self.root); pod.grab_set()
        body_names_ids = [(f"{b.name} (ID: {b.id})", b.id) for b in active_bodies]
        body_display_names = [name for name, id_val in body_names_ids]
        selected_body_display_name_var = tk.StringVar()
        
        ttk.Label(pod, text="Select Body:").grid(row=0, column=0, sticky=tk.W)
        body_selector = ttk.Combobox(pod, textvariable=selected_body_display_name_var, values=body_display_names, state='readonly', width=30)
        body_selector.grid(row=0, column=1, columnspan=2, sticky=tk.EW)
        
        # Entries for editable fields
        entries = {}
        fields_to_edit = ["Mass (kg)", "Radius (m)"] # Currently only these are easily editable live
        # Read-only display fields
        fields_to_display = ["Position (m)", "Velocity (m/s)", "Acceleration (m/s^2)"]
        
        row_idx = 1
        for field_name in fields_to_edit:
            ttk.Label(pod, text=field_name + ":").grid(row=row_idx, column=0, sticky=tk.W)
            entry = ttk.Entry(pod, width=25)
            entry.grid(row=row_idx, column=1, sticky=tk.EW)
            entries[field_name] = entry
            row_idx += 1
            
        display_labels = {}
        for field_name in fields_to_display:
            ttk.Label(pod, text=field_name + ":").grid(row=row_idx, column=0, sticky=tk.W)
            label = ttk.Label(pod, text="N/A", width=35, anchor='w') # Anchor west
            label.grid(row=row_idx, column=1, columnspan=2, sticky=tk.EW) # Span for longer text
            display_labels[field_name] = label
            row_idx += 1

        current_selected_body_id = tk.IntVar(value=-1)

        def update_inspector_display(event=None):
            selected_text = selected_body_display_name_var.get()
            if not selected_text: return
            
            body_id_to_find = next((id_val for name, id_val in body_names_ids if name == selected_text), None)
            if body_id_to_find is None: return

            current_selected_body_id.set(body_id_to_find)
            body = self.sim_engine.get_body_by_id(body_id_to_find)
            if body and not body.merged:
                entries["Mass (kg)"].delete(0, tk.END); entries["Mass (kg)"].insert(0, f"{body.mass:.3e}")
                entries["Radius (m)"].delete(0, tk.END); entries["Radius (m)"].insert(0, f"{body.radius:.3e}")
                display_labels["Position (m)"].config(text=f"[{body.pos[0]:.2e}, {body.pos[1]:.2e}, {body.pos[2]:.2e}]")
                display_labels["Velocity (m/s)"].config(text=f"[{body.vel[0]:.2e}, {body.vel[1]:.2e}, {body.vel[2]:.2e}]")
                display_labels["Acceleration (m/s^2)"].config(text=f"[{body.acc[0]:.2e}, {body.acc[1]:.2e}, {body.acc[2]:.2e}]")
            else: # Clear fields if body not found or merged
                for entry in entries.values(): entry.delete(0, tk.END); entry.insert(0, "N/A")
                for label in display_labels.values(): label.config(text="N/A")
        
        body_selector.bind("<<ComboboxSelected>>", update_inspector_display)
        if body_display_names: body_selector.set(body_display_names[0]); update_inspector_display()

        def apply_changes():
            body_id = current_selected_body_id.get()
            if body_id == -1: messagebox.showerror("Error", "No body selected."); return
            body = self.sim_engine.get_body_by_id(body_id)
            if not body or body.merged: messagebox.showerror("Error", "Selected body not found or merged."); return
            
            try:
                new_mass = float(entries["Mass (kg)"].get())
                new_radius = float(entries["Radius (m)"].get())
                if new_mass <= 0 or new_radius <=0: raise ValueError("Mass/Radius must be positive.")
                
                body.mass = new_mass
                body.radius = new_radius
                # If simulation is running, changing mass/radius will affect subsequent calculations.
                # For simplicity, we don't recalculate accelerations immediately here;
                # they will be updated in the next simulation step.
                messagebox.showinfo("Applied", f"Changes applied to {body.name}.")
                update_inspector_display() # Refresh display with potentially formatted numbers
                if not self.is_running: self._update_visualization() # Update main view if paused
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"{e}")
        
        ttk.Button(pod, text="Apply Changes", command=apply_changes).grid(row=row_idx, column=0, pady=10)
        ttk.Button(pod, text="Close", command=pod.destroy).grid(row=row_idx, column=1, pady=10)
        self.root.wait_window(pod)

    def open_scenario_pod(self):
        pod = tk.Toplevel(self.root); pod.title("Scenario Manager"); pod.transient(self.root); pod.grab_set()
        def save():
            fp = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if not fp: return
            try:
                data = {
                    "G": self.sim_engine.G, "dt": self.sim_engine.dt,
                    "integrator_type": self.sim_engine.integrator_type,
                    "collision_model": self.sim_engine.collision_model,
                    "total_sim_time": self.total_sim_time_var.get(),
                    "initial_bodies": self.initial_body_config_dicts,
                    "simulation_mode": self.simulation_mode.get() # Save sim mode
                }
                with open(fp, 'w') as f: json.dump(data, f, indent=4)
                messagebox.showinfo("Save", "Scenario saved.")
            except Exception as e: messagebox.showerror("Save Error", f"{e}")
        def load():
            fp = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if not fp: return
            try:
                with open(fp, 'r') as f: data = json.load(f)
                self.sim_engine.G = data.get("G", self.sim_engine.G)
                self.sim_engine.dt = data.get("dt", self.sim_engine.dt)
                self.sim_engine.integrator_type = data.get("integrator_type", self.sim_engine.integrator_type)
                self.sim_engine.collision_model = data.get("collision_model", self.sim_engine.collision_model)
                self.total_sim_time_var.set(data.get("total_sim_time", self.total_sim_time_var.get()))
                self.simulation_mode.set(data.get("simulation_mode", "pre_defined")) # Load sim mode

                loaded_bodies = []
                for b_data in data.get("initial_bodies", []):
                    try: SimBody.from_dict(b_data); loaded_bodies.append(b_data)
                    except Exception as e: messagebox.showwarning("Load Warning", f"Skipping body: {e}")
                self.initial_body_config_dicts = loaded_bodies
                
                self.reset_simulation_to_initial_config() 
                self._on_mode_change_requested() # Apply loaded mode settings and UI
                messagebox.showinfo("Load", "Scenario loaded. Sim reset."); pod.destroy()
            except Exception as e: messagebox.showerror("Load Error", f"{e}")
        ttk.Button(pod, text="Load Scenario", command=load).pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(pod, text="Save Current", command=save).pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(pod, text="Close", command=pod.destroy).pack(pady=10)
        self.root.wait_window(pod)

    def open_node_editor_placeholder(self):
        # This is just a placeholder as requested.
        if self.app_state.get() == "simulating": # Update main app state if opened from sim view
            self._set_app_state("node_editor_placeholder") 
        
        messagebox.showinfo("Node Editor", "[Node Editor Feature - Future Implementation with VisPy]")
        # In a real scenario, this might open a modal Toplevel that takes focus.
        # For now, just show a message and keep the sim view.
