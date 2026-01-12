import sys
import numpy as np
import os
import json
import time
import csv
import math # For pi and power in radius calculation

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout,
    QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QGroupBox, QColorDialog,
    QRadioButton, QStatusBar, QScrollArea, QCheckBox, QSizePolicy, QSlider
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QColor, QPalette, QDoubleValidator

import vispy.scene
from vispy.scene import visuals
from vispy.color import Color, Colormap

from physics_engine import SimBody, SimulationEngine

# --- Helper for Scientific Notation Input ---
class SciLineEdit(QLineEdit):
    """
    A QLineEdit subclass that handles scientific notation input and display.
    It allows flexible input (e.g., "1.23e20", "1.23*10^20") and formats output
    to standard scientific notation for large/small numbers.
    """
    def __init__(self, initial_value=0.0, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignRight)
        self._value = float(initial_value) # Internal float value
        self.setValue(initial_value) # Format initial value for display
        self.editingFinished.connect(self._parse_and_validate)

    def _parse_and_validate(self):
        """Parses the text in the line edit and updates the internal float value."""
        text = self.text().strip()
        if not text: # Handle empty input
            self.setValue(0.0)
            return

        try:
            if 'e' in text.lower():
                val = float(text)
            elif '*' in text and '^' in text:
                parts_mult = text.split('*')
                if len(parts_mult) != 2: raise ValueError("Invalid format for N*10^M")
                base = float(parts_mult[0].strip())
                exp_part = parts_mult[1].lower().replace('e','10').split('^')
                if len(exp_part) != 2: raise ValueError("Invalid format for 10^M")
                exponent = float(exp_part[1].strip())
                val = base * (10 ** exponent)
            else:
                val = float(text)

            self.setValue(val)
        except ValueError:
            QMessageBox.warning(self.parent(), "Invalid Input",
                                f"Could not parse '{text}' as a valid number. Reverting to previous value.")
            self.setValue(self._value)

    def setValue(self, value):
        """Sets the internal float value and formats the text for display."""
        try:
            self._value = float(value)
            if abs(self._value) > 1e5 or (abs(self._value) < 1e-3 and self._value != 0):
                self.setText(f"{self._value:.3e}")
            else:
                self.setText(f"{self._value:.3f}")
        except ValueError:
            self.setText("Error")

    def value(self):
        """Returns the current internal float value."""
        return self._value

# --- Dialogs ---

class AddBodyDialog(QDialog):
    ASSUMED_DENSITY_KG_M3 = 5510

    def __init__(self, parent=None, next_body_id=0, initial_pos=None, initial_vel=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Celestial Body")
        self.setMinimumWidth(450)

        self.layout = QFormLayout(self)
        self.name_edit = QLineEdit(f"Body{next_body_id}")
        self.mass_sci_edit = SciLineEdit(1.0e20, self)

        self.pos_x_edit = QDoubleSpinBox(); self.pos_x_edit.setRange(-1e15, 1e15); self.pos_x_edit.setDecimals(2); self.pos_x_edit.setGroupSeparatorShown(True)
        self.pos_y_edit = QDoubleSpinBox(); self.pos_y_edit.setRange(-1e15, 1e15); self.pos_y_edit.setDecimals(2); self.pos_y_edit.setGroupSeparatorShown(True)
        self.pos_z_edit = QDoubleSpinBox(); self.pos_z_edit.setRange(-1e15, 1e15); self.pos_z_edit.setDecimals(2); self.pos_z_edit.setGroupSeparatorShown(True)

        self.vel_x_edit = QDoubleSpinBox(); self.vel_x_edit.setRange(-1e7, 1e7); self.vel_x_edit.setDecimals(2); self.vel_x_edit.setGroupSeparatorShown(True)
        self.vel_y_edit = QDoubleSpinBox(); self.vel_y_edit.setRange(-1e7, 1e7); self.vel_y_edit.setDecimals(2); self.vel_y_edit.setGroupSeparatorShown(True)
        self.vel_z_edit = QDoubleSpinBox(); self.vel_z_edit.setRange(-1e7, 1e7); self.vel_z_edit.setDecimals(2); self.vel_z_edit.setGroupSeparatorShown(True)

        self.radius_edit = QDoubleSpinBox()
        self.radius_edit.setDecimals(0); self.radius_edit.setRange(1.0, 1e12)
        self.radius_edit.setSuffix(" m"); self.radius_edit.setGroupSeparatorShown(True)

        initial_mass = self.mass_sci_edit.value()
        self.radius_edit.setValue(AddBodyDialog.calculate_radius_from_mass(initial_mass)) # Changed to static call
        self.mass_sci_edit.editingFinished.connect(self.update_radius_from_mass_input)
        self.calc_radius_button = QPushButton("Calc Radius from Mass")
        self.calc_radius_button.clicked.connect(self.update_radius_from_mass_input)

        self.color_button = QPushButton("Choose Color")
        self.chosen_color = QColor(np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255))
        self.color_button.setStyleSheet(f"background-color: {self.chosen_color.name()}; color: {('black' if self.chosen_color.lightnessF() > 0.5 else 'white')};")
        self.color_button.clicked.connect(self.pick_color)

        if initial_pos is not None:
            self.pos_x_edit.setValue(initial_pos[0]); self.pos_y_edit.setValue(initial_pos[1]); self.pos_z_edit.setValue(initial_pos[2])
        else:
            self.pos_x_edit.setValue(np.random.uniform(-1e8, 1e8)); self.pos_y_edit.setValue(np.random.uniform(-1e8, 1e8)); self.pos_z_edit.setValue(0.0)
        if initial_vel is not None:
            self.vel_x_edit.setValue(initial_vel[0]); self.vel_y_edit.setValue(initial_vel[1]); self.vel_z_edit.setValue(initial_vel[2])
        else:
            self.vel_x_edit.setValue(np.random.uniform(-100, 100)); self.vel_y_edit.setValue(np.random.uniform(-100, 100)); self.vel_z_edit.setValue(0.0)

        self.layout.addRow("Name:", self.name_edit)
        self.layout.addRow("Mass (kg, e.g., 1.23e20):", self.mass_sci_edit)
        radius_layout = QHBoxLayout(); radius_layout.addWidget(self.radius_edit); radius_layout.addWidget(self.calc_radius_button)
        self.layout.addRow("Radius:", radius_layout)
        pos_layout = QHBoxLayout(); pos_layout.addWidget(self.pos_x_edit); pos_layout.addWidget(self.pos_y_edit); pos_layout.addWidget(self.pos_z_edit)
        self.layout.addRow("Position (x,y,z) m:", pos_layout)
        vel_layout = QHBoxLayout(); vel_layout.addWidget(self.vel_x_edit); vel_layout.addWidget(self.vel_y_edit); vel_layout.addWidget(self.vel_z_edit)
        self.layout.addRow("Velocity (x,y,z) m/s:", vel_layout)
        self.layout.addRow("Color:", self.color_button)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

    @staticmethod # Changed to staticmethod
    def calculate_radius_from_mass(mass_kg, density_kg_m3=None):
        _density = density_kg_m3 if density_kg_m3 is not None else AddBodyDialog.ASSUMED_DENSITY_KG_M3
        if mass_kg <= 0: return 1.0
        volume_m3 = mass_kg / _density
        radius_m = ( (3 * volume_m3) / (4 * math.pi) )**(1/3)
        return max(1.0, radius_m)

    def update_radius_from_mass_input(self):
        mass = self.mass_sci_edit.value()
        calculated_radius = AddBodyDialog.calculate_radius_from_mass(mass) # Changed to static call
        self.radius_edit.setValue(calculated_radius)

    def pick_color(self):
        color = QColorDialog.getColor(self.chosen_color, self)
        if color.isValid():
            self.chosen_color = color
            self.color_button.setStyleSheet(f"background-color: {self.chosen_color.name()}; color: {('black' if self.chosen_color.lightnessF() > 0.5 else 'white')};")

    def get_body_data(self):
        if self.exec_() == QDialog.Accepted:
            mass_val = self.mass_sci_edit.value()
            if mass_val <= 0: QMessageBox.warning(self, "Invalid Mass", "Mass must be positive."); return None
            radius_val = self.radius_edit.value()
            if radius_val <=0: QMessageBox.warning(self, "Invalid Radius", "Radius must be positive."); return None
            pos = [self.pos_x_edit.value(), self.pos_y_edit.value(), self.pos_z_edit.value()]
            vel = [self.vel_x_edit.value(), self.vel_y_edit.value(), self.vel_z_edit.value()]
            return {"name": self.name_edit.text(), "mass": mass_val, "pos": pos, "vel": vel,
                    "radius": radius_val, "color": self.chosen_color.name()}
        return None

class SystemConfigDialog(QDialog):
    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("System Configuration")
        self.layout = QFormLayout(self)
        self.g_const_edit = QDoubleSpinBox(); self.g_const_edit.setDecimals(15); self.g_const_edit.setRange(1e-20, 1e20); self.g_const_edit.setGroupSeparatorShown(True)
        self.dt_edit = QDoubleSpinBox(); self.dt_edit.setSuffix(" s"); self.dt_edit.setRange(0.01, 1e9); self.dt_edit.setGroupSeparatorShown(True)
        self.integrator_combo = QComboBox(); self.integrator_combo.addItems(['rk4', 'verlet'])
        self.collision_combo = QComboBox(); self.collision_combo.addItems(['ignore', 'elastic', 'merge'])
        if current_config:
            self.g_const_edit.setValue(current_config.get('G', 6.674e-11))
            self.dt_edit.setValue(current_config.get('dt', 3600.0))
            self.integrator_combo.setCurrentText(current_config.get('integrator', 'rk4'))
            self.collision_combo.setCurrentText(current_config.get('collision_model', 'ignore'))
        self.layout.addRow("Gravitational Constant (G):", self.g_const_edit)
        self.layout.addRow("Time Step (dt):", self.dt_edit)
        self.layout.addRow("Integrator:", self.integrator_combo)
        self.layout.addRow("Collision Model:", self.collision_combo)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)
    def get_config_data(self):
        if self.exec_() == QDialog.Accepted:
            return {"G": self.g_const_edit.value(), "dt": self.dt_edit.value(),
                    "integrator": self.integrator_combo.currentText(), "collision_model": self.collision_combo.currentText(),}
        return None

class ObjectInspectorDialog(QDialog):
    ASSUMED_DENSITY_KG_M3 = 5510

    def __init__(self, parent_app, sim_engine, initial_body_configs, selected_body_id=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.sim_engine = sim_engine
        self.initial_body_configs = initial_body_configs

        self.setWindowTitle("Object Inspector")
        self.setMinimumWidth(500)
        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.body_select_combo = QComboBox()
        self.populate_body_selector()
        self.body_select_combo.currentIndexChanged.connect(self.on_body_selected_changed)
        form_layout.addRow("Select Body:", self.body_select_combo)
        self.name_edit = QLineEdit()
        self.mass_sci_edit = SciLineEdit(0.0, self)
        self.radius_display_label = QLabel("Calculated: N/A m")
        self.mass_sci_edit.editingFinished.connect(self.update_radius_display_from_mass)
        self.color_button = QPushButton()
        self.current_qcolor = QColor("#FFFFFF")
        self.color_button.clicked.connect(self.pick_body_color)
        form_layout.addRow("Name:", self.name_edit)
        form_layout.addRow("Mass (kg, e.g., 1.23e20):", self.mass_sci_edit)
        form_layout.addRow("Radius (auto from mass):", self.radius_display_label)
        form_layout.addRow("Color:", self.color_button)
        self.pos_label = QLabel("N/A"); self.vel_label = QLabel("N/A"); self.acc_label = QLabel("N/A")
        form_layout.addRow("Current Position (m):", self.pos_label)
        form_layout.addRow("Current Velocity (m/s):", self.vel_label)
        form_layout.addRow("Current Acceleration (m/s²):", self.acc_label)
        self.layout.addLayout(form_layout)
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Changes"); self.apply_button.clicked.connect(self.apply_changes)
        self.delete_button = QPushButton("Delete Body"); self.delete_button.setStyleSheet("background-color: #A04040; color: white;"); self.delete_button.clicked.connect(self.delete_body)
        self.close_button = QPushButton("Close"); self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.apply_button); button_layout.addWidget(self.delete_button)
        button_layout.addStretch(); button_layout.addWidget(self.close_button)
        self.layout.addLayout(button_layout)

        if selected_body_id is not None: self.select_body_in_combo(selected_body_id)
        else: self.body_select_combo.setCurrentIndex(0); self.load_body_data(None)

    @staticmethod # Changed to staticmethod
    def calculate_radius_from_mass(mass_kg, density_kg_m3=None):
        _density = density_kg_m3 if density_kg_m3 is not None else ObjectInspectorDialog.ASSUMED_DENSITY_KG_M3
        if mass_kg <= 0: return 1.0
        volume_m3 = mass_kg / _density
        radius_m = ( (3 * volume_m3) / (4 * math.pi) )**(1/3)
        return max(1.0, radius_m)

    def update_radius_display_from_mass(self):
        mass = self.mass_sci_edit.value()
        radius = ObjectInspectorDialog.calculate_radius_from_mass(mass) # Changed to static call
        self.radius_display_label.setText(f"{radius:.3e} m (density: {self.ASSUMED_DENSITY_KG_M3} kg/m³)")

    def populate_body_selector(self):
        self.body_select_combo.blockSignals(True)
        self.body_select_combo.clear()
        self.body_select_combo.addItem("--- Select a Body ---", userData=None)
        for i, b_dict in enumerate(self.initial_body_configs):
            body_id = b_dict.get('id', f"temp_id_{i}")
            display_name = f"{b_dict.get('name', 'Unnamed')} (ID: {body_id})"
            self.body_select_combo.addItem(display_name, userData=body_id)
        self.body_select_combo.blockSignals(False)

    def select_body_in_combo(self, body_id_to_select):
        for i in range(self.body_select_combo.count()):
            if self.body_select_combo.itemData(i) == body_id_to_select:
                self.body_select_combo.setCurrentIndex(i); return
        self.body_select_combo.setCurrentIndex(0); self.load_body_data(None)

    def on_body_selected_changed(self, index):
        if index >= 0: self.load_body_data(self.body_select_combo.itemData(index))

    def pick_body_color(self):
        color = QColorDialog.getColor(self.current_qcolor, self)
        if color.isValid():
            self.current_qcolor = color
            self.color_button.setStyleSheet(f"background-color: {self.current_qcolor.name()}; color: {('black' if self.current_qcolor.lightnessF() > 0.5 else 'white')};")
            self.color_button.setText(self.current_qcolor.name())

    def load_body_data(self, body_id):
        self.selected_body_id = body_id
        if body_id is None: # Placeholder selected
            self.name_edit.clear(); self.mass_sci_edit.setValue(0); self.radius_display_label.setText("N/A m")
            self.color_button.setText("Choose Color"); self.color_button.setStyleSheet("")
            self.pos_label.setText("N/A"); self.vel_label.setText("N/A"); self.acc_label.setText("N/A")
            self.apply_button.setEnabled(False); self.delete_button.setEnabled(False)
            return

        body_config_dict = next((b for b in self.initial_body_configs if b.get('id') == body_id), None)
        if body_config_dict:
            self.name_edit.setText(body_config_dict.get('name', ''))
            mass_val = body_config_dict.get('mass', 0); self.mass_sci_edit.setValue(mass_val)
            radius_val = ObjectInspectorDialog.calculate_radius_from_mass(mass_val) # Changed to static call
            self.radius_display_label.setText(f"{radius_val:.3e} m (density: {self.ASSUMED_DENSITY_KG_M3} kg/m³)")
            color_str = body_config_dict.get('color', '#FFFFFF'); self.current_qcolor = QColor(color_str)
            self.color_button.setStyleSheet(f"background-color: {self.current_qcolor.name()}; color: {('black' if self.current_qcolor.lightnessF() > 0.5 else 'white')};")
            self.color_button.setText(self.current_qcolor.name())
            live_body = self.sim_engine.get_body_by_id(body_id)
            if live_body and not live_body.merged:
                self.pos_label.setText(f"[{live_body.pos[0]:.3e}, {live_body.pos[1]:.3e}, {live_body.pos[2]:.3e}]")
                self.vel_label.setText(f"[{live_body.vel[0]:.3e}, {live_body.vel[1]:.3e}, {live_body.vel[2]:.3e}]")
                self.acc_label.setText(f"[{live_body.acc[0]:.3e}, {live_body.acc[1]:.3e}, {live_body.acc[2]:.3e}]")
                self.apply_button.setEnabled(True); self.delete_button.setEnabled(True)
            else:
                self.pos_label.setText("N/A (merged/not in engine)"); self.vel_label.setText("N/A"); self.acc_label.setText("N/A")
                self.apply_button.setEnabled(True); self.delete_button.setEnabled(True)
        else: self.load_body_data(None) # Fallback to clear if config not found

    def apply_changes(self):
        if self.selected_body_id is None: QMessageBox.warning(self, "No Body Selected", "Select a body."); return
        body_config_dict = next((b for b in self.initial_body_configs if b.get('id') == self.selected_body_id), None)
        if not body_config_dict: QMessageBox.critical(self, "Error", f"Config for ID {self.selected_body_id} not found."); return
        try:
            new_name = self.name_edit.text(); new_mass = self.mass_sci_edit.value()
            if new_mass <= 0: raise ValueError("Mass must be positive.")
            new_radius = ObjectInspectorDialog.calculate_radius_from_mass(new_mass); new_color = self.current_qcolor.name() # Changed to static call
            body_config_dict.update({'name': new_name, 'mass': new_mass, 'radius': new_radius, 'color': new_color})
            live_body = self.sim_engine.get_body_by_id(self.selected_body_id)
            if live_body and not live_body.merged:
                live_body.name = new_name; live_body.mass = new_mass; live_body.radius = new_radius; live_body.color = new_color
                if not self.parent_app.is_running: self.parent_app._update_visualization()
            QMessageBox.information(self, "Changes Applied", f"Changes applied to '{new_name}'. Reset sim for full effect.")
            self.populate_body_selector(); self.select_body_in_combo(self.selected_body_id)
        except ValueError as e: QMessageBox.critical(self, "Invalid Input", str(e))
        except Exception as e: QMessageBox.critical(self, "Error Applying Changes", str(e))

    def delete_body(self):
        if self.selected_body_id is None: QMessageBox.warning(self, "No Body Selected", "Select body to delete."); return
        body_config_to_delete = next((b for b in self.initial_body_configs if b.get('id') == self.selected_body_id), None)
        if not body_config_to_delete: QMessageBox.critical(self, "Error", f"Config for ID {self.selected_body_id} not found."); return
        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Delete '{body_config_to_delete.get('name', 'Unnamed')}' (ID: {self.selected_body_id})?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.initial_body_configs[:] = [b for b in self.initial_body_configs if b.get('id') != self.selected_body_id]
            if self.parent_app.camera_mode == "follow_body" and self.parent_app.camera_target_body_id == self.selected_body_id:
                self.parent_app.set_camera_mode("free")
            QMessageBox.information(self, "Body Deleted", f"Body ID {self.selected_body_id} removed. Resetting.")
            self.parent_app.reset_simulation_to_initial_config()
            self.populate_body_selector()
            if self.body_select_combo.count() > 1: self.body_select_combo.setCurrentIndex(1)
            else: self.load_body_data(None); self.accept()


class NBodyVisPyApp(QMainWindow):
    MIN_MASS_FOR_VIS_SCALE = 1e15; MAX_MASS_FOR_VIS_SCALE = 2e30
    MIN_VIS_SIZE_PX = 3; MAX_VIS_SIZE_PX = 60

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisPy N-Body Gravitational Simulator")
        self.setGeometry(50, 50, 1800, 1000)
        self.sim_engine = SimulationEngine()
        self.initial_body_config_dicts = []
        self._min_mass_log_ref = np.log10(self.MIN_MASS_FOR_VIS_SCALE)
        self._max_mass_log_ref = np.log10(self.MAX_MASS_FOR_VIS_SCALE)
        if abs(self._max_mass_log_ref - self._min_mass_log_ref) < 1e-6: self._max_mass_log_ref = self._min_mass_log_ref + 1.0
        self.is_running = False; self.current_simulation_mode = "pre_defined"
        self.total_sim_time_s = 30 * 24 * 3600.0
        self.precalculated_frames_body_dicts = []; self.precalculated_frame_times = []; self.animation_frame_index = 0
        self.time_scale_multiplier = 1.0
        self.camera_mode = "free"; self.camera_target_body_id = -1
        self.trail_data = []; self.max_trail_length = 700; self.trail_width = 2.0; self.trail_method = 'gl'
        self.object_inspector_dialog = None
        self.autoscale_active = False # New attribute for autoscale state
        self._setup_ui(); self._load_default_scenario()
        self.sim_timer = QTimer(self); self.sim_timer.timeout.connect(self._simulation_step_tick)
        self._update_ui_states()

    def _setup_ui(self):
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.control_panel = QWidget()
        self.control_panel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.control_panel.setMinimumWidth(350); self.control_panel.setMaximumWidth(500)
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget(); self.control_layout_scrollable = QVBoxLayout(scroll_content_widget)
        scroll_area.setWidget(scroll_content_widget); self.control_panel_layout.addWidget(scroll_area)

        sim_mode_group = QGroupBox("Simulation Mode"); sim_mode_layout = QVBoxLayout()
        self.rb_predefined = QRadioButton("Pre-defined Simulation"); self.rb_predefined.setChecked(True)
        self.rb_predefined.toggled.connect(lambda: self._on_simulation_mode_changed("pre_defined"))
        self.rb_realtime = QRadioButton("Real-time Calculation")
        self.rb_realtime.toggled.connect(lambda: self._on_simulation_mode_changed("real_time"))
        sim_mode_layout.addWidget(self.rb_predefined); sim_mode_layout.addWidget(self.rb_realtime)
        self.total_sim_time_edit = QDoubleSpinBox(); self.total_sim_time_edit.setSuffix(" days")
        self.total_sim_time_edit.setRange(0.1, 10000); self.total_sim_time_edit.setValue(self.total_sim_time_s / (24*3600.0))
        self.total_sim_time_edit.valueChanged.connect(lambda val: setattr(self, 'total_sim_time_s', val * 24 * 3600.0))
        form_layout_sim_time = QFormLayout(); form_layout_sim_time.addRow("Total Sim Duration:", self.total_sim_time_edit)
        sim_mode_layout.addLayout(form_layout_sim_time); sim_mode_group.setLayout(sim_mode_layout)
        self.control_layout_scrollable.addWidget(sim_mode_group)

        sim_controls_group = QGroupBox("Simulation Controls"); sim_controls_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ Play"); self.play_button.clicked.connect(self.toggle_simulation)
        self.reset_button = QPushButton("↺ Reset"); self.reset_button.clicked.connect(self.reset_simulation_from_button)
        sim_controls_layout.addWidget(self.play_button); sim_controls_layout.addWidget(self.reset_button)
        sim_controls_group.setLayout(sim_controls_layout); self.control_layout_scrollable.addWidget(sim_controls_group)

        config_group = QGroupBox("Configuration"); config_layout = QVBoxLayout()
        self.add_body_button = QPushButton("Add Body"); self.add_body_button.clicked.connect(self.open_add_body_dialog)
        self.system_config_button = QPushButton("System Config"); self.system_config_button.clicked.connect(self.open_system_config_dialog)
        self.obj_inspector_button = QPushButton("Object Inspector"); self.obj_inspector_button.clicked.connect(self.open_object_inspector)
        self.node_editor_button = QPushButton("Node Editor (NYI)"); self.node_editor_button.clicked.connect(lambda: QMessageBox.information(self, "Node Editor", "Node Editor feature planned."))
        config_layout.addWidget(self.add_body_button); config_layout.addWidget(self.system_config_button)
        config_layout.addWidget(self.obj_inspector_button); config_layout.addWidget(self.node_editor_button)
        config_group.setLayout(config_layout); self.control_layout_scrollable.addWidget(config_group)

        scenario_group = QGroupBox("Scenarios"); scenario_h_layout = QHBoxLayout()
        self.import_csv_button = QPushButton("Import CSV"); self.import_csv_button.clicked.connect(self.import_scenario_csv)
        self.export_csv_button = QPushButton("Export CSV"); self.export_csv_button.clicked.connect(self.export_csv_data)
        self.load_scenario_button = QPushButton("Load Scenario (.json)"); self.load_scenario_button.clicked.connect(self.load_scenario)
        self.save_scenario_button = QPushButton("Save Scenario (.json)"); self.save_scenario_button.clicked.connect(self.save_scenario)
        scenario_h_layout.addWidget(self.import_csv_button, 1); scenario_h_layout.addWidget(self.export_csv_button, 1)
        scenario_h_layout.addWidget(self.load_scenario_button, 1); scenario_h_layout.addWidget(self.save_scenario_button, 1)
        scenario_group.setLayout(scenario_h_layout); self.control_layout_scrollable.addWidget(scenario_group)

        vis_group = QGroupBox("Visualization & Camera"); vis_layout = QFormLayout()
        self.time_scale_slider = QSlider(Qt.Horizontal); self.time_scale_slider.setRange(1, 1000); self.time_scale_slider.setValue(100)
        self.time_scale_slider.setTickPosition(QSlider.TicksBelow); self.time_scale_slider.setTickInterval(100)
        self.time_scale_slider.valueChanged.connect(self._update_time_scale_value)
        self.time_scale_label = QLabel("1.00x"); speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.time_scale_slider); speed_layout.addWidget(self.time_scale_label)
        vis_layout.addRow("Animation Speed:", speed_layout)

        self.cb_autoscale_view = QCheckBox("Autoscale View") # New Autoscale CheckBox
        self.cb_autoscale_view.toggled.connect(self._on_autoscale_toggled)
        vis_layout.addRow(self.cb_autoscale_view) # Add to layout

        self.cb_auto_rotate = QCheckBox("Auto-Rotate Camera"); self.cb_auto_rotate.setChecked(True)
        vis_layout.addRow(self.cb_auto_rotate)
        self.cb_show_axis = QCheckBox("Show Coordinate Axis"); self.cb_show_axis.setChecked(True)
        self.cb_show_axis.toggled.connect(self._toggle_axis_visibility)
        vis_layout.addRow(self.cb_show_axis)

        camera_mode_layout = QHBoxLayout()
        self.rb_cam_free = QRadioButton("Free"); self.rb_cam_free.setChecked(True); self.rb_cam_free.toggled.connect(lambda checked: self.set_camera_mode("free") if checked else None)
        self.rb_cam_follow = QRadioButton("Follow Body"); self.rb_cam_follow.toggled.connect(lambda checked: self.set_camera_mode("follow_body") if checked else None)
        self.rb_cam_com = QRadioButton("Follow CoM"); self.rb_cam_com.toggled.connect(lambda checked: self.set_camera_mode("follow_com") if checked else None)
        camera_mode_layout.addWidget(self.rb_cam_free); camera_mode_layout.addWidget(self.rb_cam_follow); camera_mode_layout.addWidget(self.rb_cam_com)
        vis_layout.addRow("Camera Mode:", camera_mode_layout)
        self.follow_body_combo = QComboBox(); self.follow_body_combo.setEnabled(False)
        self.follow_body_combo.currentIndexChanged.connect(self.on_follow_target_changed)
        vis_layout.addRow("Follow Target (if Body):", self.follow_body_combo)
        vis_group.setLayout(vis_layout); self.control_layout_scrollable.addWidget(vis_group)
        self.control_layout_scrollable.addStretch(1); self.main_layout.addWidget(self.control_panel)

        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor=Color("#101018"))
        self.canvas.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.main_layout.addWidget(self.canvas.native, 1)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = vispy.scene.cameras.TurntableCamera(fov=30, distance=2e9, elevation=30, azimuth=-60, up='+z')
        self.body_markers = visuals.Markers(parent=self.view.scene); self.trail_lines = []
        self.axis = visuals.XYZAxis(parent=self.view.scene); s = 1e8
        self.axis.transform = vispy.visuals.transforms.STTransform(translate=(0,0,0), scale=(s,s,s))
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_time_label = QLabel("Time: 0.00 days"); self.status_state_label = QLabel("State: IDLE")
        self.status_bar.addPermanentWidget(self.status_time_label); self.status_bar.addPermanentWidget(QLabel(" | "))
        self.status_bar.addPermanentWidget(self.status_state_label)

    def _on_autoscale_toggled(self, checked):
        self.autoscale_active = checked
        if checked:
            # Immediately apply autoscale if possible
            if hasattr(self, 'body_markers') and self.body_markers.visible: # Check if visualization is ready
                 # Get current bodies from visualization if available, else from engine
                current_bodies_for_vis = []
                if self.current_simulation_mode == "pre_defined" and self.precalculated_frames_body_dicts and \
                   self.animation_frame_index < len(self.precalculated_frames_body_dicts):
                    frame_data = self.precalculated_frames_body_dicts[self.animation_frame_index]
                    for b_data in frame_data:
                         current_bodies_for_vis.append({'pos': np.array(b_data['pos']), 'radius': b_data['radius']}) # Add other needed keys if any
                else: # Real-time or before pre-calc animation starts
                    active_engine_bodies = [b for b in self.sim_engine.bodies if not b.merged]
                    for body in active_engine_bodies:
                        current_bodies_for_vis.append({'pos': body.pos, 'radius': body.radius})
                
                if current_bodies_for_vis:
                    self._autoscale_camera_logic(current_bodies_for_vis)
                else: # No bodies, set a default view
                    self.view.camera.center = (0,0,0)
                    self.view.camera.distance = 2e9 

        self._update_ui_states() # Update enabled state of other camera controls

    def _autoscale_camera_logic(self, bodies_for_render):
        if not bodies_for_render:
            self.view.camera.center = (0, 0, 0)
            self.view.camera.distance = 2e9 # Default distance
            return

        all_pos = np.array([b['pos'] for b in bodies_for_render])
        
        if all_pos.shape[0] == 0:
            self.view.camera.center = (0, 0, 0)
            self.view.camera.distance = 2e9
            return

        min_coords = np.min(all_pos, axis=0)
        max_coords = np.max(all_pos, axis=0)
        
        new_center = (min_coords + max_coords) / 2.0
        self.view.camera.center = tuple(new_center)

        min_cam_dist = 1e3 # Minimum camera distance
        padding_factor = 1.5 # To ensure bodies are not at the very edge

        if all_pos.shape[0] == 1:
            # Single body: distance based on its radius
            radius = bodies_for_render[0].get('radius', 1e5) # Default radius if not present
            # Estimate distance so the body takes up a small portion of the view
            # This is a heuristic and might need fov adjustment for perfection
            new_distance = radius * 20 
            new_distance = max(new_distance, min_cam_dist)
        else:
            dimensions = max_coords - min_coords
            max_extent = np.max(dimensions)
            
            if max_extent < 1e-3: # All points are virtually coincident
                avg_radius = np.mean([b.get('radius', 1e5) for b in bodies_for_render])
                # Estimate extent based on average radius and number of bodies
                max_extent = avg_radius * 2 * (len(bodies_for_render)**(1/3.0) if len(bodies_for_render) > 0 else 1.0)
                max_extent = max(max_extent, 1e6) # Default extent if still too small

            fov_rad = np.deg2rad(self.view.camera.fov)
            if np.tan(fov_rad / 2) < 1e-6: # Avoid division by zero or very small numbers if FOV is tiny
                new_distance = 2e9 # Default large distance
            else:
                # Basic formula: distance = size / (2 * tan(FOV/2))
                new_distance = (max_extent / (2 * np.tan(fov_rad / 2)))
            
            new_distance *= padding_factor
            new_distance = max(new_distance, min_cam_dist)

        self.view.camera.distance = new_distance

    def _update_time_scale_value(self, value):
        self.time_scale_multiplier = value / 100.0
        self.time_scale_label.setText(f"{self.time_scale_multiplier:.2f}x")

    def _toggle_axis_visibility(self, checked):
        if hasattr(self, 'axis'): self.axis.visible = checked
        self.canvas.update()

    def reset_simulation_from_button(self):
        self.reset_simulation_to_initial_config()
        self.status_bar.showMessage("Simulation reset to initial conditions by user.", 3000)

    def set_camera_mode(self, mode):
        # This function is only relevant if autoscale is OFF
        if self.autoscale_active:
            # If autoscale is on, ensure radio buttons reflect a "disabled" or "autoscale" state
            # For now, _update_ui_states handles disabling them.
            return

        self.camera_mode = mode
        self.follow_body_combo.setEnabled(mode == "follow_body")
        if mode == "free":
            self.camera_target_body_id = -1
        elif mode == "follow_body":
            if self.follow_body_combo.count() > 0:
                current_combo_data = self.follow_body_combo.currentData()
                if current_combo_data is not None:
                    self.camera_target_body_id = current_combo_data
                elif self.follow_body_combo.count() > 1:
                    self.follow_body_combo.setCurrentIndex(1)
                    self.camera_target_body_id = self.follow_body_combo.itemData(1)
                else:
                    self.camera_mode = "free"
                    if hasattr(self, 'rb_cam_free'): self.rb_cam_free.setChecked(True)
                    self.camera_target_body_id = -1
            else:
                self.camera_mode = "free"
                if hasattr(self, 'rb_cam_free'): self.rb_cam_free.setChecked(True)
                self.camera_target_body_id = -1
        elif mode == "follow_com":
            self.camera_target_body_id = -1
        self._update_camera()

    def on_follow_target_changed(self, index):
        if self.autoscale_active: return # Ignore if autoscale is active

        if index >= 0 and self.camera_mode == "follow_body":
            body_id = self.follow_body_combo.itemData(index)
            if body_id is not None:
                self.camera_target_body_id = body_id
            else:
                self.camera_target_body_id = -1
            self._update_camera()

    def populate_follow_combo(self):
        self.follow_body_combo.blockSignals(True)
        self.follow_body_combo.clear()
        self.follow_body_combo.addItem("--- Select a Body ---", userData=None)
        current_target_still_exists = False
        for body in self.sim_engine.bodies:
            if not body.merged:
                self.follow_body_combo.addItem(f"{body.name} (ID: {body.id})", userData=body.id)
                if body.id == self.camera_target_body_id:
                    self.follow_body_combo.setCurrentText(f"{body.name} (ID: {body.id})")
                    current_target_still_exists = True
        if not current_target_still_exists and self.camera_mode == "follow_body":
            if self.follow_body_combo.count() > 1:
                self.follow_body_combo.setCurrentIndex(1)
                self.camera_target_body_id = self.follow_body_combo.itemData(1)
            else:
                # No actual bodies left, if camera_mode was 'follow_body', it should ideally switch.
                # For now, just clear target_id. set_camera_mode handles switching if target is lost.
                self.camera_target_body_id = -1 
                if not self.autoscale_active: # Only switch if not in autoscale
                    self.set_camera_mode("free")
                    if hasattr(self, 'rb_cam_free'): self.rb_cam_free.setChecked(True)
                self.follow_body_combo.setCurrentIndex(0)
        self.follow_body_combo.blockSignals(False)


    def _update_camera(self):
        # This method is for manual camera modes (free, follow, com)
        # It should NOT run if autoscale is active.
        if self.autoscale_active:
            return

        if self.camera_mode == "follow_body" and self.camera_target_body_id != -1:
            target_body = self.sim_engine.get_body_by_id(self.camera_target_body_id)
            if target_body and not target_body.merged:
                self.view.camera.center = tuple(target_body.pos)
            else:
                self.set_camera_mode("free")
                if hasattr(self, 'rb_cam_free'): self.rb_cam_free.setChecked(True)
        elif self.camera_mode == "follow_com":
            com_pos, _ = self.sim_engine.get_center_of_mass()
            if np.any(np.isnan(com_pos)) or not self.sim_engine.bodies or not any(not b.merged for b in self.sim_engine.bodies):
                self.view.camera.center = (0,0,0)
            else:
                self.view.camera.center = tuple(com_pos)
        self.canvas.update()

    def _on_simulation_mode_changed(self, mode):
        if self.is_running: self.pause_simulation()
        self.current_simulation_mode = mode
        self.reset_simulation_to_initial_config(); self._update_ui_states()

    def _update_ui_states(self):
        is_predefined = (self.current_simulation_mode == "pre_defined")
        self.total_sim_time_edit.setEnabled(is_predefined)
        
        can_export_csv = is_predefined and bool(self.precalculated_frames_body_dicts) and not self.is_running
        if hasattr(self, 'export_csv_button'):
            self.export_csv_button.setEnabled(can_export_csv)

        if self.is_running:
            self.play_button.setText("❚❚ Pause"); self.play_button.setEnabled(True)
            self.reset_button.setEnabled(False)
        else:
            self.reset_button.setEnabled(True)
            if is_predefined:
                play_text = "▶ Play Pre-calc"
                if self.precalculated_frames_body_dicts and \
                   0 < self.animation_frame_index < len(self.precalculated_frames_body_dicts):
                    play_text = "▶ Resume Anim"
                self.play_button.setText(play_text)
            else: self.play_button.setText("▶ Play Real-time")
            self.play_button.setEnabled(True)

        status_text = "IDLE"
        if self.is_running: status_text = "ANIMATING" if is_predefined else "RUNNING"
        elif is_predefined and self.precalculated_frames_body_dicts and \
             self.animation_frame_index >= len(self.precalculated_frames_body_dicts):
            status_text = "FINISHED (Anim)"
        
        self.status_state_label.setText(f"State: {status_text} ({self.current_simulation_mode.replace('_',' ').title()})")
        self.status_time_label.setText(f"Time: {self.sim_engine.time_elapsed / (24*3600):.2f} days")
        
        # Update camera control enabled states based on autoscale
        are_cam_modes_configurable = not self.autoscale_active
        self.rb_cam_free.setEnabled(are_cam_modes_configurable)
        self.rb_cam_follow.setEnabled(are_cam_modes_configurable)
        self.rb_cam_com.setEnabled(are_cam_modes_configurable)
        # follow_body_combo's enabled state also depends on whether "Follow Body" mode is selected
        self.follow_body_combo.setEnabled(are_cam_modes_configurable and self.camera_mode == "follow_body")
        
        self.populate_follow_combo()

    def _load_default_scenario(self):
        self.sim_engine.clear_bodies(); self.initial_body_config_dicts = []
        m_earth = 5.972e24; m_moon = 7.348e22; dist_em = 3.844e8
        G_const = self.sim_engine.G
        if G_const <= 0: QMessageBox.critical(self, "Config Error", "G must be positive."); return
        M_total = m_earth + m_moon
        r_earth_from_com = (m_moon / M_total) * dist_em; r_moon_from_com = (m_earth / M_total) * dist_em
        pos_earth = np.array([-r_earth_from_com, 0.0, 0.0]); pos_moon = np.array([r_moon_from_com, 0.0, 0.0])
        try: omega = np.sqrt(G_const * M_total / (dist_em**3))
        except ZeroDivisionError: QMessageBox.critical(self, "Math Error", "dist_em cannot be zero."); return
        v_earth_mag = omega * r_earth_from_com; v_moon_mag = omega * r_moon_from_com
        vel_earth = np.array([0.0, v_earth_mag, 0.0]); vel_moon = np.array([0.0, -v_moon_mag, 0.0])
        try:
            earth_radius = AddBodyDialog.calculate_radius_from_mass(m_earth) # Changed to static call
            moon_radius = AddBodyDialog.calculate_radius_from_mass(m_moon)   # Changed to static call
            b1_dict = SimBody(0, "Earth", m_earth, pos_earth, vel_earth, earth_radius, '#1E90FF').to_dict()
            b2_dict = SimBody(1, "Moon", m_moon, pos_moon, vel_moon, moon_radius, '#D3D3D3').to_dict()
        except (ValueError, TypeError) as e: QMessageBox.critical(self, "Body Error", f"{e}"); return
        self.initial_body_config_dicts = [b1_dict, b2_dict]
        self.total_sim_time_s = 60 * 24 * 3600.0
        self.total_sim_time_edit.setValue(self.total_sim_time_s / (24*3600.0))
        self.view.camera.distance = dist_em * 3.0; self.view.camera.center = (0,0,0)
        self.reset_simulation_to_initial_config()

    def reset_simulation_to_initial_config(self):
        self.pause_simulation()
        self.sim_engine.clear_bodies()
        current_max_id = -1
        for b_dict in self.initial_body_config_dicts:
            try:
                body_id = b_dict.get('id')
                if body_id is None: body_id = self.sim_engine.next_body_id; b_dict['id'] = body_id
                if body_id > current_max_id: current_max_id = body_id
                b_dict['radius'] = ObjectInspectorDialog.calculate_radius_from_mass(b_dict['mass']) # Changed to static call
                sim_body_instance = SimBody.from_dict(b_dict)
                self.sim_engine.add_body_instance(sim_body_instance)
            except (ValueError, TypeError, KeyError) as e:
                QMessageBox.critical(self, "Config Error", f"Failed to load body '{b_dict.get('name', 'Unknown')}': {e}. Skipping.")
                continue
        self.sim_engine.next_body_id = current_max_id + 1

        self.sim_engine.reset_time_and_trails()
        self.precalculated_frames_body_dicts = []; self.precalculated_frame_times = []; self.animation_frame_index = 0

        active_bodies_for_trails = [b for b in self.sim_engine.bodies if not b.merged]
        self.trail_data = [[] for _ in active_bodies_for_trails]
        for trail_line_visual in self.trail_lines: trail_line_visual.parent = None
        self.trail_lines = []

        for i, body in enumerate(active_bodies_for_trails):
            line = visuals.Line(pos=np.array([[0,0,0],[0,0,0]]),
                                color=Color(body.color, alpha=0.6).rgba,
                                method=self.trail_method, width=self.trail_width,
                                parent=self.view.scene, connect='strip', antialias=True)
            self.trail_lines.append(line)

        self._update_visualization()
        if self.autoscale_active: # Apply autoscale after reset if active
            active_engine_bodies = [b for b in self.sim_engine.bodies if not b.merged]
            bodies_for_render = [{'pos': body.pos, 'radius': body.radius} for body in active_engine_bodies]
            if bodies_for_render:
                self._autoscale_camera_logic(bodies_for_render)
        self._update_ui_states()

    def toggle_simulation(self):
        if self.is_running: self.pause_simulation()
        else:
            self.is_running = True
            if self.current_simulation_mode == "pre_defined":
                if not self.initial_body_config_dicts: QMessageBox.warning(self, "No Initial Config", "Configure bodies first for pre-defined simulation."); self.is_running = False; self._update_ui_states(); return
                if not self.precalculated_frames_body_dicts:
                    if not self._precalculate_simulation_data(): self.is_running = False; self._update_ui_states(); return
                if self.precalculated_frames_body_dicts:
                    if self.animation_frame_index >= len(self.precalculated_frames_body_dicts): self.animation_frame_index = 0
            elif self.current_simulation_mode == "real_time":
                if not self.sim_engine.bodies or not any(not b.merged for b in self.sim_engine.bodies):
                    QMessageBox.warning(self, "No Active Bodies", "Add active bodies first for real-time simulation."); self.is_running = False; self._update_ui_states(); return
                self.sim_engine._calculate_accelerations()
            interval_ms = int(33 / self.time_scale_multiplier)
            self.sim_timer.start(max(10, interval_ms))
            self.status_bar.showMessage("Simulation started.", 2000)
        self._update_ui_states()

    def pause_simulation(self):
        if self.is_running:
            self.is_running = False
            self.sim_timer.stop()
            self.status_bar.showMessage("Simulation paused.", 2000)
        self._update_ui_states()

    def _simulation_step_tick(self):
        if not self.is_running: self.sim_timer.stop(); return

        if self.current_simulation_mode == "real_time":
            self.sim_engine.simulation_step()
            self._update_visualization()
        elif self.current_simulation_mode == "pre_defined":
            if not self.precalculated_frames_body_dicts: self.pause_simulation(); return
            if self.animation_frame_index < len(self.precalculated_frames_body_dicts):
                frame_body_dicts = self.precalculated_frames_body_dicts[self.animation_frame_index]
                frame_time = self.precalculated_frame_times[self.animation_frame_index]
                temp_bodies_for_vis = []
                temp_trails_for_vis = []
                for b_data in frame_body_dicts:
                    temp_bodies_for_vis.append({'pos': np.array(b_data['pos']), 'color': b_data['color'],
                                                'radius': b_data['radius'], 'id': b_data['id'], 'mass': b_data['mass']})
                    temp_trails_for_vis.append([np.array(p) for p in b_data.get('trail', [])])
                self._update_visualization(bodies_data=temp_bodies_for_vis, trails_data_list=temp_trails_for_vis, current_time=frame_time)
                self.animation_frame_index += 1
            else: self.animation_frame_index = 0
        
        # Camera update logic is now primarily within _update_visualization
        # self._update_camera() # This call is now conditional inside _update_visualization
        
        interval_ms = int(33 / self.time_scale_multiplier)
        self.sim_timer.setInterval(max(10, interval_ms))
        self._update_ui_states()

    def _precalculate_simulation_data(self):
        if hasattr(self, 'export_csv_button'): self.export_csv_button.setEnabled(False)
        self.status_bar.showMessage("Pre-calculating simulation... Please wait."); QApplication.processEvents()
        self.sim_engine.clear_bodies()
        current_max_id = -1
        for b_dict in self.initial_body_config_dicts:
            try:
                body_id = b_dict.get('id', self.sim_engine.next_body_id)
                b_dict['id'] = body_id
                if body_id > current_max_id: current_max_id = body_id
                b_dict['radius'] = ObjectInspectorDialog.calculate_radius_from_mass(b_dict['mass']) # Changed to static call
                sim_body_instance = SimBody.from_dict(b_dict)
                self.sim_engine.add_body_instance(sim_body_instance)
            except (ValueError, TypeError, KeyError) as e:
                 QMessageBox.critical(self, "Pre-calc Body Load Error", f"Error loading body '{b_dict.get('name', '')}' for pre-calc: {e}. Skipping."); return False
        self.sim_engine.next_body_id = current_max_id + 1
        self.sim_engine.reset_time_and_trails()
        self.precalculated_frames_body_dicts = []; self.precalculated_frame_times = []; self.animation_frame_index = 0
        try:
            total_duration = self.total_sim_time_s; dt_val = self.sim_engine.dt
            if total_duration <= 0 or dt_val <= 0: raise ValueError("Total simulation time and time step must be positive.")
            num_steps = int(total_duration / dt_val)
            if num_steps == 0: QMessageBox.information(self, "Info", "Simulation time too short. No calculation."); return True
        except ValueError as e: QMessageBox.critical(self, "Input Error", f"Invalid simulation time or time step: {e}"); return False
        if self.sim_engine.bodies: self.sim_engine._calculate_accelerations()
        for step in range(num_steps):
            self.sim_engine.simulation_step()
            frame_states = []
            active_bodies_in_engine_this_step = [b for b in self.sim_engine.bodies if not b.merged]
            for body_in_engine in active_bodies_in_engine_this_step:
                body_dict = body_in_engine.to_dict()
                body_dict['trail'] = [p.tolist() for p in body_in_engine.trail]
                frame_states.append(body_dict)
            self.precalculated_frames_body_dicts.append(frame_states)
            self.precalculated_frame_times.append(self.sim_engine.time_elapsed)
            if step % (max(1, num_steps // 20)) == 0:
                self.status_bar.showMessage(f"Pre-calculating... {step*100/num_steps:.0f}%"); QApplication.processEvents()
        self.status_bar.showMessage("Pre-calculation complete.", 3000)
        QMessageBox.information(self, "Pre-calculation Complete", f"{num_steps} steps calculated.")
        return True

    def _update_visualization(self, bodies_data=None, trails_data_list=None, current_time=None):
        if current_time is None: current_time = self.sim_engine.time_elapsed
        positions, colors, sizes = [], [], []
        source_bodies_for_render = []
        current_trails_for_render = []

        if bodies_data is not None:
            source_bodies_for_render = bodies_data
            current_trails_for_render = trails_data_list if trails_data_list else []
        else:
            active_engine_bodies = [b for b in self.sim_engine.bodies if not b.merged]
            if len(self.trail_data) != len(active_engine_bodies):
                old_trail_map = {body.id: [] for body in self.sim_engine.bodies if not body.merged}
                for i, body_in_engine in enumerate(self.sim_engine.bodies):
                    if not body_in_engine.merged and i < len(self.trail_data):
                        old_trail_map[body_in_engine.id] = self.trail_data[i]
                self.trail_data = [old_trail_map.get(body.id, []) for body in active_engine_bodies]
            for i, body in enumerate(active_engine_bodies):
                source_bodies_for_render.append({'pos': body.pos, 'color': body.color,
                                                 'radius': body.radius, 'id': body.id, 'mass': body.mass})
                if i < len(self.trail_data):
                    self.trail_data[i].append(body.pos.copy())
                    if len(self.trail_data[i]) > self.max_trail_length: self.trail_data[i].pop(0)
            current_trails_for_render = self.trail_data[:len(active_engine_bodies)]

        if not source_bodies_for_render:
            self.body_markers.set_data(np.empty((0,3)));
            for tl in self.trail_lines: tl.visible = False
            # Handle camera if autoscale is on and no bodies
            if self.autoscale_active:
                self._autoscale_camera_logic([]) # Pass empty list to reset view
            self.canvas.update(); self._update_ui_states(); return

        min_mass_log_ref = self._min_mass_log_ref; max_mass_log_ref = self._max_mass_log_ref
        min_vis_size_px = self.MIN_VIS_SIZE_PX; max_vis_size_px = self.MAX_VIS_SIZE_PX
        for i, body_info in enumerate(source_bodies_for_render):
            positions.append(body_info['pos']); colors.append(Color(body_info['color']).rgba)
            current_mass_log = np.log10(max(1e-9, body_info.get('mass', 1e-9)))
            clamped_mass_log = max(min_mass_log_ref, min(max_mass_log_ref, current_mass_log))
            scale_factor = (clamped_mass_log - min_mass_log_ref) / (max_mass_log_ref - min_mass_log_ref) if (max_mass_log_ref - min_mass_log_ref) != 0 else 0.5
            vis_size = min_vis_size_px + (max_vis_size_px - min_vis_size_px) * scale_factor
            sizes.append(max(2, min(vis_size, 100)))
        if positions: self.body_markers.set_data(np.array(positions), face_color=np.array(colors), edge_color=Color('grey', alpha=0.2).rgba, size=np.array(sizes))
        else: self.body_markers.set_data(np.empty((0,3)))

        num_bodies_rendering = len(source_bodies_for_render)
        if len(self.trail_lines) != num_bodies_rendering:
            for tl_vis in self.trail_lines: tl_vis.parent = None
            self.trail_lines = []
            for i in range(num_bodies_rendering):
                body_info_for_trail = source_bodies_for_render[i]
                trail_color = Color(body_info_for_trail['color'], alpha=0.6).rgba
                new_line = visuals.Line(pos=np.array([[0,0,0],[0,0,0]]),
                                         parent=self.view.scene, method=self.trail_method,
                                         width=self.trail_width, color=trail_color,
                                         connect='strip', antialias=True)
                self.trail_lines.append(new_line)
        for i in range(num_bodies_rendering):
            if i < len(current_trails_for_render) and i < len(self.trail_lines):
                trail_points = current_trails_for_render[i]
                if trail_points and len(trail_points) >= 2:
                    pos_data = np.array(trail_points, dtype=np.float32)
                    if pos_data.ndim == 2 and pos_data.shape[1] == 3:
                        self.trail_lines[i].set_data(pos=pos_data); self.trail_lines[i].visible = True
                    else: self.trail_lines[i].visible = False
                else: self.trail_lines[i].visible = False
        for i in range(num_bodies_rendering, len(self.trail_lines)): self.trail_lines[i].visible = False

        # Camera update logic
        if self.autoscale_active:
            self._autoscale_camera_logic(source_bodies_for_render)
        else:
            self._update_camera() # Handles follow modes etc.

        cam_event = self.view.camera._event_value
        camera_being_dragged = (cam_event is not None and hasattr(cam_event, 'buttons') and any(cam_event.buttons))
        if self.cb_auto_rotate.isChecked() and not camera_being_dragged:
             self.view.camera.azimuth += 0.1
        self.canvas.update(); self._update_ui_states()

    def open_add_body_dialog(self):
        next_id = self.sim_engine.next_body_id
        cam_center = self.view.camera.center if self.view.camera.center is not None else np.array([0,0,0])
        cam_dist = self.view.camera.distance if self.view.camera.distance is not None else 1e8
        suggested_pos = cam_center + np.random.normal(0, cam_dist * 0.1, 3)
        dialog = AddBodyDialog(self, next_body_id=next_id, initial_pos=suggested_pos)
        body_data = dialog.get_body_data()
        if body_data:
            new_body_dict = SimBody(next_id, body_data["name"], body_data["mass"],
                                    body_data["pos"], body_data["vel"],
                                    body_data["radius"], body_data["color"]).to_dict()
            self.initial_body_config_dicts.append(new_body_dict)
            if not self.is_running: self.reset_simulation_to_initial_config() # This will trigger autoscale if active
            else: QMessageBox.information(self, "Body Added", f"Body '{body_data['name']}' added. Reset to include.")
            self._update_ui_states()

    def open_system_config_dialog(self):
        current_config = {'G': self.sim_engine.G, 'dt': self.sim_engine.dt, 'integrator': self.sim_engine.integrator_type, 'collision_model': self.sim_engine.collision_model}
        dialog = SystemConfigDialog(self, current_config)
        new_config = dialog.get_config_data()
        if new_config:
            self.sim_engine.G = new_config['G']; self.sim_engine.dt = new_config['dt']
            self.sim_engine.integrator_type = new_config['integrator']; self.sim_engine.collision_model = new_config['collision_model']
            QMessageBox.information(self, "Config Updated", "System parameters updated. Reset to apply.")
            if not self.is_running: self.reset_simulation_to_initial_config()
            self._update_ui_states()

    def open_object_inspector(self):
        if self.object_inspector_dialog is None or not self.object_inspector_dialog.isVisible():
            selected_id_for_inspector = None
            if self.camera_mode == "follow_body" and self.camera_target_body_id != -1 and not self.autoscale_active: # Only use if not autoscaling
                selected_id_for_inspector = self.camera_target_body_id
            self.object_inspector_dialog = ObjectInspectorDialog(self, self.sim_engine, self.initial_body_config_dicts, selected_id_for_inspector)
            self.object_inspector_dialog.show(); self.object_inspector_dialog.activateWindow()
        else: self.object_inspector_dialog.activateWindow()

    def save_scenario(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Scenario", "", "JSON Files (*.json)")
        if not filepath: return
        try:
            cam_center_data = [0,0,0]
            if self.view.camera.center is not None:
                cam_center_data = list(self.view.camera.center) if isinstance(self.view.camera.center, (list, tuple)) else self.view.camera.center.tolist()
            scenario_data = {"G": self.sim_engine.G, "dt": self.sim_engine.dt, "integrator_type": self.sim_engine.integrator_type,
                             "collision_model": self.sim_engine.collision_model, "total_sim_time_s": self.total_sim_time_s,
                             "simulation_mode": self.current_simulation_mode, "initial_bodies": self.initial_body_config_dicts,
                             "camera_state": {"distance": self.view.camera.distance, "center": cam_center_data,
                                              "fov": self.view.camera.fov, "azimuth": self.view.camera.azimuth, "elevation": self.view.camera.elevation,},
                             "autoscale_active": self.autoscale_active # Save autoscale state
                             }
            with open(filepath, 'w') as f: json.dump(scenario_data, f, indent=4)
            self.status_bar.showMessage(f"Scenario saved to {os.path.basename(filepath)}", 3000)
        except Exception as e: QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")

    def load_scenario(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Scenario", "", "JSON Files (*.json)")
        if not filepath: return
        try:
            with open(filepath, 'r') as f: scenario_data = json.load(f)
            self.sim_engine.G = scenario_data.get("G", self.sim_engine.G)
            self.sim_engine.dt = scenario_data.get("dt", self.sim_engine.dt)
            self.sim_engine.integrator_type = scenario_data.get("integrator_type", self.sim_engine.integrator_type)
            self.sim_engine.collision_model = scenario_data.get("collision_model", self.sim_engine.collision_model)
            self.total_sim_time_s = scenario_data.get("total_sim_time_s", self.total_sim_time_s)
            self.total_sim_time_edit.setValue(self.total_sim_time_s / (24*3600.0))
            new_mode = scenario_data.get("simulation_mode", "pre_defined")
            self.rb_predefined.blockSignals(True); self.rb_realtime.blockSignals(True)
            if new_mode == "pre_defined": self.rb_predefined.setChecked(True)
            else: self.rb_realtime.setChecked(True)
            self.rb_predefined.blockSignals(False); self.rb_realtime.blockSignals(False)
            self.current_simulation_mode = new_mode
            loaded_initial_bodies = []
            for b_data in scenario_data.get("initial_bodies", []):
                try: SimBody.from_dict(b_data); loaded_initial_bodies.append(b_data)
                except Exception as e: QMessageBox.warning(self, "Load Warning", f"Skipping body: {e}")
            self.initial_body_config_dicts = loaded_initial_bodies
            
            self.autoscale_active = scenario_data.get("autoscale_active", False) # Load autoscale state
            self.cb_autoscale_view.setChecked(self.autoscale_active) # Update checkbox

            cam_state = scenario_data.get("camera_state")
            if cam_state and not self.autoscale_active: # Only load camera state if not autoscaling
                self.view.camera.distance = cam_state.get("distance", self.view.camera.distance)
                center_val = cam_state.get("center");
                if center_val is not None: self.view.camera.center = tuple(center_val)
                self.view.camera.fov = cam_state.get("fov", self.view.camera.fov)
                self.view.camera.azimuth = cam_state.get("azimuth", self.view.camera.azimuth)
                self.view.camera.elevation = cam_state.get("elevation", self.view.camera.elevation)
            
            self.reset_simulation_to_initial_config() # This will apply autoscale if active
            self.status_bar.showMessage(f"Scenario loaded from {os.path.basename(filepath)}", 3000)
        except Exception as e: QMessageBox.critical(self, "Load Error", f"Failed to load: {e}")

    def export_csv_data(self):
        if self.current_simulation_mode != "pre_defined" or not self.precalculated_frames_body_dicts:
            QMessageBox.warning(self, "Export Error", "CSV export is only for completed pre-defined simulations."); return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Simulation Data as CSV", "", "CSV Files (*.csv)")
        if not filepath: return
        try:
            self.status_bar.showMessage("Exporting CSV data... Please wait."); QApplication.processEvents()
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Parameter", "Value"])
                writer.writerow(["G", self.sim_engine.G]); writer.writerow(["TimeStep", self.sim_engine.dt])
                writer.writerow(["TotalSimTime_Predefined", self.total_sim_time_s])
                writer.writerow(["Integrator", self.sim_engine.integrator_type]); writer.writerow(["CollisionModel", self.sim_engine.collision_model])
                writer.writerow(["SimulationMode", self.current_simulation_mode]); writer.writerow([])
                for b_init_dict in self.initial_body_config_dicts:
                    name = b_init_dict.get('name', f'Body{b_init_dict.get("id", "Unknown")}')
                    writer.writerow([f"{name}_ID", b_init_dict.get('id', 'N/A')])
                    writer.writerow([f"{name}_Mass", b_init_dict['mass']])
                    writer.writerow([f"{name}_InitialPosX", b_init_dict['pos'][0]])
                    writer.writerow([f"{name}_InitialPosY", b_init_dict['pos'][1]])
                    writer.writerow([f"{name}_InitialPosZ", b_init_dict['pos'][2]])
                    writer.writerow([f"{name}_InitialVelX", b_init_dict['vel'][0]])
                    writer.writerow([f"{name}_InitialVelY", b_init_dict['vel'][1]])
                    writer.writerow([f"{name}_InitialVelZ", b_init_dict['vel'][2]])
                    writer.writerow([f"{name}_Radius", b_init_dict['radius']])
                    writer.writerow([f"{name}_Color", b_init_dict['color']])
                    writer.writerow([])
                writer.writerow([])

                data_header = ["Time"]; body_ids_in_first_frame_order = []; body_names_map = {}
                if self.precalculated_frames_body_dicts:
                    initial_names_map = {b['id']: b.get('name', f"Body{b['id']}") for b in self.initial_body_config_dicts}
                    for b_dict_frame0 in self.precalculated_frames_body_dicts[0]:
                        body_id = b_dict_frame0['id']
                        name = initial_names_map.get(body_id, b_dict_frame0.get('name', f'Body{body_id}'))
                        body_ids_in_first_frame_order.append(body_id)
                        body_names_map[body_id] = name
                        data_header.extend([f"{name}_Px", f"{name}_Py", f"{name}_Pz", f"{name}_Vx", f"{name}_Vy", f"{name}_Vz"])
                writer.writerow(data_header)

                for i, frame_time in enumerate(self.precalculated_frame_times):
                    row = [frame_time]
                    frame_data_map = {b_data['id']: b_data for b_data in self.precalculated_frames_body_dicts[i]}
                    for body_id in body_ids_in_first_frame_order:
                        b_data_this_frame = frame_data_map.get(body_id)
                        if b_data_this_frame:
                            row.extend(b_data_this_frame['pos']); row.extend(b_data_this_frame['vel'])
                        else: row.extend(["N/A"] * 6)
                    writer.writerow(row)
            self.status_bar.showMessage(f"CSV data exported to {os.path.basename(filepath)}", 3000)
            QMessageBox.information(self, "Export Successful", f"Data exported to {filepath}")
        except Exception as e:
            self.status_bar.showMessage(f"CSV export failed: {e}", 5000); QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def _parse_raw_kv_to_body_data(self, raw_kv_dict, source_row_info=""):
        parsed_data = {}
        temp_id_from_csv = None
        temp_name_from_csv = None
        for key_raw, value_str in raw_kv_dict.items():
            key_upper = key_raw.strip().upper()
            if "_ID" in key_upper:
                try:
                    temp_id_from_csv = int(value_str)
                    idx = key_upper.rfind("_ID")
                    temp_name_from_csv = key_raw.strip()[:idx]
                    if not temp_name_from_csv:
                        temp_name_from_csv = f"UnnamedParsed_{temp_id_from_csv}"
                        print(f"Warning ({source_row_info}): Key '{key_raw}' implies an ID but no name prefix. Using default name '{temp_name_from_csv}'.")
                    parsed_data['id_from_csv'] = temp_id_from_csv
                    parsed_data['name_from_csv'] = temp_name_from_csv
                except ValueError:
                    print(f"Warning ({source_row_info}): Could not parse ID '{value_str}' for key '{key_raw}'. ID will be auto-assigned.")
                break
        if temp_name_from_csv is None:
            print(f"Error ({source_row_info}): Crucial 'Name_ID' key not found. Skipping body block."); return None
        prefix = temp_name_from_csv + "_"
        def get_val(param_suffix, data_type, default_val_for_param):
            full_key_to_try = prefix + param_suffix
            raw_val_str = raw_kv_dict.get(full_key_to_try)
            if raw_val_str is not None:
                try: return data_type(raw_val_str)
                except ValueError: print(f"Warning ({source_row_info}): Parse error for '{raw_val_str}' for '{full_key_to_try}'. Using default."); return default_val_for_param
            return default_val_for_param
        parsed_data['mass'] = get_val("Mass", float, -1.0)
        pos_x = get_val("InitialPosX", float, 0.0); pos_y = get_val("InitialPosY", float, 0.0); pos_z = get_val("InitialPosZ", float, 0.0)
        parsed_data['pos'] = [pos_x, pos_y, pos_z]
        vel_x = get_val("InitialVelX", float, 0.0); vel_y = get_val("InitialVelY", float, 0.0); vel_z = get_val("InitialVelZ", float, 0.0)
        parsed_data['vel'] = [vel_x, vel_y, vel_z]
        parsed_data['radius'] = get_val("Radius", float, -1.0)
        parsed_data['color'] = get_val("Color", str, f"#{np.random.randint(0,256**3-1):06X}")
        return parsed_data

    def import_scenario_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Bodies from CSV", "", "CSV Files (*.csv)")
        if not filepath: return
        raw_body_blocks_kv = []; current_body_kv = {}; first_body_block_started = False
        try:
            with open(filepath, mode='r', newline='', encoding='utf-8-sig') as csvfile:
                dialect = None
                try: sample = csvfile.read(2048); csvfile.seek(0); dialect = csv.Sniffer().sniff(sample, delimiters=',;\t| ')
                except csv.Error: print("CSV Sniffer could not detect dialect, defaulting to comma.")
                reader = csv.reader(csvfile, dialect=dialect) if dialect else csv.reader(csvfile)
                for row_num, row in enumerate(reader):
                    if not row or len(row) < 2 or not any(cell.strip() for cell in row): continue
                    key_raw = row[0].strip(); value_raw = row[1].strip()
                    is_id_key = "_ID" in key_raw.upper()
                    if is_id_key:
                        if first_body_block_started and current_body_kv: raw_body_blocks_kv.append(current_body_kv)
                        current_body_kv = {key_raw: value_raw}; first_body_block_started = True
                    elif first_body_block_started and current_body_kv: current_body_kv[key_raw] = value_raw
                    else: print(f"Skipping row {row_num + 1} (before first 'Name_ID' or malformed): {row}")
            if first_body_block_started and current_body_kv: raw_body_blocks_kv.append(current_body_kv)
            if not raw_body_blocks_kv: QMessageBox.information(self, "CSV Import", "No valid body blocks found."); return
            parsed_csv_bodies_prelim = []
            for i, raw_kv in enumerate(raw_body_blocks_kv):
                body_data = self._parse_raw_kv_to_body_data(raw_kv, source_row_info=f"Block {i+1}")
                if body_data: parsed_csv_bodies_prelim.append(body_data)
            if not parsed_csv_bodies_prelim: QMessageBox.information(self, "CSV Import", "No bodies parsed."); return
            reply = QMessageBox.question(self, "Import Mode", f"{len(parsed_csv_bodies_prelim)} bodies parsed. Append or Replace?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Cancel: return
            final_new_configs = []; ids_currently_taken = set(); next_id_to_assign = self.sim_engine.next_body_id
            if reply == QMessageBox.StandardButton.Yes: # Append
                for b_existing in self.initial_body_config_dicts:
                    if 'id' in b_existing: ids_currently_taken.add(b_existing['id']); next_id_to_assign = max(next_id_to_assign, b_existing['id'] + 1)
            for prelim_data in parsed_csv_bodies_prelim:
                final_id = None; id_from_csv = prelim_data.get('id_from_csv'); name_from_csv = prelim_data.get('name_from_csv', "UnnamedImport")
                if id_from_csv is not None:
                    if id_from_csv in ids_currently_taken: print(f"Warning: CSV ID {id_from_csv} for '{name_from_csv}' conflicts. Assigning new."); final_id = next_id_to_assign
                    else: final_id = id_from_csv
                else: print(f"Warning: No ID for '{name_from_csv}' in CSV. Assigning new."); final_id = next_id_to_assign
                if final_id is not None: next_id_to_assign = max(next_id_to_assign, final_id + 1)
                if final_id is not None: ids_currently_taken.add(final_id)
                else: print(f"Error: Could not determine ID for '{name_from_csv}'. Skipping."); continue
                mass = prelim_data['mass']; radius = prelim_data['radius']
                if mass <= 0: print(f"Error: Body '{name_from_csv}' (ID: {final_id}) invalid mass. Skipping."); continue
                if radius <= 0: print(f"Warning: Body '{name_from_csv}' (ID: {final_id}) invalid radius. Auto-calculating."); radius = ObjectInspectorDialog.calculate_radius_from_mass(mass) # Static call
                final_new_configs.append({'id': final_id, 'name': name_from_csv, 'mass': mass, 'pos': prelim_data['pos'], 'vel': prelim_data['vel'], 'radius': radius, 'color': prelim_data['color']})
            if not final_new_configs: QMessageBox.information(self, "CSV Import", "No valid bodies to add."); return
            if reply == QMessageBox.StandardButton.Yes: self.initial_body_config_dicts.extend(final_new_configs)
            else: self.initial_body_config_dicts = final_new_configs
            self.reset_simulation_to_initial_config()
            QMessageBox.information(self, "CSV Import Successful", f"{len(final_new_configs)} bodies imported/updated. Sim reset.")
        except FileNotFoundError: QMessageBox.critical(self, "Import Error", f"File not found: {filepath}")
        except csv.Error as ce: QMessageBox.critical(self, "CSV Reading Error", f"Error reading CSV: {ce}")
        except Exception as e: import traceback; QMessageBox.critical(self, "Import Error", f"Unexpected error: {e}\n{traceback.format_exc()}")

    def closeEvent(self, event):
        if hasattr(self, 'object_inspector_dialog') and self.object_inspector_dialog and self.object_inspector_dialog.isVisible():
            self.object_inspector_dialog.close()
        self.sim_timer.stop()
        super().closeEvent(event)