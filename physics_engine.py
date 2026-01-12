import numpy as np

# --- Physics Simulation Core ---
class SimBody:
    """Represents a single celestial body in the simulation."""
    def __init__(self, id_val, name, mass, pos, vel, radius=1.0, color='blue'):
        try:
            self.id = int(id_val)
            self.name = str(name)
            self.mass = float(mass)
            if self.mass <= 0: raise ValueError("Mass must be positive.")
            
            if not (isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 3):
                raise ValueError("Position must be a 3-element list/tuple/array.")
            self.pos = np.array(pos, dtype=float)
            
            if not (isinstance(vel, (list, tuple, np.ndarray)) and len(vel) == 3):
                raise ValueError("Velocity must be a 3-element list/tuple/array.")
            self.vel = np.array(vel, dtype=float)
            
            self.acc = np.zeros(3, dtype=float) # Acceleration vector
            self.radius = float(radius)
            if self.radius <= 0: raise ValueError("Radius must be positive.")
            self.color = str(color)
            self.trail = [] # Stores historical positions for drawing trails
            self.merged = False # Flag to indicate if body has been merged

        except (ValueError, TypeError) as e:
            raise type(e)(f"Error initializing SimBody '{name}': {e}")

    def to_dict(self):
        """Converts SimBody object to a dictionary for serialization."""
        return {
            "id": self.id, "name": self.name, "mass": self.mass,
            "pos": self.pos.tolist(), "vel": self.vel.tolist(),
            "radius": self.radius, "color": self.color
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a SimBody object from a dictionary."""
        return cls(data["id"], data["name"], data["mass"],
                   np.array(data["pos"]), np.array(data["vel"]),
                   data["radius"], data["color"])
    
    def clear_trail(self):
        self.trail = []

    def add_to_trail(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > 1000: 
            self.trail.pop(0)


class SimulationEngine:
    """Manages the simulation state, physics calculations, and integration."""
    def __init__(self):
        self.bodies = [] 
        self.G = 6.674e-11 
        self.dt = 3600.0 
        self.time_elapsed = 0.0 
        self.integrator_type = 'rk4' 
        self.collision_model = 'ignore' # 'ignore', 'elastic', 'merge'
        self.next_body_id = 0 

    def add_body_instance(self, body_instance):
        if not isinstance(body_instance, SimBody):
            raise TypeError("Only SimBody instances can be added to the engine.")
        
        current_ids = {b.id for b in self.bodies}
        if body_instance.id in current_ids:
             body_instance.id = self.next_body_id 
        
        self.bodies.append(body_instance)
        self.next_body_id = max(self.next_body_id, body_instance.id + 1)


    def add_new_body(self, name, mass, pos, vel, radius, color):
        try:
            body = SimBody(self.next_body_id, name, mass, pos, vel, radius, color)
            self.bodies.append(body)
            self.next_body_id += 1
            return body
        except (ValueError, TypeError) as e:
            raise type(e)(f"Error creating body '{name}': {e}")

    def get_body_by_id(self, body_id):
        try:
            target_id = int(body_id)
            for body in self.bodies:
                if body.id == target_id:
                    return body
        except ValueError:
            return None 
        return None
        
    def clear_bodies(self):
        self.bodies = []
        self.next_body_id = 0 

    def _calculate_accelerations(self):
        for body_i in self.bodies:
            if body_i.merged: continue # Skip merged bodies
            body_i.acc = np.zeros(3, dtype=float) 
            for body_j in self.bodies:
                if body_j.merged or body_i is body_j:
                    continue 

                r_vec = body_j.pos - body_i.pos
                r_mag_sq = np.sum(r_vec**2)
                
                if r_mag_sq < 1e-18: 
                    continue 
                
                r_mag = np.sqrt(r_mag_sq)
                
                force_mag_over_mass_i = self.G * body_j.mass / r_mag_sq
                body_i.acc += force_mag_over_mass_i * (r_vec / r_mag)

    def _handle_collisions_elastic(self):
        for i in range(len(self.bodies)):
            b1 = self.bodies[i]
            if b1.merged: continue
            for j in range(i + 1, len(self.bodies)): 
                b2 = self.bodies[j]
                if b2.merged: continue

                dist_vec = b1.pos - b2.pos
                dist = np.linalg.norm(dist_vec)
                min_dist_for_collision = b1.radius + b2.radius

                if dist < min_dist_for_collision and dist > 1e-9: 
                    n_vec = dist_vec / dist 
                    v_rel = b1.vel - b2.vel 
                    v_rel_n = np.dot(v_rel, n_vec) 

                    if v_rel_n < 0: 
                        m1, m2 = b1.mass, b2.mass
                        v1_n_initial = np.dot(b1.vel, n_vec)
                        v2_n_initial = np.dot(b2.vel, n_vec)

                        v1_n_final = (v1_n_initial * (m1 - m2) + 2 * m2 * v2_n_initial) / (m1 + m2)
                        v2_n_final = (v2_n_initial * (m2 - m1) + 2 * m1 * v1_n_initial) / (m1 + m2)

                        b1.vel += (v1_n_final - v1_n_initial) * n_vec
                        b2.vel += (v2_n_final - v2_n_initial) * n_vec
                        
                        overlap = min_dist_for_collision - dist
                        separation_factor = 1.01 
                        b1.pos += n_vec * (overlap * m2 / (m1 + m2)) * separation_factor
                        b2.pos -= n_vec * (overlap * m1 / (m1 + m2)) * separation_factor
                elif dist <= 1e-9: 
                    b1.pos += np.random.rand(3) * b1.radius * 0.1 
                    b2.pos -= np.random.rand(3) * b2.radius * 0.1
    
    def _handle_collisions_merge(self):
        bodies_to_remove_indices = set()
        new_bodies_to_add = []

        for i in range(len(self.bodies)):
            if i in bodies_to_remove_indices or self.bodies[i].merged:
                continue
            b1 = self.bodies[i]

            for j in range(i + 1, len(self.bodies)):
                if j in bodies_to_remove_indices or self.bodies[j].merged:
                    continue
                b2 = self.bodies[j]

                dist_vec = b1.pos - b2.pos
                dist = np.linalg.norm(dist_vec)
                min_dist_for_collision = b1.radius + b2.radius

                if dist < min_dist_for_collision: # Collision detected
                    # Conserve momentum for the new merged body
                    m_total = b1.mass + b2.mass
                    new_vel = (b1.mass * b1.vel + b2.mass * b2.vel) / m_total
                    
                    # Position of new body: CoM of the two colliding bodies
                    new_pos = (b1.mass * b1.pos + b2.mass * b2.pos) / m_total
                    
                    # New radius (e.g., conserving volume, assuming density is constant, r_new^3 = r1^3 + r2^3)
                    new_radius = (b1.radius**3 + b2.radius**3)**(1/3)
                    
                    # New body properties
                    new_name = f"Merged({b1.name}+{b2.name})"
                    # Color: average or dominant? For simplicity, take b1's or a new default.
                    new_color = b1.color if b1.mass >= b2.mass else b2.color 

                    merged_body = SimBody(self.next_body_id, new_name, m_total, new_pos, new_vel, new_radius, new_color)
                    self.next_body_id += 1
                    new_bodies_to_add.append(merged_body)
                    
                    # Mark original bodies for removal (conceptually)
                    b1.merged = True 
                    b2.merged = True
                    bodies_to_remove_indices.add(i)
                    bodies_to_remove_indices.add(j)
                    break # b1 has merged, move to next i

        # Filter out merged bodies and add new ones
        self.bodies = [b for idx, b in enumerate(self.bodies) if not b.merged]
        self.bodies.extend(new_bodies_to_add)


    def _verlet_step(self):
        active_bodies = [b for b in self.bodies if not b.merged]
        if not active_bodies: return

        for body in active_bodies:
            body.pos += body.vel * self.dt + 0.5 * body.acc * self.dt**2
        
        acc_old_map = {body.id: body.acc.copy() for body in active_bodies}
        
        self._calculate_accelerations() # Will use only non-merged bodies

        for body in active_bodies:
            if body.id in acc_old_map: # Ensure body wasn't just created in a merge
                body.vel += 0.5 * (acc_old_map[body.id] + body.acc) * self.dt

    def _get_accel_for_rk4_substep(self, temp_pos_of_current_body, all_other_bodies_states_for_substep):
        acc = np.zeros(3, dtype=float)
        for other_b_state in all_other_bodies_states_for_substep:
            # Assuming other_b_state does not include merged bodies or they are handled
            r_vec = np.array(other_b_state['pos']) - temp_pos_of_current_body
            r_mag_sq = np.sum(r_vec**2)
            if r_mag_sq < 1e-18: continue 
            r_mag = np.sqrt(r_mag_sq)
            
            force_mag_over_mass = self.G * other_b_state['mass'] / r_mag_sq
            acc += force_mag_over_mass * (r_vec / r_mag)
        return acc

    def _rk4_step_for_body(self, body, other_bodies_current_states_list):
        k1_pos_deriv = body.vel.copy()
        k1_vel_deriv = body.acc.copy() # a(t)

        temp_pos_k2 = body.pos + 0.5 * k1_pos_deriv * self.dt
        acc_k2 = self._get_accel_for_rk4_substep(temp_pos_k2, other_bodies_current_states_list)
        k2_pos_deriv = body.vel + 0.5 * k1_vel_deriv * self.dt
        k2_vel_deriv = acc_k2
        
        temp_pos_k3 = body.pos + 0.5 * k2_pos_deriv * self.dt
        acc_k3 = self._get_accel_for_rk4_substep(temp_pos_k3, other_bodies_current_states_list)
        k3_pos_deriv = body.vel + 0.5 * k2_vel_deriv * self.dt
        k3_vel_deriv = acc_k3
        
        temp_pos_k4 = body.pos + k3_pos_deriv * self.dt
        acc_k4 = self._get_accel_for_rk4_substep(temp_pos_k4, other_bodies_current_states_list)
        k4_pos_deriv = body.vel + k3_vel_deriv * self.dt
        k4_vel_deriv = acc_k4
        
        body.pos += (k1_pos_deriv + 2*k2_pos_deriv + 2*k3_pos_deriv + k4_pos_deriv) * self.dt / 6.0
        body.vel += (k1_vel_deriv + 2*k2_vel_deriv + 2*k3_vel_deriv + k4_vel_deriv) * self.dt / 6.0

    def _rk4_step(self):
        active_bodies = [b for b in self.bodies if not b.merged]
        if not active_bodies: return
        
        # Calculate accelerations a(t) based on current P(t) for non-merged bodies
        self._calculate_accelerations()

        other_bodies_states_for_each = []
        for i in range(len(active_bodies)):
            others_list = []
            for j in range(len(active_bodies)):
                if i == j: continue
                b_other = active_bodies[j]
                # Ensure we're using current data of non-merged bodies
                others_list.append({'id': b_other.id, 'pos': b_other.pos.copy(), 'mass': b_other.mass, 'radius': b_other.radius})
            other_bodies_states_for_each.append(others_list)

        for i, body in enumerate(active_bodies):
            self._rk4_step_for_body(body, other_bodies_states_for_each[i])
        
        # Recalculate accelerations a(t+dt) based on new positions P(t+dt)
        self._calculate_accelerations()


    def simulation_step(self):
        active_bodies = [b for b in self.bodies if not b.merged]
        if not active_bodies: return 

        for body in active_bodies:
            body.add_to_trail()

        if self.integrator_type == 'verlet':
            self._verlet_step()
        else: 
            self._rk4_step()

        # Handle collisions AFTER integration step
        if self.collision_model == 'elastic':
            self._handle_collisions_elastic()
        elif self.collision_model == 'merge':
            self._handle_collisions_merge()
            # After merge, accelerations might need recalculation if it affects next step logic,
            # but typically _calculate_accelerations is called at start of next full step or RK4.
            # If Verlet, the new acc is calculated anyway.
            # If RK4, the a(t+dt) from this step's end is used as a(t) for next step.
            # We might need to update accelerations if a merge happened right here.
            if any(b.merged for b in self.bodies if not b.merged): # If any active body just merged
                 self._calculate_accelerations() # Re-calculate for the now potentially different system

        self.time_elapsed += self.dt
        
        # Clean up bodies marked as merged (if not already done by collision handlers)
        # _handle_collisions_merge already rebuilds self.bodies, so this might be redundant here
        # self.bodies = [b for b in self.bodies if not b.merged]


    def reset_time_and_trails(self):
        self.time_elapsed = 0.0
        for body in self.bodies:
            body.clear_trail()
            body.merged = False # Reset merged flag on full reset
    
    def get_system_energy(self):
        active_bodies = [b for b in self.bodies if not b.merged]
        if not active_bodies: return 0.0, 0.0, 0.0
        
        kinetic_energy = sum(0.5 * b.mass * np.sum(b.vel**2) for b in active_bodies)
        potential_energy = 0.0
        
        for i in range(len(active_bodies)):
            for j in range(i + 1, len(active_bodies)): 
                b1, b2 = active_bodies[i], active_bodies[j]
                r_vec = b2.pos - b1.pos
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-9: 
                    potential_energy -= self.G * b1.mass * b2.mass / r_mag
        return kinetic_energy, potential_energy, kinetic_energy + potential_energy

    def get_center_of_mass(self, body_id_list=None):
        target_bodies_temp = []
        if body_id_list is None: 
            target_bodies_temp = [b for b in self.bodies if not b.merged]
        else: 
            for bid in body_id_list:
                body = self.get_body_by_id(bid)
                if body and not body.merged: target_bodies_temp.append(body)
        
        if not target_bodies_temp: return np.zeros(3), np.zeros(3) 
        
        total_mass = sum(b.mass for b in target_bodies_temp)
        if abs(total_mass) < 1e-18: 
            return np.zeros(3), np.zeros(3) 

        com_pos = sum(b.mass * b.pos for b in target_bodies_temp) / total_mass
        com_vel = sum(b.mass * b.vel for b in target_bodies_temp) / total_mass
        return com_pos, com_vel