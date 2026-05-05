import numpy as np

class Rocket3D:
    def __init__(self, v0, theta_deg, phi_deg, masse=100,
                 Cd=0.3, A=0.05, x0=0, y0=0, z0=0):
        self.v0 = v0
        self.theta = np.radians(theta_deg)
        self.phi = np.radians(phi_deg)
        self.masse = masse
        self.Cd = Cd
        self.A = A
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.g = 9.81
        self.rho = 1.225
        self.vx0 = v0 * np.cos(self.theta) * np.cos(self.phi)
        self.vy0 = v0 * np.sin(self.theta)
        self.vz0 = v0 * np.cos(self.theta) * np.sin(self.phi)

    def _derivees(self, state):
        x, y, z, vx, vy, vz = state
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v > 0:
            k = 0.5 * self.rho * self.Cd * self.A / self.masse
            ax = -k * v * vx
            ay = -self.g - k * v * vy
            az = -k * v * vz
        else:
            ax, ay, az = 0, -self.g, 0
        return np.array([vx, vy, vz, ax, ay, az])

    def trajectoire_rk4(self, dt=0.1):
        t_l, x_l, y_l, z_l = [], [], [], []
        t = 0
        state = np.array([self.x0, self.y0, self.z0, self.vx0, self.vy0, self.vz0])
        while True:
            t_l.append(t); x_l.append(state[0]); y_l.append(state[1]); z_l.append(state[2])
            k1 = self._derivees(state)
            k2 = self._derivees(state + dt/2 * k1)
            k3 = self._derivees(state + dt/2 * k2)
            k4 = self._derivees(state + dt * k3)
            ns = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            if ns[1] < 0 and state[1] >= 0:
                f = state[1] / (state[1] - ns[1])
                t_l.append(t - dt + f * dt)
                x_l.append(state[0] + f * (ns[0] - state[0]))
                y_l.append(0.0)
                z_l.append(state[2] + f * (ns[2] - state[2]))
                break
            state = ns
        return np.array(t_l), np.array(x_l), np.array(y_l), np.array(z_l)


class ThreatGenerator3D:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.v0_range = (150, 400)
        self.theta_range = (30, 75)
        self.phi_range = (-20, 20)
        self.masse_range = (50, 200)
        self.Cd_range = (0.2, 0.5)
        self.x0_range = (-8000, -3000)
        self.z0_range = (-2000, 2000)

    def generer_une_menace(self):
        return Rocket3D(
            v0=np.random.uniform(*self.v0_range),
            theta_deg=np.random.uniform(*self.theta_range),
            phi_deg=np.random.uniform(*self.phi_range),
            masse=np.random.uniform(*self.masse_range),
            Cd=np.random.uniform(*self.Cd_range),
            x0=np.random.uniform(*self.x0_range),
            z0=np.random.uniform(*self.z0_range)
        )

    def generer_salve(self, n=5):
        return [self.generer_une_menace() for _ in range(n)]
