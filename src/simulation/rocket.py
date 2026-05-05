import numpy as np

class Rocket:


    def __init__(self, v0, angle_deg, x0=0, y0=0):

        self.v0 = v0
        self.angle = np.radians(angle_deg)  
        self.x0 = x0
        self.y0 = y0
        self.g = 9.81  

        # vitesse initiale
        self.vx0 = v0 * np.cos(self.angle) 
        self.vy0 = v0 * np.sin(self.angle)  

    def temps_de_vol(self):
        """Calcule la durée totale du vol jusqu'à l'impact au sol."""
        return (2 * self.vy0) / self.g

    def portee(self):
        """Calcule la distance horizontale totale (portée)."""
        return (self.v0**2 * np.sin(2 * self.angle)) / self.g

    def hauteur_max(self):
        """Calcule l'altitude maximale atteinte."""
        return (self.vy0**2) / (2 * self.g)

    def position(self, t):
        """
        Calcule la position (x, y) à l'instant t.
        Retourne None si la roquette a déjà touché le sol.
        """
        x = self.x0 + self.vx0 * t
        y = self.y0 + self.vy0 * t - 0.5 * self.g * t**2

        # La roquette ne va pas sous le sol
        if y < 0:
            return None
        return x, y

    def trajectoire_complete(self, dt=0.1):
        """
        Génère tous les points de la trajectoire.
        dt = pas de temps en secondes (plus petit = plus précis)
        """
        T = self.temps_de_vol()

        # Créer un tableau de temps de 0 à T par pas de dt
        t_array = np.arange(0, T, dt)

        # Calculer x et y pour chaque instant
        x_array = self.x0 + self.vx0 * t_array
        y_array = self.y0 + self.vy0 * t_array - 0.5 * self.g * t_array**2

        # Garder uniquement les points où y >= 0
        masque = y_array >= 0

        return t_array[masque], x_array[masque], y_array[masque]

    def vitesse(self, t):
        """Calcule la vitesse (vx, vy) à l'instant t."""
        vx = self.vx0  # constante
        vy = self.vy0 - self.g * t
        return vx, vy

    def infos(self):
        """Affiche un résumé des caractéristiques de la roquette."""
        print(f"=== Roquette ===")
        print(f"Vitesse initiale  : {self.v0} m/s")
        print(f"Angle de tir      : {np.degrees(self.angle):.1f}°")
        print(f"Vitesse X initiale: {self.vx0:.2f} m/s")
        print(f"Vitesse Y initiale: {self.vy0:.2f} m/s")
        print(f"Temps de vol      : {self.temps_de_vol():.2f} s")
        print(f"Portée            : {self.portee():.2f} m")
        print(f"Hauteur maximale  : {self.hauteur_max():.2f} m")



class RocketWithDrag:
    """
    Roquette avec résistance de l'air.
    Utilise la méthode d'Euler pour intégration numérique.
    """

    def __init__(self, v0, angle_deg, masse=100, 
                 Cd=0.3, A=0.05, x0=0, y0=0):
        self.v0 = v0
        self.angle = np.radians(angle_deg)
        self.masse = masse
        self.Cd = Cd
        self.A = A
        self.x0 = x0
        self.y0 = y0

        self.g = 9.81
        self.rho = 1.225

        self.vx0 = v0 * np.cos(self.angle)
        self.vy0 = v0 * np.sin(self.angle)

    def _force_drag(self, vx, vy):
        v = np.sqrt(vx**2 + vy**2)
        if v == 0:
            return 0, 0
        F_drag = 0.5 * self.rho * self.Cd * self.A * v**2
        F_drag_x = -F_drag * (vx / v)
        F_drag_y = -F_drag * (vy / v)
        return F_drag_x, F_drag_y

    def trajectoire_euler(self, dt=0.01):
        t_list, x_list, y_list = [], [], []
        vx_list, vy_list = [], []

        t = 0
        x, y = self.x0, self.y0
        vx, vy = self.vx0, self.vy0

        while True:
            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            vx_list.append(vx)
            vy_list.append(vy)

            Fdx, Fdy = self._force_drag(vx, vy)
            ax = Fdx / self.masse
            ay = -self.g + Fdy / self.masse

            new_vx = vx + ax * dt
            new_vy = vy + ay * dt
            new_x = x + vx * dt
            new_y = y + vy * dt

            # Si y passe en dessous de 0 → interpoler
            if new_y < 0 and y >= 0:
                frac = y / (y - new_y)
                t_list.append(t + frac * dt)
                x_list.append(x + frac * (new_x - x))
                y_list.append(0.0)
                vx_list.append(vx + frac * (new_vx - vx))
                vy_list.append(vy + frac * (new_vy - vy))
                break

            vx, vy = new_vx, new_vy
            x, y = new_x, new_y
            t += dt

        return (np.array(t_list), np.array(x_list), 
                np.array(y_list), np.array(vx_list), np.array(vy_list))



class RocketRK4(RocketWithDrag):


    def _derivees(self, state):
        x, y, vx, vy = state
        Fdx, Fdy = self._force_drag(vx, vy)
        ax = Fdx / self.masse
        ay = -self.g + Fdy / self.masse
        return np.array([vx, vy, ax, ay])

    def trajectoire_rk4(self, dt=0.1):
        t_list, x_list, y_list = [], [], []

        t = 0
        state = np.array([self.x0, self.y0, self.vx0, self.vy0])

        while True:
            t_list.append(t)
            x_list.append(state[0])
            y_list.append(state[1])

            k1 = self._derivees(state)
            k2 = self._derivees(state + dt/2 * k1)
            k3 = self._derivees(state + dt/2 * k2)
            k4 = self._derivees(state + dt * k3)

            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt

            # Si y passe en dessous de 0 → interpoler
            if new_state[1] < 0 and state[1] >= 0:
                frac = state[1] / (state[1] - new_state[1])
                x_impact = state[0] + frac * (new_state[0] - state[0])
                t_impact = t - dt + frac * dt

                t_list.append(t_impact)
                x_list.append(x_impact)
                y_list.append(0.0)
                break

            state = new_state

        return np.array(t_list), np.array(x_list), np.array(y_list)



class ThreatGenerator:
    """
    Génère des roquettes aléatoires depuis des zones de lancement.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)  # pour reproduire les mêmes résultats

        # Zone de lancement (position de l'ennemi)
        self.launch_zone_x = (-8000, -3000)  # mètres
        self.launch_zone_y = (0, 0)           # au niveau du sol

        # Plages de paramètres réalistes
        self.v0_range     = (150, 400)   # m/s
        self.angle_range  = (30, 75)     # degrés
        self.masse_range  = (50, 200)    # kg
        self.Cd_range     = (0.2, 0.5)   # coefficient drag

    def generer_une_menace(self):
        """Génère une roquette avec des paramètres aléatoires."""
        x0  = np.random.uniform(*self.launch_zone_x)
        v0  = np.random.uniform(*self.v0_range)
        ang = np.random.uniform(*self.angle_range)
        m   = np.random.uniform(*self.masse_range)
        Cd  = np.random.uniform(*self.Cd_range)

        return RocketRK4(v0=v0, angle_deg=ang, 
                         masse=m, Cd=Cd, x0=x0, y0=0)

    def generer_salve(self, n=5):
        """Génère une salve de n roquettes."""
        return [self.generer_une_menace() for _ in range(n)]
