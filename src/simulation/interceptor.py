import numpy as np
from scipy import interpolate

class Interceptor:
    def __init__(self, x0, y0, vitesse=800, kill_radius=30):
        self.x0 = x0
        self.y0 = y0
        self.vitesse = vitesse
        self.kill_radius = kill_radius

    def _distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def pure_pursuit(self, cible_x, cible_y, cible_t, dt=0.1):
        mx, my = self.x0, self.y0
        missile_x, missile_y, missile_t = [mx], [my], [0]
        distances = []
        t, intercepte, idx_cible = 0, False, 0
        while t < cible_t[-1]:
            while idx_cible < len(cible_t) - 1 and cible_t[idx_cible] < t:
                idx_cible += 1
            cx, cy = cible_x[idx_cible], cible_y[idx_cible]
            d = self._distance(mx, my, cx, cy)
            distances.append(d)
            if d < self.kill_radius:
                intercepte = True
                break
            angle = np.arctan2(cy - my, cx - mx)
            mx += self.vitesse * np.cos(angle) * dt
            my += self.vitesse * np.sin(angle) * dt
            t += dt
            missile_x.append(mx)
            missile_y.append(my)
            missile_t.append(t)
        return {
            'missile_x': np.array(missile_x), 'missile_y': np.array(missile_y),
            'missile_t': np.array(missile_t), 'intercepte': intercepte,
            'distance_min': min(distances) if distances else float('inf'),
            'temps_interception': t if intercepte else None, 'methode': 'Pure Pursuit'
        }

    def proportional_navigation(self, cible_x, cible_y, cible_t, N=4, dt=0.01):
        mx, my = self.x0, self.y0
        t_fine = np.arange(cible_t[0], cible_t[-1], dt)
        if len(t_fine) < 2:
            return {
                'missile_x': np.array([mx]), 'missile_y': np.array([my]),
                'missile_t': np.array([0]), 'intercepte': False,
                'distance_min': float('inf'),
                'temps_interception': None, 'methode': 'Proportional Navigation'
            }
        interp_x = interpolate.interp1d(cible_t, cible_x, fill_value='extrapolate')
        interp_y = interpolate.interp1d(cible_t, cible_y, fill_value='extrapolate')
        cxf = interp_x(t_fine)
        cyf = interp_y(t_fine)
        min_alt = 200
        best_idx = len(t_fine) // 4
        for i in range(len(t_fine)):
            d = self._distance(mx, my, cxf[i], cyf[i])
            t_missile = d / self.vitesse
            if t_missile <= t_fine[i] and cyf[i] > min_alt:
                best_idx = i
                break
        angle_init = np.arctan2(cyf[best_idx] - my, cxf[best_idx] - mx)
        vxm = self.vitesse * np.cos(angle_init)
        vym = self.vitesse * np.sin(angle_init)
        missile_x, missile_y, missile_t = [mx], [my], [0]
        distances = []
        t, intercepte, lambda_prev = 0, False, None
        step = 0
        while t < t_fine[-1] and step < len(t_fine) - 1:
            cx, cy = cxf[step], cyf[step]
            d = self._distance(mx, my, cx, cy)
            distances.append(d)
            if d < self.kill_radius:
                intercepte = True
                break
            if my < -10:
                break
            lambda_current = np.arctan2(cy - my, cx - mx)
            if lambda_prev is not None:
                lambda_dot = (lambda_current - lambda_prev) / dt
                while lambda_dot > np.pi / dt:
                    lambda_dot -= 2 * np.pi / dt
                while lambda_dot < -np.pi / dt:
                    lambda_dot += 2 * np.pi / dt
            else:
                lambda_dot = 0
            lambda_prev = lambda_current
            if step < len(t_fine) - 2:
                vxc = (cxf[step+1] - cxf[step]) / dt
                vyc = (cyf[step+1] - cyf[step]) / dt
            else:
                vxc, vyc = 0, 0
            dx, dy = cx - mx, cy - my
            v_closing = -((dx * (vxc - vxm) + dy * (vyc - vym)) / d)
            if v_closing < 50:
                v_closing = 50
            a_n = N * v_closing * lambda_dot
            a_n = np.clip(a_n, -40 * 9.81, 40 * 9.81)
            v_angle = np.arctan2(vym, vxm)
            vxm += -a_n * np.sin(v_angle) * dt
            vym += a_n * np.cos(v_angle) * dt
            v_total = np.sqrt(vxm**2 + vym**2)
            vxm = vxm / v_total * self.vitesse
            vym = vym / v_total * self.vitesse
            mx += vxm * dt
            my += vym * dt
            t += dt
            step += 1
            missile_x.append(mx)
            missile_y.append(my)
            missile_t.append(t)
        return {
            'missile_x': np.array(missile_x), 'missile_y': np.array(missile_y),
            'missile_t': np.array(missile_t), 'intercepte': intercepte,
            'distance_min': min(distances) if distances else float('inf'),
            'temps_interception': t if intercepte else None,
            'methode': 'Proportional Navigation'
        }


class EngagementManager:
    def __init__(self, zone_x_min, zone_x_max, interceptor_pos=(0, 0),
                 missile_vitesse=800, kill_radius=30):
        self.zone_x_min = zone_x_min
        self.zone_x_max = zone_x_max
        self.interceptor_pos = interceptor_pos
        self.missile_vitesse = missile_vitesse
        self.kill_radius = kill_radius

    def impact_dans_zone(self, cible_x, cible_y):
        x_impact = cible_x[-1]
        return self.zone_x_min <= x_impact <= self.zone_x_max

    def _calculer_delai_lancement(self, cible_x, cible_y, cible_t):
        mx, my = self.interceptor_pos
        for i in range(len(cible_t)):
            d = np.sqrt((cible_x[i] - mx)**2 + (cible_y[i] - my)**2)
            t_vol_missile = d / self.missile_vitesse
            x_cible = cible_x[i]
            dans_zone_x = self.zone_x_min - 2000 <= x_cible <= self.zone_x_max + 1000
            altitude_ok = cible_y[i] > 300
            if t_vol_missile <= cible_t[i] and dans_zone_x and altitude_ok:
                delai = max(0, cible_t[i] - t_vol_missile - 1.0)
                return delai
        return 0

    def evaluer_menace(self, cible_x, cible_y, cible_t):
        x_impact = cible_x[-1]
        dans_zone = self.impact_dans_zone(cible_x, cible_y)
        return {
            'x_impact': x_impact, 'dans_zone': dans_zone,
            'decision': 'INTERCEPTER' if dans_zone else 'IGNORER',
        }

    def gerer_salve(self, salve):
        resultats = []
        missiles_utilises = 0
        interceptions = 0
        ignorees = 0
        for roquette in salve:
            t_c, x_c, y_c = roquette.trajectoire_rk4(dt=0.1)
            evaluation = self.evaluer_menace(x_c, y_c, t_c)
            if evaluation['decision'] == 'INTERCEPTER':
                delai = self._calculer_delai_lancement(x_c, y_c, t_c)
                idx_debut = np.searchsorted(t_c, delai)
                if idx_debut >= len(t_c) - 10:
                    idx_debut = 0
                x_c_tronque = x_c[idx_debut:]
                y_c_tronque = y_c[idx_debut:]
                t_c_tronque = t_c[idx_debut:] - t_c[idx_debut]
                missile = Interceptor(
                    x0=self.interceptor_pos[0], y0=self.interceptor_pos[1],
                    vitesse=self.missile_vitesse, kill_radius=self.kill_radius
                )
                r = missile.proportional_navigation(x_c_tronque, y_c_tronque, t_c_tronque)
                evaluation['interception'] = r
                evaluation['delai_lancement'] = delai
                missiles_utilises += 1
                if r['intercepte']:
                    interceptions += 1
            else:
                evaluation['interception'] = None
                evaluation['delai_lancement'] = None
                ignorees += 1
            resultats.append({
                'roquette': roquette,
                'trajectoire': (t_c, x_c, y_c),
                'evaluation': evaluation
            })
        return {
            'details': resultats, 'total_menaces': len(salve),
            'menaces_dangereuses': missiles_utilises, 'ignorees': ignorees,
            'interceptions': interceptions, 'missiles_utilises': missiles_utilises,
            'taux_interception': interceptions / max(missiles_utilises, 1) * 100
        }
