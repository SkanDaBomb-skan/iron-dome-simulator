import numpy as np
from scipy import interpolate

class Interceptor3D:
    def __init__(self, x0, y0, z0, vitesse=800, kill_radius=30):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.vitesse = vitesse
        self.kill_radius = kill_radius

    def _distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def pure_pursuit_3d(self, cx, cy, cz, ct, dt=0.1):
        mx, my, mz = self.x0, self.y0, self.z0
        mxl, myl, mzl, mtl = [mx], [my], [mz], [0]
        dists = []
        t, ok, ic = 0, False, 0
        while t < ct[-1]:
            while ic < len(ct)-1 and ct[ic] < t:
                ic += 1
            d = self._distance(mx, my, mz, cx[ic], cy[ic], cz[ic])
            dists.append(d)
            if d < self.kill_radius:
                ok = True
                break
            dx = cx[ic] - mx
            dy = cy[ic] - my
            dz = cz[ic] - mz
            dist_total = np.sqrt(dx**2 + dy**2 + dz**2)
            mx += self.vitesse * (dx / dist_total) * dt
            my += self.vitesse * (dy / dist_total) * dt
            mz += self.vitesse * (dz / dist_total) * dt
            t += dt
            mxl.append(mx); myl.append(my); mzl.append(mz); mtl.append(t)
        return {
            'missile_x': np.array(mxl), 'missile_y': np.array(myl),
            'missile_z': np.array(mzl), 'missile_t': np.array(mtl),
            'intercepte': ok,
            'distance_min': min(dists) if dists else float('inf'),
            'temps_interception': t if ok else None,
            'methode': 'Pure Pursuit 3D'
        }

    def proportional_navigation_3d(self, cx, cy, cz, ct, N=4, dt=0.01):
        mx, my, mz = self.x0, self.y0, self.z0
        t_fine = np.arange(ct[0], ct[-1], dt)
        if len(t_fine) < 2:
            return {
                'missile_x': np.array([mx]), 'missile_y': np.array([my]),
                'missile_z': np.array([mz]), 'missile_t': np.array([0]),
                'intercepte': False, 'distance_min': float('inf'),
                'temps_interception': None, 'methode': 'PN 3D'
            }
        interp_x = interpolate.interp1d(ct, cx, fill_value='extrapolate')
        interp_y = interpolate.interp1d(ct, cy, fill_value='extrapolate')
        interp_z = interpolate.interp1d(ct, cz, fill_value='extrapolate')
        cxf = interp_x(t_fine)
        cyf = interp_y(t_fine)
        czf = interp_z(t_fine)
        bi = len(t_fine) // 4
        for i in range(len(t_fine)):
            d = self._distance(mx, my, mz, cxf[i], cyf[i], czf[i])
            if d / self.vitesse <= t_fine[i] and cyf[i] > 200:
                bi = i
                break
        ddx = cxf[bi] - mx
        ddy = cyf[bi] - my
        ddz = czf[bi] - mz
        di = np.sqrt(ddx**2 + ddy**2 + ddz**2)
        vxm = self.vitesse * ddx / di
        vym = self.vitesse * ddy / di
        vzm = self.vitesse * ddz / di
        mxl, myl, mzl, mtl = [mx], [my], [mz], [0]
        dists = []
        t, ok, s = 0, False, 0
        prev_az, prev_el = None, None
        while t < t_fine[-1] and s < len(t_fine) - 1:
            tcx, tcy, tcz = cxf[s], cyf[s], czf[s]
            d = self._distance(mx, my, mz, tcx, tcy, tcz)
            dists.append(d)
            if d < self.kill_radius:
                ok = True
                break
            if my < -10:
                break
            ddx, ddy, ddz = tcx - mx, tcy - my, tcz - mz
            az = np.arctan2(ddz, ddx)
            el = np.arctan2(ddy, np.sqrt(ddx**2 + ddz**2))
            if prev_az is not None:
                az_dot = (az - prev_az) / dt
                el_dot = (el - prev_el) / dt
                while az_dot > np.pi/dt: az_dot -= 2*np.pi/dt
                while az_dot < -np.pi/dt: az_dot += 2*np.pi/dt
                while el_dot > np.pi/dt: el_dot -= 2*np.pi/dt
                while el_dot < -np.pi/dt: el_dot += 2*np.pi/dt
            else:
                az_dot, el_dot = 0, 0
            prev_az, prev_el = az, el
            if s < len(t_fine) - 2:
                vxc = (cxf[s+1] - cxf[s]) / dt
                vyc = (cyf[s+1] - cyf[s]) / dt
                vzc = (czf[s+1] - czf[s]) / dt
            else:
                vxc, vyc, vzc = 0, 0, 0
            vc = -((ddx*(vxc-vxm) + ddy*(vyc-vym) + ddz*(vzc-vzm)) / d)
            if vc < 50: vc = 50
            a_az = np.clip(N * vc * az_dot, -40*9.81, 40*9.81)
            a_el = np.clip(N * vc * el_dot, -40*9.81, 40*9.81)
            v_az = np.arctan2(vzm, vxm)
            v_el = np.arctan2(vym, np.sqrt(vxm**2 + vzm**2))
            vxm += -a_az * np.sin(v_az) * dt
            vzm += a_az * np.cos(v_az) * dt
            vym += a_el * np.cos(v_el) * dt
            vt = np.sqrt(vxm**2 + vym**2 + vzm**2)
            vxm = vxm/vt * self.vitesse
            vym = vym/vt * self.vitesse
            vzm = vzm/vt * self.vitesse
            mx += vxm*dt; my += vym*dt; mz += vzm*dt
            t += dt; s += 1
            mxl.append(mx); myl.append(my); mzl.append(mz); mtl.append(t)
        return {
            'missile_x': np.array(mxl), 'missile_y': np.array(myl),
            'missile_z': np.array(mzl), 'missile_t': np.array(mtl),
            'intercepte': ok,
            'distance_min': min(dists) if dists else float('inf'),
            'temps_interception': t if ok else None,
            'methode': 'PN 3D'
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
        """
        Calcule le moment optimal pour lancer le missile.
        On attend que la roquette soit assez proche pour
        intercepter AU-DESSUS de la zone protégée.
        """
        mx, my = self.interceptor_pos
        zone_centre = (self.zone_x_min + self.zone_x_max) / 2

        # Chercher le moment où le missile peut intercepter
        # au-dessus de la zone protégée
        for i in range(len(cible_t)):
            # Distance missile → position cible à cet instant
            d = np.sqrt((cible_x[i] - mx)**2 + (cible_y[i] - my)**2)
            t_vol_missile = d / self.missile_vitesse

            # Le missile arrive à temps ET la cible est au-dessus de la zone
            x_cible = cible_x[i]
            dans_zone_x = self.zone_x_min - 2000 <= x_cible <= self.zone_x_max + 1000
            altitude_ok = cible_y[i] > 300

            if t_vol_missile <= cible_t[i] and dans_zone_x and altitude_ok:
                # Calculer le délai : lancer pour arriver juste à temps
                delai = max(0, cible_t[i] - t_vol_missile - 1.0)
                return delai

        return 0  # lancer immédiatement si pas de solution

    def evaluer_menace(self, cible_x, cible_y, cible_t):
        x_impact = cible_x[-1]
        dans_zone = self.impact_dans_zone(cible_x, cible_y)
        return {
            'x_impact': x_impact,
            'dans_zone': dans_zone,
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
                # Calculer le délai de lancement optimal
                delai = self._calculer_delai_lancement(x_c, y_c, t_c)

                # Tronquer la trajectoire cible à partir du délai
                idx_debut = np.searchsorted(t_c, delai)
                if idx_debut >= len(t_c) - 10:
                    idx_debut = 0

                x_c_tronque = x_c[idx_debut:]
                y_c_tronque = y_c[idx_debut:]
                t_c_tronque = t_c[idx_debut:] - t_c[idx_debut]

                missile = Interceptor(
                    x0=self.interceptor_pos[0],
                    y0=self.interceptor_pos[1],
                    vitesse=self.missile_vitesse,
                    kill_radius=self.kill_radius
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
            'details': resultats,
            'total_menaces': len(salve),
            'menaces_dangereuses': missiles_utilises,
            'ignorees': ignorees,
            'interceptions': interceptions,
            'missiles_utilises': missiles_utilises,
            'taux_interception': interceptions / max(missiles_utilises, 1) * 100
        }

class EngagementManager3D:
    def __init__(self, zone_x_min, zone_x_max, zone_z_min, zone_z_max,
                 interceptor_pos=(0, 0, 0), missile_vitesse=800, kill_radius=30, stock_missiles=10):
        self.zone_x_min = zone_x_min
        self.zone_x_max = zone_x_max
        self.zone_z_min = zone_z_min
        self.zone_z_max = zone_z_max
        self.interceptor_pos = interceptor_pos
        self.missile_vitesse = missile_vitesse
        self.kill_radius = kill_radius
        self.stock_missiles = stock_missiles

    def impact_dans_zone(self, x_impact, z_impact):
        return (self.zone_x_min <= x_impact <= self.zone_x_max and
                self.zone_z_min <= z_impact <= self.zone_z_max)

    def calculer_score_priorite(self, tc, xc, yc, zc, alpha=1.0, beta=1.0):
        tti = tc[-1]
        x_c = (self.zone_x_min + self.zone_x_max) / 2
        z_c = (self.zone_z_min + self.zone_z_max) / 2
        d_epi = np.sqrt((xc[-1] - x_c)**2 + (zc[-1] - z_c)**2)
        r_defense = np.sqrt(((self.zone_x_max - self.zone_x_min)/2)**2 + ((self.zone_z_max - self.zone_z_min)/2)**2)
        
        if tti <= 0: tti = 0.01
        if r_defense <= 0: r_defense = 1.0
        
        sp = alpha * (1.0 / tti) + beta * (1.0 - d_epi / r_defense)
        return sp

    def gerer_salve(self, salve):
        resultats = []
        missiles_utilises = 0
        interceptions = 0
        ignorees_hors_zone = 0
        echecs_manque_missiles = 0

        # Phase 1 : Evaluation TEWA
        menaces_evaluees = []
        for roquette in salve:
            tc, xc, yc, zc = roquette.trajectoire_rk4(dt=0.1)
            dans_zone = self.impact_dans_zone(xc[-1], zc[-1])
            
            if dans_zone:
                sp = self.calculer_score_priorite(tc, xc, yc, zc)
                menaces_evaluees.append({
                    'roquette': roquette,
                    'trajectoire': (tc, xc, yc, zc),
                    'dans_zone': True,
                    'score_sp': sp
                })
            else:
                ignorees_hors_zone += 1
                resultats.append({
                    'roquette': roquette,
                    'trajectoire': (tc, xc, yc, zc),
                    'dans_zone': False,
                    'interception': None,
                    'statut': 'Ignorée (Hors zone)'
                })

        # Phase 2 : Tri par priorité
        menaces_evaluees.sort(key=lambda x: x['score_sp'], reverse=True)

        # Phase 3 : Engagement
        for menace in menaces_evaluees:
            roquette = menace['roquette']
            tc, xc, yc, zc = menace['trajectoire']
            
            if self.stock_missiles > 0:
                self.stock_missiles -= 1
                missiles_utilises += 1
                
                missile = Interceptor3D(
                    *self.interceptor_pos,
                    vitesse=self.missile_vitesse,
                    kill_radius=self.kill_radius
                )
                r = missile.proportional_navigation_3d(xc, yc, zc, tc)
                
                if r['intercepte']:
                    interceptions += 1
                    
                resultats.append({
                    'roquette': roquette,
                    'trajectoire': (tc, xc, yc, zc),
                    'dans_zone': True,
                    'interception': r,
                    'score_sp': menace['score_sp'],
                    'statut': 'Engagée'
                })
            else:
                echecs_manque_missiles += 1
                resultats.append({
                    'roquette': roquette,
                    'trajectoire': (tc, xc, yc, zc),
                    'dans_zone': True,
                    'interception': None,
                    'score_sp': menace['score_sp'],
                    'statut': 'ECHEC (Manque de missiles)'
                })

        dangereuses = missiles_utilises + echecs_manque_missiles
        return {
            'details': resultats, 
            'total': len(salve),
            'dangereuses': dangereuses,
            'ignorees': ignorees_hors_zone,
            'manque_missiles': echecs_manque_missiles,
            'interceptions': interceptions,
            'missiles_utilises': missiles_utilises,
            'taux': (interceptions / max(dangereuses, 1)) * 100
        }
