import numpy as np
from scipy import interpolate
import scipy.linalg as linalg

class Radar:
    def __init__(self, sigma=30, frequence=10):
        self.sigma = sigma
        self.frequence = frequence
        self.dt_mesure = 1.0 / frequence

    def observer(self, t_vrai, x_vrai, y_vrai, duree_observation=5.0):
        masque = t_vrai <= duree_observation
        t_obs_base = t_vrai[masque]
        x_obs_base = x_vrai[masque]
        y_obs_base = y_vrai[masque]
        indices = np.arange(0, len(t_obs_base), max(1, int(1.0 / (self.frequence * 0.1))))
        t_obs = t_obs_base[indices]
        x_vrai_obs = x_obs_base[indices]
        y_vrai_obs = y_obs_base[indices]
        x_obs = x_vrai_obs + np.random.normal(0, self.sigma, len(t_obs))
        y_obs = y_vrai_obs + np.random.normal(0, self.sigma, len(t_obs))
        y_obs = np.maximum(y_obs, 0)
        return {'t': t_obs, 'x': x_obs, 'y': y_obs,
                'x_vrai': x_vrai_obs, 'y_vrai': y_vrai_obs, 'n_mesures': len(t_obs)}

class KalmanFilter:
    """
    Unscented Kalman Filter (UKF) pour suivi et prediction de trajectoire balistique.
    
    Estime dynamiquement le vecteur d'etat [x, y, vx, vy, k] ou k est le 
    coefficient de trainee balistique (drag). L'UKF utilise la propagation
    par points sigma, eliminant le besoin de Jacobiennes et augmentant la
    precision sur la non-linearite du vol.
    """
    
    def __init__(self, sigma_processus=5.0, sigma_mesure=30.0):
        self.g = 9.81
        self.sigma_processus = sigma_processus
        self.sigma_mesure = sigma_mesure
        self.k_drag_init = 0.000086
        
        # Dimensions
        self.n = 5
        
        # Parametres Merwe Scaled Sigma Points
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Poids
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lam))
            self.Wc[i] = 1.0 / (2 * (self.n + self.lam))

    def _generer_points_sigma(self, x, P):
        try:
            U = linalg.cholesky((self.n + self.lam) * P)
        except linalg.LinAlgError:
            U = linalg.cholesky((self.n + self.lam) * (P + np.eye(self.n) * 1e-8))
            
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for k in range(self.n):
            sigmas[k + 1] = x + U[k]
            sigmas[self.n + k + 1] = x - U[k]
        return sigmas

    def _derivees(self, state, k):
        """Calcule les derivees pour RK4 : [vx, vy, ax, ay]."""
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)
        if v > 0 and k > 0:
            ax = -k * v * vx
            ay = -self.g - k * v * vy
        else:
            ax = 0
            ay = -self.g
        return np.array([vx, vy, ax, ay])
        
    def _rk4_step(self, X, dt):
        """Propage l'etat [x, y, vx, vy, k] sur dt via RK4."""
        x, y, vx, vy, k = X
        state = np.array([x, y, vx, vy])
        k1 = self._derivees(state, k)
        k2 = self._derivees(state + dt/2 * k1, k)
        k3 = self._derivees(state + dt/2 * k2, k)
        k4 = self._derivees(state + dt * k3, k)
        
        new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return np.array([new_state[0], new_state[1], new_state[2], new_state[3], k])

    def filtrer(self, t_obs, x_obs, y_obs):
        if len(t_obs) < 2:
            return np.array([[x_obs[0], y_obs[0], 0, 0, self.k_drag_init]])

        dt = t_obs[1] - t_obs[0]
        K_SCALE = 10000.0
        
        # Initialisation robuste pour la Vitesse
        n_init = min(5, len(t_obs))
        vx0 = np.polyfit(t_obs[:n_init], x_obs[:n_init], 1)[0]
        vy0 = np.polyfit(t_obs[:n_init], y_obs[:n_init], 1)[0]
        X = np.array([x_obs[0], y_obs[0], vx0, vy0, self.k_drag_init * K_SCALE])
        
        P = np.diag([
            self.sigma_mesure**2, self.sigma_mesure**2,
            (self.sigma_mesure)**2, (self.sigma_mesure)**2,
            0.5**2
        ])
        
        # On FOCUS sur la vitesse (Q tres petit = vitesse ultra lisse)
        # Et on donne moins d'importance a k (Q tres petit = k ne bouge presque pas)
        sigma_a = 2.0  # La vitesse doit suivre une physique propre
        Q = np.diag([
            (0.5 * sigma_a * dt**2)**2, (0.5 * sigma_a * dt**2)**2,
            (sigma_a * dt)**2, (sigma_a * dt)**2,
            0.01**2  # k est bloque, il n'a pas le droit d'absorber le bruit
        ])
        R = np.eye(2) * self.sigma_mesure**2
        
        etats = []
        for j in range(len(t_obs)):
            # --- PREDICTION UKF ---
            sigmas = self._generer_points_sigma(X, P)
            sigmas_f = np.zeros_like(sigmas)
            
            for i in range(2 * self.n + 1):
                state_physique = sigmas[i].copy()
                state_physique[4] /= K_SCALE
                
                state_pred = self._rk4_step(state_physique, dt)
                
                state_pred[4] *= K_SCALE
                sigmas_f[i] = state_pred
                
            X_pred = np.dot(self.Wm, sigmas_f)
            
            P_pred = Q.copy()
            for i in range(2 * self.n + 1):
                y_diff = sigmas_f[i] - X_pred
                P_pred += self.Wc[i] * np.outer(y_diff, y_diff)
                
            # --- MISE A JOUR UKF ---
            sigmas_h = np.zeros((2 * self.n + 1, 2))
            for i in range(2 * self.n + 1):
                sigmas_h[i] = [sigmas_f[i, 0], sigmas_f[i, 1]]
                
            zp = np.dot(self.Wm, sigmas_h)
            
            S = R.copy()
            for i in range(2 * self.n + 1):
                y_diff = sigmas_h[i] - zp
                S += self.Wc[i] * np.outer(y_diff, y_diff)
                
            Pxz = np.zeros((self.n, 2))
            for i in range(2 * self.n + 1):
                Pxz += self.Wc[i] * np.outer(sigmas_f[i] - X_pred, sigmas_h[i] - zp)
                
            K_gain = np.dot(Pxz, linalg.inv(S))
            z_obs = np.array([x_obs[j], y_obs[j]])
            
            X = X_pred + np.dot(K_gain, z_obs - zp)
            P = P_pred - np.dot(K_gain, np.dot(S, K_gain.T))
            
            X[4] = np.clip(X[4], 0.1, 8.0)
            
            etat = X.copy()
            etat[4] = etat[4] / K_SCALE
            etats.append(etat)
            
        return np.array(etats)

    def _simuler_trajectoire(self, etat, force_k=None, dt=0.01):
        if len(etat) == 5:
            x, y, vx, vy, k = etat
        else:
            x, y, vx, vy = etat
            k = self.k_drag_init
            
        if force_k is not None:
            k = force_k
            
        state = np.array([x, y, vx, vy], dtype=float)
        preds = [state.copy()]

        for _ in range(50000):
            k1 = self._derivees(state, k)
            k2 = self._derivees(state + dt/2 * k1, k)
            k3 = self._derivees(state + dt/2 * k2, k)
            k4 = self._derivees(state + dt * k3, k)
            
            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

            if new_state[1] < 0 and state[1] >= 0:
                frac = state[1] / (state[1] - new_state[1])
                x_impact = state[0] + frac * (new_state[0] - state[0])
                preds.append(np.array([x_impact, 0, 0, 0]))
                break
            
            state = new_state
            preds.append(state.copy())
            
            if state[1] < -100:
                break

        return np.array(preds)

    def pipeline_complet(self, t_obs, x_obs, y_obs):
        etats = self.filtrer(t_obs, x_obs, y_obs)
        etat_final = etats[-1]
        k_estime = etat_final[4]

        preds_brut = self._simuler_trajectoire(etat_final, force_k=0)
        preds_drag = self._simuler_trajectoire(etat_final, force_k=k_estime)

        return {
            'etats_filtres': etats,
            'predictions_brutes': preds_brut,
            'predictions': preds_drag,
            'x_impact_brut': preds_brut[-1][0],
            'x_impact': preds_drag[-1][0],
            'k_estime': k_estime
        }

class PolynomialPredictor:
    def __init__(self, deg_x=2, deg_y=2):
        self.deg_x = deg_x; self.deg_y = deg_y
    def entrainer(self, t_obs, x_obs, y_obs):
        self.coeff_x = np.polyfit(t_obs, x_obs, self.deg_x)
        self.coeff_y = np.polyfit(t_obs, y_obs, self.deg_y)
    def predire(self, t_futur):
        return np.polyval(self.coeff_x, t_futur), np.polyval(self.coeff_y, t_futur)
    def predire_trajectoire(self, t_obs, x_obs, y_obs, dt=0.1):
        self.entrainer(t_obs, x_obs, y_obs)
        t_f = np.arange(t_obs[0], t_obs[-1]*10, dt)
        x_p, y_p = self.predire(t_f)
        idx = np.where(y_p < 0)[0]
        if len(idx) > 0:
            i = idx[0]; t_f = t_f[:i+1]; x_p = x_p[:i+1]; y_p = y_p[:i+1]; y_p[-1] = 0
        return t_f, x_p, y_p
