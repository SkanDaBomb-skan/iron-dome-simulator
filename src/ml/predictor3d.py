import numpy as np
import scipy.linalg as linalg

class Radar3D:
    def __init__(self, sigma=30, frequence=10):
        self.sigma=sigma;self.frequence=frequence
    def observer(self, t_v, x_v, y_v, z_v, duree=8.0):
        m=t_v<=duree;to=t_v[m];xo=x_v[m];yo=y_v[m];zo=z_v[m]
        ind=np.arange(0,len(to),max(1,int(1.0/(self.frequence*0.1))))
        t=to[ind];xr=xo[ind];yr=yo[ind];zr=zo[ind]
        return {'t':t,'x':xr+np.random.normal(0,self.sigma,len(t)),
                'y':np.maximum(yr+np.random.normal(0,self.sigma,len(t)),0),
                'z':zr+np.random.normal(0,self.sigma,len(t)),'n':len(t)}

class UKF3D:
    """
    Unscented Kalman Filter en 3D.
    Etat : [x, y, z, vx, vy, vz, k] (7 dimensions)
    k = coefficient de drag estime conjointement.
    Utilise la propagation par points sigma (Merwe Scaled),
    eliminant le besoin de Jacobiennes analytiques.
    """
    def __init__(self, sigma_mesure=30.0):
        self.g = 9.81
        self.sm = sigma_mesure
        self.n = 7  # dimension de l'etat

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

        # Matrice d'observation H (3x7) et bruit de mesure R (3x3)
        self.H = np.zeros((3, 7))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1
        self.R = np.eye(3) * sigma_mesure**2

    def _generer_points_sigma(self, x, P):
        """Genere les 2n+1 points sigma a partir de l'etat et de la covariance."""
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

    def _f(self, s, dt):
        """Transition non-lineaire (gravite + drag 3D)."""
        x, y, z, vx, vy, vz, k = s
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        if v > 0 and k > 0:
            ax = -k * v * vx
            ay = -self.g - k * v * vy
            az = -k * v * vz
        else:
            ax = 0; ay = -self.g; az = 0
        return np.array([
            x + vx*dt + 0.5*ax*dt**2,
            y + vy*dt + 0.5*ay*dt**2,
            z + vz*dt + 0.5*az*dt**2,
            vx + ax*dt, vy + ay*dt, vz + az*dt, k
        ])

    def filtrer(self, t_obs, x_obs, y_obs, z_obs):
        """Filtrage UKF 3D : estime [x,y,z,vx,vy,vz,k]."""
        if len(t_obs) < 2:
            return np.array([[x_obs[0], y_obs[0], z_obs[0], 0, 0, 0, 0.0001]])

        dt = t_obs[1] - t_obs[0]
        K_SCALE = 10000.0

        # Initialisation robuste
        n_init = min(5, len(t_obs))
        vx0 = np.polyfit(t_obs[:n_init], x_obs[:n_init], 1)[0]
        vy0 = np.polyfit(t_obs[:n_init], y_obs[:n_init], 1)[0]
        vz0 = np.polyfit(t_obs[:n_init], z_obs[:n_init], 1)[0]

        X = np.array([x_obs[0], y_obs[0], z_obs[0], vx0, vy0, vz0, 0.00008 * K_SCALE])

        P = np.diag([
            self.sm**2, self.sm**2, self.sm**2,
            self.sm**2, self.sm**2, self.sm**2,
            0.3**2
        ])

        sigma_a = 2.0
        Q = np.diag([
            (0.5*sigma_a*dt**2)**2, (0.5*sigma_a*dt**2)**2, (0.5*sigma_a*dt**2)**2,
            (sigma_a*dt)**2, (sigma_a*dt)**2, (sigma_a*dt)**2,
            0.001**2
        ])

        etats = []
        for j in range(len(t_obs)):
            # --- PREDICTION UKF ---
            sigmas = self._generer_points_sigma(X, P)
            sigmas_f = np.zeros_like(sigmas)

            for i in range(2 * self.n + 1):
                state = sigmas[i].copy()
                state[6] /= K_SCALE
                state_pred = self._f(state, dt)
                state_pred[6] *= K_SCALE
                sigmas_f[i] = state_pred

            X_pred = np.dot(self.Wm, sigmas_f)

            P_pred = Q.copy()
            for i in range(2 * self.n + 1):
                d = sigmas_f[i] - X_pred
                P_pred += self.Wc[i] * np.outer(d, d)

            # --- MISE A JOUR UKF ---
            sigmas_h = np.zeros((2 * self.n + 1, 3))
            for i in range(2 * self.n + 1):
                sigmas_h[i] = [sigmas_f[i, 0], sigmas_f[i, 1], sigmas_f[i, 2]]

            zp = np.dot(self.Wm, sigmas_h)

            S = self.R.copy()
            for i in range(2 * self.n + 1):
                dz = sigmas_h[i] - zp
                S += self.Wc[i] * np.outer(dz, dz)

            Pxz = np.zeros((self.n, 3))
            for i in range(2 * self.n + 1):
                Pxz += self.Wc[i] * np.outer(sigmas_f[i] - X_pred, sigmas_h[i] - zp)

            K_gain = np.dot(Pxz, linalg.inv(S))
            z_obs_vec = np.array([x_obs[j], y_obs[j], z_obs[j]])

            X = X_pred + np.dot(K_gain, z_obs_vec - zp)
            P = P_pred - np.dot(K_gain, np.dot(S, K_gain.T))

            X[6] = np.clip(X[6], 0.1, 8.0)

            etat = X.copy()
            etat[6] = etat[6] / K_SCALE
            etats.append(etat)

        return np.array(etats)

    def simuler_impact(self, etat, k_override=None, dt=0.1):
        """Simule jusqu'a l'impact, retourne la trajectoire predite."""
        x, y, z, vx, vy, vz = etat[:6]
        k = k_override if k_override is not None else (etat[6] if len(etat) > 6 else 0)
        preds = [np.array([x, y, z, vx, vy, vz])]
        for _ in range(5000):
            v = np.sqrt(vx**2 + vy**2 + vz**2)
            ax = -k*v*vx if v > 0 and k > 0 else 0
            ay = -self.g - k*v*vy if v > 0 and k > 0 else -self.g
            az = -k*v*vz if v > 0 and k > 0 else 0
            nx = x + vx*dt + 0.5*ax*dt**2
            ny = y + vy*dt + 0.5*ay*dt**2
            nz = z + vz*dt + 0.5*az*dt**2
            nvx = vx + ax*dt; nvy = vy + ay*dt; nvz = vz + az*dt
            if ny < 0 and y >= 0:
                f = y / (y - ny)
                preds.append(np.array([x + f*(nx-x), 0, z + f*(nz-z), 0, 0, 0]))
                break
            x, y, z, vx, vy, vz = nx, ny, nz, nvx, nvy, nvz
            preds.append(np.array([x, y, z, vx, vy, vz]))
        return np.array(preds)

    def pipeline_complet(self, t_obs, x_obs, y_obs, z_obs):
        """Pipeline UKF 3D complet."""
        etats = self.filtrer(t_obs, x_obs, y_obs, z_obs)
        preds = self.simuler_impact(etats[-1])
        preds_brut = self.simuler_impact(etats[-1], k_override=0)
        return {
            'etats_filtres': etats[:,:6],
            'predictions': preds, 'predictions_brutes': preds_brut,
            'x_impact': preds[-1][0], 'z_impact': preds[-1][2],
            'x_impact_brut': preds_brut[-1][0], 'z_impact_brut': preds_brut[-1][2],
            'k_estime': etats[-1][6]}


import torch
import torch.nn as nn

class RocketLSTM3D(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # [x_impact, z_impact]
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = hn[-1]  # dernier état caché
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
