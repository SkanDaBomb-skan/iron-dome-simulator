import numpy as np
import sys
sys.path.append('src')

from simulation.rocket3d import Rocket3D, ThreatGenerator3D
from simulation.interceptor3d import EngagementManager3D

def test_tewa():
    print("Test du TEWA et de la gestion des stocks de missiles...")
    
    # Manager avec seulement 3 missiles en stock
    em = EngagementManager3D(
        zone_x_min=-1000, zone_x_max=1000,
        zone_z_min=-1000, zone_z_max=1000,
        interceptor_pos=(0, 0, 0),
        missile_vitesse=800,
        stock_missiles=3
    )

    salve = []
    # Roquettes dans la zone (dangereuses)
    for _ in range(5):
        # Tirs tres verticaux (TTI court) ou longs (TTI long) pour varier les scores
        v0 = np.random.uniform(200, 300)
        th = np.random.uniform(60, 80)
        r = Rocket3D(v0=v0, theta_deg=th, phi_deg=0, masse=100, Cd=0.3, x0=-3000, z0=0)
        salve.append(r)
        
    # Roquette hors de la zone (ignoree)
    r_hors = Rocket3D(v0=300, theta_deg=45, phi_deg=80, masse=100, Cd=0.3, x0=-8000, z0=8000)
    salve.append(r_hors)
    
    resultats = em.gerer_salve(salve)
    
    print(f"\nTotal menaces: {resultats['total']}")
    print(f"Menaces dangereuses: {resultats['dangereuses']}")
    print(f"Ignorees (hors zone): {resultats['ignorees']}")
    print(f"Echecs (manque missiles): {resultats['manque_missiles']}")
    print(f"Missiles utilises: {resultats['missiles_utilises']}")
    print(f"Interceptions: {resultats['interceptions']}")
    print("\nDetails par ordre de traitement :")
    for r in resultats['details']:
        statut = r.get('statut', 'Inconnu')
        sp = r.get('score_sp', 0)
        print(f" - Statut: {statut:<30} | Score Sp: {sp:.4f}")

if __name__ == '__main__':
    test_tewa()
