import numpy as np

class ScoringSystem:
    def evaluer_scenario(self, resultats):
        details = resultats['details']
        altitudes, temps, distances_min = [], [], []
        for r in details:
            ev = r['evaluation']
            if ev['interception'] is not None and ev['interception']['intercepte']:
                altitudes.append(ev['interception']['missile_y'][-1])
                temps.append(ev['interception']['temps_interception'])
                distances_min.append(ev['interception']['distance_min'])
        score = {
            'taux_interception': resultats['taux_interception'],
            'total_menaces': resultats['total_menaces'],
            'menaces_dangereuses': resultats['menaces_dangereuses'],
            'interceptions': resultats['interceptions'],
            'ignorees': resultats['ignorees'],
            'missiles_utilises': resultats['missiles_utilises'],
            'efficacite': resultats['interceptions'] / max(resultats['missiles_utilises'], 1),
            'altitude_moyenne': np.mean(altitudes) if altitudes else 0,
            'altitude_min': np.min(altitudes) if altitudes else 0,
            'temps_moyen': np.mean(temps) if temps else 0,
            'distance_min_moyenne': np.mean(distances_min) if distances_min else 0,
        }
        s = 0
        s += min(score['taux_interception'], 100) * 0.4
        s += score['efficacite'] * 100 * 0.2
        s += min(score['altitude_moyenne'] / 1000, 1) * 100 * 0.2
        s += max(0, (1 - score['temps_moyen'] / 30)) * 100 * 0.2
        score['score_global'] = round(s, 1)
        return score

    def afficher_score(self, score, nom_scenario=""):
        print(f"\n{'='*50}")
        print(f"  RAPPORT DE SCORE - {nom_scenario}")
        print(f"{'='*50}")
        print(f"  Menaces totales        : {score['total_menaces']}")
        print(f"  Menaces dangereuses    : {score['menaces_dangereuses']}")
        print(f"  Ignorees (hors zone)   : {score['ignorees']}")
        print(f"  Missiles utilises      : {score['missiles_utilises']}")
        print(f"  Interceptions reussies : {score['interceptions']}")
        print(f"  Taux d'interception    : {score['taux_interception']:.1f}%")
        print(f"  Efficacite             : {score['efficacite']*100:.1f}%")
        print(f"  Altitude moy. intercep.: {score['altitude_moyenne']:.0f} m")
        print(f"  Temps moyen intercep.  : {score['temps_moyen']:.2f} s")
        print(f"  SCORE GLOBAL           : {score['score_global']} / 100")
        print(f"{'='*50}")
