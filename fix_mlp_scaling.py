import json

filepath = 'notebooks/phase2_prediction.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if not cell.get('source'):
        continue
    
    src = "".join(cell['source'])
    if "class MLPPredictor:" in src:
        new_source = [
            "from sklearn.neural_network import MLPRegressor\n",
            "from sklearn.pipeline import make_pipeline\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.compose import TransformedTargetRegressor\n",
            "\n",
            "class MLPPredictor:\n",
            "    def __init__(self):\n",
            "        # Un réseau de neurones avec 2 couches cachées de 64 neurones\n",
            "        base_model = make_pipeline(\n",
            "            StandardScaler(),\n",
            "            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42, early_stopping=True)\n",
            "        )\n",
            "        # IMPORTANT : Normaliser les cibles (y) est crucial pour un MLP !\n",
            "        self.model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())\n",
            "        self.est_entraine = False\n",
            "        \n",
            "    def extraire_features(self, t_obs, x_obs, y_obs):\n",
            "        if len(t_obs) < 2:\n",
            "            return np.zeros(6)\n",
            "        \n",
            "        n = min(5, len(t_obs))\n",
            "        vx = np.polyfit(t_obs[:n], x_obs[:n], 1)[0]\n",
            "        vy = np.polyfit(t_obs[:n], y_obs[:n], 1)[0]\n",
            "        return np.array([x_obs[0], y_obs[0], vx, vy, x_obs[-1], y_obs[-1]])\n",
            "        \n",
            "    def entrainer(self, trajectories, cibles):\n",
            "        X, y = [], []\n",
            "        for t_obs, x_obs, y_obs in trajectories:\n",
            "            features = self.extraire_features(t_obs, x_obs, y_obs)\n",
            "            X.append(features)\n",
            "        self.model.fit(X, cibles)\n",
            "        self.est_entraine = True\n",
            "        \n",
            "    def predire(self, t_obs, x_obs, y_obs):\n",
            "        if not self.est_entraine:\n",
            "            raise ValueError(\"Le modele n'est pas entraine !\")\n",
            "        features = self.extraire_features(t_obs, x_obs, y_obs)\n",
            "        return self.model.predict([features])[0]\n"
        ]
        
        # We need to keep the generation code at the end of the cell
        tail = []
        keep = False
        for line in cell['source']:
            if "def generer_dataset_mlp" in line:
                keep = True
            if keep:
                tail.append(line)
        
        cell['source'] = new_source + ["\n"] + tail

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed MLPPredictor target scaling")
