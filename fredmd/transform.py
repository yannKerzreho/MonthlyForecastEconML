import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'fredmd/fredmd.csv' # J'ai mis à jour selon le chemin de ton erreur
OUTPUT_FILE = 'fredmd/nice_fredmd.csv'

def transform_series(series, code):
    """
    Applique la transformation FRED-MD.
    Retourne la série transformée et le nombre de lignes 'perdues' au début (lag).
    """
    s = pd.to_numeric(series, errors='coerce')
    
    expected_lag = 0
    
    if code == 1:   # Level
        res = s
        expected_lag = 0
    elif code == 2: # First Difference
        res = s.diff()
        expected_lag = 1
    elif code == 3: # Second Difference
        res = s.diff().diff()
        expected_lag = 2
    elif code == 4: # Log
        res = np.log(s)
        expected_lag = 0
    elif code == 5: # Log First Difference
        res = np.log(s).diff()
        expected_lag = 1
    elif code == 6: # Log Second Difference
        res = np.log(s).diff().diff()
        expected_lag = 2
    else:
        res = s
        expected_lag = 0
        
    return res, expected_lag

def main():
    print(f"--- Démarrage du traitement de {INPUT_FILE} ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERREUR : Le fichier '{INPUT_FILE}' n'a pas été trouvé.")
        return

    # 1. Lecture
    try:
        df_raw = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"ERREUR lecture CSV : {e}")
        return

    # Récupération des codes (Ligne index 0)
    try:
        # CORRECTION FUTUREWARNING : Conversion en float d'abord, puis fillna, puis int
        transf_codes = df_raw.iloc[0, 1:].astype(float).fillna(1).astype(int)
    except ValueError:
        print("ERREUR : Les codes de transformation posent problème.")
        return

    # Données brutes (à partir de l'index 1)
    data = df_raw.iloc[1:].copy()
    data.reset_index(drop=True, inplace=True)

    date_col = df_raw.columns[0]
    data[date_col] = pd.to_datetime(data[date_col])
    
    # CORRECTION PERFORMANCEWARNING : On utilise un dictionnaire pour éviter la fragmentation
    transformed_data_dict = {date_col: data[date_col]}
    lags_expected = {}

    print("Traitement des colonnes...")

    # 2. Transformation
    for col_name in df_raw.columns[1:]:
        code = transf_codes[col_name]
        original_series = data[col_name]
        
        transformed_series, lag = transform_series(original_series, code)
        
        # Ajout au dictionnaire au lieu du DataFrame
        transformed_data_dict[col_name] = transformed_series
        lags_expected[col_name] = lag

    # Création du DataFrame d'un seul coup (très rapide et non fragmenté)
    df_transformed = pd.DataFrame(transformed_data_dict)

    # 3. Nettoyage du "Ragged Edge" (Les 10 dernières lignes)
    print("\n--- Nettoyage du 'Ragged Edge' (fin de fichier) ---")
    
    # On isole les 10 dernières lignes
    tail_indices = df_transformed.index[-10:]
    
    # On identifie les lignes dans ce tail qui ont au moins un NaN
    rows_with_nan_in_tail = df_transformed.loc[tail_indices].isna().any(axis=1)
    indices_to_drop = rows_with_nan_in_tail[rows_with_nan_in_tail].index
    
    if len(indices_to_drop) > 0:
        print(f"Suppression de {len(indices_to_drop)} ligne(s) incomplète(s) à la fin du fichier (Ragged Edge).")
        df_transformed = df_transformed.drop(indices_to_drop).reset_index(drop=True)
    else:
        print("Aucune ligne incomplète détectée dans les 10 dernières dates.")

    print(f"Lignes restantes : {len(df_transformed)}")

    # 4. Rapport intelligent sur les NaN
    print("\n--- Rapport de validité des données ---")
    print("(Affiche uniquement les colonnes où des données manquent au-delà de la transformation mathématique)")
    
    something_printed = False
    
    for col in df_transformed.columns[1:]:
        series = df_transformed[col]
        nan_indices = series.index[series.isna()]
        
        if len(nan_indices) > 0:
            # Liste des index qui sont NaN
            nan_list = list(nan_indices)
            last_nan_idx = nan_list[-1]
            expected_lag = lags_expected.get(col, 0)
            
            # On vérifie si le dernier NaN dépasse le lag attendu
            # Note: Si expected_lag = 1, l'index 0 est normalement NaN.
            if last_nan_idx >= expected_lag:
                last_nan_date = df_transformed.loc[last_nan_idx, date_col]
                date_str = last_nan_date.strftime('%Y-%m-%d')
                
                # Date de début des données complètes
                if last_nan_idx + 1 < len(df_transformed):
                    start_valid_date = df_transformed.loc[last_nan_idx + 1, date_col].strftime('%Y-%m-%d')
                    print(f"Colonne '{col}' (Code {transf_codes[col]}) : Incomplète jusqu'au {date_str}. (Valide dès {start_valid_date})")
                else:
                    print(f"Colonne '{col}' (Code {transf_codes[col]}) : Vide ou entièrement NaN à la fin.")
                
                something_printed = True
    
    if not something_printed:
        print("Toutes les colonnes sont complètes (mis à part les NaNs initiaux normaux).")

    # 5. Sauvegarde
    df_transformed.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- Terminé. Fichier sauvegardé : {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()