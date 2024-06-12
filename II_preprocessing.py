from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns


def check_tab(data, colonne_cible):

    if data.empty:
        print("Le tableau est vide.")
        return False

    if not all(isinstance(col, str) for col in data.columns):
        print("Le tableau doit avoir des en-têtes de colonnes valides.")
        return False

    # types de données
    types_colonnes = data.dtypes
    for col, dtype in types_colonnes.items():
        if dtype == 'object' and not data[col].apply(lambda x: isinstance(x, (str, type(None)))).all():
            print(f"La colonne '{col}' contient des types de données hétérogènes.")
            return False

    # présence de valeurs manquantes
    valeurs_manquantes = data.isnull().sum().sum()
    if valeurs_manquantes > 0:
        print(f"Le tableau contient {valeurs_manquantes} valeurs manquantes.")
        return False

    # taille du tableau
    if len(data) < 10:  # Ce seuil peut être ajusté selon les besoins
        print("Le tableau contient trop peu de données pour un traitement IA efficace.")
        return False

    # colonne cible présente
    if colonne_cible not in data.columns:
        print(f"La colonne cible '{colonne_cible}' n'est pas présente dans le tableau.")
        return False

    return True


def correlation_matrix(data, colonne_cible, seuil_corr):

    print(f"Nombre total de features initiales : {len(data.columns)}")

    corr_matrix = data.corr()

    print("Matrice de corrélation :")
    print(corr_matrix)

    # Enregistrer la matrice de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('visualization/correlation_matrix.png')
    plt.close()

    features_importantes = corr_matrix[colonne_cible][abs(corr_matrix[colonne_cible]) >= seuil_corr].index.tolist()
    if colonne_cible in features_importantes:
        features_importantes.remove(colonne_cible)

    # Supprimer les features inutiles
    sorted_data = data[features_importantes + [colonne_cible]]

    # Afficher les features conservées et leur nombre
    print("Features conservées :")
    print(features_importantes)
    print(f"Nombre de features conservées : {len(features_importantes)}")

    sorted_corr_matrix = sorted_data.corr()

    # Afficher la matrice de corrélation
    print("Matrice de corrélation :")
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Matrice de Corrélation')
    plt.savefig('visualization/sorted_corr_matrix.png')
    plt.close()

    return sorted_data


def scale_data(df, method='standardisation'):

    if method not in ['standardisation', 'normalisation']:
        raise ValueError("La méthode doit être 'standardisation' ou 'normalisation'")

    # Sélectionner les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64']).columns

    # Appliquer la normalisation ou standardisation
    if method == 'standardisation':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'normalisation':
        scaler = MinMaxScaler()


    df_copy = df.copy()  
    df_copy.loc[:, colonnes_numeriques] = scaler.fit_transform(df_copy[colonnes_numeriques])

    return df_copy