import requests
import schedule
import time

# URL du fichier Excel
url = "https://agriculture.ec.europa.eu/document/download/62d01488-33a0-4601-a841-ca48fa11d999_en?filename=eu-milk-historical-price-series_en.xlsx"

# Fonction pour télécharger et lire les données
def download_and_save_excel():
    response = requests.get(url)
    with open('initial_datas/Ire_EU_Milk_Prices.xlsx', 'wb') as f:
        f.write(response.content)
    print("Fichier Excel téléchargé et enregistré.")

# Planifier la tâche pour qu'elle s'exécute chaque semaine
schedule.every().week.do(download_and_save_excel)

if __name__ == '__main__':
    # Boucle pour exécuter la tâche planifiée
    while True:
        schedule.run_pending()
        time.sleep(1)  # Attend une seconde avant de vérifier à nouveau
