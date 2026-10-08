# download_sceneparse150.py
import os
import urllib.request
import zipfile

def download_file(url, dest):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to: {dest}")

def extract_zip(path, dest):
    print(f"Extracting: {path}")
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(dest)
    print(f"Extracted to: {dest}")

def download_sceneparse150(root="ADE20K"):
    os.makedirs(root, exist_ok=True)

    # URL officielle MIT ADE20K
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    zip_path = os.path.join(root, "ADEChallengeData2016.zip")

    # Télécharger si absent
    if not os.path.exists(zip_path):
        print("Téléchargement du dataset ADE20K (SceneParse150)...")
        download_file(url, zip_path)
    else:
        print("Archive déjà présente, skip download.")

    # Extraire
    print("Extraction...")
    extract_zip(zip_path, root)

    print("\nDataset ADE20K prêt !")
    print(f"Structure disponible dans : {root}/")

if __name__ == "__main__":
    download_sceneparse150()
