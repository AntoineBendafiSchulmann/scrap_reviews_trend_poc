import os
import sys
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

if len(sys.argv) < 2:
    print("Usage: python scrape_trustpilot.py <site>")
    sys.exit(1)

DOMAIN = sys.argv[1]
SITE_NAME = DOMAIN.replace(".fr", "").replace(".com", "").replace(".net", "").replace(".org", "").capitalize()
OUTPUT_FILE = "trustpilot_reviews.txt"

chrome_options = Options()
chrome_options.add_argument("--headless") 
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

page = 1
total_reviews = 0

while True:
    url = f"https://fr.trustpilot.com/review/{DOMAIN}?page={page}"
    print(f"Chargement de la page {page} → {url}")
    driver.get(url)
    time.sleep(random.uniform(3, 6))

    reviews = driver.find_elements(By.CLASS_NAME, "styles_reviewContent__44s_M")

    if not reviews:
        print(f"✅ Fin de l'extraction : Aucune donnée trouvée à la page {page}.")
        break 

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, review in enumerate(reviews, start=1):
            try:
                text = review.find_element(By.CLASS_NAME, "typography_body-l__v5JLj").text.strip()
                text = " ".join(text.splitlines())  
                f.write(f"{DOMAIN}\t{DOMAIN}\t{SITE_NAME}\t4.0\tREVIEW_{page}_{i}\t5\t{text}\n")
                total_reviews += 1
            except Exception as e:
                print(f"Erreur pour l'avis {i} sur la page {page}: {e}")

    print(f"✅ {len(reviews)} avis ajoutés depuis la page {page}.")
    
    page += 1 

print(f"Extraction terminée : {total_reviews} avis enregistrés dans `{OUTPUT_FILE}`.")
driver.quit()
