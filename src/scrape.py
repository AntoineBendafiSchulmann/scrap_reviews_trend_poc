import sys
import uuid
import time
from decimal import Decimal
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

from src.yelp_api import get_restaurants_by_location

def scrape_reviews_selenium(alias, max_reviews=50):
    data = []
    if not alias:
        return data
    url = f"https://www.yelp.com/biz/{alias}"
    driver = webdriver.Chrome()
    try:
        driver.get(url)
        time.sleep(3)
        elements = driver.find_elements(By.XPATH, '//p[contains(@class,"comment__09f24__D0cxf")]')
        for e in elements[:max_reviews]:
            try:
                review_text = e.text
            except:
                review_text = ""
            try:
                rating_el = e.find_element(
                    By.XPATH,
                    './/div[@role="img" and contains(@aria-label,"star rating")]'
                )
                rating_str = rating_el.get_attribute("aria-label")
                rating_value = rating_str.split(" ")[0]
            except:
                rating_value = "0"
            try:
                date_el = e.find_element(By.XPATH, './/span[contains(@class,"css-1e4fdj9")]')
                time_created = date_el.text
            except:
                time_created = ""
            try:
                rating_decimal = Decimal(rating_value)
            except:
                rating_decimal = Decimal("0")
            data.append({
                "text": review_text,
                "rating": rating_decimal,
                "time_created": time_created
            })
    finally:
        driver.quit()
    return data

def main():
    if len(sys.argv) > 1:
        location = sys.argv[1]
    else:
        location = "Paris"

    restaurants = get_restaurants_by_location(location=location, limit=10)


    with open("reviews_output.txt", "a", encoding="utf-8") as f:
        for r in restaurants:
            rest_id = r.get("id", "")
            alias = r.get("alias", "")
            name = r.get("name", "")
            rating_float = r.get("rating", 0)
            rating_dec = Decimal(str(rating_float))

            if not alias:
                continue

            reviews = scrape_reviews_selenium(alias, 5)
            for rev in reviews:

                review_id = str(uuid.uuid4())
                text = rev.get("text", "").replace("\n", " ")
                review_rating = rev.get("rating", Decimal("0"))
                time_created = rev.get("time_created", "")

                line = (
                    f"{rest_id}\t"
                    f"{alias}\t"
                    f"{name}\t"
                    f"{rating_dec}\t"
                    f"{review_id}\t"
                    f"{review_rating}\t"
                    f"{time_created}\t"
                    f"{text}\n"
                )
                f.write(line)

if __name__ == "__main__":
    main()
