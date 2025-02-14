import requests
from src.config import YELP_API_KEY, BASE_YELP_URL

def get_restaurants_by_location(term="restaurants", location="Paris", limit=10):
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}"
    }
    params = {
        "term": term,
        "location": location,
        "limit": limit
    }
    url = f"{BASE_YELP_URL}/businesses/search"
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("businesses", [])

def get_reviews(business_id):
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}"
    }
    url = f"{BASE_YELP_URL}/businesses/{business_id}/reviews"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 404:
        print(f"[WARNING] 404 Not Found for ID: {business_id}. Skipping reviews.")
        return []

    response.raise_for_status()
    
    data = response.json()
    return data.get("reviews", [])