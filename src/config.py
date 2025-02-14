import os
from dotenv import load_dotenv

load_dotenv() 

YELP_API_KEY = os.environ.get("YELP_API_KEY", "")
BASE_YELP_URL = "https://api.yelp.com/v3"