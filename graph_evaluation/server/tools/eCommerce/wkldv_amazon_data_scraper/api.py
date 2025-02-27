import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amazon_search_results(searchquery: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[wkalidev-amazon-scraper](uhttps://wkalidev-amazon-scraper.herokuapp.com/rl) is a tool that allows us to scrape any online resource. It collects the HTML from any web page using a simple API and it provides ready to process data. It’s great for extracting product information, processing real estate, HR, or financial data, and even tracking information for a specific market. Using wkalidev-amazon-scraper, we can get all the information needed from a specific Amazon product page."
    
    """
    url = f"https://wkldv-amazon-data-scraper.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wkldv-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

