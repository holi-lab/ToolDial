import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scraper(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Scraper" endpoint of the Web Scraping API is responsible for extracting data from websites while simulating a real browser. With its Headless Browser capability, the endpoint enables users to bypass restrictions, solve captchas, and scrape dynamic websites with ease. The endpoint can be used for high-level web scraping tasks, making it an ideal choice for businesses, data analysts, and developers who need to extract data from websites quickly and efficiently. By providing the endpoint with the necessary input parameters, users can initiate the scraping process and receive the extracted data in the desired format. This endpoint is a powerful tool for anyone who needs to extract data from websites and can be easily integrated into existing workflows and systems."
    
    """
    url = f"https://web-scraping-api.p.rapidapi.com/scrape"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

