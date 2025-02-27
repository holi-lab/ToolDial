import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_app_details(country: str, appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of App Store Listing"
    country: Enter the two letter country code to search in. Default is **us**. Examples include:

- us
- ca
- gb
- de
etc
        appid: Enter the app ID that you want to retrieve, for example **544007664**. You can find this ID from the app store listing URL
        
    """
    url = f"https://apple-app-store-scraper.p.rapidapi.com/v1/appstore"
    querystring = {'country': country, 'appid': appid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-app-store-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

