import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_scrape(url: str, ua: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/v1/scrape API endpoint is used for scraping.
		When you call this endpoint, a chrome browser runs behind the scenes."
    url: URL of the website which you will scrape
        ua: ua agent is used to change the user agent.
The underlying browser will be Chrome - no matter what ua is set.
        
    """
    url = f"https://scrapingbuddy.p.rapidapi.com/v1/scrape"
    querystring = {'url': url, }
    if ua:
        querystring['ua'] = ua
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapingbuddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

