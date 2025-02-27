import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_date(date: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the most discussed stocks on Reddit."
    date: Choose one: `today` | `yesterday` | `this_week` | `last_week` | `this_month` | `last_month`
        page: Enter a page number
        
    """
    url = f"https://wallstreetbets.p.rapidapi.com/"
    querystring = {'date': date, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wallstreetbets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

