import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def public_holiday_endpoint(day: str, month: str, year: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Public Holiday API allows you to gett the public, local, religious, and other holidays of a particular country."
    day: he day to get the holiday(s) from, in the format of 1-31.
        month: The month to get the holiday(s) from, in the format of 1-12 (e.g., 1 is January, 2 is February, etc).
        year: The year to get the holiday(s) from.
        country: The country's two letter ISO 3166-1 alpha-2 code.
        
    """
    url = f"https://public-holidays3.p.rapidapi.com/v1"
    querystring = {'day': day, 'month': month, 'year': year, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-holidays3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

