import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_10_years_of_price_to_earnings(bse_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 10 years of historical Price to Earnings ratio for Indian companies listed on BSE. All you need to do is pass in the BSE Id of the company."
    
    """
    url = f"https://indian-stock-exchange1.p.rapidapi.com/api/getPE"
    querystring = {'bse_id': bse_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-stock-exchange1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

