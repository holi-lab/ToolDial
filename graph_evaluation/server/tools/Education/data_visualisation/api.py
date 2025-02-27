import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getting_data(x_rapidapi_key: str='demo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this api is for the sales of supermarkets in three different cities"
    
    """
    url = f"https://data_visualisation_.p.rapidapi.com/api/get-data/"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "data_visualisation_.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

