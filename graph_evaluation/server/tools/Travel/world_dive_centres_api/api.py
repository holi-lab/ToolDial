import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_dive_operators_by_a_country_or_a_region(country: str='phuket', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns a json list of dive sites which match the region or country entered as the query. The range of results depend but there is usually around 100-500 sites for each country. There are around 15'000 dive sites listed in the database."
    
    """
    url = f"https://world-dive-centres-api.p.rapidapi.com/api/divecentres"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-dive-centres-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

