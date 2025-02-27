import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_historicalfigures(offset: int=None, name: str='julius caesar', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Historical Figures API endpoint. Returns a list of up to 10 people that match the search parameters. Either **name** and/or **title** parameter is required."
    offset: number of results to offset pagination.
        name: name of the person to search. Includes partial results (e.g. julius will match Julius Caesar).
        
    """
    url = f"https://historical-figures-by-api-ninjas.p.rapidapi.com/v1/historicalfigures"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "historical-figures-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

