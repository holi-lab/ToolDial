import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_motorcycles(model: str='Ninja', offset: int=None, make: str='Kawasaki', year: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Motorcycles API endpoint. Returns up to 30 motorcycle results matching the input name parameters. For searches that yield > 30 results, please use the offset parameter.
		
		Either **make** or **model** parameter must be set."
    model: name of motorcycle model. Supports partial matching (e.g. Ninja will match Ninja 650).
        offset: number of results to offset for pagination. Default is 0.
        make: name of manufacturer/brand. Supports partial matching (e.g. Harley will match Harley-Davidson).
        year: release year of motorcycle model. Must be in the form of YYYY (e.g. 2022).
        
    """
    url = f"https://motorcycles-by-api-ninjas.p.rapidapi.com/v1/motorcycles"
    querystring = {}
    if model:
        querystring['model'] = model
    if offset:
        querystring['offset'] = offset
    if make:
        querystring['make'] = make
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycles-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

