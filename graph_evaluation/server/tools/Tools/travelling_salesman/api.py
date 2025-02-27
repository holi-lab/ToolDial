import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exampleproblem(locationcount: int=10, edgetype: str='Euclidean2D', name: str='ExampleProblem', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://travelling-salesman.p.rapidapi.com/tsm/exampleProblem"
    querystring = {}
    if locationcount:
        querystring['locationCount'] = locationcount
    if edgetype:
        querystring['edgeType'] = edgetype
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelling-salesman.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

