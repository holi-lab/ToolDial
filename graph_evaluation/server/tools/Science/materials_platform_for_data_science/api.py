import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mpds(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve materials data, see www.mpds.io"
    q: JSON-serialized URL-encoded Object with the retrieval criteria, e.g. {"elements":"Ag-K"}
        
    """
    url = f"https://materials-platform-for-data-science.p.rapidapi.com/facet"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "materials-platform-for-data-science.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

