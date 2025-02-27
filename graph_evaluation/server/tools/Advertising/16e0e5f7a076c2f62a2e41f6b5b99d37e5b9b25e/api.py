import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def venture(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Link to any suggestions on a nail"
    
    """
    url = f"https://16e0e5f7a076c2f62a2e41f6b5b99d37e5b9b25e.p.rapidapi.com/employees/id"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "16e0e5f7a076c2f62a2e41f6b5b99d37e5b9b25e.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

