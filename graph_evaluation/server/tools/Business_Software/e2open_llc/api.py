import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def demand_sensing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates accurate near-term forecasts that reflect current market realities using real-time data, automation and machine learning algorithms."
    
    """
    url = f"https://e2open-llc.p.rapidapi.comhttps://www.e2open.com/intelligent-applications/demand-sensing/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e2open-llc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

