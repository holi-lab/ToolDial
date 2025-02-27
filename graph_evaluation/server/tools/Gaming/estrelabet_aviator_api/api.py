import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def estrelabet_aviator_latest(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will retrieve the latest results from Aviator at Estrelabet.
		
		- The first result is the most recent
		- The last result is the oldest"
    
    """
    url = f"https://estrelabet-aviator-api.p.rapidapi.com/estrelabet-aviator-latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "estrelabet-aviator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

