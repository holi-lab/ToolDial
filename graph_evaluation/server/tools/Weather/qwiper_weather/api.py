import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def m_t_o_d_aujourd_hui_niamey_niger(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtenez la température, les conditions du ciel, les vents, l'humidité, etc. pour la ville de Niamey."
    
    """
    url = f"https://qwiper-weather.p.rapidapi.com/?json=true"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qwiper-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

