import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cotizaciones_basico(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve las ultimas cotizaciones del dólar oficial, euro oficial, dólar blue, real oficial y contado con liquidación."
    
    """
    url = f"https://cotizaciones-argentinas-por-downtack-com.p.rapidapi.com/cotizaciones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cotizaciones-argentinas-por-downtack-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

