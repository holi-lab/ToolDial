import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def city_name(city_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gibt Informationen zu der angegebenen Stadt zurück. Bei Mehrdeutigkeit werden maximal drei Ergebnisse zurückgegeben. Zurückgegebene Werte sind: Einwohnerzahl, Durchschnittsalter, umliegende Orte."
    
    """
    url = f"https://german-cities.p.rapidapi.com/{city_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "german-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

