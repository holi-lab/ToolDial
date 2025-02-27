import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def steuersignal(zip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Abrufen des Wärmepumpensteuersignals für einen Ort in Deutschland. Das Ergebnis enthält eine Vorhersage für die kommenden 72 Stunden."
    zip: Postleitzahl eines Ortes in Deutschland
        
    """
    url = f"https://warmepumpen-steuerungssignal.p.rapidapi.com/heatpump"
    querystring = {'zip': zip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "warmepumpen-steuerungssignal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

