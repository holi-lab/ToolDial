import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def adverts(outdated: bool=None, limit: int=50, type: int=0, zipcodes: str='[77500,93200,..]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Récupérez les informations sur les annonces immobilières françaises présentes et passées issues d'une 20aine de plateformes
		______________________
		Get detailed info on French Real Estate Adverts"
    
    """
    url = f"https://annonces-immobilieres-france.p.rapidapi.com/Adverts"
    querystring = {}
    if outdated:
        querystring['outdated'] = outdated
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if zipcodes:
        querystring['zipcodes'] = zipcodes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "annonces-immobilieres-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

