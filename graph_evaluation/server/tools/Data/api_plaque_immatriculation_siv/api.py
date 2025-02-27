import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vehicule_info2(immatriculation: str='AA-123-BC', token: str='TokenDemoRapidapi', host_name: str='https://apiplaqueimmatriculation.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* Récupérer les informations des véhicules simplement avec notre API PLAQUE IMMATRICULATION rapide flexible .
		
		Les données sont demandées à notre api plaque immatriculation en temps réel à partir de sources de données officielles du gouvernement, de sorte que les informations que vous voyez soient en temps réel et non mises en cache, ou stockées."
    
    """
    url = f"https://api-plaque-immatriculation-siv.p.rapidapi.com/get-vehicule-info"
    querystring = {}
    if immatriculation:
        querystring['immatriculation'] = immatriculation
    if token:
        querystring['token'] = token
    if host_name:
        querystring['host_name'] = host_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-plaque-immatriculation-siv.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vehicule_info(token: str='TokenDemoRapidapi', host_name: str='https://apiplaqueimmatriculation.com', immatriculation: str='AA-123-BC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* Récupérer les informations des véhicules simplement avec notre API PLAQUE IMMATRICULATION rapide flexible .
		
		Les données sont demandées à notre api plaque immatriculation en temps réel à partir de sources de données officielles du gouvernement, de sorte que les informations que vous voyez soient en temps réel et non mises en cache, ou stockées."
    
    """
    url = f"https://api-plaque-immatriculation-siv.p.rapidapi.com/get-vehicule-info"
    querystring = {}
    if token:
        querystring['token'] = token
    if host_name:
        querystring['host_name'] = host_name
    if immatriculation:
        querystring['immatriculation'] = immatriculation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-plaque-immatriculation-siv.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

