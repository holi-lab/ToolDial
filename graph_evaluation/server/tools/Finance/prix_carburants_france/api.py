import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def avoir_les_prix_des_carburants_par_ville(ville: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Avoir tous les prix des carburants concernant une seule ville"
    
    """
    url = f"https://prix-carburants-france.p.rapidapi.com/instant/{ville}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prix-carburants-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def avoir_tous_les_prix_dans_toute_la_france(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ce endpoint va retourner toutes les infos concernant les prix de carburants dans toute la France"
    
    """
    url = f"https://prix-carburants-france.p.rapidapi.com/instant"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prix-carburants-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

