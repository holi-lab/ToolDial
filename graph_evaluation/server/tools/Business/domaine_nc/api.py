import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_domaine(ext: str, nom: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Récupère les détails d'un domaine, cela permet de rechercher en mode API le contenu disponible en web sur [DOMAINE.nc](https://www.domaine.nc/whos)."
    ext: Chaque extension obéit à ses propres règles d'enregistrement, avec des variantes en termes d'identification des titulaires, de présence locale ou de justificatifs à fournir.

Ce sont les extensions disponibles en Nouvelle-Calédonie, qui sont :

- `nc` : extension générique
- `asso.nc` : réservé aux associations
- `nom.nc` : réservé aux particuliers qui désirerai deposer leur nom de famille 

        nom: Nom de domaine, voir [quelques exemples](https://www.domaine.nc/whos?who=AA) sur le site web.
        
    """
    url = f"https://domaine-nc.p.rapidapi.com/domaines/{nom}/{ext}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domaine-nc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_api_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Status de l'API"
    
    """
    url = f"https://domaine-nc.p.rapidapi.com/actuator/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domaine-nc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_domaines(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all domain names"
    
    """
    url = f"https://domaine-nc.p.rapidapi.com/domaines"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domaine-nc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

