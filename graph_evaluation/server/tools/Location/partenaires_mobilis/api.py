import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def partenaire_par_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Récupérer un partenaire par son id"
    id: id du partenaire fourni par API.
        
    """
    url = f"https://partenaires-mobilis.p.rapidapi.com/api/partenaires/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "partenaires-mobilis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_partenaires(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Récupérer tous les partenaires."
    
    """
    url = f"https://partenaires-mobilis.p.rapidapi.com/api/partenaires"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "partenaires-mobilis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def g_olocaliser_les_partenaires(nearbydistance: str='2000', nearbylon: str='166.448744', nearbylat: str='-22.302828', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Récupérer tous les partenaires, ou récupérer tous les partenaires dans un rayon donné autour d'un point donné :
		
		- **nearByLon**= {longitude du centre du cercle de la zone de recherche}
		- **nearByLat**= {latitude du centre du cercle de la zone de recherche}
		- **nearByDistance**= {rayon du cercle de la zone de recherche}"
    
    """
    url = f"https://partenaires-mobilis.p.rapidapi.com/api/partenaires"
    querystring = {}
    if nearbydistance:
        querystring['nearByDistance'] = nearbydistance
    if nearbylon:
        querystring['nearByLon'] = nearbylon
    if nearbylat:
        querystring['nearByLat'] = nearbylat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "partenaires-mobilis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Donne l'état de santé de l'API"
    
    """
    url = f"https://partenaires-mobilis.p.rapidapi.com/actuator/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "partenaires-mobilis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

