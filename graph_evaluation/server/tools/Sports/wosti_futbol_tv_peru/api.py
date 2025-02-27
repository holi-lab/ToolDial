import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getteams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener el listado de equipos de fútbol televisados en Perú."
    
    """
    url = f"https://wosti-futbol-tv-peru.p.rapidapi.com/api/Teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futbol-tv-peru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompetitions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener el listado de competiciones de partidos de fútbol televisados en Perú.
		
		> Este endpoints no requiere de ningún parámetro."
    
    """
    url = f"https://wosti-futbol-tv-peru.p.rapidapi.com/api/Competitions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futbol-tv-peru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener el listado de eventos de partidos de fútbol televisados en Perú.
		
		> Este endpoints no requiere de ningún parámetro."
    
    """
    url = f"https://wosti-futbol-tv-peru.p.rapidapi.com/api/Events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futbol-tv-peru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventsid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener el evento por identificador único  del listado de eventos de partidos de fútbol televisados en Perú.
		
		> Este endpoints requiere de un parámetro denominado Id."
    
    """
    url = f"https://wosti-futbol-tv-peru.p.rapidapi.com/api/Events"
    querystring = {'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futbol-tv-peru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompetitionsid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener competición por identificador único del listado de competiciones de partidos de fútbol televisados en Perú.
		
		> Este endpoints requiere de un parámetro denominado Id."
    
    """
    url = f"https://wosti-futbol-tv-peru.p.rapidapi.com/api/Competitions"
    querystring = {'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futbol-tv-peru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

