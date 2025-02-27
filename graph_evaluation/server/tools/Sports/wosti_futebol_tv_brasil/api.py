import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getteams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Veja a lista de times de futebol televisionados do Brasil."
    
    """
    url = f"https://wosti-futebol-tv-brasil.p.rapidapi.com/api/Teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futebol-tv-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompetitionid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtenha a competição por identificador único na lista de competições de partidas de futebol televisionadas no Brasil.
		
		> Este endpoint requer um parâmetro chamado Id."
    
    """
    url = f"https://wosti-futebol-tv-brasil.p.rapidapi.com/api/Competitions"
    querystring = {'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futebol-tv-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventsid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtenha o evento por identificador único na lista de eventos de partidas de futebol televisionadas no Brasil.
		
		> Este endpoint requer um parâmetro chamado Id."
    
    """
    url = f"https://wosti-futebol-tv-brasil.p.rapidapi.com/api/Events"
    querystring = {'Id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futebol-tv-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompetitions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Veja a lista de jogos de futebol televisionados do Brasil."
    
    """
    url = f"https://wosti-futebol-tv-brasil.p.rapidapi.com/api/Competitions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futebol-tv-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Veja a lista de eventos de partidas de futebol televisionadas no Brasil."
    
    """
    url = f"https://wosti-futebol-tv-brasil.p.rapidapi.com/api/Events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wosti-futebol-tv-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

