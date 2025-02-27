import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def country_specific_transfers(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest confirmed transfers for a specific country: 
		Current available countries / markets / leagues: 
		- "**en**" (England) : Premier League
		- "**es**" (Spain) : La Liga
		- "**de**" (Germany) : Bundesliga
		- "**it**" (Italy) : Serie A
		- "**fr**" (France) : Ligue 1
		- "**pt**" (Portugal) : Liga
		- "**nl**" (Netherlands) : Eredivisie
		- "**br**" (Brazil) : Brasileirão
		- "**mx**" (Mexico) : Torneo Clausura
		- "**ru**" (Russia) : Premier Liga
		- "**ch**" (Switzerland) : Super League
		- "**dk**" (Denmark) : SAS-Ligaen
		- "**tr**" (Turkey) : Super Lig
		- "**ar**" (Argentina) : Primera Division
		- "**gk**" (Greece) : Super League"
    
    """
    url = f"https://soccer-transfers.p.rapidapi.com/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-transfers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_transfers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all latest confirmed transfers."
    
    """
    url = f"https://soccer-transfers.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-transfers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

