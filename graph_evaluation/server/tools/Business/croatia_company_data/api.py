import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_mbs_or_oib(tipidentifikatora: str, expand_relations: str, identifikator: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Croatian company register by either MBS or OIB company identifiers (osobni identifikacijski broj)"
    
    """
    url = f"https://croatia-company-data.p.rapidapi.com/default/HR/subjekt_detalji"
    querystring = {'tipIdentifikatora': tipidentifikatora, 'expand_relations': expand_relations, 'identifikator': identifikator, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "croatia-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

