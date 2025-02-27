import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company_profile(kvk: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a Dutch company from it's KVK number."
    
    """
    url = f"https://netherlands-company-data.p.rapidapi.com/default/NL/basisprofielen/{kvk}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netherlands-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_kvk_number(pagina: int, aantal: int, kvknummer: int, handelsnaam: str=None, vestigingsnummer: int=None, rsin: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Dutch Company register by name (handelsnaam) or by KVK, RSIN or vestigingsnummer identifiers."
    
    """
    url = f"https://netherlands-company-data.p.rapidapi.com/default/NL/zoeken"
    querystring = {'pagina': pagina, 'aantal': aantal, 'kvkNummer': kvknummer, }
    if handelsnaam:
        querystring['handelsnaam'] = handelsnaam
    if vestigingsnummer:
        querystring['vestigingsnummer'] = vestigingsnummer
    if rsin:
        querystring['rsin'] = rsin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netherlands-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

