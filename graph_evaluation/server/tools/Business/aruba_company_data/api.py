import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcompanybyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Aruba business register by company ID"
    
    """
    url = f"https://aruba-company-data.p.rapidapi.com/api/v1/bedrijf/public/HANDELSREGISTER/{is_id}/0"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aruba-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(includeinactief: str, searchterm: str, take: int, includeactief: str, skip: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Aruba company register by text"
    
    """
    url = f"https://aruba-company-data.p.rapidapi.com/api/v1/bedrijf/public/search"
    querystring = {'includeInactief': includeinactief, 'searchTerm': searchterm, 'take': take, 'includeActief': includeactief, 'skip': skip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aruba-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

