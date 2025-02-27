import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(search: str, ceptype: str=None, exact: str=None, similar: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    
    """
    url = f"https://busca-endereco.p.rapidapi.com/"
    querystring = {'search': search, }
    if ceptype:
        querystring['cepType'] = ceptype
    if exact:
        querystring['exact'] = exact
    if similar:
        querystring['similar'] = similar
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "busca-endereco.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

