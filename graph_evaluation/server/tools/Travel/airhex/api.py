import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airlines_check_in_url(iata: str, md5apikey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    iata: Airline's IATA code
        md5apikey: Please register at http://airhex.com to use your own API key
        
    """
    url = f"https://airhex-airhex-v1.p.rapidapi.com/content/check-in/"
    querystring = {'iata': iata, }
    if md5apikey:
        querystring['md5apikey'] = md5apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airhex-airhex-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_marketing_names(iata: str, md5apikey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Purified IATA database"
    iata: Airline's IATA code
        md5apikey: Please register at http://airhex.com to use your own API key
        
    """
    url = f"https://airhex-airhex-v1.p.rapidapi.com/content/marketing-names/"
    querystring = {'iata': iata, }
    if md5apikey:
        querystring['md5apikey'] = md5apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airhex-airhex-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

