import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nyhetene_fra_en_spesifikk_avis(avisid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for listen av alle artiklene i forhold til klima fra en spesifikk avis."
    
    """
    url = f"https://norsk-klima-nyheter.p.rapidapi.com/nyheter/{avisid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "norsk-klima-nyheter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nyhetene_fra_alle_norsk_aviser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for listen av alle nyhetene i forhold til klima fra alle norsk aviser."
    
    """
    url = f"https://norsk-klima-nyheter.p.rapidapi.com/nyheter"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "norsk-klima-nyheter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

