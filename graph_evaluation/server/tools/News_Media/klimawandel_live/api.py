import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hole_news_beitr_ge_von_einem_spezifischen_medium(mediumid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dieser Endpunkt bringt alle News-Beiträge über Klimawandel von einem spezifischen Medium."
    
    """
    url = f"https://klimawandel-live1.p.rapidapi.com/news/{mediumid}"
    querystring = {}
    if mediumid:
        querystring['mediumId'] = mediumid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "klimawandel-live1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hole_alle_news_beitr_ge_zu_klimawandel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dieser Endpunkt bringt alle News-Beiträge über Klimawandel von diversen Schweizer Medien."
    
    """
    url = f"https://klimawandel-live1.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "klimawandel-live1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

