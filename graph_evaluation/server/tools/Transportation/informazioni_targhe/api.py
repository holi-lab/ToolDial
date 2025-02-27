import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def controlla_stato_elaborazione_job(job: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Controlla lo stato di elaborazione di un job creato"
    
    """
    url = f"https://informazioni-targhe.p.rapidapi.com/job/status"
    querystring = {'job': job, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "informazioni-targhe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recupera_risultato_elaborazione_job(job: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ottieni i dati risultanti dall'elaborazione delle targhe inviate"
    
    """
    url = f"https://informazioni-targhe.p.rapidapi.com/job/retrieve"
    querystring = {'job': job, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "informazioni-targhe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

