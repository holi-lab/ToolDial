import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def asia_lab_end(asialab: str=None, covid_19: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Asia Lab Endpoints"
    
    """
    url = f"https://asia-lab1.p.rapidapi.com/"
    querystring = {}
    if asialab:
        querystring['AsiaLab'] = asialab
    if covid_19:
        querystring['covid-19'] = covid_19
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asia-lab1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

