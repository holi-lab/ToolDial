import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def predict_age(name: str, apikey: str, country_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agify.io predicts the age of a person given their name."
    
    """
    url = f"https://agify-io.p.rapidapi.com/"
    querystring = {'name': name, 'apikey': apikey, }
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "agify-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

