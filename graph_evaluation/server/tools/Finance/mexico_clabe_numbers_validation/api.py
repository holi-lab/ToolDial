import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mexico_clabe_validation(clabe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Validates Mexico CLABE Numbers by the method prescribed for verifying the Mexico CLABE Numbers. This API helps in ensuring accuracy of Mexico CLABE Numbers."
    
    """
    url = f"https://mexico-clabe-numbers-validation.p.rapidapi.com/"
    querystring = {'clabe': clabe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mexico-clabe-numbers-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

