import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cpf_validation(cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint make a validation on cpf number, return a success message and true or false"
    cpf: Inform a cpf number with or withou mask
        
    """
    url = f"https://cpf_validator.p.rapidapi.com/cpf"
    querystring = {'cpf': cpf, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cpf_validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

