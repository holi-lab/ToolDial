import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gender_and_ethnicity(fullname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns gender and ethnicity of a full name"
    fullname: Full name to include first name, middle name (optional) and last name.
        
    """
    url = f"https://diversity-data-io.p.rapidapi.com/"
    querystring = {'fullname': fullname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diversity-data-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

