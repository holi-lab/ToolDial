import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ie_validate(state: str, ie: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate the ie number"
    state: Inform the state acronym ex: SP, PE. BA
        ie: Inform the ie number with or withou mask
        
    """
    url = f"https://ie-validator.p.rapidapi.com/ie"
    querystring = {'state': state, 'ie': ie, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ie-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

