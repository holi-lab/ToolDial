import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcarinfos(plaque: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all car information using Spain car plate, ex :"1995CCX""
    
    """
    url = f"https://api-license-plate-spain-matricula-api-espana.p.rapidapi.com/"
    querystring = {'plaque': plaque, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-license-plate-spain-matricula-api-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

