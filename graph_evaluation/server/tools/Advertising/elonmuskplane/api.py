import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def elon_plane_enp(eplane: str='EP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows Elon Musk's Gulfstream G650ER plane"
    
    """
    url = f"https://elonmuskplane.p.rapidapi.com/plane/elon"
    querystring = {}
    if eplane:
        querystring['EPlane'] = eplane
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elonmuskplane.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

