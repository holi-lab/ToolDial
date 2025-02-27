import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tracking_userid_recuperadiario(authorization: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the user diary detail, with spedition status, and update"
    
    """
    url = f"https://tracking-login.p.rapidapi.com/tracking/{userid}/recuperaDiario"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tracking-login.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

