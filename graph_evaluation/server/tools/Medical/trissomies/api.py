import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def default_endpoint(examinationdate: str, fetalcrl: str, fetalheartrate: str, dateofbirth: str, nuchaltranslucency: str, previoust18: bool, previoust13: bool, previoust21: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Execute the API."
    
    """
    url = f"https://trissomies.p.rapidapi.com/{examinationdate}/{dateofbirth}/{previoust21}/{previoust18}/{previoust13}/{fetalcrl}/{nuchaltranslucency}/{fetalheartrate}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trissomies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

