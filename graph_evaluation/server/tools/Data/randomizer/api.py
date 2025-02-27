import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getrandom(length: int, case: str='all', extra: str='true', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a random sequence of characters matching the provided query attributes"
    length: Sequence length
        case: Force sequence to match case argument
        extra: Allow the use of special characters 

 **true** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; @#é$%&ùµ£-_!èçàù?^) 

 **extended** &nbsp;&nbsp;&nbsp;&nbsp; |"(§{})°¨[*]´`,;./:+=~
        type: Sequence type 

 **all** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; string + number + special
        
    """
    url = f"https://randomizer.p.rapidapi.com/randomizer/{length}"
    querystring = {}
    if case:
        querystring['case'] = case
    if extra:
        querystring['extra'] = extra
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "randomizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

