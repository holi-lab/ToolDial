import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ocks_perf6m(x_rapidapi_key: str='a8e4f24d21msh497089d72e59bf3p1377e8jsn2be6a846ed17', x_rapidapi_host: str='tradingradar.p.rapidapi.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get FSTE MIB stocks PERFORMANCE (6M)"
    
    """
    url = f"https://tradingradar.p.rapidapi.com/ocks/perf6M"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tradingradar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

