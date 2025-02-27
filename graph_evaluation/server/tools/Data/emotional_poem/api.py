import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpoem(type: str='max', method: str='cos', pleasure: int=0, delight: int=0, sorrow: int=0, anger: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/app/api/getPoem
		method:only: ["cos", "dot", "cross", "euler"]
		type:only: ["max", "min"]
		delight:0.0-1.0
		anger:0.0-1.0
		sorrow:0.0-1.0
		pleasure:0.0-1.0"
    type: type:only: [\"max\", \"min\"]

        method: only: [\"cos\", \"dot\", \"cross\", \"euler\"]
default:\"cos\"
        pleasure: pleasure:0.0-1.0
default:0.0
        delight: delight:0.0-1.0
default:0.0
        sorrow: sorrow:0.0-1.0
default:0.0
        anger: anger:0.0-1.0
default:0.0
        
    """
    url = f"https://emotional-poem.p.rapidapi.com/app/api/getPoem"
    querystring = {}
    if type:
        querystring['type'] = type
    if method:
        querystring['method'] = method
    if pleasure:
        querystring['pleasure'] = pleasure
    if delight:
        querystring['delight'] = delight
    if sorrow:
        querystring['sorrow'] = sorrow
    if anger:
        querystring['anger'] = anger
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emotional-poem.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

