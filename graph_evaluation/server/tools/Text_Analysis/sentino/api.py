import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def questionnaire_sentino(content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "They are sampled from original items so the first 30 are best for Big5, first 60 are best for BFAS and 90 for NEO.  The more items scored the better."
    
    """
    url = f"https://sentino.p.rapidapi.com/questionnaire/neo.sentino.90"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentino.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

