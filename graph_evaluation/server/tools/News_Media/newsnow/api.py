import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(q: str='everything', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NewsNow API: Real-time news aggregation and delivery service, providing developers and businesses with global news articles and headlines, offering extensive coverage, flexible querying, and easy integration."
    
    """
    url = f"https://newsnow.p.rapidapi.com/"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

