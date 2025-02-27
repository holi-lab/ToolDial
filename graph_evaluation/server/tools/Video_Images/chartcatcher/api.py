import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def plot(chart: str='barplot', x: str='1,2,3,4,5,6,7,8,9,10', xlabel: str='x-axis', style: str='dark', y: str='1,4,9,16,25,36,49,64,81,100', ylabel: str='y-axis', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "plot endpoint"
    
    """
    url = f"https://chartcatcher.p.rapidapi.com/plot"
    querystring = {}
    if chart:
        querystring['chart'] = chart
    if x:
        querystring['x'] = x
    if xlabel:
        querystring['xlabel'] = xlabel
    if style:
        querystring['style'] = style
    if y:
        querystring['y'] = y
    if ylabel:
        querystring['ylabel'] = ylabel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chartcatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

