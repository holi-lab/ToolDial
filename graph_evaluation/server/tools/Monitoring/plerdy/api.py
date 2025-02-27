import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def heatmap(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With Plerdy Heatmaps we see exactly where the clicks were on the website.  All clicks on the website are recorded and displayed in real time.  Which helps to understand how users behave on the website."
    
    """
    url = f"https://plerdy.p.rapidapi.comhttps://www.plerdy.com/heatmap/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plerdy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

