import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reset(is_class: str='CLASS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Might happen time to time you want to recreate you model and clean it for a new training. In order to do it "/reset" API let you remove the default model or a class of it "/reset/{class}". 
		This is especially  useful when you init the model with a wrong data set."
    
    """
    url = f"https://anomaly-score.p.rapidapi.com/reset/{is_class}"
    querystring = {}
    if is_class:
        querystring['class'] = is_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anomaly-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

