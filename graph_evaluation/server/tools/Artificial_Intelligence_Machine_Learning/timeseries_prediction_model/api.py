import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prediction(periods: int, values: str, dates: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "predict next period with dates and values"
    periods: how many days do you want to predict
        
    """
    url = f"https://timeseries-prediction-model.p.rapidapi.com/prediction"
    querystring = {'periods': periods, 'values': values, 'dates': dates, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timeseries-prediction-model.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

