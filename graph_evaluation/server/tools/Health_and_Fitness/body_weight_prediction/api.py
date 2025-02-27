import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def body_weigh_prediction(hip: int=100, neck: int=38, thigh: int=100, age: int=32, abdomen: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point takes a "GET" request with url/ as a parameter and returns the prediction of the weight"
    
    """
    url = f"https://body-weight-prediction.p.rapidapi.com/"
    querystring = {}
    if hip:
        querystring['Hip'] = hip
    if neck:
        querystring['Neck'] = neck
    if thigh:
        querystring['Thigh'] = thigh
    if age:
        querystring['Age'] = age
    if abdomen:
        querystring['Abdomen'] = abdomen
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "body-weight-prediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

