import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def seconds_simplified(data_type: str, figure: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert seconds to get days, hours, minutes, seconds. (Seconds) -> Days: 1, Hours: 3, Minutes: 46, Seconds: 40"
    
    """
    url = f"https://epoch-converter.p.rapidapi.com/epoch-converter"
    querystring = {'data_type': data_type, 'figure': figure, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epoch-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def start_end_date_to_time(figure: str, data_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert datetime to get start timestamp and end timestamp. (year, month, day Hours:Minutes:Seconds) -> 1674864000.0, 1674950399.0"
    
    """
    url = f"https://epoch-converter.p.rapidapi.com/epoch-converter"
    querystring = {'figure': figure, 'data_type': data_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epoch-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def date_to_time(figure: str, data_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert datetime to timestamp. (year, month, day Hours:Minutes:Seconds) -> 1674919760"
    
    """
    url = f"https://epoch-converter.p.rapidapi.com/epoch-converter"
    querystring = {'figure': figure, 'data_type': data_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epoch-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_to_date(data_type: str, figure: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert timestamp to datetime. (year, month, day Hours:Minutes:Seconds)"
    
    """
    url = f"https://epoch-converter.p.rapidapi.com/epoch-converter"
    querystring = {'data_type': data_type, 'figure': figure, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "epoch-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

