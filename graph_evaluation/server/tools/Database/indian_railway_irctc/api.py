import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_train_live_status(is_id: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Train Live Status By ID you got while searching trains in get Train By ID API.
		
		before calling this API to get the live train status you need the id of train, which you will get in gettrainId api"
    
    """
    url = f"https://indian-railway-irctc.p.rapidapi.com/getTrainLiveStatusById"
    querystring = {'id': is_id, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-railway-irctc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_train_info_by_train_number(trainno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get train info by train number.
		This is important API to get the live train status you need the id of train, which you will get here"
    trainno: trainno can be train number or train name also
        
    """
    url = f"https://indian-railway-irctc.p.rapidapi.com/getTrainId"
    querystring = {'trainno': trainno, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-railway-irctc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

