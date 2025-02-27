import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_tripdetails_mfref(mfref: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrives the TripDetails based on the  MFRef Number
		- MFRef"
    
    """
    url = f"https://asr-hub.p.rapidapi.com/api/TripDetails/{mfref}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asr-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_schedulechangeaccept_mfref_flightid(flightid: int, mfref: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accept Schedule Change based on the UserInputs
		 - MFRef
		 - Flight Id"
    
    """
    url = f"https://asr-hub.p.rapidapi.com/api/ScheduleChangeAccept/{mfref}/{flightid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asr-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

