import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gstin(gstin: str, content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Details about GSTIN (GST number) and following 
		1. Latest Return filing list
		2. GST Compliance Classification
		3. HSN/SAC"
    
    """
    url = f"https://gst-return-status.p.rapidapi.com/free/gstin/{gstin}"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gst-return-status.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

