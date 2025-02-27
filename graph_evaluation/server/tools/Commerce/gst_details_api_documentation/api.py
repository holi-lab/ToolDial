import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gst_number_search_tool_gstin_verification_online(gst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GST Number Search Tool & GSTIN Verification Online"
    
    """
    url = f"https://gst-details-api-documentation.p.rapidapi.com/GetGSTDetails"
    querystring = {'GST': gst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gst-details-api-documentation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

