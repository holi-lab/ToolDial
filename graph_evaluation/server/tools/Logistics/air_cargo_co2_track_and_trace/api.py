import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pull_track(awb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By providing a valid AWB, you can get tracking information for the shipment.
		
		Answers are 10s in average. However, it takes between 1s to 5min to get the information from the airline system but the API has to answer in 30s which generating timeouts errors.
		In such case, you have to build the logic to try again 10mins after a timeout to avoid this or to use the subscription method."
    awb: provide valid AWB number
        
    """
    url = f"https://air-cargo-co2-track-and-trace.p.rapidapi.com/track"
    querystring = {'awb': awb, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-cargo-co2-track-and-trace.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

