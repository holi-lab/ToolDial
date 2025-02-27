import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historic_views_and_subscribers(channelid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns historic views and subscribers for a channel, along with basic details of the channel (name, handle, thumbnails, etc.).
		
		**Note:**
		 `viewsHist`/ `subsHist` data format: 
		`[[timestampInMilliSecondsA, valueA], [timestampInMilliSecondsB, valueB], ...]`."
    
    """
    url = f"https://historical-youtube-data.p.rapidapi.com/hytapi/channel"
    querystring = {'channelId': channelid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "historical-youtube-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

