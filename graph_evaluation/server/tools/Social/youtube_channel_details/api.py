import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_channel_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube Channel Details"
    id: **Channel ID** starting with the `UC` prefix, or **Channel URL**

e.g. `https://www.youtube.com/user/PewDiePie`
e.g. `UC-lHJZR3Gqxm24_Vd_AJ5Yw`
        
    """
    url = f"https://youtube-channel-details.p.rapidapi.com/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-channel-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

