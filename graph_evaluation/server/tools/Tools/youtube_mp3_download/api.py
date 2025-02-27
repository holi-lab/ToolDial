import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mp3(is_id: str, is_return: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert to MP3 at the default bit rate."
    id: Youtube Video Id
        is_return: Enables back button in the download page to enable the user to come back to the source site.
Value must be `1`
        
    """
    url = f"https://youtube-mp3-download1.p.rapidapi.com/dl"
    querystring = {'id': is_id, }
    if is_return:
        querystring['return'] = is_return
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp3-download1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

