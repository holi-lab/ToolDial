import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mp3(is_id: str, cut: int=None, sstart: str=None, send: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert to MP3 at the default bit rate.
		Quota cost will apply according to the length of the mp3 file."
    id: Quota cost will apply according to the length of the mp3 file.
Quota is calculated as follows:
if length <= 30 then 1
if 30 < length <= 60 then 2
if 60 < length <= 90 then 3
if 90 < length <= 120 then 4
if 120 < length <= 150 then 5
if 150 < length <= 180 then 6
        cut: Cut the mp3 according to the provided sStart and sEnd parameters
The value must be provided as 1
The quota Cost is +1.
        sstart: Starting point for cutting the mp3.
Format must be HH:MM:SS

        send: End point for cutting the mp3.
Format must be HH:MM:SS

        
    """
    url = f"https://youtube-mp36.p.rapidapi.com/dl"
    querystring = {'id': is_id, }
    if cut:
        querystring['cut'] = cut
    if sstart:
        querystring['sStart'] = sstart
    if send:
        querystring['sEnd'] = send
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp36.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

