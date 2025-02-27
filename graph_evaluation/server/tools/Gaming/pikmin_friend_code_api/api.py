import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_friend_codes(random: str='true', skip: str='5', limit: str='25', sort: str='-1', qr_code: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of friend codes. Can limit numbers, skip and randomize friend codes"
    random: Returns objects in random order. Set true to enable.
        skip: Where the response will start. E.g., 5 starts at the 5th result. Used for pagination.
        limit: Limit of friend code objects returned
        sort: Order of date added. 1 is ascending order and -1 is descending order. 
        qr_code: QR Code in a base64 string. Set to true to enable.
        
    """
    url = f"https://pikmin-friend-code-api.p.rapidapi.com/friendcode"
    querystring = {}
    if random:
        querystring['random'] = random
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if qr_code:
        querystring['qr_code'] = qr_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pikmin-friend-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

