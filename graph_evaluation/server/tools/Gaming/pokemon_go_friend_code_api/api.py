import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_friend_codes(sort: str='-1', skip: int=5, qr_code: str='true', limit: int=25, random: str='true', filter: str='mystic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of friend codes. Can limit numbers, skip, random friend codes, and filter by team."
    sort: Order of date added. 1 is ascending order and -1 is descending order. 
        skip: Where the response will start. E.g., 5 starts at the 5th result. Used for pagination.
        qr_code: QR Code in a base64 string. Set to true to enable.
        limit: Limit of friend code objects returned
        random: Returns objects in random order. Set true to enable.
        filter: Filter by the team name. E.g.,  Valor, Instinct, Mystic, or, Neutral
        
    """
    url = f"https://pokemon-go-friend-code-api.p.rapidapi.com/friendcode"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if skip:
        querystring['skip'] = skip
    if qr_code:
        querystring['qr_code'] = qr_code
    if limit:
        querystring['limit'] = limit
    if random:
        querystring['random'] = random
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-go-friend-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

