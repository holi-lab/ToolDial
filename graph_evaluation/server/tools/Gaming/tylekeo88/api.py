import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ty_le_bong_da(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "7mcn chuyen cung cap ket qua bong da truc tuyen duoc cap nhat lien tuc nhanh chong va chinh xac Ben canh do ban se duoc cap nhat tylebongda cung ty le ma cao moi ngay Truy cap ngay de co the soi tylekeo88 bong da moi ngay cac ban nhe
		Xem them
		https://flipboard.com/@7mcn/
		https://trello.com/7mcn/cards
		https://www.behance.net/tylebongda68
		https://www.flickr.com/people/193865928@N07/
		https://twitter.com/tylebongda88
		https://www.youtube.com/channel/UC1EMUdTOHWsP1n5VJpL7fjg/about"
    
    """
    url = f"https://tylekeo88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tylekeo88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

