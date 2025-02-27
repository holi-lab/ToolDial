import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def movierulz(manuch: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Movierulz](url) is a piracy website infamous for allowing users to download pirated movies.This notorious online portal is responsible for streaming the latest English, Bollywood, Punjabi, Malayalam, Tamil and Telugu movies before their release or as soon as they are exhibited in theatres.
		[https://www.gamblingagora.com/2021/05/movierulz-download-hd-movies-from-movie.html](url)"
    
    """
    url = f"https://movierulz.p.rapidapi.com/"
    querystring = {}
    if manuch:
        querystring['manuch'] = manuch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movierulz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

