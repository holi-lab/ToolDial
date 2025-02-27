import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_status_cat_image(status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    status: 100 101 200 201 202 204 206 207 300 301 303 304 305 307 400 401 402 403 404 405 406 408 409 410 411 413 414 416 417 418 422 423 424 425 426 429 431 444 450 500 502 503 506 507 508 509 599
        
    """
    url = f"https://community-http-status-cats.p.rapidapi.com/{status}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-http-status-cats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

