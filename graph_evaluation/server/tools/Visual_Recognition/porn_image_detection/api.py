import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_detection(image_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Request Parameters:**
		
		- image_url==>	**[Required]**	Image URL
		
		**Response Objects Description:**
		The probability of picture detection results is divided into the following 5 categories:
		
		- Porn==>	Adult erotic pictures
		- Sexy==>	Sexy pictures
		- Neutral==>	Normal picture
		- Drawing==>	Normal painting
		- Hentai==>	Porn painting"
    
    """
    url = f"https://porn-image-detection.p.rapidapi.com/nsfw/{image_url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-image-detection.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

