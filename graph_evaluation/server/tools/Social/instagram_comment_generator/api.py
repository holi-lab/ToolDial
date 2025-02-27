import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_comment_from_shortcode(shortcode: str, style: str='professional', number: int=2, type: str='short', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a shortcode to an Instagram post and receive a comment back.
		E.g. The short code of this link (https://www.instagram.com/p/Cmjutz9sUBo/) would be "Cmjutz9sUBo"."
    
    """
    url = f"https://instagram-comment-generator.p.rapidapi.com/getcomment"
    querystring = {'shortcode': shortcode, }
    if style:
        querystring['style'] = style
    if number:
        querystring['number'] = number
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-comment-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

