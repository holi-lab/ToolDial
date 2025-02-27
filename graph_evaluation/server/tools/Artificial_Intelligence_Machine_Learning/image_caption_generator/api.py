import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generates_image_captions(imageurl: str, useemojis: bool=None, vibe: str=None, limit: int=3, usehashtags: bool=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates descriptive captions for a given image. When `useEmojis`/`useHashtags` is set to `true`, the generated captions will include emojis/hashtags. A maximum of three captions are returned."
    imageurl: Url of the image
        useemojis: If true, emojis will be added to generate captions.
        vibe: Choose the vibe of the generated captions.
        limit: Number of captions returned. Must be <=3.
        usehashtags: If true, hashtags will be added to generate captions.
        
    """
    url = f"https://image-caption-generator2.p.rapidapi.com/v2/captions"
    querystring = {'imageUrl': imageurl, }
    if useemojis:
        querystring['useEmojis'] = useemojis
    if vibe:
        querystring['vibe'] = vibe
    if limit:
        querystring['limit'] = limit
    if usehashtags:
        querystring['useHashtags'] = usehashtags
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-caption-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

