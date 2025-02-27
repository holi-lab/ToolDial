import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_id(front_imgurl: str, side: str, back_imgurl: str=None, img_size: int=None, webhook: str=None, img_source: int=None, img_crop: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API endpoint SafetyNet provides the ability to scan and validate an ID document passed into it. It is capable of verifying Id’s from 200+ countries. It not only returns if the ID is authentic or not, but also mentions reason. It can also identify the type of the ID document (driver’s license, national id, passport etc.). It also returns if the ID image passed is tampered with or digitally manipulated. SafetyNet supports processing of driver’s licenses, state IDs, other govt issued IDs, custom IDs, passports, medical insurance cards etc. For complete list of IDs supported please go to following  "URL: https://app.safetynet.ai/api/api_reference/SafetyNet List of Supported Ids.pdf""
    
    """
    url = f"https://id-verification1.p.rapidapi.com/verifyID/verify"
    querystring = {'front_imgurl': front_imgurl, 'side': side, }
    if back_imgurl:
        querystring['back_imgurl'] = back_imgurl
    if img_size:
        querystring['img_size'] = img_size
    if webhook:
        querystring['webhook'] = webhook
    if img_source:
        querystring['img_source'] = img_source
    if img_crop:
        querystring['img_crop'] = img_crop
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "id-verification1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

