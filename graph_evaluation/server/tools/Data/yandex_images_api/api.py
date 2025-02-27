import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_images_from_yandex_search_results(page: int, query: str, domain: str, imageheight: int=None, region: str=None, imageorientation: str=None, imagewidth: int=None, imageonsite: str=None, imagesize: str=None, imagecolor: str=None, onlyrecentimages: bool=None, imagetype: str=None, imagefile: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use advanced filtering to get specific images from yandex! Remember to set from which domain you want data. Default is yandex.com but  yandex.ru, .by, .kz, .uz are also available.  Set page number in order to get more results. 0 is the first page, 1 is second and so on."
    page: 0 - first page
1 - second page
...
        domain: get search results from:
- yandex.com
- yandex.ru
- yandex.by
- yandex.kz
- yandex.uz
        region: e.g. Paris, France
        imageonsite: e.g. https://google.com/
        
    """
    url = f"https://yandex-images-api.p.rapidapi.com/getimages"
    querystring = {'page': page, 'query': query, 'domain': domain, }
    if imageheight:
        querystring['imageHeight'] = imageheight
    if region:
        querystring['region'] = region
    if imageorientation:
        querystring['imageOrientation'] = imageorientation
    if imagewidth:
        querystring['imageWidth'] = imagewidth
    if imageonsite:
        querystring['imageOnSite'] = imageonsite
    if imagesize:
        querystring['imageSize'] = imagesize
    if imagecolor:
        querystring['imageColor'] = imagecolor
    if onlyrecentimages:
        querystring['onlyRecentImages'] = onlyrecentimages
    if imagetype:
        querystring['imageType'] = imagetype
    if imagefile:
        querystring['imageFile'] = imagefile
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-images-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets server time."
    
    """
    url = f"https://yandex-images-api.p.rapidapi.com/getservertime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-images-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

