import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_webdata(url: str, generic: bool=None, imgarr: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service gets a json object with data from the url you provided."
    url: Url you want to crawl. No "hashtags" (#) on the url. Only urls that return html.
        generic: Default is FALSE. "true" will try to get a single generic image from the url if not explicitly provided at the html, ie. first image on website. Using imgarr parameter ignore this feature.
        imgarr: Default is FALSE. "false" for getting a single image from the url, "true" for getting all images in html.
        
    """
    url = f"https://webdata-crawler.p.rapidapi.com/webdata?url={url}&imgarr={imgarr}&generic={generic}"
    querystring = {}
    if generic:
        querystring['generic'] = generic
    if imgarr:
        querystring['imgarr'] = imgarr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webdata-crawler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_images(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service gets an array of images fetched from all the img tags retrieved by the provided url"
    url: Url you want to crawl. No "hashtags" (#) on the url. Only urls that return html.
        
    """
    url = f"https://webdata-crawler.p.rapidapi.com/images?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webdata-crawler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

