import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbookcoverurl(languagecode: str, isbn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a book cover image url by isbn and language code"
    
    """
    url = f"https://book-cover-api2.p.rapidapi.com/api/public/books/v1/cover/url"
    querystring = {'languageCode': languagecode, 'isbn': isbn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-cover-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookcoverimage(isbn: str, languagecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a book cover image by isbn and language code"
    
    """
    url = f"https://book-cover-api2.p.rapidapi.com/api/public/books/v1/cover/image"
    querystring = {'isbn': isbn, 'languageCode': languagecode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-cover-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

