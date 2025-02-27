import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getsingleverse(verse: int, chapter: int, book: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single verse based on the book, chapter and verse number."
    
    """
    url = f"https://niv-bible.p.rapidapi.com/row"
    querystring = {'Verse': verse, 'Chapter': chapter, 'Book': book, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "niv-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

