import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbooks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List Of All Books."
    
    """
    url = f"https://ajith-holy-bible.p.rapidapi.com/GetBooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-holy-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchapter(book: str, chapter: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    book: Enter one of the books from Bible.
        chapter: Enter chapter number from given book.
        
    """
    url = f"https://ajith-holy-bible.p.rapidapi.com/GetChapter"
    querystring = {'Book': book, 'chapter': chapter, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-holy-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getverses(book: str, chapter: int, versefrom: int, verseto: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    book: Enter one of the books from Bible.
        chapter: Enter chapter number from given book.
        versefrom: Enter Verse to start.
        verseto: Enter Verse to End.
        
    """
    url = f"https://ajith-holy-bible.p.rapidapi.com/GetVerses"
    querystring = {'Book': book, 'chapter': chapter, 'VerseFrom': versefrom, 'VerseTo': verseto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-holy-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getverseofachapter(book: str, chapter: int, verse: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://ajith-holy-bible.p.rapidapi.com/GetVerseOfaChapter"
    querystring = {'Book': book, 'chapter': chapter, 'Verse': verse, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-holy-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

