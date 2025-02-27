import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def unscramble_words_solver(dictionary: str, group_by_length: str, page_size: str, word_sorting: str, letters: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Solve your unscrambled words with our unscramble API"
    letters: change this to get your unscrambled word, enter the word here
        
    """
    url = f"https://word-finder-english.p.rapidapi.com/api/search"
    querystring = {'dictionary': dictionary, 'group_by_length': group_by_length, 'page_size': page_size, 'word_sorting': word_sorting, 'letters': letters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-finder-english.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

