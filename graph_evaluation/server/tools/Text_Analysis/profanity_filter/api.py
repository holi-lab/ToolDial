import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_text(check_word: str, add: str=None, range: str=None, r: str='*', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to detect and filter out profanity / offensive form a given text. It is only available for English words. You can also censor out words in text content."
    check_word: Use this parameter to filter text for profanity / offensive word. MAX_LENGTH = `[ 2 .. 700 ] characters`
        add: You can use the optional `add` parameter with a comma separated list of words to be added to the selected `range` profanity list. Accepts `[ 2 .. 250 ] characters` in length).
The `add` parameter is case-insensitive, so the case of you entry is not important.
        range: You can use the optional `range` parameter to set the level of filtering range. Default: '`low`'
`'low'` :  mild database
`'mid'` : includes all database words in `'low'` database but does not include all `'high'` database of words
`'high'`: strict large database words which also includes `'low'` and `'mid'` database
        r: Censor replacement string - You can use the optional `r` parameter to get output of censor words in `word_filtered` response data. Must be a valid punctuation. MAX_LENGTH `<= 1 character`
        
    """
    url = f"https://profanity-filter.p.rapidapi.com/api/v1/filter"
    querystring = {'check_word': check_word, }
    if add:
        querystring['add'] = add
    if range:
        querystring['range'] = range
    if r:
        querystring['r'] = r
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "profanity-filter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

