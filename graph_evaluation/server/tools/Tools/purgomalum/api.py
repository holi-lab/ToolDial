import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def remove_profanity(text: str, add: str=None, fill_text: str=None, fill_char: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calling the web service involves a simple HTTP GET request. The only two requirements for using PurgoMalum are the "text" parameter on the query string and the request-type name as part of the path in the request"
    text: Input text to be processed.
        add: A comma separated list of words to be added to the profanity list. Accepts letters, numbers, underscores (_) and commas (,). Accepts up to 10 words (or 200 maximum characters in length). The PurgoMalum filter is case-insensitive, so the case of you entry is not important.
        fill_text: Text used to replace any words matching the profanity list. Accepts letters, numbers, underscores (_) tildes (~), exclamation points (!), dashes/hyphens (-), equal signs (=), pipes (|), single quotes ('), double quotes ("), asterisks (*), open and closed curly brackets ({ }), square brackets ([ ]) and parentheses (). Maximum length of 20 characters. When not used, the default is an asterisk (*) fill.
        fill_char: Single character used to replace any words matching the profanity list. Fills designated character to length of word replaced. Accepts underscore (_) tilde (~), dash/hyphen (-), equal sign (=), pipe (|) and asterisk (*). When not used, the default is an asterisk (*) fill.
        
    """
    url = f"https://community-purgomalum.p.rapidapi.com/json"
    querystring = {'text': text, }
    if add:
        querystring['add'] = add
    if fill_text:
        querystring['fill_text'] = fill_text
    if fill_char:
        querystring['fill_char'] = fill_char
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-purgomalum.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_if_text_contains_profanity(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-purgomalum.p.rapidapi.com/containsprofanity"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-purgomalum.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

