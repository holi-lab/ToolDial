import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def spell_check(mode: str, text: str, precontexttext: str=None, postcontexttext: str=None, mkt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check spelling."
    mode: Mode of spellcheck:
- **proof** - Meant to provide Office Word like spelling corrections. It can correct long queries, provide casing corrections and suppresses aggressive corrections.
- **spell** - Meant to provide Search engine like spelling corrections. It will correct small queries(up to length 9 tokens) without any casing changes and will be more optimized (perf and relevance) towards search like queries.
        text: The text string to check for spelling and grammar errors.
        precontexttext: A string that gives context to the text string. For example, the text string petal is valid; however, if you set preContextText to bike, the context changes and the text string becomes not valid. In this case, the API will suggest that you change petal to pedal (as in bike pedal).
        postcontexttext: A string that gives context to the text string. For example, the text string read is valid; however, if you set postContextText to carpet, the context changes and the text string becomes not valid. In this case, the API will suggest that you change read to red (as in red carpet).
        mkt: For proof mode, only support en-us, es-es, pt-br, For spell mode, support all language codes.
        
    """
    url = f"https://bing-spell-check2.p.rapidapi.com/spellcheck"
    querystring = {'mode': mode, 'text': text, }
    if precontexttext:
        querystring['preContextText'] = precontexttext
    if postcontexttext:
        querystring['postContextText'] = postcontexttext
    if mkt:
        querystring['mkt'] = mkt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-spell-check2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

