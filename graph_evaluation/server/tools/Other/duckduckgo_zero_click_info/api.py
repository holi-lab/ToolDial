import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zero_click_info(q: str, callback: str='process_duckduckgo', no_html: int=1, no_redirect: int=1, skip_disambig: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our Zero-click Info API gives you free access to much of our topic summaries, categories, disambiguation, !bang redirects, definitions and more."
    q: Search terms
        callback: Function to callback (for a JSONP formatted response).
        no_html: 1 to remove HTML from text, e.g. bold and italics.
        no_redirect: 1 to skip HTTP redirects (for !bang commands - see http://duckduckgo.com/bang.html).
        skip_disambig: 1 to skip disambiguation (D) Type.
        
    """
    url = f"https://duckduckgo-duckduckgo-zero-click-info.p.rapidapi.com/"
    querystring = {'q': q, }
    if callback:
        querystring['callback'] = callback
    if no_html:
        querystring['no_html'] = no_html
    if no_redirect:
        querystring['no_redirect'] = no_redirect
    if skip_disambig:
        querystring['skip_disambig'] = skip_disambig
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duckduckgo-duckduckgo-zero-click-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

