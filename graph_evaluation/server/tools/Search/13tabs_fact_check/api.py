import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_13tabs_fact_finder(page: int, q: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Debunk misinformation and fake news shared on the social media network or news portals with the help of 13TABS Fact Check API."
    page: Page Number
        q: Query string
        lang: The 'lang' parameter is not required but it can be used to filter the result. Supported languages and its value for the parameter is mentioned below. 

**Language**  --  **value**

Arabic   --   ar
Bengali (Bangla)   --   bn
Dutch    --    nl
English    --    en
Estonian    --    et
Farsi    --    fa
Finnish    --    fi
French    --    fr
German    --    de
Greek    --    el
Gujarati    --    gu
Hindi    --    hi
Indonesian    --    id
Italian    --    it
Kannada    --    kn
Marathi    --    mr
Malayalam    --    ml
Nepali    --    ne
Polish    --    pl
Portuguese    --    pt
Punjabi    --    pa
Russian    --    ru
Spanish    --    es
Tamil    --    ta
Telugu    --    te
Urdu    --    ur
        
    """
    url = f"https://13tabs-fact-check.p.rapidapi.com/api/facts/index.jsp"
    querystring = {'page': page, 'q': q, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "13tabs-fact-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

