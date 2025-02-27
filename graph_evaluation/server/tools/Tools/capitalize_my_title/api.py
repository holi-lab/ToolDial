import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettitle(title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint for Capitalize My Title"
    title: The {title} input is the title that you want to capitalize. Without styles specified, the title will be capitalized using [APA style rules in title case](https://capitalizemytitle.com/#APAStyle).
        
    """
    url = f"https://capitalize-my-title.p.rapidapi.com/title/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capitalize-my-title.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettitlewithstyleandsubstyle(title: str, style: str, substyle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint for Capitalize My Title with style and sub-style"
    style: The {style} input can be any of the following:

- apa: Capitalize using the APA manual style guidelines.
- chicago: Capitalize using the Chicago Manual of Style guidelines.
- ap: Capitalize using the Associated Press Stylebook 2020 (55th edition).
- mla: Capitalize using the MLA Handbook, 8th edition style rules.
- nyt: Capitalize with the NY Times style rules.
- ama: Capitalize using the American Medical Association (AMA) Manual of Style, 11th edition
- bb: Capitalize using the The Bluebook, 21st edition
- wikipedia: Capitalize with the Wikipedia style rules.
- email: Capitalize with standard email style rules.
        substyle: The {substyle} input can be any of the following:

-  title-case: Capitalize the first letter of important words.
- sentence-case: Capitalize only the first word of each sentence. [More details](https://capitalizemytitle.com/#WhatiosTitleCase).
- uppercase: Uppercase all letters.
- lowercase: Lowercase all letters.
- first-letter: Capitalize the first letter of every word.
- alt-case: Capitalize every other letter of your text starting with the first letter being capitalized.
- toggle-case: Change the case of every letter in your string.
        
    """
    url = f"https://capitalize-my-title.p.rapidapi.com/title/{style}/{substyle}/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capitalize-my-title.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettitlewithstyle(style: str, title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint for Capitalize My Title with one style."
    style: The {style} input can be any of the following:

- apa: Capitalize using the APA manual style guidelines.
- chicago: Capitalize using the Chicago Manual of Style guidelines.
- ap: Capitalize using the Associated Press Stylebook 2020 (55th edition).
- mla: Capitalize using the MLA Handbook, 8th edition style rules.
- nyt: Capitalize with the NY Times style rules.
- ama: Capitalize using the American Medical Association (AMA) Manual of Style, 11th edition
- bb: Capitalize using the The Bluebook, 21st edition
- wikipedia: Capitalize with the Wikipedia style rules.
- email: Capitalize with standard email style rules.
-  title-case: Capitalize the first letter of important words.
- sentence-case: Capitalize only the first word of each sentence. [More details](https://capitalizemytitle.com/#WhatiosTitleCase).
- uppercase: Uppercase all letters.
- lowercase: Lowercase all letters.
- first-letter: Capitalize the first letter of every word.
- alt-case: Capitalize every other letter of your text starting with the first letter being capitalized.
- toggle-case: Change the case of every letter in your string.
        title: The {title} input is the title that you want to capitalize.
        
    """
    url = f"https://capitalize-my-title.p.rapidapi.com/title/{style}/{title}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capitalize-my-title.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

