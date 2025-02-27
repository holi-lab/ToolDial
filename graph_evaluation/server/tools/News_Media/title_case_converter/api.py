import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def title_case(title: str, style: str, preserveallcaps: str=None, usestraightquotes: bool=None, tagspeciesnames: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts a title to title case. See the About page for a detailed explanation of the parameters."
    title: The text to be converted to title case. Maximum length: 300 characters
        style: Specifies the title case style to be used. Possible values are: `AMA`, `AP`, `APA`,`BB`,  `CMOS`, `MLA`, `NYT`, `WP`
        preserveallcaps: Specifies whether words in all caps should be kept unchanged. Possible values are: `0` (default), `false`, `1`, `true`, `auto`.
        usestraightquotes: Specifies if quotes should be output as \\\"straight\\\" quotes instead of “curly” quotes. Possible values are: `0` (default), `false`, `1`, `true`.
        tagspeciesnames: Specifies if scientific species names should be tagged. Possible values are: `0` (default, no tagging), `1` (the complete species name is tagged using `<ScientificName>`), `2` (the parts of the species name that are customarily written in italics are tagged using `<em>`).
        
    """
    url = f"https://title-case-converter.p.rapidapi.com/v1/TitleCase"
    querystring = {'title': title, 'style': style, }
    if preserveallcaps:
        querystring['preserveAllCaps'] = preserveallcaps
    if usestraightquotes:
        querystring['useStraightQuotes'] = usestraightquotes
    if tagspeciesnames:
        querystring['tagSpeciesNames'] = tagspeciesnames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "title-case-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

