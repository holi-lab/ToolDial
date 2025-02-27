import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tweetsanalyzer(query: str, start: str, end: str, maxtweets: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The positivity score of a user is calculated using text analysis in BioMegiXTweet Analyzer."
    query: Search tweets of your interest by typing a keyword, #tag, a phrase, or a sentence.
        start: yyyy-mm-dd
        end: yyyy-mm-dd
        
    """
    url = f"https://biomegix-tweets-analyzer.p.rapidapi.com/tweetsanalyzer/"
    querystring = {'query': query, 'start': start, 'end': end, 'maxtweets': maxtweets, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "biomegix-tweets-analyzer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

