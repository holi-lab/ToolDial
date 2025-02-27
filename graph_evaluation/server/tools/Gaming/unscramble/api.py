import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def unscramble(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the word or jumbled letters and we will unscramble it .
		For eg.
		Input :-
		[Paramater] word : the word to unscramble
		
		Output :-
		Unscrambled words grouped by length and sorted
		
		Example :-
		Input
		api/unscramble?word=river
		
		Output :-
		{
		    "word": "river",
		    "unscrambled": {
		        "5 Letter Words": [
		            "river"
		        ],
		        "4 Letter Words": [
		            "rier",
		            "rive",
		            "veri",
		            "vier",
		            "vire"
		        ],
		        "3 Letter Words": [
		            "err",
		            "ire",
		            "rev",
		            "rie",
		            "vei",
		            "vie"
		        ],
		        "2 Letter Words": [
		            "er",
		            "ie",
		            "re"
		        ]
		    }
		}"
    
    """
    url = f"https://unscramble1.p.rapidapi.com/unscramble"
    querystring = {'word': word, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unscramble1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

