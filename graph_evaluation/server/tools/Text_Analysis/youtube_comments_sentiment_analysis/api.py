import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analyse_sentiment_from_youtube_comments(lang: str, video: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return a JSON with:
		- All comments and it's respective score and type (positive, negative or neutro)
		- Total comments
		- Total of positive comments
		- Total of negative comments
		- Total of neutros comments
		- Greatest positive score
		- Greatest negative score"
    
    """
    url = f"https://youtube-comments-sentiment-analysis.p.rapidapi.com/analysis"
    querystring = {'lang': lang, 'video': video, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-comments-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

