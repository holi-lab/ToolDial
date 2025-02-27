import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def notion_page_extraction(page_id: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to extract text from a Notion page. To use this service, you'll need to provide the page_id of the desired Notion page.
		
		Please note, access to private Notion pages requires an integration token. You can generate this token by setting up a private integration in your Notion settings.
		
		With the page_id and integration token, our API can then fetch and return the page data in a format ready for use with large language models.
		
		To better illustrate this process, we've provided an example using a specific Notion page created for this purpose. This endpoint simplifies data extraction, enabling you to focus on building remarkable products with large language models."
    
    """
    url = f"https://text-for-gpt.p.rapidapi.com/notion"
    querystring = {'page_id': page_id, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-for-gpt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_transcript_extraction(source: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to extract transcripts from a YouTube video. To use this service, you'll need to provide the source parameter, which is the URL of the desired YouTube video.
		
		Our API will then fetch and convert the video's transcript into a text format ready for use with large language models.
		
		No additional tokens or permissions are required - just provide the video URL and let our service do the rest.
		
		This endpoint simplifies the process of data extraction from YouTube, allowing you to focus more on developing powerful applications with large language models."
    
    """
    url = f"https://text-for-gpt.p.rapidapi.com/youtube"
    querystring = {'source': source, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-for-gpt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

