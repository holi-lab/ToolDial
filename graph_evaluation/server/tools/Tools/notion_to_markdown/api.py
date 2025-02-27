import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_page_as_markdown(auth: str, pageid: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/notion-to-md` route is a GET request that converts a specified Notion page into Markdown format. This route is designed to be used with the Notion API and requires an authorization token (`auth`) and a Notion page ID (`pageId`) as parameters.
		
		The `auth` parameter is the Notion API token. The `pageId` parameter is the ID of the Notion page you want to convert to Markdown.
		
		The route also accepts an optional `format` parameter, which determines the format of the response. If `format` is set to `json`, the response will be a JSON object containing the Markdown text. If `format` is not specified or set to any other value, the response will be a plain text string containing the Markdown text.
		
		The route is configured not to parse child pages.
		
		If the conversion is successful, the route returns the Markdown text as a response. If an error occurs during the conversion, the route returns an error response with a status code of 400. The error response includes the error message from the Notion API if the error is an `APIResponseError`.
		
		
		Usage:
		
		```http
		GET /notion-to-md?auth=<your_notion_api_token>&pageId=<notion_page_id>&format=json
		```
		
		Replace `<your_notion_api_token>` with your Notion API token and `<notion_page_id>` with the ID of the Notion page you want to convert to Markdown. If you want the response in JSON format, include `&format=json` in the request URL."
    auth: This is your Notion API token, which is required to authenticate your request. This token gives the API access to your Notion workspace.  To obtain token create an internal integration in Notion:
https://www.notion.so/help/create-integrations-with-the-notion-api#create-an-internal-integration
        pageid:  This is the ID of the Notion page that you want to convert to Markdown. You can find the page ID in the URL of the page in the Notion app. This parameter is required.
        format: This is an optional parameter that determines the format of the response. If format is set to json, the response will be a JSON object containing the Markdown text. If format is not specified or set to any other value, the response will be a plain text string containing the Markdown text.
        
    """
    url = f"https://notion-to-markdown.p.rapidapi.com/notion-to-md"
    querystring = {'auth': auth, 'pageId': pageid, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "notion-to-markdown.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

