import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_markdown(domarkdown: str=None, html: str=None, output: str=None, preview: bool=None, showframe: bool=None, tags: str=None, u: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter a url or some HTML code to convert the page to Markdown. One parameter between "u" and "html" must be submitted with the request."
    domarkdown: (any value will run this) used without the other parameters (including u), and takes a text parameter containing url-encoded Markdown text. It returns raw HTML
        html: HTML text to be Markdownified. May be a full page or snippet, and can be submitted as urlencoded text in a GET call or straight using POST (suggested for larger requests).
        output: (optional, default markdown) type of text to return (json*, url (encoded), or markdown). There's also an 'nv' output mode that will generate a Notational Velocity/nvALT url for creating a note from resulting text. If you need the 'nvalt://' handler for older versions, use 'nvalt' for the output mode instead. Might be useful for some people.
        preview: (optional, default 0) whether to run the result back through Markdown and provide HTML instead of Markdown
        showframe: determines whether or not the output is encased in the HTML frame for viewing/copying
        tags: (optional) if the output type is "nv" or "nvalt", adding this key with a url-encoded string of space or comma separated tags will include them when importing into NV/nvALT
        u: url encoded URI to parse
        
    """
    url = f"https://thefosk-heck-yes-markdown.p.rapidapi.com/go/"
    querystring = {}
    if domarkdown:
        querystring['domarkdown'] = domarkdown
    if html:
        querystring['html'] = html
    if output:
        querystring['output'] = output
    if preview:
        querystring['preview'] = preview
    if showframe:
        querystring['showframe'] = showframe
    if tags:
        querystring['tags'] = tags
    if u:
        querystring['u'] = u
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefosk-heck-yes-markdown.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

