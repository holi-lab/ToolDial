import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_articles(language: str, sort: str=None, required_props: str=None, cursor: str=None, to: str=None, limit: int=None, is_from: str=None, q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get, filter, smart search google news articles."
    language: Supported by 10 languages. 

Language request codes:

    TR, EN, FR, DE,  IT, ZH, ES, RU, KO, PT
        sort: You can arrange the articles systematically in groups; separate according to type, class, etc.

Sorting refers to ordering articles in an increasing or decreasing manner according to some linear relationship among the article items. You can order the articles and arrange them in a sequence ordered by category with similar properties.

Expected values:

date:asc
date:desc

_score:asc
_score:desc
        required_props: Required props are delimited by commas. Each property must have a name. If one of the property names is missing from the article, that property is excluded.

Example:

title, image


        cursor: This defines the returned value of the \\\"next_cursor\\\". It allows you to navigate to next article list.
        to: \\\"to\\\" when used defines the date of items published before the specified date.

Format:
ISO date standart format is used. 
YYYY-MM-DDTHH:mm:ss

Example:
2022-10-17T18:41:34.306Z
        limit: With \\\"limit,\\\" you can specify the number of articles required per request. Maximum limit is set to 100 articles per request.
        is_from: \\\"from\\\" when used defines the date of items published after the specified date.

Format:
ISO date standart format is used. 
YYYY-MM-DDTHH:mm:ss

Example:
2022-10-17T18:41:34.306Z


        q: The search term or terms used to locate the article.

\\\"AND\\\" or \\\"OR\\\" may be placed between the keywords.

If you wish to group the keywords, you may include them in parenthesis.

For instance: (cats and dogs) OR (fish)

If you wish to omit articles by term, you can do so by prefixing the keyword with a (-) sign.

Example: cats -dogs
(This will return articles featuring cats but exclude those with the term \\\"dog.\\\"


        
    """
    url = f"https://google-news-api1.p.rapidapi.com/search"
    querystring = {'language': language, }
    if sort:
        querystring['sort'] = sort
    if required_props:
        querystring['required_props'] = required_props
    if cursor:
        querystring['cursor'] = cursor
    if to:
        querystring['to'] = to
    if limit:
        querystring['limit'] = limit
    if is_from:
        querystring['from'] = is_from
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

