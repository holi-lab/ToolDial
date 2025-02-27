import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def post_search(query: str, after: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches Reddit posts."
    query: The comma-separated query for the search. Supports the following term types:

`site:{site_name}` - search only posts where the domain matches {site_name}.

`-site:{site_name}` - search only posts where the domain does not match {site_name}.

`/r/{subreddit}` - search only posts from the subreddit {subreddit}.

`-/r/{subreddit}` - search only posts not from the subreddit {subreddit}.

`{term}` - search only posts with titles containing the term {term}.

`-{term}` - search only posts with titles not containing the term {term}.

`score:{score}` - search only posts with score at least {score}.

`before:{YYYY-mm-dd}`, `after:{YYYY-mm-dd}` - search only posts within the date range. `before` is inclusive, `after` is not.
        after: The previous result's `sort_key` value. Used for pagination.
        
    """
    url = f"https://socialgrep.p.rapidapi.com/search/posts"
    querystring = {'query': query, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "socialgrep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_search(query: str, after: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches Reddit comments."
    query: The comma-separated query for the search. Supports the following term types:

`/r/{subreddit}` - search only comments from the subreddit {subreddit}.

`-/r/{subreddit}` - search only comments not from the subreddit {subreddit}.

`{term}` - search only comments containing the term {term}.

`-{term}` - search only comments not containing the term {term}.

`score:{score}` - search only comments with score at least {score}.

`before:{YYYY-mm-dd}`, `after:{YYYY-mm-dd}` - search only comments within the date range. `before` is inclusive, `after` is not.

`post:{post_id}` - search only comments for the given post.
        after: The previous result's `sort_key` value. Used for pagination.
        
    """
    url = f"https://socialgrep.p.rapidapi.com/search/comments"
    querystring = {'query': query, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "socialgrep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

