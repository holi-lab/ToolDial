import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_posts_from_subreddit(get_subreddit_posts: str, subreddit: str='askreddit', nsfw: bool=None, time: str='all', full_data: bool=None, proxy: str=None, sort: str='hot', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Posts from Subreddit"
    subreddit: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"bitcoin\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        nsfw: (boolean, optional): Indicates whether to include NSFW (Not Safe for Work) posts in the search results. Default value is **True**.
        time: (string, optional): Specifies the time range for the search results. Possible values are \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**year**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**month**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**week**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**day**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**hour**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        full_data: (boolean, optional): Indicates whether to include the full data of each post in the search results. Default value is **False**.
        proxy: If no proxy value is provided (default is None), the search request will be made directly to the Reddit API without using a proxy.

Also you can use proxy https/socks5:
example:
with auth
socks5:127.0.0.1:1088:login:pass
http:127.0.0.1:8080:login:pass
without auth
socks5:127.0.0.1:1088
http:127.0.0.1:8080
        sort: (string, optional): Specifies the sorting order of the search results. Possible values are \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**hot**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**top**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**new**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"comments\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        limit: (integer, optional): Specifies the maximum number of search results to retrieve. Must be between 1 and 250. Default value is 10.
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{get_subreddit_posts}"
    querystring = {}
    if subreddit:
        querystring['subreddit'] = subreddit
    if nsfw:
        querystring['nsfw'] = nsfw
    if time:
        querystring['time'] = time
    if full_data:
        querystring['full_data'] = full_data
    if proxy:
        querystring['proxy'] = proxy
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_subreddits(search_subreddits: str, full_data: bool=None, proxy: str=None, keyword: str='bitcoin', sort: str='relevance', time: str='all', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint utilizes the Reddit API's search functionality to retrieve the Subreddits.
		To obtain the best results, it is recommended to use appropriate search parameters, including the keyword, sorting order, time range, and limiting the number of results to a reasonable value.
		Setting the limit parameter to its maximum value of 250 allows you to retrieve the maximum number of search results in a single request."
    full_data: (boolean, optional): Indicates whether to include the full data of each post in the search results. Default value is **False**.
        proxy: If no proxy value is provided (default is None), the search request will be made directly to the Reddit API without using a proxy.

Also you can use proxy https/socks5:
example:
with auth
socks5:127.0.0.1:1088:login:pass
http:127.0.0.1:8080:login:pass
without auth
socks5:127.0.0.1:1088
http:127.0.0.1:8080
        keyword: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\"bitcoin\\\\\\\".
        sort: (string, optional): Specifies the sorting order of the search results. Possible values are \\\\\\\"**relevance**\\\\\\\", \\\\\\\"**hot**\\\\\\\",\\\\\\\"**top**\\\\\\\", \\\\\\\"**new**\\\\\\\", and \\\\\\\"comments\\\\\\\". Default value is \\\\\\\"**relevance**\\\\\\\".
        time: (string, optional): Specifies the time range for the search results. Possible values are \\\\\\\"**all**\\\\\\\", \\\\\\\"**year**\\\\\\\", \\\\\\\"**month**\\\\\\\",\\\\\\\"**week**\\\\\\\", \\\\\\\"**day**\\\\\\\", and \\\\\\\"**hour**\\\\\\\". Default value is \\\\\\\"**all**\\\\\\\".
        limit: (integer, optional): Specifies the maximum number of search results to retrieve. Must be between **1** and **250**. Default value is **10**.
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{search_subreddits}"
    querystring = {}
    if full_data:
        querystring['full_data'] = full_data
    if proxy:
        querystring['proxy'] = proxy
    if keyword:
        querystring['keyword'] = keyword
    if sort:
        querystring['sort'] = sort
    if time:
        querystring['time'] = time
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_comments(search_comments: str, full_data: bool=None, proxy: str=None, restrict_sr: bool=None, time: str='all', limit: int=10, sort: str='relevance', keyword: str='bitcoin', nsfw: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint utilizes the Reddit API's search functionality to retrieve the comments.
		To obtain the best results, it is recommended to use appropriate search parameters, including the keyword, sorting order, time range, and limiting the number of results to a reasonable value.
		Setting the limit parameter to its maximum value of 250 allows you to retrieve the maximum number of search results in a single request."
    full_data: (boolean, optional): Indicates whether to include the full data of each post in the search results. Default value is **False**.
        proxy: If no proxy value is provided (default is None), the search request will be made directly to the Reddit API without using a proxy.

Also you can use proxy https/socks5:
example:
with auth
socks5:127.0.0.1:1088:login:pass
http:127.0.0.1:8080:login:pass
without auth
socks5:127.0.0.1:1088
http:127.0.0.1:8080
        restrict_sr: (boolean, optional): Indicates whether to restrict the search results to the specified subreddit. Default value is **True**.
        time: (string, optional): Specifies the time range for the search results. Possible values are \\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"**year**\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"**month**\\\\\\\\\\\\\\\",\\\\\\\\\\\\\\\"**week**\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"**day**\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\"**hour**\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\".
        limit: (integer, optional): Specifies the maximum number of search results to retrieve. Must be between **1** and **250**. Default value is **10**.
        sort: (string, optional): Specifies the sorting order of the search results. Possible values are \\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"**hot**\\\\\\\\\\\\\\\",\\\\\\\\\\\\\\\"**top**\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\"**new**\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\"comments\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\".
        keyword: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\\\\\\\\\"bitcoin\\\\\\\\\\\\\\\".
        nsfw: (boolean, optional): Indicates whether to include NSFW (Not Safe for Work) posts in the search results. Default value is **True**.
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{search_comments}"
    querystring = {}
    if full_data:
        querystring['full_data'] = full_data
    if proxy:
        querystring['proxy'] = proxy
    if restrict_sr:
        querystring['restrict_sr'] = restrict_sr
    if time:
        querystring['time'] = time
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if keyword:
        querystring['keyword'] = keyword
    if nsfw:
        querystring['nsfw'] = nsfw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(search_users: str, proxy: str=None, time: str='all', sort: str='relevance', limit: int=10, full_data: bool=None, keyword: str='bitcoin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint utilizes the Reddit API's search functionality to retrieve the users.
		To obtain the best results, it is recommended to use appropriate search parameters, including the keyword, sorting order, time range, and limiting the number of results to a reasonable value.
		Setting the limit parameter to its maximum value of 250 allows you to retrieve the maximum number of search results in a single request."
    proxy: If no proxy value is provided (default is None), the search request will be made directly to the Reddit API without using a proxy.

Also you can use proxy https/socks5:
example:
with auth
socks5:127.0.0.1:1088:login:pass
http:127.0.0.1:8080:login:pass
without auth
socks5:127.0.0.1:1088
http:127.0.0.1:8080
        time: (string, optional): Specifies the time range for the search results. Possible values are \\\\\\\"**all**\\\\\\\", \\\\\\\"**year**\\\\\\\", \\\\\\\"**month**\\\\\\\",\\\\\\\"**week**\\\\\\\", \\\\\\\"**day**\\\\\\\", and \\\\\\\"**hour**\\\\\\\". Default value is \\\\\\\"**all**\\\\\\\".
        sort: (string, optional): Specifies the sorting order of the search results. Possible values are \\\\\\\"**relevance**\\\\\\\", \\\\\\\"**hot**\\\\\\\",\\\\\\\"**top**\\\\\\\", \\\\\\\"**new**\\\\\\\", and \\\\\\\"comments\\\\\\\". Default value is \\\\\\\"**relevance**\\\\\\\".
        limit: (integer, optional): Specifies the maximum number of search results to retrieve. Must be between **1** and **250**. Default value is **10**.
        full_data: (boolean, optional): Indicates whether to include the full data of each post in the search results. Default value is **False**.
        keyword: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\"bitcoin\\\\\\\".
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{search_users}"
    querystring = {}
    if proxy:
        querystring['proxy'] = proxy
    if time:
        querystring['time'] = time
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if full_data:
        querystring['full_data'] = full_data
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shadowban_checker(shadowban: str, proxy: str=None, timeout: int=5, username: str='John', full: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to check if a Reddit user is shadowbanned. Shadowbanning is a practice where a user's actions and content are hidden from the public without their knowledge. This endpoint provides a way to detect if a user is affected by shadowbanning on Reddit.
		
		Parameters:
		
		username (string, optional): Specifies the username of the Reddit user to check for shadowbanning. The default value is set to "John".
		full (boolean, optional): Determines whether additional user information should be returned in the response. If set to True, it includes details such as the user's join date, post karma, comment karma, and verified mail status. By default, it is set to False.
		proxy (string, optional): Specifies the proxy server to be used for the request. If required, you can provide the proxy server's details to route the request through a specific network or location. The default value is None, indicating no proxy is used.
		timeout (integer, optional): Specifies the maximum time, in seconds, to wait for a response from the Reddit API. The default timeout is set to 5 seconds.
		Returns:
		
		The endpoint returns a JSON response with the following fields:
		
		is_shadowbanned (boolean): Indicates whether the specified Reddit user is shadowbanned. If True, the user is shadowbanned. If False, the user is not shadowbanned.
		If full is set to True, the response also includes the following additional fields:
		username (string): The username of the Reddit user.
		join_date (string): The date and time when the user joined Reddit.
		post_karma (string): The amount of karma the user has gained from posting.
		comment_karma (string): The amount of karma the user has gained from commenting.
		verified_mail (string): Indicates whether the user has a verified email associated with their account.
		Error Handling:
		
		The endpoint handles potential errors during the process. If the provided username is invalid, a ValueError is raised with an HTTPException status code of 400 and an error message indicating the issue.
		
		Possible errors during the request process, such as proxy connection failures, request timeouts, or internal server errors, are handled as HTTPExceptions with status codes 500 and appropriate error details.
		
		Note:
		
		The endpoint utilizes the Reddit API to fetch information about the user.
		Proper validation and handling of the username parameter are performed to ensure the input is valid and avoid potential errors.
		The endpoint supports the usage of proxy servers for the request if required.
		The timeout parameter allows controlling the maximum waiting time for the response from the Reddit API.
		The endpoint provides basic information about the user and helps identify if they are shadowbanned on Reddit."
    username: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"bitcoin\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{shadowban}"
    querystring = {}
    if proxy:
        querystring['proxy'] = proxy
    if timeout:
        querystring['timeout'] = timeout
    if username:
        querystring['username'] = username
    if full:
        querystring['full'] = full
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_posts(search_posts: str, nsfw: bool=None, full_data: bool=None, time: str='all', sort: str='relevance', keyword: str='bitcoin', proxy: str=None, restrict_sr: bool=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint utilizes the Reddit API's search functionality to retrieve the posts.
		To obtain the best results, it is recommended to use appropriate search parameters, including the keyword, sorting order, time range, and limiting the number of results to a reasonable value.
		Setting the limit parameter to its maximum value of 250 allows you to retrieve the maximum number of search results in a single request."
    search_posts: This endpoint allows you to search for posts on Reddit. It retrieves a list of posts based on the specified search parameters. The search results can be filtered and sorted based on various criteria such as relevance, popularity, and time.Also in one request you can get up to 250 results.
        nsfw: (boolean, optional): Indicates whether to include NSFW (Not Safe for Work) posts in the search results. Default value is **True**.
        full_data: (boolean, optional): Indicates whether to include the full data of each post in the search results. Default value is **False**.
        time: (string, optional): Specifies the time range for the search results. Possible values are \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**year**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**month**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**week**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**day**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**hour**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**all**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        sort: (string, optional): Specifies the sorting order of the search results. Possible values are \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**hot**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**top**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**new**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", and \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"comments\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"**relevance**\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        keyword: (string, optional): Specifies the keyword to search for in the posts. Default value is \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"bitcoin\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\".
        proxy: If no proxy value is provided (default is None), the search request will be made directly to the Reddit API without using a proxy.

Also you can use proxy https/socks5:
example:
with auth
socks5:127.0.0.1:1088:login:pass
http:127.0.0.1:8080:login:pass
without auth
socks5:127.0.0.1:1088
http:127.0.0.1:8080
        restrict_sr: (boolean, optional): Indicates whether to restrict the search results to the specified subreddit. Default value is **True**.
        limit: (integer, optional): Specifies the maximum number of search results to retrieve. Must be between 1 and 250. Default value is 10.
        
    """
    url = f"https://reddit-fast-search.p.rapidapi.com/api/{search_posts}"
    querystring = {}
    if nsfw:
        querystring['nsfw'] = nsfw
    if full_data:
        querystring['full_data'] = full_data
    if time:
        querystring['time'] = time
    if sort:
        querystring['sort'] = sort
    if keyword:
        querystring['keyword'] = keyword
    if proxy:
        querystring['proxy'] = proxy
    if restrict_sr:
        querystring['restrict_sr'] = restrict_sr
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-fast-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

