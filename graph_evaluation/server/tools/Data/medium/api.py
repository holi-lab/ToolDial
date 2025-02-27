import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_lists(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of *list_ids* created by the user."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/lists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_fans(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of *user_ids* of the people who clapped on the article."
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/fans"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_s_html(article_id: str, fullpage: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the plain **HTML** of a Medium Article, for the corresponding `article_id`"
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/html"
    querystring = {}
    if fullpage:
        querystring['fullpage'] = fullpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_responses(list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of *response_ids* (same as *article_ids*) on the given Medium List."
    
    """
    url = f"https://medium2.p.rapidapi.com/list/{list_id}/responses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_articles(list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of *articles_ids* present in the given List (Medium List)."
    
    """
    url = f"https://medium2.p.rapidapi.com/list/{list_id}/articles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_info(list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list-related information such as Id, Name, Author, Description, Thumbnail Image URL, Creation Datetime, Last Modified Datetime, number of articles in the list, claps, voters, and comments."
    
    """
    url = f"https://medium2.p.rapidapi.com/list/{list_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_articles(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of *article_ids* of the related posts. (Length = 5)
		
		**Note:** If the article is self-published, related posts will belong to the same author, else related posts will belong to the publication in which the article is published."
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/related"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_s_content(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the content of an article/story from Medium, for the corresponding *article_id*"
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_tags(tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of related tags for any given tag.
		
		Example: For tag *blockchain*, it will return *'cryptocurrency', 'bitcoin', 'ethereum',  'crypto',  'ico',  'technology',  'defi',  'nft',  'fintech'*."
    
    """
    url = f"https://medium2.p.rapidapi.com/related_tags/{tag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tag_info(tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns tag-related information.
		
		- Name
		- Followers Count
		- Number of stories 
		- Number of writers
		- Number of latest stories
		- Number of latest writers"
    
    """
    url = f"https://medium2.p.rapidapi.com/tag/{tag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_posts(topic_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of latest "curated/distributed" posts (*article_ids*) for a topic/niche (as classified by the Medium platform). Example of a topic/niche can be: 
		 - "blockchain" 
		 - "relationships" 
		 - "mental-health", etc ...
		
		These are known as *topic-slugs*. At any given moment, this endpoint will return a list of 25 articles (*article_ids*)."
    
    """
    url = f"https://medium2.p.rapidapi.com/latestposts/{topic_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_writers(topic_slug: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of top writers (*user_ids*) within a particular topic/niche (e.g. "blockchain", "relationships", "artificial-intelligence").
		
		**Note**: You can use `count` query string parameter to limit the number of results. The maximum number of top writers within a topic/niche will be 250."
    
    """
    url = f"https://medium2.p.rapidapi.com/top_writers/{topic_slug}"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_feeds(tag: str, mode: str, after: int=0, count: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of `article_ids` (default count = 25) for the given `tag` and `mode`.
		
		Mode:
		- `hot` : For getting trending articles
		- `new` : For getting latest articles
		- `top_year` : For getting best articles of the year
		- `top_month` : For getting best articles of the month
		- `top_week` : For getting best articles of the week
		- `top_all_time`: For getting best article of all time
		
		To get the subsequent top feeds, it can also take a query string parameter called `after`, where, `after` < 250.
		
		To limit the number of top feeds, use another query string parameter called `count`, where `count` < 25."
    mode: -     `hot`: For getting trending articles
-     `new`: For getting the latest articles
-     `top_year`: For getting the best articles of the year
-     `top_month`: For getting the best articles of the month
-     `top_week`: For getting the best articles of the week
-     `top_all_time`: For getting the best article of all time
        
    """
    url = f"https://medium2.p.rapidapi.com/topfeeds/{tag}/{mode}"
    querystring = {}
    if after:
        querystring['after'] = after
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_publications(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of *publication_ids* for the given search query results. (Max Length = 1000)"
    
    """
    url = f"https://medium2.p.rapidapi.com/search/publications"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_lists(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of *list_ids* for the given search query results. (Max Length = 1000)"
    
    """
    url = f"https://medium2.p.rapidapi.com/search/lists"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of *user_ids* for the given search query results. (Max Length = 1000)"
    
    """
    url = f"https://medium2.p.rapidapi.com/search/users"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_tags(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of *tags* for the given search query results. (Max Length = 1000)"
    
    """
    url = f"https://medium2.p.rapidapi.com/search/tags"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_articles(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of *articles_ids* for the given search query results. (Max Length = 1000)"
    
    """
    url = f"https://medium2.p.rapidapi.com/search/articles"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_responses(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of responses (`response_ids`, same as `article_ids`) for a given article (`article_id`)
		
		Note: To see the content of the response, use the "Get Article's Content" endpoint"
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/responses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_s_markdown(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the **markdown** of an article/story from Medium, for the corresponding `article_id`"
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}/markdown"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_interests(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of tags that the given user follows."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/interests"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_articles_by_user(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list `article_ids` of the top 10 articles on the user's profile, for a given `user_id`"
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/top_articles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers(user_id: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of `user_ids` of the followers for the given user. 
		
		Use `count` query parameter (optional) to limit the number of results. (count <= 1500)
		
		*Note:* The length of this followers' list might be different from what you get in the "Get User Info" Endpoint. It's because, this list doesn't include Medium Users who left the platform.
		
		If you really need the exact followers' count, use this endpoint to get the followers' list and take its length as the exact followers' count"
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/followers"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user/author related information like Full name, Bio, Followers count, Following count, Twitter username, Profile-image URL, User ID, etc ... It takes "user_id" (String) as the path parameter.
		
		If you don't know the "user_id", then you can get it from the endpoint '/user/id_for/{username}'."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following(user_id: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of *user_ids* of the author's/user's followings. 
		
		Use `count` query parameter (optional) to limit the number of results. (count <= 1500)
		
		Note: Currently, this list does not contain the *publication_ids* of the publications that the user/author is following."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/following"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_by_user(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of *articles_ids* written by the author/user."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/articles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_id(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a unique ID (user_id: String) associated with the user/author's unique username."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/id_for/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_publications(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of *publication_ids* where the user is the editor and/or creator."
    
    """
    url = f"https://medium2.p.rapidapi.com/user/{user_id}/publications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_publication_articles(publication_id: str, is_from: str='2022-08-13T13:10:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of `articles_ids`, of the latest 25 articles, posted in that publication.
		
		Use the `from` query parameter to get the articles before that date and time."
    
    """
    url = f"https://medium2.p.rapidapi.com/publication/{publication_id}/articles"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_publication_newsletter(publication_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information related to the associated newsletter for the given publication. This includes their subscriber's count, name, description, unique id, etc..."
    
    """
    url = f"https://medium2.p.rapidapi.com/publication/{publication_id}/newsletter"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_publication_info(publication_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the publication related information such as Publication name, Tagline, Description, Tags, Followers Count, Twitter username, Instagram username, Facebook Page name, etc ...
		
		**Note**: If you don't know the *publication_id*, you can get it from any article published by it. Use endpoint "/article/{article_id}" to retrieve the *publication_id*."
    
    """
    url = f"https://medium2.p.rapidapi.com/publication/{publication_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_publication_id(publication_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the `publication_id` for the given `publication_slug`."
    
    """
    url = f"https://medium2.p.rapidapi.com/publication/id_for/{publication_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_welcome(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Test Endpoint**
		
		Returns the information about the Medium API"
    
    """
    url = f"https://medium2.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_article_info(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the article related information such as Title, Subtitle, Tags, Topics (assigned by Medium), Publication, Published date and time, Clap Count, Voter Count, Word Count, Reading Time, etc..."
    
    """
    url = f"https://medium2.p.rapidapi.com/article/{article_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medium2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

