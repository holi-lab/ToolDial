import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_posts(query: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Post objects which match a given search query. Searches ordered by id require at least one query or filter to be specified; searches ordered by score require at least one query and zero or more filters to be specified. Searches require an ordering and at least one search query to be specified, and allow for zero or more filters to be added. All parameters should be passed in the query string."
    query: Type of index to use. The default (and currently, the only) index is complete, which searches all posts. We may add additional index types later (e.g., an index only of recent posts, for speed.)
        order: One of: id (default), score. Searches of ordering id are returned in roughly the same order as other streams, and support pagination. Searches of ordering score are returned by a relevance score. Currently, the only ordering that supports pagination is id, and we are working on improving relevance scores.
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/posts/search"
    querystring = {}
    if query:
        querystring['query'] = query
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_channels(q: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Channel objects which match a given search query. Because channels have no inherent notion of description or name, we take textual data from common channel annotations which contain such fields, e.g. net.patter-app.settings. We also allow filtering on specific channel properties, such as channel type. No matter what query data is supplied, the search results will respect channel ACLs, and results are limited to non-private channels if the requesting access token does not have the messages scope."
    q: Searches any textual fields extracted from channel annotations. We may tweak which annotations and fields are included by this search over time.
        order: One of: popularity (default), id, or activity. Searches of ordering popularity are returned in an order that roughly matches the popularity of the channels. activity searches will order results roughly by time of last message (precise sorting of very recent messages is not guaranteed).
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/channels/search"
    querystring = {}
    if q:
        querystring['q'] = q
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_place(latitude: str, longitude: str, q: str=None, radius: str=None, count: str=None, remove_closed: str=None, altitude: str=None, horizontal_accuracy: str=None, vertical_accuracy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Performs a search for nearest places from given latitude and longitude. Optionally takes a q string to restrict matches on name (like autocomplete for Place names).  Returns a list of Places sorted by distance or distance/string match if q is provided. These are the same Place objects as returned by the previous endpoint but will also include a distance property which gives, in meters, the distance from the search centroid to the Place."
    latitude: decimal	Latitude of search location. Combined with longitude to create central point of search results.
        longitude: decimal	Longitude of search location. Combined with latitude to create central point of search results.
        q: string	The name-based search query. Can be a partial string — for example, 'cre' will find any local 'creameries' and 'ice cream' locations.
        radius: decimal	Approximate radius (in meters) of bounding circle on results. For example, supplying radius=100 will limit all locations to be within 100 meters. Defaults to 100, ranges between 0.001 and 50,000.
        count: integer	Number of results to return. Defaults to 20, ranges between 1 and 100.
        remove_closed: int (0 or 1)	Set to 0 if you would like the result to include entities which are closed (is_closed=1) or 1 if you would only prefer to see results for entities that are open. Defaults to 1.
        altitude: decimal	Altitude of search location (in meters). Not presently used to generate search results but may be later.
        horizontal_accuracy: decimal	Accuracy of latitude/longitude parameters (in meters). Not presently used to generate search results but may be later.
        vertical_accuracy: decimal	Accuracy of altitude parameter (in meters). Not presently used to generate search results but may be later.
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/places/search"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if q:
        querystring['q'] = q
    if radius:
        querystring['radius'] = radius
    if count:
        querystring['count'] = count
    if remove_closed:
        querystring['remove_closed'] = remove_closed
    if altitude:
        querystring['altitude'] = altitude
    if horizontal_accuracy:
        querystring['horizontal_accuracy'] = horizontal_accuracy
    if vertical_accuracy:
        querystring['vertical_accuracy'] = vertical_accuracy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_explore_streams(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all Explore Streams. The list of Explore Streams are dynamic and will sometimes change. Please cache them for 24 hours in your app. Also, please note that this endpoint is not paginated."
    
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/posts/stream/explore"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_the_messages_in_a_channel(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a stream of the Messages in a channel."
    channel_id: The id of the Channel to retrieve Messages from.
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/channels/{channel_id}/messages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_interactions_with_me(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the Interactions other users have had with me."
    
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/users/me/interactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_token_s_streams(key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the Streams for the current token."
    key: (Optional) Only retrieve the stream that matches the given key.
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/streams"
    querystring = {}
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_users(q: str, count: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the App.net userbase."
    q: The search query. Supports @username or #tag searches as well as normal search terms. Searches username, display name, bio information. Does not search posts.
        count: The number of Users to return, up to a maximum of 200. Defaults to 20 if not specified.
        
    """
    url = f"https://appdotnet-appnet.p.rapidapi.com/stream/0/users/search"
    querystring = {'q': q, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appdotnet-appnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

