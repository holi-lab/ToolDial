import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def about_channel(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return more info about channel"
    id: Channel id
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/channel/about"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel(next: str='4qmFsgKFARIYVUNpVEdLQTlXMEcwVEw4SG03VWZfdTlBGjpFZ1oyYVdSbGIzTVlBeUFBTUFFNEFlb0RGa05uUVZORFoycHdNazVFTkRWT2VVcHNhMmR2VFdjJTNEmgIsYnJvd3NlLWZlZWRVQ2lUR0tBOVcwRzBUTDhIbTdVZl91OUF2aWRlb3MxMDI%3D', filter: str=None, is_id: str='UCiTGKA9W0G0TL8Hm7Uf_u9A', sort: str='n', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel videos"
    next: Pagination(continuation) parameter to get next channel video, no need any other parameters if 'next' present.
Can be obtained from inside channel request result.
        filter: Filter for live streams.  Available  options:
l - live now;
p - past live streams;
        is_id: Channel id.
        sort: Sort parameter. Available  options:
n - newest;
o - oldest;
p - popular
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/channel"
    querystring = {}
    if next:
        querystring['next'] = next
    if filter:
        querystring['filter'] = filter
    if is_id:
        querystring['id'] = is_id
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_related(is_id: str, next: str='CBQSDRILWVFIc1hNZ2xDOUEYACreAQjQr43tubfzi64BCNi24eOapu-n7AEIgejpz9fcuPajAQjwqMe8v7SEuJ0BCI3CwqDWu4nahAEItNrrwNnAzuQ1CK6-ooCH-Jj5JAik8O-ahq3L1sYBCMb965f10YS4UwiNkaXwtL_gzi4I1vOMu5f7r4HeAQjEuYHvqNfimgwIzvHK75mt1Z27AQjw_7n5yaLZ3_UBCJOq5eCOo-XS_QEIocGSnpeajIsXCN2F2tj65L_4zwEI4KbhwtjP98duCI_C_IbhttbzTAi2gO-y3KbjuZgBCNbN7-m31YCKVmoPd2F0Y2gtbmV4dC1mZWVk', hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related videos"
    id: Video id
        next: Pagination(continuation) parameter to get more related videos, no need any other parameters if 'next' present.
Can be obtained from first response.
        hl: Locale/language for request
        gl: Country code  like US(default), UK, BE, etc...
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/video/related"
    querystring = {'id': is_id, }
    if next:
        querystring['next'] = next
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(next: str='Eg0SC1lRSHNYTWdsQzlBGAYyJSIRIgtZUUhzWE1nbEM5QTAAeAJCEGNvbW1lbnRzLXNlY3Rpb24%3D', is_id: str='YQHsXMglC9A', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video comments list.
		If you need sorting then use "sortTopNext" or "sortNewestNext"  fields from first response and pass it to "next" parameter."
    next: Pagination(continuation) parameter to get more comments , no need any other parameters if 'next' present. Could be used for sorting, just pass \"sortNewestNext\" or \"sortTopNext\" field values for newest or top sorting.
Can be obtained from response with \"id\" parameter in request
        is_id: Video id to get first part of comments.

        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/video/comments"
    querystring = {}
    if next:
        querystring['next'] = next
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(hl: str='en', gl: str='US', type: str='mu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of trending videos"
    hl: Locale/language for request
        gl: Country from you want get trendings like US(default), UK, BE, etc...
        type: Type of trending videos:
n - now (default)
mu - music
mo - movies
g - gaming
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/trending"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_info(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video info by id"
    id: Video id from YouTube
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/video"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist(next: str='4qmFsgJhEiRWTFBMV3dBeXBBY0ZSZ0tBSUlGcUJyOW95LVpZWm5peGFfRmoaFENBRjZCbEJVT2tOSFZRJTNEJTNEmgIiUExXd0F5cEFjRlJnS0FJSUZxQnI5b3ktWllabml4YV9Gag%3D%3D', is_id: str='PL2UMfhpwklNNI9ALzCFI-cObgnO4nQ2fu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Playlist videos"
    next: Pagination(continuation) parameter to get more playlist items, no need any other parameters if 'next' present.
Can be obtained from inside playlist request result.
        is_id: Playlist id
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/playlist"
    querystring = {}
    if next:
        querystring['next'] = next
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_videos_channels_playlists(upload_date: str='t', sort: str='v', features: str='li;hd', next: str='EogDEgVoZWxsbxr-AlNCU0NBUXRaVVVoeldFMW5iRU01UVlJQkMyMUlUMDVPWTFwaWQwUlpnZ0VMWW1VeE1rSkROWEJSVEVXQ0FRdFZNMEZUYWpGTU5sOXpXWUlCQzJaaGVrMVRRMXBuTFcxM2dnRUxaV3hrWldGSlFYWmZkMFdDQVExU1JGbFJTSE5ZVFdkc1F6bEJnZ0VMT0hwRVUybHJRMmc1Tm1PQ0FRc3pOMFU1VjNORWJVUmxaNElCQzJGaFNXcHpPRXN6YjFsdmdnRUxaMmRvUkZKS1ZuaEdlRldDQVF0clN6UXlURnB4VHpCM1FZSUJDME42VHpOaFNXVXdVbkJ6Z2dFTFNVNHdUMk5WZGtkaU5qQ0NBUXRSYTJWbGFGRTRSRjlXVFlJQkMyWk9NVU41Y2pCYVN6bE5nZ0VMZEZac1kwdHdNMkpYU0RpQ0FRdGZSQzFGT1Rsa01XSk1TWUlCQzJoQlUwNVRSSFZOY2pGUmdnRUxkREEzTVZkdE5EVnhWMDAlM0QYgeDoGCILc2VhcmNoLWZlZWQ%3D', hl: str='en', duration: str='s', gl: str='US', type: str='v', query: str='rick roll', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search any youtube content with all available filters"
    upload_date: Upload date filter.
Available options:
l - last hour;
t - today;
w - weak ago;
m - month ago;
y - year ago;
        sort: Result sort. Available options:
r - relevance;
ra - rating;
u - upload date;
v - view count;
        features: Video features.  Available options(could be joined by ';'):
h - hdr;
hd - hd;
s - subtitles;
c - cc;
3d - 3d;
3 - 360;
li - live;
lo - location;
4 - 4k;

        next: Pagination(continuation) parameter to get next result for same search query, no need any other parameters if 'next' present.
Can be obtained from inside search result.
        hl: Search language
        duration: Video duration. Available options:
s - short;
l - long;
        gl: Search location
        type: Search type. Available options:
v - video;
c - channel;
p - playlist;
        query: Search query you want to search
        
    """
    url = f"https://youtube-search-and-download.p.rapidapi.com/search"
    querystring = {}
    if upload_date:
        querystring['upload_date'] = upload_date
    if sort:
        querystring['sort'] = sort
    if features:
        querystring['features'] = features
    if next:
        querystring['next'] = next
    if hl:
        querystring['hl'] = hl
    if duration:
        querystring['duration'] = duration
    if gl:
        querystring['gl'] = gl
    if type:
        querystring['type'] = type
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

