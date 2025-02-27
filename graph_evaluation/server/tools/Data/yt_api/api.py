import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_channel_s_videos_playlists(query: str, x_cachebypass: str=None, token: str=None, geo: str=None, is_id: str='UCAuUUnT6oDeKwE6v1NGQxug', lang: str=None, forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search channel's videos and playlists, also get channel details.
		Quota cost is 1 unit."
    query: Search query
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        token: Pagination Token
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        is_id: Channel Id
        lang: Language code for localized results. Like en, gb, hi, etc
        forusername: If id not available then provide username.
Quota cost +1.

        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/search"
    querystring = {'query': query, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    if geo:
        querystring['geo'] = geo
    if is_id:
        querystring['id'] = is_id
    if lang:
        querystring['lang'] = lang
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about_channel(is_id: str, x_cachebypass: str=None, geo: str=None, lang: str=None, forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel details.
		Quota cost is 1 unit."
    id: Channel Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        lang: Language code for localized results. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/about"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if geo:
        querystring['geo'] = geo
    if lang:
        querystring['lang'] = lang
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_featured_channels(x_cachebypass: str=None, is_id: str='UCAuUUnT6oDeKwE6v1NGQxug', geo: str=None, token: str=None, forusername: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured channel listing and channel details.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        is_id: Channel Id
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        token: Pagination Token
        forusername: If id not available then provide username.
Quota cost +1.

        lang: Language code for localized results. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/channels"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if is_id:
        querystring['id'] = is_id
    if geo:
        querystring['geo'] = geo
    if token:
        querystring['token'] = token
    if forusername:
        querystring['forUsername'] = forusername
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_playlists(is_id: str, x_cachebypass: str=None, sort_by: str=None, lang: str=None, geo: str=None, token: str=None, forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist listing and channel details.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        sort_by: Sorts the channel playlists. Available options:
`date_added` [default]
`last_video_added`

        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/playlists"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if sort_by:
        querystring['sort_by'] = sort_by
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if token:
        querystring['token'] = token
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_livestreams(x_cachebypass: str=None, sort_by: str=None, lang: str=None, token: str=None, is_id: str='UCMhe8jmac-tWKU0Vj_OXDvA', geo: str=None, forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live stream listing and channel details.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        sort_by: Sorts the channel's live streams. Available options:
**newest** [default]
**popular**
**oldest**
        lang: Language code for localized results. Like en, gb, hi, etc

        token: Pagination Token
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        forusername: If id not available then provide username.
Quota cost +1.

        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/liveStreams"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if sort_by:
        querystring['sort_by'] = sort_by
    if lang:
        querystring['lang'] = lang
    if token:
        querystring['token'] = token
    if is_id:
        querystring['id'] = is_id
    if geo:
        querystring['geo'] = geo
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_shorts(x_cachebypass: str=None, token: str=None, sort_by: str=None, is_id: str='UCMhe8jmac-tWKU0Vj_OXDvA', lang: str=None, geo: str=None, forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shorts listing along with channel details
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        sort_by: Sorts the channel videos. Available options: 
**newest**  [default]
**oldest**  [deprecated]
**popular**
        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        forusername: If id not available then provide username.
Quota cost +1.

        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/shorts"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    if sort_by:
        querystring['sort_by'] = sort_by
    if is_id:
        querystring['id'] = is_id
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_videos(x_cachebypass: str=None, lang: str=None, geo: str=None, forusername: str=None, is_id: str='UCTwECeGqMZee77BjdoYtI2Q', sort_by: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel video listing and channel details.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Locale/language for request. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        forusername: If id not available then provide username.
Quota cost +1.

        is_id: Channel Id
        sort_by: Sorts the channel videos. Available options: 
**newest**  [default]
**popular**
**oldest**
        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/videos"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if forusername:
        querystring['forUsername'] = forusername
    if is_id:
        querystring['id'] = is_id
    if sort_by:
        querystring['sort_by'] = sort_by
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_home(x_cachebypass: str=None, lang: str=None, geo: str=None, forusername: str=None, is_id: str='UCAuUUnT6oDeKwE6v1NGQxug', token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel's Home Tab Listings.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.

        forusername: If id not available then provide username.
Quota cost +1.
        is_id: Channel Id
        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/home"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if forusername:
        querystring['forUsername'] = forusername
    if is_id:
        querystring['id'] = is_id
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, x_cachebypass: str=None, token: str=None, geo: str=None, type: str=None, lang: str=None, duration: str=None, upload_date: str=None, sort_by: str=None, features: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos, playlists, channels, or all of them.
		Quota cost is 2 units."
    query: Search term
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        token: Pagination Token
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        type: Search type filter options:
**video**
**channel**
**playlist**
**movie**
**show**
        lang: Locale/language for request. Like en, gb, hi, etc
        duration: Duration filter options:
**short**  - less than 4 min
**medium**  - 4 to 20 min
**long**  - more than 20 min
        upload_date: Upload Date filter options:
**hour**
**today**
**week**
**month**
**year**
        sort_by: Results Sort options:
**relevance**  [default]
**rating**
**date**
**views**
        features: Video Features options:
**HD**
**subtitles**
**CCommons**
**3D**
**Live**
**Purchased**
**4K**
**360**
**Location**
**HDR**
**VR180**

Multiple features could be joined by ','
For example: HD,subtitles
        
    """
    url = f"https://yt-api.p.rapidapi.com/search"
    querystring = {'query': query, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    if geo:
        querystring['geo'] = geo
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    if duration:
        querystring['duration'] = duration
    if upload_date:
        querystring['upload_date'] = upload_date
    if sort_by:
        querystring['sort_by'] = sort_by
    if features:
        querystring['features'] = features
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shorts_details(is_id: str, x_cachebypass: str=None, lang: str=None, params: str=None, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Youtube Shorts video info or get **Suggested** Shorts video info.
		Quota cost is 1 unit for Shorts Video Info
		Quota cost is 2 units for suggested Shorts Video Info."
    id: Available options:
- Shorts Video Id to get info or
- **WHATTOWATCH** for suggested Shorts along with info.
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        params: Shorts video param
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        
    """
    url = f"https://yt-api.p.rapidapi.com/shorts/info"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if params:
        querystring['params'] = params
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(is_id: str, x_cachebypass: str=None, lang: str=None, geo: str=None, extend: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the details of the YouTube video.
		Quota cost is 1 unit."
    id: Youtube video id or multiple ids separated by `,`
For mutli id batch request, quota cost is +1 for each extra id.
`Note: Mutli id batch request is experimental.`
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.

        extend: Extend the results to include more details like related videos/playlist, comment count, etc.
Available options:
1 = for likeCount, commentCount (~), subscriberCountText (~), relatedVideos [not supported for multi ids]
2 = for likeCount, commentCount, channelHandle

Quota cost is +1
        
    """
    url = f"https://yt-api.p.rapidapi.com/video/info"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if extend:
        querystring['extend'] = extend
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resolve_url(url: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resolve URLs like handles, etc.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://yt-api.p.rapidapi.com/resolve"
    querystring = {'url': url, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(geo: str, x_cachebypass: str=None, type: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending videos region-wise & niche-wise.
		Quota cost is 1 unit."
    geo: ISO 3166-2 country code of the region for which you want the trending data. Like US (default), UK, CA, IN, etc.
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        type: Trending type:
**now**
**music**
**games**
**movies**

Default is **now**
        lang: Locale/language for request. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/trending"
    querystring = {'geo': geo, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist(is_id: str, x_cachebypass: str=None, lang: str=None, geo: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist details and video listing.
		Quota cost is 1 unit."
    id: Playlist Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Locale/language for request. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.

        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/playlist"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_stream(is_id: str, cgeo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stream or download info.
		Quota cost is 5 units."
    id: Youtube Video Id or Shorts Id.
        cgeo: Country code in ISO 3166 format of the end user.
        
    """
    url = f"https://yt-api.p.rapidapi.com/dl"
    querystring = {'id': is_id, }
    if cgeo:
        querystring['cgeo'] = cgeo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_feed(x_cachebypass: str=None, geo: str=None, lang: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get home feeds, region-wise & niche-wise.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        lang: Locale/language for request. Like en, gb, hi, etc
        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/home"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if geo:
        querystring['geo'] = geo
    if lang:
        querystring['lang'] = lang
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_translate_download_subtitle(url: str, x_cachebypass: str=None, format: str=None, targetlang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert, translate, download the subtitle. 
		To get the subtitle url, use `Video -> Subtitles` endpoint or `Video -> Details` endpoint
		Quota cost is 1 unit."
    url: Provide the Subtitle url.
Available in the `Video -> Subtitles` or `Video -> Details` endpoint response.
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        format: Subtitle format options:
**json3** [mime: json]
**srv1** [mime: xml] [default]
**srv2** [mime: xml]
**srv3** [mime: xml]
**ttml** [mime: xml]
**vtt** [mime: text]
        targetlang: Translate to Language.
Valid codes are in the translationLanguages param of the `Video -> Subtitles` or `Video -> Info` endpoint response.
Or provide ISO-639 code of the language like es, zh-Hans, co, hi, etc

        
    """
    url = f"https://yt-api.p.rapidapi.com/subtitle"
    querystring = {'url': url, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if format:
        querystring['format'] = format
    if targetlang:
        querystring['targetLang'] = targetlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_comments(is_id: str, x_cachebypass: str=None, channelid: str=None, sort_by: str=None, token: str=None, geo: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post's comments or thread.
		Quota cost is 1 unit."
    id: Post Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        channelid: Channel Id
        sort_by: Available options:
**newest**
**top**  [default]
        token: Pagination token
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        lang: Locale/language for request. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/post/comments"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if channelid:
        querystring['channelId'] = channelid
    if sort_by:
        querystring['sort_by'] = sort_by
    if token:
        querystring['token'] = token
    if geo:
        querystring['geo'] = geo
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_details(is_id: str, x_cachebypass: str=None, channelid: str=None, geo: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post details.
		Quota cost is 1 unit."
    id: Post Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        channelid: Channel Id
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        lang: Locale/language for request. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/post/info"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if channelid:
        querystring['channelId'] = channelid
    if geo:
        querystring['geo'] = geo
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_community_posts(is_id: str, x_cachebypass: str=None, lang: str=None, token: str=None, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get community post listing and channel details.
		Quota cost is 1 unit."
    id: Channel Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        token: Pagination Token
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        
    """
    url = f"https://yt-api.p.rapidapi.com/channel/community"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if token:
        querystring['token'] = token
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subtitles(is_id: str, x_cachebypass: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available subtitles for the video.
		Quota cost is 1 unit."
    format: Subtitle format options:
**json3** [mime: json]
**srv1** [mime: xml] [default]
**srv2** [mime: xml]
**srv3** [mime: xml]
**ttml** [mime: xml]
**vtt** [mime: text]
        
    """
    url = f"https://yt-api.p.rapidapi.com/subtitles"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag(tag: str, x_cachebypass: str=None, lang: str=None, type: str=None, params: str=None, geo: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos or Shorts listings related to any hashtag.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        type: Available options are
**all**  for videos and shorts
**shorts** for only shorts
        params: Hashtag params
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        token: Pagination Token
        
    """
    url = f"https://yt-api.p.rapidapi.com/hashtag"
    querystring = {'tag': tag, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    if params:
        querystring['params'] = params
    if geo:
        querystring['geo'] = geo
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shorts_sound_attribution(x_cachebypass: str=None, lang: str=None, geo: str=None, params: str='8gU1CjMSMQoLMzFaR01oWjFlejgSCzMxWkdNaFoxZXo4GgtTQWoxZktNZVMyOCIICLiCBxICCCI%3D', token: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Shorts original sound attribution listing.
		Quota cost is 1 unit."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.

        params: Attribution param available in the Shorts info Endpoint:
soundAttribution -> params
        token: Pagination Token
        is_id: If the params value is not available then Shorts Video Id may be provided.
        
    """
    url = f"https://yt-api.p.rapidapi.com/shorts/soundAttribution"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if params:
        querystring['params'] = params
    if token:
        querystring['token'] = token
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shorts_sequence(x_cachebypass: str=None, lang: str=None, geo: str=None, params: str='GhEKCzBJNkZXMkZYX2I4GAAgASoCGA9CAGIEUkRTSA%3D%3D.Cgt4QTg3Z0ltOWdScyi56NqeBg%3D%3D', is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Shorts sequence.
		Quota cost is 2 units."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        lang: Language code for localized results. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        params: **Sequence param**
Provide either of these values:
- **sequenceContiuation** value from Shorts Info Endpoint's response for **WHATTOWATCH**
- **continuation** value from the previous request's response
        is_id: If the params value is not available then Shorts Video Id may be provided.
But it is not recommended.
        
    """
    url = f"https://yt-api.p.rapidapi.com/shorts/sequence"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    if params:
        querystring['params'] = params
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_videos_or_playlists(is_id: str, x_cachebypass: str=None, geo: str=None, token: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related videos or playlists.
		Quota cost is 1 unit."
    id: Video id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        token: Pagination token
        lang: Locale/language for request. Like en, gb, hi, etc
        
    """
    url = f"https://yt-api.p.rapidapi.com/related"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if geo:
        querystring['geo'] = geo
    if token:
        querystring['token'] = token
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(is_id: str, x_cachebypass: str=None, token: str=None, sort_by: str=None, lang: str=None, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video's or shorts' comments/thread.
		Quota cost is 1 unit."
    id: Video Id or Shorts Video Id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        token: Pagination token
        sort_by: Available options:
**newest**
**top**  [default]
        lang: Locale/language for request. Like en, gb, hi, etc
        geo: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        
    """
    url = f"https://yt-api.p.rapidapi.com/comments"
    querystring = {'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if token:
        querystring['token'] = token
    if sort_by:
        querystring['sort_by'] = sort_by
    if lang:
        querystring['lang'] = lang
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yt-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

