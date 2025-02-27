import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_channels(keyword: str='Rick Astley', lang: str=None, sortby: str=None, nexttoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for YouTube channels. Pagination scraping is supported."
    keyword: Keyword for search.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        sortby: Sorting metrics. Defaults to `relevance`.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `keyword` and `sortBy` will be ignored.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/search/channels"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if lang:
        querystring['lang'] = lang
    if sortby:
        querystring['sortBy'] = sortby
    if nexttoken:
        querystring['nextToken'] = nexttoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_details(channelid: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches details of a YouTube channel."
    channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/details"
    querystring = {'channelId': channelid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_post_comments(nexttoken: str=None, lang: str=None, postid: str='Ugkx-rW0UIVSt9Aw-ux-w16DlRW-wwKwfwnp', sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists comments of a YouTube community post. Pagination scraping is supported."
    nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `postId` and `sortBy` will be ignored.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        sortby: Sorting metrics. Defaults to `top`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/post/comments"
    querystring = {}
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if lang:
        querystring['lang'] = lang
    if postid:
        querystring['postId'] = postid
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_video_comments(videoid: str='dQw4w9WgXcQ', lang: str=None, nexttoken: str=None, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists comments of a YouTube video. Pagination scraping is supported."
    videoid: YouTube video id. The value of `v` in YouTube player URL query parameters.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `videoId` and `sortBy` will be ignored.
        sortby: Sorting metrics. Defaults to `top`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/comments"
    querystring = {}
    if videoid:
        querystring['videoId'] = videoid
    if lang:
        querystring['lang'] = lang
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_channel_videos_shorts_live(sortby: str=None, lang: str=None, type: str=None, channelid: str='UCeY0bbntWzzVIaj2z3QigXg', nexttoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists videos, shorts or live streams of a YouTube channel. Pagination scraping is supported."
    sortby: Sorting metrics. Defaults to `newest`. `oldest` is only applicable if `type` is not `shorts`.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        type: Video type. Defaults to `videos`.
        channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `channelId`, `type` and `sortBy` will be ignored.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/videos"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    if channelid:
        querystring['channelId'] = channelid
    if nexttoken:
        querystring['nextToken'] = nexttoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_channel_playlists_releases_podcasts(lang: str=None, sortby: str=None, channelid: str='UCuAXFkgsw1L7xaCfnd5JJOw', nexttoken: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists playlists, releases or podcasts of a YouTube channel. Pagination scraping is supported."
    lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        sortby: Sorting metrics. Defaults to `dateAdded`.
        channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `channelId`, `type` and `sortBy` will be ignored.
        type: * `playlists` - **Playlists** (default value)
* `releases` - **Releases** (`sortBy` will be omitted)
* `podcasts` - **Podcasts** (`sortBy` will be omitted)
* ~~`created`~~ - Deprecated. An alias for `playlists`.
* ~~`album`~~ - Deprecated. An alias for `releases`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/playlists"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if sortby:
        querystring['sortBy'] = sortby
    if channelid:
        querystring['channelId'] = channelid
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_channel_videos_and_playlists(nexttoken: str=None, channelid: str='UCuAXFkgsw1L7xaCfnd5JJOw', keyword: str='rickroll', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for videos and playlists in a YouTube Channel. Pagination scraping is supported."
    nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `channelId` and `keyword` will be ignored.
        channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        keyword: Keyword for search.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/search"
    querystring = {}
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if channelid:
        querystring['channelId'] = channelid
    if keyword:
        querystring['keyword'] = keyword
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_featured_channels(channelid: str='UCTUPsBBQA4Am8k23BYETRJg', nexttoken: str=None, type: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists subscribed or featured channels of a YouTube channel. Pagination scraping is supported."
    channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `channelId` and `type` will be ignored.
        type: * `subscribed` - **Subscriptions** (default value)
* `featured` - **Featured Channels**
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/channels"
    querystring = {}
    if channelid:
        querystring['channelId'] = channelid
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_channel_posts_poll_video_image(lang: str=None, channelid: str='UCY2ekMrWhsUVHwO3J3-PCzQ', nexttoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists poll, video, or image posts of a YouTube channel. Pagination scraping is supported."
    lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        channelid: Channel ID, custom URL name or handle. `@` is required as a prefix for a channel handle.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `channelId` will be ignored.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/channel/posts"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if channelid:
        querystring['channelId'] = channelid
    if nexttoken:
        querystring['nextToken'] = nexttoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_hashtag_videos(nexttoken: str=None, type: str=None, tag: str='meme', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists YouTube videos related to the hashtag. Pagination scraping is supported."
    nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `tag` and `type` will be ignored.
        type: Video type. Defaults to `all`.
        tag: A hashtag without `#`.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/hashtag/videos"
    querystring = {}
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if type:
        querystring['type'] = type
    if tag:
        querystring['tag'] = tag
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post_details(postid: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches details of a YouTube community post."
    lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/post/details"
    querystring = {'postId': postid, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_playlist_videos(nexttoken: str=None, playlistid: str='PLeCdlPO-XhWFzEVynMsmosfdRsIZXhZi0', lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists **available** videos of a YouTube playlist (unavailable ones won't be listed by YouTube). Pagination scraping is supported. Thumbnails won't be blurred by age safety."
    nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `playlistId` will be ignored.
        lang: Language code (ISO-639) for localized results. Default to be `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/playlist/videos"
    querystring = {}
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if playlistid:
        querystring['playlistId'] = playlistid
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_details(playlistid: str, videos: bool=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches details of a YouTube playlist (user created playlist, album or radio playlist)."
    videos: Whether to list the first page of videos. Default to be `true`.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/playlist/details"
    querystring = {'playlistId': playlistid, }
    if videos:
        querystring['videos'] = videos
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_playlists(nexttoken: str=None, keyword: str='Rick Astley', lang: str=None, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for YouTube playlists. Pagination scraping is supported. Thumbnails will not be blurred by age safety."
    nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `keyword` and `sortBy` will be ignored.
        keyword: Keyword for search.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        sortby: Sorting metrics. Defaults to `relevance`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/search/playlists"
    querystring = {}
    if nexttoken:
        querystring['nextToken'] = nexttoken
    if keyword:
        querystring['keyword'] = keyword
    if lang:
        querystring['lang'] = lang
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_videos_movies(movie: bool=None, duration: str=None, sortby: str=None, keyword: str='Rick Astley', lang: str=None, uploaddate: str=None, nexttoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for YouTube videos (movies) with optional filters. Pagination scraping is supported. Thumbnails will not be blurred by age safety."
    movie: Search for movies only. Defaults to `false`.
        duration: * `all` - **No duration limit** (default value)
* `short` - **Under 4 minutes**
* `medium` - **4 - 20 minutes**
* `long` - **Over 20 minutes**
        sortby: Sorting metrics. Defaults to `relevance`.
        keyword: Keyword for search.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        uploaddate: Upload date. Defaults to `all`.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `keyword`, `movie`, `uploadDate`, `duration` and `sortBy` will be ignored.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/search/videos"
    querystring = {}
    if movie:
        querystring['movie'] = movie
    if duration:
        querystring['duration'] = duration
    if sortby:
        querystring['sortBy'] = sortby
    if keyword:
        querystring['keyword'] = keyword
    if lang:
        querystring['lang'] = lang
    if uploaddate:
        querystring['uploadDate'] = uploaddate
    if nexttoken:
        querystring['nextToken'] = nexttoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_autocomplete_suggestions(keyword: str, lang: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists autocomplete predictions depending on the keyword."
    keyword: Keyword for search.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`.
        region: Region code (ISO 3166 alpha-2) for localized results. Defaults to `US`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/search/suggestions"
    querystring = {'keyword': keyword, }
    if lang:
        querystring['lang'] = lang
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_related_videos_and_playlists(videoid: str='dQw4w9WgXcQ', lang: str=None, nexttoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists related videos and playlists of a YouTube video. Pagination scraping is supported. It's recommended to get the first page by calling `Video > Get Video Details`, and then get subsequent pages here."
    videoid: YouTube video id. The value of `v` in YouTube player URL query parameters.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        nexttoken: A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `videoId` will be ignored.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/related"
    querystring = {}
    if videoid:
        querystring['videoId'] = videoid
    if lang:
        querystring['lang'] = lang
    if nexttoken:
        querystring['nextToken'] = nexttoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_details(videoid: str, related: bool=None, lang: str=None, audios: bool=None, videos: bool=None, subtitles: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a YouTube video, including URLs of videos, audios, thumbnails and subtitles as well as related videos and playlists."
    videoid: YouTube video id. The value of `v` in YouTube player URL query parameters.
        related: Whether to get information of related videos and playlists. Defaults to `true`.
        lang: Language code (ISO-639) for localized results. Defaults to `en-US`. Unsupported code will **fallback** to `en-US`.
        audios: Whether to get audio URLs. Defaults to `true`.
        videos: Whether to get video URLs. Defaults to `true`.
        subtitles: Whether to get subtitle URLs. Defaults to `true`.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
    querystring = {'videoId': videoid, }
    if related:
        querystring['related'] = related
    if lang:
        querystring['lang'] = lang
    if audios:
        querystring['audios'] = audios
    if videos:
        querystring['videos'] = videos
    if subtitles:
        querystring['subtitles'] = subtitles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translate_convert_download_subtitle(subtitleurl: str, format: str=None, fixoverlap: bool=None, targetlang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you translate, convert and download a subtitle of a YouTube video. Before doing this, please call endpoint `Video > Get Video Details` to obtain subtitle URLs."
    subtitleurl: Subtitle URL of a YouTube video. To get this, please call `Video > Get Video Details` first.
        format: Subtitle format. Defaults to `srt`.
        fixoverlap: Whether to fix overlapping subtitles. Defaults to `true`. Useful for auto-generated subtitles.
        targetlang: Target language (ISO-639 code) into which the subtitle will be translated. Leave blank to preserve the original language. Unsupported code will **fallback** to the original language.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles"
    querystring = {'subtitleUrl': subtitleurl, }
    if format:
        querystring['format'] = format
    if fixoverlap:
        querystring['fixOverlap'] = fixoverlap
    if targetlang:
        querystring['targetLang'] = targetlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translate_convert_download_subtitle(subtitleurl: str, format: str=None, fixoverlap: bool=None, targetlang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you translate, convert and download a subtitle of a YouTube video. Before doing this, please call endpoint `Video > Get Video Details` to obtain subtitle URLs."
    subtitleurl: Subtitle URL of a YouTube video. To get this, please call `Video > Get Video Details` first.
        format: Subtitle format. Defaults to `srt`.
        fixoverlap: Whether to fix overlapping subtitles. Defaults to `true`. Useful for auto-generated subtitles.
        targetlang: Target language (ISO-639 code) into which the subtitle will be translated. Leave blank to preserve the original language. Unsupported code will **fallback** to the original language.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles"
    querystring = {'subtitleUrl': subtitleurl, }
    if format:
        querystring['format'] = format
    if fixoverlap:
        querystring['fixOverlap'] = fixoverlap
    if targetlang:
        querystring['targetLang'] = targetlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translate_convert_download_subtitle(subtitleurl: str, format: str=None, fixoverlap: bool=None, targetlang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you translate, convert and download a subtitle of a YouTube video. Before doing this, please call endpoint `Video > Get Video Details` to obtain subtitle URLs."
    subtitleurl: Subtitle URL of a YouTube video. To get this, please call `Video > Get Video Details` first.
        format: Subtitle format. Defaults to `srt`.
        fixoverlap: Whether to fix overlapping subtitles. Defaults to `true`. Useful for auto-generated subtitles.
        targetlang: Target language (ISO-639 code) into which the subtitle will be translated. Leave blank to preserve the original language. Unsupported code will **fallback** to the original language.
        
    """
    url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles"
    querystring = {'subtitleUrl': subtitleurl, }
    if format:
        querystring['format'] = format
    if fixoverlap:
        querystring['fixOverlap'] = fixoverlap
    if targetlang:
        querystring['targetLang'] = targetlang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-media-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

