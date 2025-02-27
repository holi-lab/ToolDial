import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_multiple_video(video_group_id: str, video_id: str, format: str, sid: str=None, add_ons: str=None, auto_start: str=None, external_data: str=None, height: str='401', multiple_thumbnails: str=None, num_related_return: str=None, page: str='1', return_category_id: str=None, show_renditions: str=None, third_layer: str=None, thumbnail_sizes: str=None, transcript: str=None, width: str='480', video_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To access the data for a specific video, you simply request a URL from Aol On Network's API with the specific video ID you want."
    video_group_id: Id of video group you want to get.
        video_id: Can be retrieved from the end of the video URL.
        format: Xml or Json (Not all json calls are supported)
        sid: Account associated syndicator ID
        add_ons: Include/exclude add-ons for video
        auto_start: The video starts playing automatically
        external_data: Get external video identification
        height: Embed height value
        multiple_thumbnails: Include/exclude all thumbnails for video
        num_related_return: Number of related videos to return per video
        page: Which page of results to display
        return_category_id: Add ID of category per video
        show_renditions: Show all available renditions for the video
        third_layer: Include third layer metadata
        thumbnail_sizes: Include/exclude thumbnails sizes extracted for video
        transcript: Include/exclude transcript for video
        width: Embed width value
        video_ids: Comma separated video IDs.
        
    """
    url = f"https://community-aol-on-network.p.rapidapi.com/video/{VIDEO_ID}/info.{FORMAT}"
    querystring = {'Video_Group_Id': video_group_id, 'video_id': video_id, 'format': format, }
    if sid:
        querystring['sid'] = sid
    if add_ons:
        querystring['add_ons'] = add_ons
    if auto_start:
        querystring['auto_start'] = auto_start
    if external_data:
        querystring['external_data'] = external_data
    if height:
        querystring['height'] = height
    if multiple_thumbnails:
        querystring['multiple_thumbnails'] = multiple_thumbnails
    if num_related_return:
        querystring['num_related_return'] = num_related_return
    if page:
        querystring['page'] = page
    if return_category_id:
        querystring['return_category_id'] = return_category_id
    if show_renditions:
        querystring['show_renditions'] = show_renditions
    if third_layer:
        querystring['third_layer'] = third_layer
    if thumbnail_sizes:
        querystring['thumbnail_sizes'] = thumbnail_sizes
    if transcript:
        querystring['transcript'] = transcript
    if width:
        querystring['width'] = width
    if video_ids:
        querystring['video_ids'] = video_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-aol-on-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searching_for_videos(sort: str, search_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search our library using this request:"
    sort: most_viewed / top_rated / most_discussed / most_favorited / most_recent / featured / relevance / approval_date
        search_term: The search_term can either be free text, or 3 words separated by spaces for density search.
        
    """
    url = f"https://community-aol-on-network.p.rapidapi.com/search/{SEARCH_TERM}/videos.xml"
    querystring = {'sort': sort, 'search_term': search_term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-aol-on-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

