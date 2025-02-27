import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def subscriptions(part: str, maxresults: int=5, pagetoken: str=None, forchannelid: str=None, order: str='relevance', is_id: str=None, channelid: str='UCP4bf6IHJJQehibu6ai__cg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns subscription resources that match the API request criteria."
    part: The **part** parameter specifies a comma-separated list of one or more **subscription** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **subscription** resource, the **snippet** property contains other properties, such as a display title for the subscription. If you set **part=snippet**, the API response will also contain all of those nested properties.

The following list contains the **part** names that you can include in the parameter value:

- contentDetails
- id
- snippet
- subscriberSnippet
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        forchannelid: The **forChannelId** parameter specifies a comma-separated list of channel IDs. The API response will then only contain subscriptions matching those channels.
        order: The **order** parameter specifies the method that will be used to sort resources in the API response. The default value is **SUBSCRIPTION_ORDER_RELEVANCE**.

Acceptable values are:

- **alphabetical** – Sort alphabetically.
- **relevance** – Sort by relevance.
- **unread** – Sort by order of activity.
        is_id: The **id** parameter specifies a comma-separated list of the YouTube subscription ID(s) for the resource(s) that are being retrieved. In a **subscription** resource, the **id** property specifies the YouTube subscription ID.
        channelid: The **channelId** parameter specifies a YouTube channel ID. The API will only return that channel's subscriptions.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/subscriptions/"
    querystring = {'part': part, }
    if maxresults:
        querystring['maxResults'] = maxresults
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if forchannelid:
        querystring['forChannelId'] = forchannelid
    if order:
        querystring['order'] = order
    if is_id:
        querystring['id'] = is_id
    if channelid:
        querystring['channelId'] = channelid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channelsections(part: str, channelid: str='UC_x5XG1OV2P6uZZ5FSM9Ttw', hl: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of `channelSection` resources that match the API request criteria."
    part: The **part** parameter specifies a comma-separated list of one or more **channelSection** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **channelSection** resource, the **snippet** property contains other properties, such as a display title for the section. If you set **part=snippet**, the API response will also contain all of those nested properties.

The following list contains the **part** names that you can include in the parameter value:

- contentDetails
- id
- snippet
        channelid: The **channelId** parameter specifies a YouTube channel ID. If a request specifies a value for this parameter, the API will only return the specified channel's sections.
        hl: *This parameter has been deprecated.* The **hl** parameter provided support for retrieving localized metadata for a channel section. However, this functionality has been deprecated in YouTube Studio and in the YouTube app.
        is_id: The **id** parameter specifies a comma-separated list of IDs that uniquely identify the **channelSection** resources that are being retrieved. In a **channelSection** resource, the **id** property specifies the section's ID.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/channelSections/"
    querystring = {'part': part, }
    if channelid:
        querystring['channelId'] = channelid
    if hl:
        querystring['hl'] = hl
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channels(part: str, hl: str=None, categoryid: str=None, maxresults: int=5, forusername: str='GoogleDevelopers', pagetoken: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of zero or more `channel` resources that match the request criteria."
    part: The **part** parameter specifies a comma-separated list of one or more **channel** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **channel** resource, the **contentDetails** property contains other properties, such as the **uploads** properties. As such, if you set **part=contentDetails**, the API response will also contain all of those nested properties.

The following list contains the **part** names that you can include in the parameter value:

- auditDetails
- brandingSettings
- contentDetails
- contentOwnerDetails
- id
- localizations
- snippet
- statistics
- status
- topicDetails
        hl: The **hl** parameter instructs the API to retrieve localized resource metadata for a specific application language that the YouTube website supports. The parameter value must be a language code included in the list returned by the `i18nLanguages` method.

If localized resource details are available in that language, the resource's **snippet.localized** object will contain the localized values. However, if localized details are not available, the **snippet.localized** object will contain resource details in the resource's default language.
        categoryid: *This parameter has been deprecated.* The **categoryId** parameter specified a YouTube guide category and could be used to request YouTube channels associated with that category.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        forusername: The **forUsername** parameter specifies a YouTube username, thereby requesting the channel associated with that username.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        is_id: The **id** parameter specifies a comma-separated list of the YouTube channel ID(s) for the resource(s) that are being retrieved. In a **channel** resource, the **id** property specifies the channel's YouTube channel ID.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/channels/"
    querystring = {'part': part, }
    if hl:
        querystring['hl'] = hl
    if categoryid:
        querystring['categoryId'] = categoryid
    if maxresults:
        querystring['maxResults'] = maxresults
    if forusername:
        querystring['forUsername'] = forusername
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activities(part: str, channelid: str, regioncode: str=None, publishedbefore: str=None, maxresults: int=5, publishedafter: str=None, pagetoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of channel activity events that match the request criteria. For example, you can retrieve events associated with a particular channel or with the user's own channel."
    part: The **part** parameter specifies a comma-separated list of one or more **activity** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in an **activity** resource, the **snippet** property contains other properties that identify the type of activity, a display title for the activity, and so forth. If you set **part=snippet**, the API response will also contain all of those nested properties.

The following list contains the part names that you can include in the parameter value:

- contentDetails
- id
- snippet
        channelid: The **channelId** parameter specifies a unique YouTube channel ID. The API will then return a list of that channel's activities.
        regioncode: The **regionCode** parameter instructs the API to return results for the specified country. The parameter value is an [ISO 3166-1 alpha-2](http://www.iso.org/iso/country_codes/iso_3166_code_lists/country_names_and_code_elements.htm) country code. YouTube uses this value when the authorized user's previous activity on YouTube does not provide enough information to generate the activity feed.
        publishedbefore: The publishedBefore parameter specifies the date and time before which an activity must have occurred for that activity to be included in the API response. If the parameter value specifies a day, but not a time, then any activities that occurred that day will be excluded from the result set. The value is specified in [ISO 8601](https://www.w3.org/TR/NOTE-datetime) (**YYYY-MM-DDThh:mm:ss.sZ**) format.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        publishedafter: The **publishedAfter** parameter specifies the earliest date and time that an activity could have occurred for that activity to be included in the API response. If the parameter value specifies a day, but not a time, then any activities that occurred that day will be included in the result set. The value is specified in [ISO 8601](https://www.w3.org/TR/NOTE-datetime) (**YYYY-MM-DDThh:mm:ss.sZ**) format.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/activities/"
    querystring = {'part': part, 'channelId': channelid, }
    if regioncode:
        querystring['regionCode'] = regioncode
    if publishedbefore:
        querystring['publishedBefore'] = publishedbefore
    if maxresults:
        querystring['maxResults'] = maxresults
    if publishedafter:
        querystring['publishedAfter'] = publishedafter
    if pagetoken:
        querystring['pageToken'] = pagetoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos(part: str, videocategoryid: str=None, maxheight: int=None, maxresults: int=5, is_id: str='Q8TXgCzxEnw', pagetoken: str=None, regioncode: str=None, maxwidth: int=None, chart: str=None, hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of videos that match the API request parameters."
    part: The **part** parameter specifies a comma-separated list of one or more **video** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **video** resource, the **snippet** property contains the **channelId**, **title**, **description**, **tags** and **categoryId** properties. As such, if you set **part=snippet**, the API response will contain all of those properties.

The following list contains the **part** names that you can include in the parameter value:

- contentDetails
- fileDetails
- id
- liveStreamingDetails
- localizations
- player
- processingDetails
- recordingDetails
- snippet
- statistics
- status
- suggestions
- topicDetails
        videocategoryid: The **videoCategoryId** parameter identifies the video category for which the chart should be retrieved. This parameter can only be used in conjunction with the **chart** parameter. By default, charts are not restricted to a particular category. The default value is **0**.
        maxheight: The **maxHeight** parameter specifies the maximum height of the embedded player returned in the **player.embedHtml** property. You can use this parameter to specify that instead of the default dimensions, the embed code should use a height appropriate for your application layout. If the **maxWidth** parameter is also provided, the player may be shorter than the **maxHeight** in order to not violate the maximum width. Acceptable values are **72** to **8192**, inclusive.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set.

**Note:** This parameter is supported for use in conjunction with the **myRating** parameter, but it is not supported for use in conjunction with the **id** parameter. Acceptable values are **1** to **50**, inclusive. The default value is **5**.
        is_id: The **id** parameter specifies a comma-separated list of the YouTube video ID(s) for the resource(s) that are being retrieved. In a **video** resource, the **id** property specifies the video's ID.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.

**Note:** This parameter is supported for use in conjunction with the **myRating** parameter, but it is not supported for use in conjunction with the **id** parameter.
        regioncode: The **regionCode** parameter instructs the API to select a video chart available in the specified region. This parameter can only be used in conjunction with the **chart** parameter. The parameter value is an ISO 3166-1 alpha-2 country code.
        maxwidth: The **maxWidth** parameter specifies the maximum width of the embedded player returned in the **player.embedHtml** property. You can use this parameter to specify that instead of the default dimensions, the embed code should use a width appropriate for your application layout.

If the **maxHeight** parameter is also provided, the player may be narrower than **maxWidth** in order to not violate the maximum height. Acceptable values are **72** to **8192**, inclusive.
        chart: The **chart** parameter identifies the chart that you want to retrieve.

Acceptable values are:

- **mostPopular** – Return the most popular videos for the specified content region and video category.
        hl: The **hl** parameter instructs the API to retrieve localized resource metadata for a specific application language that the YouTube website supports. The parameter value must be a language code included in the list returned by the `i18nLanguages` method.

If localized resource details are available in that language, the resource's **snippet.localized** object will contain the localized values. However, if localized details are not available, the **snippet.localized** object will contain resource details in the resource's default language.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/videos/"
    querystring = {'part': part, }
    if videocategoryid:
        querystring['videoCategoryId'] = videocategoryid
    if maxheight:
        querystring['maxHeight'] = maxheight
    if maxresults:
        querystring['maxResults'] = maxresults
    if is_id:
        querystring['id'] = is_id
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if regioncode:
        querystring['regionCode'] = regioncode
    if maxwidth:
        querystring['maxWidth'] = maxwidth
    if chart:
        querystring['chart'] = chart
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(part: str, videolicense: str=None, relatedtovideoid: str=None, videotype: str=None, videosyndicated: str=None, videoembeddable: str=None, videoduration: str=None, videodimension: str=None, videodefinition: str=None, safesearch: str='moderate', topicid: str=None, videocaption: str=None, pagetoken: str=None, videocategoryid: str=None, type: str='video,channel,playlist', regioncode: str=None, relevancelanguage: str=None, q: str=None, publishedbefore: str=None, publishedafter: str=None, order: str='relevance', maxresults: int=5, locationradius: str=None, channeltype: str=None, eventtype: str=None, channelid: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of search results that match the query parameters specified in the API request. By default, a search result set identifies matching `video`, `channel` and `playlist` resources, but you can also configure queries to only retrieve a specific type of resource."
    part: The **part** parameter specifies a comma-separated list of one or more **search** resource properties that the API response will include. Set the parameter value to **snippet**.
        videolicense: The **videoLicense** parameter filters search results to only include videos with a particular license. YouTube lets video uploaders choose to attach either the Creative Commons license or the standard YouTube license to each of their videos. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Return all videos, regardless of which license they have, that match the query parameters.
- **creativeCommon** – Only return videos that have a Creative Commons license. Users can reuse videos with this license in other videos that they create. [Learn more](http://www.google.com/support/youtube/bin/answer.py?answer=1284989).
- **youtube** – Only return videos that have the standard YouTube license.
        relatedtovideoid: The **relatedToVideoId** parameter retrieves a list of videos that are related to the video that the parameter value identifies. The parameter value must be set to a YouTube video ID and, if you are using this parameter, the **type** parameter must be set to **video**.

Note that if the **relatedToVideoId** parameter is set, the only other supported parameters are **part**, **maxResults**, **pageToken**, **regionCode**, **relevanceLanguage**, **safeSearch**, **type** (which must be set to **video**) and **fields**.
        videotype: The **videoType** parameter lets you restrict a search to a particular type of videos. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Return all videos.
- **episode** – Only retrieve episodes of shows.
- **movie** – Only retrieve movies.
        videosyndicated: The **videoSyndicated** parameter lets you to restrict a search to only videos that can be played outside youtube.com. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Return all videos, syndicated or not.
- **true** – Only retrieve syndicated videos.2
        videoembeddable: The **videoEmbeddable** parameter lets you to restrict a search to only videos that can be embedded into a webpage. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Return all videos, embeddable or not.
- **true** – Only retrieve embeddable videos.
        videoduration: The **videoDuration** parameter filters video search results based on their duration. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Do not filter video search results based on their duration. This is the default value.
- **long** – Only include videos longer than 20 minutes.
- **medium** – Only include videos that are between four and 20 minutes long (inclusive).
- **short** – Only include videos that are less than four minutes long.
        videodimension: The **videoDimension** parameter lets you restrict a search to only retrieve 2D or 3D videos. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **2d** – Restrict search results to exclude 3D videos.
- **3d** – Restrict search results to only include 3D videos.
- **any** – Include both 3D and non-3D videos in returned results. This is the default value.
        videodefinition: The **videoDefinition** parameter lets you restrict a search to only include either high definition (HD) or standard definition (SD) videos. HD videos are available for playback in at least 720p, though higher resolutions, like 1080p, might also be available. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Return all videos, regardless of their resolution.
- **high** – Only retrieve HD videos.
- **standard** – Only retrieve videos in standard definition.
        safesearch: The **safeSearch** parameter indicates whether the search results should include restricted content as well as standard content.

Acceptable values are:

- **moderate** – YouTube will filter some content from search results and, at the least, will filter content that is restricted in your locale. Based on their content, search results could be removed from search results or demoted in search results. This is the default parameter value.
- **none** – YouTube will not filter the search result set.
- **strict** – YouTube will try to exclude all restricted content from the search result set. Based on their content, search results could be removed from search results or demoted in search results.
        topicid: The **topicId** parameter indicates that the API response should only contain resources associated with the specified topic. The value identifies a Freebase topic ID.

**Important:** Due to the deprecation of Freebase and the Freebase API, the **topicId** parameter started working differently as of February 27, 2017. At that time, YouTube started supporting a small set of curated topic IDs, and you can only use that smaller set of IDs as values for this parameter.
        videocaption: The **videoCaption** parameter indicates whether the API should filter video search results based on whether they have captions. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **any** – Do not filter results based on caption availability.
- **closedCaption** – Only include videos that have captions.
- **none** – Only include videos that do not have captions.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        videocategoryid: The **videoCategoryId** parameter filters video search results based on their category. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.
        type: The **type** parameter restricts a search query to only retrieve a particular type of resource. The value is a comma-separated list of resource types. The default value is **video,channel,playlist**.

Acceptable values are:

- channel
- playlist
- video
        regioncode: The **regionCode** parameter instructs the API to return search results for videos that can be viewed in the specified country. The parameter value is an [ISO 3166-1 alpha-2](http://www.iso.org/iso/country_codes/iso_3166_code_lists/country_names_and_code_elements.htm) country code.
        relevancelanguage: The **relevanceLanguage** parameter instructs the API to return search results that are most relevant to the specified language. The parameter value is typically an [ISO 639-1 two-letter language code](http://www.loc.gov/standards/iso639-2/php/code_list.php). However, you should use the values **zh-Hans** for simplified Chinese and **zh-Hant** for traditional Chinese. Please note that results in other languages will still be returned if they are highly relevant to the search query term.
        q: The **q** parameter specifies the query term to search for.

Your request can also use the Boolean NOT (-) and OR (|) operators to exclude videos or to find videos that are associated with one of several search terms. For example, to search for videos matching either 'boating' or 'sailing', set the **q** parameter value to **boating|sailing**. Similarly, to search for videos matching either 'boating' or 'sailing' but not 'fishing', set the **q** parameter value to **boating|sailing -fishing**. Note that the pipe character must be URL-escaped when it is sent in your API request. The URL-escaped value for the pipe character is **%7C**.
        publishedbefore: The **publishedBefore** parameter indicates that the API response should only contain resources created before or at the specified time. The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
        publishedafter: The **publishedAfter** parameter indicates that the API response should only contain resources created at or after the specified time. The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
        order: The **order** parameter specifies the method that will be used to order resources in the API response. The default value is **relevance**.

Acceptable values are:

- **date** – Resources are sorted in reverse chronological order based on the date they were created.
- **rating** – Resources are sorted from highest to lowest rating.
- **relevance** – Resources are sorted based on their relevance to the search query. This is the default value for this parameter.
- **title** – Resources are sorted alphabetically by title.
- **videoCount** – Channels are sorted in descending order of their number of uploaded videos.
- **viewCount** – Resources are sorted from highest to lowest number of views. For live broadcasts, videos are sorted by number of concurrent viewers while the broadcasts are ongoing.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        locationradius: The **locationRadius** parameter, in conjunction with the **location** parameter, defines a circular geographic area.

The parameter value must be a floating point number followed by a measurement unit. Valid measurement units are **m**, **km**, **ft** and **mi**. For example, valid parameter values include **1500m**, **5km**, **10000ft** and **0.75mi**. The API does not support **locationRadius** parameter values larger than 1000 kilometers.

**Note:** See the definition of the **location** parameter for more information.
        channeltype: The **channelType** parameter lets you restrict a search to a particular type of channel.

Acceptable values are:

- **any** – Return all channels.
- **show** – Only retrieve shows.
        eventtype: The **eventType** parameter restricts a search to broadcast events. If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.

Acceptable values are:

- **completed** – Only include completed broadcasts.
- **live** – Only include active broadcasts.
- **upcoming** – Only include upcoming broadcasts.
        channelid: The **channelId** parameter indicates that the API response should only contain resources created by the channel.

**Note:** Search results are constrained to a maximum of 500 videos if your request specifies a value for the **channelId** parameter and sets the **type** parameter value to **video**, but it does not also set one of the **forContentOwner**, **forDeveloper** or **forMine** filters.
        location: The **location** parameter, in conjunction with the **locationRadius** parameter, defines a circular geographic area and also restricts a search to videos that specify, in their metadata, a geographic location that falls within that area. The parameter value is a string that specifies latitude/longitude coordinates e.g. (**37.42307,-122.08427**).

- The **location** parameter value identifies the point at the center of the area.
- The **locationRadius** parameter specifies the maximum distance that the location associated with a video can be from that point for the video to still be included in the search results.

The API returns an error if your request specifies a value for the **location** parameter but does not also specify a value for the **locationRadius** parameter.

Note: If you specify a value for this parameter, you must also set the **type** parameter's value to **video**.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/search/"
    querystring = {'part': part, }
    if videolicense:
        querystring['videoLicense'] = videolicense
    if relatedtovideoid:
        querystring['relatedToVideoId'] = relatedtovideoid
    if videotype:
        querystring['videoType'] = videotype
    if videosyndicated:
        querystring['videoSyndicated'] = videosyndicated
    if videoembeddable:
        querystring['videoEmbeddable'] = videoembeddable
    if videoduration:
        querystring['videoDuration'] = videoduration
    if videodimension:
        querystring['videoDimension'] = videodimension
    if videodefinition:
        querystring['videoDefinition'] = videodefinition
    if safesearch:
        querystring['safeSearch'] = safesearch
    if topicid:
        querystring['topicId'] = topicid
    if videocaption:
        querystring['videoCaption'] = videocaption
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if videocategoryid:
        querystring['videoCategoryId'] = videocategoryid
    if type:
        querystring['type'] = type
    if regioncode:
        querystring['regionCode'] = regioncode
    if relevancelanguage:
        querystring['relevanceLanguage'] = relevancelanguage
    if q:
        querystring['q'] = q
    if publishedbefore:
        querystring['publishedBefore'] = publishedbefore
    if publishedafter:
        querystring['publishedAfter'] = publishedafter
    if order:
        querystring['order'] = order
    if maxresults:
        querystring['maxResults'] = maxresults
    if locationradius:
        querystring['locationRadius'] = locationradius
    if channeltype:
        querystring['channelType'] = channeltype
    if eventtype:
        querystring['eventType'] = eventtype
    if channelid:
        querystring['channelId'] = channelid
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlists(part: str, maxresults: int=5, pagetoken: str=None, is_id: str=None, channelid: str='UC_x5XG1OV2P6uZZ5FSM9Ttw', hl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of playlists that match the API request parameters. For example, you can retrieve all playlists that the authenticated user owns, or you can retrieve one or more playlists by their unique IDs."
    part: The **part** parameter specifies a comma-separated list of one or more **playlist** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **playlist** resource, the **snippet** property contains properties like **author**, **title**, **description** and **timeCreated**. As such, if you set **part=snippet**, the API response will contain all of those properties.

The following list contains the **part** names that you can include in the parameter value:

- contentDetails
- id
- localizations
- player
- snippet
- status
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        is_id: The **id** parameter specifies a comma-separated list of the YouTube playlist ID(s) for the resource(s) that are being retrieved. In a **playlist** resource, the **id** property specifies the playlist's YouTube playlist ID.
        channelid: This value indicates that the API should only return the specified channel's playlists.
        hl: The **hl** parameter instructs the API to retrieve localized resource metadata for a specific application language that the YouTube website supports. The parameter value must be a language code included in the list returned by the `i18nLanguages` method.

If localized resource details are available in that language, the resource's **snippet.localized** object will contain the localized values. However, if localized details are not available, the **snippet.localized** object will contain resource details in the resource's default language.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/playlists/"
    querystring = {'part': part, }
    if maxresults:
        querystring['maxResults'] = maxresults
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if is_id:
        querystring['id'] = is_id
    if channelid:
        querystring['channelId'] = channelid
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlistitems(part: str, videoid: str=None, pagetoken: str=None, is_id: str=None, maxresults: int=5, playlistid: str='PLOU2XLYxmsILMUsDRrVoRRlL29p8LHFIT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of playlist items that match the API request parameters. You can retrieve all of the playlist items in a specified playlist or retrieve one or more playlist items by their unique IDs."
    part: The **part** parameter specifies a comma-separated list of one or more **playlistItem** resource properties that the API response will include.

If the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a **playlistItem** resource, the **snippet** property contains numerous fields, including the **title**, **description**, **position** and **resourceId** properties. As such, if you set **part=snippet**, the API response will contain all of those properties.

The following list contains the **part** names that you can include in the parameter value:

- contentDetails
- id
- snippet
- status
        videoid: The **videoId** parameter specifies that the request should return only the playlist items that contain the specified video.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** and **prevPageToken** properties identify other pages that could be retrieved.
        is_id: The **id** parameter specifies a comma-separated list of one or more unique playlist item IDs.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are **0** to **50**, inclusive. The default value is **5**.
        playlistid: The **playlistId** parameter specifies the unique ID of the playlist for which you want to retrieve playlist items. Note that even though this is an optional parameter, every request to retrieve playlist items must specify a value for either the **id** parameter or the **playlistId** parameter.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/playlistItems/"
    querystring = {'part': part, }
    if videoid:
        querystring['videoId'] = videoid
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if is_id:
        querystring['id'] = is_id
    if maxresults:
        querystring['maxResults'] = maxresults
    if playlistid:
        querystring['playlistId'] = playlistid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commentthreads(part: str, searchterms: str=None, textformat: str='html', pagetoken: str=None, channelid: str=None, maxresults: int=20, order: str='time', videoid: str='Q8TXgCzxEnw', allthreadsrelatedtochannelid: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of comment threads that match the API request parameters."
    part: The **part** parameter specifies a comma-separated list of one or more **commentThread** resource properties that the API response will include.

The following list contains the **part** names that you can include in the parameter value:

- id
- replies
- snippet
        searchterms: The **searchTerms** parameter instructs the API to limit the API response to only contain comments that contain the specified search terms.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter.
        textformat: Set this parameter's value to **html** or **plainText** to instruct the API to return the comments left by users in html formatted or in plain text. The default value is **html**.

Acceptable values are:

- **html** – Returns the comments in HTML format. This is the default value.
- **plainText** – Returns the comments in plain text format.
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** property identifies the next page of the result that can be retrieved.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter.
        channelid: The **channelId** parameter instructs the API to return comment threads containing comments about the specified channel. (The response will not include comments left on videos that the channel uploaded.)
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter. Acceptable values are **1** to **100**, inclusive. The default value is **20**.
        order: The **order** parameter specifies the order in which the API response should list comment threads. Valid values are:

- **time** - Comment threads are ordered by time. This is the default behavior.
- **relevance** - Comment threads are ordered by relevance.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter.
        videoid: The **videoId** parameter instructs the API to return comment threads associated with the specified video ID.
        allthreadsrelatedtochannelid: The **allThreadsRelatedToChannelId** parameter instructs the API to return all comment threads associated with the specified channel. The response can include comments about the channel or about the channel's videos.
        is_id: The **id** parameter specifies a comma-separated list of comment thread IDs for the resources that should be retrieved.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/commentThreads/"
    querystring = {'part': part, }
    if searchterms:
        querystring['searchTerms'] = searchterms
    if textformat:
        querystring['textFormat'] = textformat
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if channelid:
        querystring['channelId'] = channelid
    if maxresults:
        querystring['maxResults'] = maxresults
    if order:
        querystring['order'] = order
    if videoid:
        querystring['videoId'] = videoid
    if allthreadsrelatedtochannelid:
        querystring['allThreadsRelatedToChannelId'] = allthreadsrelatedtochannelid
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i18nlanguages(part: str, hl: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of application languages that the YouTube website supports."
    part: The **part** parameter specifies the **i18nLanguage** resource properties that the API response will include. Set the parameter value to **snippet**.
        hl: The **hl** parameter specifies the language that should be used for text values in the API response. The default value is **en_US**.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/i18nLanguages/"
    querystring = {'part': part, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(part: str, pagetoken: str=None, textformat: str='html', parentid: str='Ugi79__QZLYRu3gCoAEC', maxresults: int=20, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of comments that match the API request parameters."
    part: The **part** parameter specifies a comma-separated list of one or more `comment` resource properties that the API response will include.

The following list contains the **part** names that you can include in the parameter value:

- id
- snippet
        pagetoken: The **pageToken** parameter identifies a specific page in the result set that should be returned. In an API response, the **nextPageToken** property identifies the next page of the result that can be retrieved.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter.
        textformat: This parameter indicates whether the API should return comments formatted as HTML or as plain text. The default value is **html**.

Acceptable values are:

- **html** – Returns the comments in HTML format. This is the default value.
- **plainText** – Returns the comments in plain text format.
        parentid: The **parentId** parameter specifies the ID of the comment for which replies should be retrieved.

**Note:** YouTube currently supports replies only for top-level comments. However, replies to replies may be supported in the future.
        maxresults: The **maxResults** parameter specifies the maximum number of items that should be returned in the result set.

**Note:** This parameter is not supported for use in conjunction with the **id** parameter. Acceptable values are **1** to **100**, inclusive. The default value is **20**.
        is_id: The **id** parameter specifies a comma-separated list of comment IDs for the resources that are being retrieved. In a **comment** resource, the **id** property specifies the comment's ID.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/comments/"
    querystring = {'part': part, }
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if textformat:
        querystring['textFormat'] = textformat
    if parentid:
        querystring['parentId'] = parentid
    if maxresults:
        querystring['maxResults'] = maxresults
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i18nregions(part: str, hl: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of content regions that the YouTube website supports."
    part: The **part** parameter specifies the **i18nRegion** resource properties that the API response will include. Set the parameter value to **snippet**.
        hl: The **hl** parameter specifies the language that should be used for text values in the API response. The default value is **en_US**.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/i18nRegions/"
    querystring = {'part': part, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videocategories(part: str, is_id: str=None, regioncode: str='US', hl: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of categories that can be associated with YouTube videos."
    part: The **part** parameter specifies the **videoCategory** resource properties that the API response will include. Set the parameter value to **snippet**.
        is_id: The **id** parameter specifies a comma-separated list of video category IDs for the resources that you are retrieving.
        regioncode: The **regionCode** parameter instructs the API to return the list of video categories available in the specified country. The parameter value is an [ISO 3166-1 alpha-2](http://www.iso.org/iso/country_codes/iso_3166_code_lists/country_names_and_code_elements.htm) country code.
        hl: The **hl** parameter specifies the language that should be used for text values in the API response. The default value is **en_US**.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/videoCategories/"
    querystring = {'part': part, }
    if is_id:
        querystring['id'] = is_id
    if regioncode:
        querystring['regionCode'] = regioncode
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def captions(part: str, videoid: str, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of caption tracks that are associated with a specified video."
    part: The **part** parameter specifies the **caption** resource parts that the API response will include.

The list below contains the **part** names that you can include in the parameter value:

- id
- snippet
        videoid: The **videoId** parameter specifies the YouTube video ID of the video for which the API should return caption tracks.
        is_id: The **id** parameter specifies a comma-separated list of IDs that identify the **caption** resources that should be retrieved. Each ID must identify a caption track associated with the specified video.
        
    """
    url = f"https://youtube-v311.p.rapidapi.com/captions/"
    querystring = {'part': part, 'videoId': videoid, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v311.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

