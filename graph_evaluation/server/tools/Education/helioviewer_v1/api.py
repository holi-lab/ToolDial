import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettile(is_id: int, x: int, y: int, imagescale: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a single image tile to be used in the Helioviewer.org Viewport.  Tiles are 512x512 pixel PNG images, generated for a given image scale from the intermediary JPEG2000 image files.
		
		        Use the `getClosestImage` API endpoint to obtain the desired image identifier for the `id` parameter."
    id: Unique image identifier.
        x: Tile position.
        y: Tile position.
        imagescale: Image scale in arcseconds per pixel.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getTile/"
    querystring = {'id': is_id, 'x': x, 'y': y, 'imageScale': imagescale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkyoutubeauth(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check to see if Helioveiwer.org is authorized to interact with a user's YouTube account via the current browser session."
    callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/checkYouTubeAuth/"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def requeuemovie(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Re-generate a custom movie that is no longer cached on disk.  Once the movie has been successfully queued for regeneration, the Movie ID can be used to check on the status of the movie (via `getMovieStatus`) and to download the movie (via `downloadMovie`)."
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/reQueueMovie/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdatasources(verbose: bool=None, enable: str='[Yohkoh,STEREO_A,STEREO_B]', callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a hierarchial list of the available datasources.
		
		        Optional parameter `verbose` is exists for compatability with JHelioviewer.  It outputs the hierarchical list in an alternative format and limits the list of available datasources to a known set (SDO and SOHO).  JHelioviewer may not operate properly if new datasources appear in the feed without a client-side updgrade.  To explicitly include new sources, use the optional `enable` parameter."
    enable: Comma-separated list of observatories to enable.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getDataSources/"
    querystring = {}
    if verbose:
        querystring['verbose'] = verbose
    if enable:
        querystring['enable'] = enable
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getclosestimage(date: str, callback: str=None, sourceid: int=14, observatory: str='SDO', instrument: str='AIA', detector: str='AIA', measurement: str='335', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the image data that is closest to the requested date/time.  Return the associated metadata from the helioviewer database and the XML header of the JPEG2000 image file.
		
		        Either `sourceId` must be specified, or the combination of `observatory`, `instrument`, `detector`, and `measurement`."
    date: Desired date/time of the image. ISO 8601 combined UTC date and time UTC format.
        callback: Wrap the response object in a function call of your choosing.
        sourceid: Unique image datasource identifier.  Can be specified in lieu of the `observatory`, `instrument`, `detector`, and `measurement` parameters.
        observatory: Observatory name.
        instrument: Instrument name.
        detector: Detector name.
        measurement: Measurement name.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getClosestImage/"
    querystring = {'date': date, }
    if callback:
        querystring['callback'] = callback
    if sourceid:
        querystring['sourceId'] = sourceid
    if observatory:
        querystring['observatory'] = observatory
    if instrument:
        querystring['instrument'] = instrument
    if detector:
        querystring['detector'] = detector
    if measurement:
        querystring['measurement'] = measurement
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def queuemovie(starttime: str, endtime: str, layers: str, events: str, eventslabels: bool, imagescale: int, format: str='mp4', framerate: str='15', maxframes: str='300', scale: bool=None, scaletype: str='earth', scalex: int=-1000, scaley: int=-500, movielength: int=4, watermark: bool=None, width: str='1920', height: str='1200', x0: str='0', y0: str='0', x1: str='-5000', y1: str='-5000', x2: str='5000', y2: str='5000', callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a custom movie by submitting a request to the movie generation queue.  The response returned will provide you with a unique Movie ID that can be used to check on the status of your movie (via `getMovieStatus`) and to download your movie (via `downloadMovie`)."
    starttime: Desired date and time of the first frame of the movie. ISO 8601 combined UTC date and time UTC format.
        endtime: Desired date and time of the final frame of the movie. ISO 8601 combined UTC date and time UTC format.
        layers: Image datasource layer(s) to include in the movie.
        events: List feature/event types and FRMs to use to annoate the movie.  Use the empty string to indicate that no feature/event annotations should be shown.
        eventslabels: Optionally annotate each event marker with a text label.
        imagescale: Image scale in arcseconds per pixel.
        format: Movie format (`mp4`, `webm`, `flv`). Default value is `mp4`.
        framerate: Movie frames per second.  15 frames per second by default.
        maxframes: Maximum number of frames in the movie.  May be capped by the server.
        scale: Optionally overlay an image scale indicator.
        scaletype: Image scale indicator.
        scalex: Horizontal offset of the image scale indicator in arcseconds with respect to the center of the Sun.
        scaley: Vertical offset of the image scale indicator in arcseconds with respect to the center of the Sun.
        movielength: Movie length in seconds.
        watermark: Optionally overlay a Helioviewer.org watermark image.  Enabled by default.
        width: Width of the field of view in pixels. (Used in conjunction width `x0`,`y0`, and `height`).
        height: Height of the field of view in pixels. (Used in conjunction width `x0`,`y0`, and `width`).
        x0: The horizontal offset of the center of the field of view from the center of the Sun.  Used in conjunction with `y0`, `width`, and `height`.
        y0: The vertical offset of the center of the field of view from the center of the Sun.  Used in conjunction with `x0`, `width`, and `height`.
        x1: The horizontal offset of the top-left corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `y1`, `x2`, and `y2`.
        y1: The vertical offset of the top-left corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `x2`, and `y2`.
        x2: The horizontal offset of the bottom-right corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `y1`, and `y2`.
        y2: The vertical offset of the bottom-right corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `y1`, and `x2`.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/queueMovie/"
    querystring = {'startTime': starttime, 'endTime': endtime, 'layers': layers, 'events': events, 'eventsLabels': eventslabels, 'imageScale': imagescale, }
    if format:
        querystring['format'] = format
    if framerate:
        querystring['frameRate'] = framerate
    if maxframes:
        querystring['maxFrames'] = maxframes
    if scale:
        querystring['scale'] = scale
    if scaletype:
        querystring['scaleType'] = scaletype
    if scalex:
        querystring['scaleX'] = scalex
    if scaley:
        querystring['scaleY'] = scaley
    if movielength:
        querystring['movieLength'] = movielength
    if watermark:
        querystring['watermark'] = watermark
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if x0:
        querystring['x0'] = x0
    if y0:
        querystring['y0'] = y0
    if x1:
        querystring['x1'] = x1
    if y1:
        querystring['y1'] = y1
    if x2:
        querystring['x2'] = x2
    if y2:
        querystring['y2'] = y2
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shortenurl(querystring: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shorten a Helioviewer.org URL with the bit.ly URL shortening web service."
    querystring: URL-encoded query string.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/shortenURL/"
    querystring = {'queryString': querystring, }
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uploadmovietoyoutube(is_id: str, title: str='AIA 94 (2014-02-26 16:14:25 - 2014-02-27 15:37:49 UTC)', description: str='This movie was produced by Helioviewer.org. See the original at http://helioviewer.org/?movieId=y9tX5 or download a high-quality version from http://helioviewer.org/api/?action=downloadMovie&id=y9tX5&format=mp4&hq=true', tags: str='SDO,AIA,94', share: bool=None, html: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Share an existing user-generated movie to YouTube."
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        title: Movie title.
        description: Move description.
        tags: Movie keyword tags (comma separated).
        share: Optionally share the movie with the Helioviewer community.
        html: Optionally output response as HTML instead of JSON.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/uploadMovieToYouTube/"
    querystring = {'id': is_id, }
    if title:
        querystring['title'] = title
    if description:
        querystring['description'] = description
    if tags:
        querystring['tags'] = tags
    if share:
        querystring['share'] = share
    if html:
        querystring['html'] = html
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadmovie(is_id: str, format: str, hq: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a custom movie in one of three file formats."
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        format: Movie Format (`mp4`, `webm`, or `flv`).
        hq: Optionally download a higher-quality movie file (valid for .mp4 movies only, ignored otherwise).
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/downloadMovie/"
    querystring = {'id': is_id, 'format': format, }
    if hq:
        querystring['hq'] = hq
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadscreenshot(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a custom screenshot (that was generated using the `takeScreenshot` API endpoint)."
    id: Unique screenshot identifier (provided by the response to a `takeScreenshot` request).
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/downloadScreenshot/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getyoutubeauth(is_id: str, title: str, description: str, tags: str, share: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request authorization from the user via a Google / YouTube authorization flow to permit Helioviewer to upload videos on behalf of the user."
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        title: Movie title.
        description: Move description.
        tags: Movie keyword tags (comma separated).
        share: Optionally share the movie with the Helioviewer community.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getYouTubeAuth/"
    querystring = {'id': is_id, 'title': title, 'description': description, 'tags': tags, }
    if share:
        querystring['share'] = share
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playmovie(is_id: str, format: str, height: str, hq: bool=None, width: int=846, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Output an HTML web page with the requested movie embedded within."
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        format: Movie format (mp4, webm, or flv).
        height: Height of embedded movie player in pixels. Defaults to the actual height of the movie itself.
        hq: Optionally download a higher-quality movie file (valid for .mp4 movies only, ignored otherwise).
        width: Width of embedded movie player in pixels.  Defaults to the actual width of the movie itself.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/playMovie/"
    querystring = {'id': is_id, 'format': format, 'height': height, }
    if hq:
        querystring['hq'] = hq
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def takescreenshot(date: str, imagescale: int, layers: str, eventlabels: bool, events: str='[AR,HMI_HARP;SPoCA,1],[CH,all,1]', scale: bool=None, scaletype: str='earth', scalex: int=-1000, scaley: int=-500, width: str='1920', height: str='1200', x0: str='0', y0: str='0', x1: str='-5000', y1: str='-5000', x2: str='5000', y2: str='5000', display: bool=None, watermark: bool=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a custom screenshot.
		
		        You must specify values for either `x1`, `y1`, `x2`, and `y2`
		        or `x0`, `y0`, `width` and `height`.
		
		        By default, the response is a JSON object containing a unique screenshot identifier (`id`) that can be used to with the `downloadScreenshot` API endpoint.
		
		        Set the `display` parameter to `true` to directly return the screenshot as binary PNG image data in the response.
		
		        Please note that each request causes the server to generate a screenshot from scratch and is resource intensive.  For performance reasons, you should cache the response if you simply intend to serve exactly the same screenshot to multiple users."
    date: Desired date/time of the image. ISO 8601 combined UTC date and time UTC format.
        imagescale: Image scale in arcseconds per pixel.
        layers: Image datasource layer(s) to include in the screenshot.
        eventlabels: Optionally annotate each event marker with a text label.
        events: List feature/event types and FRMs to use to annoate the movie.  Use the empty string to indicate that no feature/event annotations should be shown.
        scale: Optionally overlay an image scale indicator.
        scaletype: Image scale indicator.
        scalex: Horizontal offset of the image scale indicator in arcseconds with respect to the center of the Sun.
        scaley: Vertical offset of the image scale indicator in arcseconds with respect to the center of the Sun.
        width: Width of the field of view in pixels. (Used in conjunction width `x0`,`y0`, and `height`).
        height: Height of the field of view in pixels. (Used in conjunction width `x0`,`y0`, and `width`).
        x0: The horizontal offset of the center of the field of view from the center of the Sun.  Used in conjunction with `y0`, `width`, and `height`.
        y0: The vertical offset of the center of the field of view from the center of the Sun.  Used in conjunction with `x0`, `width`, and `height`.
        x1: The horizontal offset of the top-left corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `y1`, `x2`, and `y2`.
        y1: The vertical offset of the top-left corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `x2`, and `y2`.
        x2: The horizontal offset of the bottom-right corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `y1`, and `y2`.
        y2: The vertical offset of the bottom-right corner of the field of view with respect to the center of the Sun (in arcseconds). Used in conjunction with `x1`, `y1`, and `x2`.
        display: Set to `true` to directly output binary PNG image data.  Default is `false` (which outputs a JSON object).
        watermark: Optionally overlay a watermark consisting of a Helioviewer logo and the datasource abbreviation(s) and timestamp(s) displayed in the screenshot.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/takeScreenshot/"
    querystring = {'date': date, 'imageScale': imagescale, 'layers': layers, 'eventLabels': eventlabels, }
    if events:
        querystring['events'] = events
    if scale:
        querystring['scale'] = scale
    if scaletype:
        querystring['scaleType'] = scaletype
    if scalex:
        querystring['scaleX'] = scalex
    if scaley:
        querystring['scaleY'] = scaley
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if x0:
        querystring['x0'] = x0
    if y0:
        querystring['y0'] = y0
    if x1:
        querystring['x1'] = x1
    if y1:
        querystring['y1'] = y1
    if x2:
        querystring['x2'] = x2
    if y2:
        querystring['y2'] = y2
    if display:
        querystring['display'] = display
    if watermark:
        querystring['watermark'] = watermark
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnewsfeed(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the XML RSS feed of the official Helioviewer Project Blog."
    callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getNewsFeed/"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuservideos(num: int=10, since: str=None, force: bool=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a listing (in descending time order) of the most recently user-generated movies that have been publicly shared to YouTube.  Result set is limited to the value requested or default value of the `num` parameter (unless truncated when the date value of the `since` parameter is reached)."
    num: Number of shared user-generated movies to include in the response.  Default is 10.
        since: Optionally truncate result set if this date/time is reached.  ISO 8601 combined UTC date and time UTC format.
        force: Optionally bypass cache to retrieve most up-to-date data.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getUserVideos/"
    querystring = {}
    if num:
        querystring['num'] = num
    if since:
        querystring['since'] = since
    if force:
        querystring['force'] = force
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getjp2header(is_id: int, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the XML header embedded in a JPEG2000 image. Includes the FITS header as well as a section of Helioviewer-specific metadata."
    id: Unique JP2 image identifier.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getJP2Header/"
    querystring = {'id': is_id, }
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getjp2image(date: str, sourceid: int=14, observatory: str='SDO', instrument: str='AIA', detector: str='AIA', measurement: str='335', jpip: bool=None, json: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a JP2 image for the specified datasource that is the closest match in time to the `date` requested.
		
		        Either `sourceId` must be specified, or the combination of `observatory`, `instrument`, `detector`, and `measurement`."
    date: Desired date/time of the JP2 image. ISO 8601 combined UTC date and time UTC format.
        sourceid: Unique image datasource identifier.
        observatory: Observatory name.
        instrument: Instrument name.
        detector: Detector name.
        measurement: Measurement name.
        jpip: Optionally return a JPIP URI instead of the binary data of the image itself.
        json: Optionally return a JSON object.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getJP2Image/"
    querystring = {'date': date, }
    if sourceid:
        querystring['sourceId'] = sourceid
    if observatory:
        querystring['observatory'] = observatory
    if instrument:
        querystring['instrument'] = instrument
    if detector:
        querystring['detector'] = detector
    if measurement:
        querystring['measurement'] = measurement
    if jpip:
        querystring['jpip'] = jpip
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getjpx(starttime: str, endtime: str, measurement: str='335', sourceid: int=14, observatory: str='SDO', instrument: str='AIA', detector: str='AIA', linked: bool=None, verbose: bool=None, jpip: bool=None, cadence: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate and (optionally) download a custom JPX movie of the specified datasource.
		
		        Either `sourceId` must be specified, or the combination of `observatory`, `instrument`, `detector`, and `measurement`."
    starttime: Date/Time for the beginning of the JPX movie data. ISO 8601 combined UTC date and time UTC format.
        endtime: Date/Time for the end of the JPX movie data.  ISO 8601 combined UTC date and time UTC format.
        measurement: Measurement name.
        sourceid: Unique image datasource identifier.  Can be specified in lieu of the `observatory`, `instrument`, `detector`, and `measurement` parameters.
        observatory: Observatory name.
        instrument: Instrument name.
        detector: Detector name.
        linked: Generate a `linked` JPX file containing image pointers instead of data for each individual frame in the series. Currently, only JPX image series support this feature.
        verbose: If set to `true,` the JSON response will include timestamps for each frame in the resulting movie and any warning messages associated with the request, in addition to the JPX movie file URI.
        jpip: Optionally return a JPIP URI string instead of the binary data of the movie itself, or instead of an HTTP URI in the JSON response (if `verbose` is set to `true`).
        cadence: The desired amount of time (in seconds) between each frame in the movie.

                If no cadence is specified, the server will attempt to select an optimal cadence.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getJPX/"
    querystring = {'startTime': starttime, 'endTime': endtime, }
    if measurement:
        querystring['measurement'] = measurement
    if sourceid:
        querystring['sourceId'] = sourceid
    if observatory:
        querystring['observatory'] = observatory
    if instrument:
        querystring['instrument'] = instrument
    if detector:
        querystring['detector'] = detector
    if linked:
        querystring['linked'] = linked
    if verbose:
        querystring['verbose'] = verbose
    if jpip:
        querystring['jpip'] = jpip
    if cadence:
        querystring['cadence'] = cadence
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmoviestatus(is_id: str, format: str, verbose: bool=None, callback: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: Unique movie identifier (provided by the response to a `queueMovie` request).
        format: Movie format (`mp4`, `webm`, or `flv`).
        verbose: Optionally include extra metadata in the response.
        callback: Wrap the response object in a function call of your choosing.
        
    """
    url = f"https://helioviewer-v1.p.rapidapi.com/api/v1/getMovieStatus/"
    querystring = {'id': is_id, 'format': format, }
    if verbose:
        querystring['verbose'] = verbose
    if callback:
        querystring['callback'] = callback
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helioviewer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

