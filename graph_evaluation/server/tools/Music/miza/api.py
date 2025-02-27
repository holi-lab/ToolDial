import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_file(path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a file hosted by the server. This link is usually obtained by replacing the /p/ path in the /merge and /edit endpoint responses with /f/ or /d/. The request fails if the file requested does not exist or has been deleted."
    
    """
    url = f"https://miza.p.rapidapi.com/d/{path}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_address_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the host (server) and remote (your) IP addresses. A fast and responsive endpoint that is also handy for checking whether the service is online."
    
    """
    url = f"https://miza.p.rapidapi.com/ip"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_trimming_extending(url: str, start: str='-', fmt: str='mp3', end: str='45m', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trims a song or video, outputting as a URL (similar to the concatenation function). Like the output from concat, one must send a GET request to the URL returned in order to retrieve the actual output.
		Specify trim start and end after the URL, optionally omitting them by replacing them with an empty string or dash (-). Accepts input in number of seconds, as well as time formats such as *2h30m* or *1:53:30*.
		If the trim end passes the end of the song, it will automatically be extended by looping back from the beginning. This is *much* faster than concatenating multiple copies of the same song, as it skips re-encoding every single copy, allowing it to produce 10-hour extensions of songs or videos in seconds. Output format can additionally be specified after trim end.
		See https://mizabot.xyz/downloader for more info as well as test usage of this endpoint."
    
    """
    url = f"https://miza.p.rapidapi.com/ytdlp"
    querystring = {'url': url, }
    if start:
        querystring['start'] = start
    if fmt:
        querystring['fmt'] = fmt
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_search(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for a song from popular song platforms.
		The API will search YouTube for the item by default. Prepend *scsearch:* to the search term to specify a search from SoundCloud, *spsearch:* for Spotify, or *bcsearch:* for BandCamp.
		Unlike all other endpoints provided by this API, this endpoint can actually be used infinitely free of charge or quota. Simply direct your request directly to [http://i.mizabot.xyz/ytdl?search=](http://i.mizabot.xyz/ytdl?search=) instead of rapidapi's forwarded URL. This may be done for all other endpoints too, however there will be a rate limit in place."
    
    """
    url = f"https://miza.p.rapidapi.com/ytdl"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_concatenation(u1: str, u3: str='https://cdn.discordapp.com/attachments/688253918890688521/829266927003107338/paladin.ogg', fmt: str='ogg', u2: str='http://i.mizabot.xyz/d/Bc5SdZDrKg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Joins one or more audio or video files into a single file.
		Inputs are specified by URLs specified by query parameters "u1", "u2", etc. These URLs may be playlists, which will perform the concat operation on all contained tracks.
		Output format supports most video and audio formats. See https://mizabot.xyz/downloader for list of available formats, as well as test usage of this API.
		May or may not be the same input format, codec, or framerate/resolution/aspect ratio (if it is a video), but will take much longer to provide a response compared to the download endpoint, because it must finish all conversion before it can begin streaming.
		Redirects to a URL which will return the output file. This download link will be reusable, but temporary."
    
    """
    url = f"https://miza.p.rapidapi.com/ytdlc"
    querystring = {'u1': u1, }
    if u3:
        querystring['u3'] = u3
    if fmt:
        querystring['fmt'] = fmt
    if u2:
        querystring['u2'] = u2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_extraction(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces information on all songs in a playlist (YouTube, Spotify, SoundCloud, etc), with much faster response time than most other available tools. Currently O(1) for YouTube, O(log n) for Spotify, and O(log n) for SoundCloud. See [this article](http://i.mizabot.xyz/f/Bc9AUlCRkA) for the algorithm behind the YouTube playlists specifically."
    
    """
    url = f"https://miza.p.rapidapi.com/ytdl"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "miza.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

