import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def live_geo_v3(host: str, jsonp: str, path: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns geographical information about the most recent visitors. This call returns sampled data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        limit: The max number of recent data to return. Default: 100.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/geo/v3/"
    querystring = {'host': host, 'jsonp': jsonp, 'path': path, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_engagement_series(jsonp: str, human: str, start: str, end: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns series of the engaged seconds where the default time span of each data point is 5 minutes. You should use this call if you want to see a more granular picture of your data."
    jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: The start of today.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        limit: An integer or string that specifies the number of snapshots to return. e.g. 100 or time span from start to return snapshots for e.g. 10minutes, 3days, respectively. Default: the entire time span between start and end.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/engagement/series/"
    querystring = {'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_social_stats(host: str, jsonp: str, human: str, start: str, end: str, properties: str, fields: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns social activity summarized over the given timespan. You should use this call if you want to discover broad information about your data where we already did the work providing the max, min, average or median."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: 30 days ago.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        properties: The stat to apply to the data. Can be one or a comma separated list of max, min, avg, median. Default: max.
        fields: One or a comma separated list of: tw_url_mentions: how many times your site's url is mentioned on Twitter in the given time span. tw_handle_mentions: the amount of times your Twitter handle is mentioned on Twitter in the given time span. fb_page_likes: the number of likes on your Facebook page in the given time span. fb_domain_activity: the number of times someone likes or shares a page on your website through the Facebook widget in the given time span. Default: tw_url_mentions,fb_domain_activity.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/social/stats/"
    querystring = {'host': host, 'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'properties': properties, 'fields': fields, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_social_series(jsonp: str, human: str, start: str, end: str, limit: str, fields: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns series of the social activity where the default time span between each data point is 5 minutes. You should use this call if you want to see a more granular picture of your data."
    jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: The start of today.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        limit: An integer or string that specifies the number of snapshots to return. e.g. 100 or time span from start to return snapshots for e.g. 10minutes, 3days, respectively. Default: the entire time span between start and end.
        fields: One or a comma separated list of: tw_url_mentions: how many times your site's url is mentioned on Twitter in the given time span. tw_handle_mentions: the amount of times your Twitter handle is mentioned on Twitter in the given time span. fb_page_likes: the number of likes on your Facebook page in the given time span. fb_domain_activity: the number of times someone likes or shares a page on your website through the Facebook widget in the given time span. Default: tw_url_mentions,fb_domain_activity.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/social/series/"
    querystring = {'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'limit': limit, 'fields': fields, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_histogram_v3(host: str, jsonp: str, keys: str, breaks: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a histogram where field values are summarized within ranges, as specified by breaks. The resulting histogram is returned as a list with one more element than the number of breaks. This call returns real-time data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        keys: The keys represent what data to return. To save on space, short names are used for values in several of the real-time API calls. One or a comma separated list of: pagetimer: Time to finish loading the dom. time_spent: Number of seconds on the page. domain: The domain name of the document (what's in the browser bar). uid: The chartbeat account. host: The reported domain (the dashboard the data goes to). title: Page title. new: First time visitor for the site in the last 30 days. path: Path of the page from location.pathname. referrer: Referrer from document.referrer. token: Temporary uuidevent's page session (regenerated when moving to another page). user: User token. window_height: window.innerHeight or document.body.offsetHeight. scroll_top: window.pageYOffset or document.body.scrollTop or document.documentElement.scrollTop. page_height: document.body.scrollHeight. read: The number of people reading. write: The number of people writing. idle: The number of people idle.
        breaks: How to break the histogram data. e.g. 1,2,10 would return values from 0-1,1-2,2-10,10+. This function only works with numeric field values.
        path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/histogram/v3"
    querystring = {'host': host, 'jsonp': jsonp, 'keys': keys, 'breaks': breaks, 'path': path, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_engagement_stats(jsonp: str, human: str, start: str, end: str, properties: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns engaged seconds summarized over the given timespan. You should use this call if you want to discover broad information about your data where we already did the work providing the max or min."
    jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: 30 days ago.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        properties: The stat to apply to the data. Can be one or a comma separated list of max, min. Default: max.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/engagement/stats/"
    querystring = {'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'properties': properties, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_summary_v3(host: str, jsonp: str, keys: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns either numeric or categorical summaries of event fields given a host and optional path. Numeric summaries include min, max, sum, nonzero observations, observations and sum of squares. Categorical summaries include field keys with associated counts. This call return real-time data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        keys: The keys represent what data to return. To save on space, short names are used for values in several of the real-time API calls. One or a comma separated list of: pagetimer: Time to finish loading the dom. time_spent: Number of seconds on the page. domain: The domain name of the document (what's in the browser bar). uid: The chartbeat account. host: The reported domain (the dashboard the data goes to). title: Page title. new: First time visitor for the site in the last 30 days. path: Path of the page from location.pathname. referrer: Referrer from document.referrer. token: Temporary uuidevent's page session (regenerated when moving to another page). user: User token. window_height: window.innerHeight or document.body.offsetHeight. scroll_top: window.pageYOffset or document.body.scrollTop or document.documentElement.scrollTop. page_height: document.body.scrollHeight. read: The number of people reading. write: The number of people writing. idle: The number of people idle.
        path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/summary/v3"
    querystring = {'host': host, 'jsonp': jsonp, 'keys': keys, 'path': path, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_traffic_series(host: str, jsonp: str, human: str, start: str, end: str, limit: str, fields: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns series of the traffic sources and/or page load time where the default time span of each data point is 5 minutes. You should use this call if you want to see a more granular picture of your data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: The start of today.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        limit: An integer or string that specifies the number of snapshots to return. e.g. 100 or time span from start to return snapshots for e.g. 10minutes, 3days, respectively. Default: the entire time span between start and end.
        fields: One or a comma separated list of: return: the number of returning visitors. new: the number of new visitors. people: the number of people on the domain. read: the number of people reading on the domain. domload: the DOM load time. engaged_time_avg: the average enagaged time. write: the number of people writing on the domain. idle: the number of people idle on the domain. internal: the number of people coming from an internal referrer. social: the number of people coming from social services. Default: people.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/traffic/series/"
    querystring = {'host': host, 'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'limit': limit, 'fields': fields, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_quick_stats_v3(host: str, jsonp: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an overview summary of the people on your domain right now. The number of people on a page, reading, writing, idle, etc... This call returns real-time data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/quickstats/v3"
    querystring = {'host': host, 'jsonp': jsonp, 'path': path, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_top_pages_v3(host: str, jsonp: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of pages, ordered by which ones have most visitors right now. This call returns real-time data."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in/
        limit: limit		 The max number of pages to return. Default: 10.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/toppages/v3"
    querystring = {'host': host, 'jsonp': jsonp, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_referrers_v3(host: str, jsonp: str, path: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of referrers for the domain."
    host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        limit: The max number of referrers to return. Default: 10.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/referrers/v3/"
    querystring = {'host': host, 'jsonp': jsonp, 'path': path, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_path_summary_v3(jsonp: str, host: str, keys: str, types: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns summary data (refer to the summary API call), but for top paths of given a host. This call returns real-time data."
    jsonp: The name of a function to wrap the return data in.
        host: The domain of the site you would like to query represented as a string.
        keys: The keys represent what data to return. To save on space, short names are used for values in several of the real-time API calls. One or a comma separated list of: pagetimer: Time to finish loading the dom. time_spent: Number of seconds on the page. domain: The domain name of the document (what's in the browser bar). uid: The chartbeat account. host: The reported domain (the dashboard the data goes to). title: Page title. new: First time visitor for the site in the last 30 days. path: Path of the page from location.pathname. referrer: Referrer from document.referrer. token: Temporary uuidevent's page session (regenerated when moving to another page). user: User token. window_height: window.innerHeight or document.body.offsetHeight. scroll_top: window.pageYOffset or document.body.scrollTop or document.documentElement.scrollTop. page_height: document.body.scrollHeight. read: The number of people reading. write: The number of people writing. idle: The number of people idle.
        types: How to return the key data, either n or s (numeric or string). Default: n
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/pathsummary/v3"
    querystring = {'jsonp': jsonp, 'host': host, 'keys': keys, 'types': types, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_recent_v3(path: str, host: str, jsonp: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about the most recent visitors to a given host. This call returns real-time data."
    path: A specific path. If not given, data is from all paths. e.g. the path of http://chartbeat.com/signup/ is /signup/.
        host: The domain of the site you would like to query represented as a string.
        jsonp: The name of a function to wrap the return data in.
        limit: The max number of recent visitors to return. Default: 50.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/live/recent/v3"
    querystring = {'path': path, 'host': host, 'jsonp': jsonp, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_traffic_stats(jsonp: str, human: str, start: str, end: str, properties: str, fields: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns traffic sources and/or page load time summarized over the given timespan. You should use this call if you want to discover broad information about your data where we already did the work providing the max or min."
    jsonp: The name of a function to wrap the return data in.
        human: A boolean that tells the api call to return human readable start and end time in the form YYYY-mm-dd HH:MM:SS, as opposed to the unix timestamp. Default: false.
        start: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS or a time delta, where the time delta specified is start time prior to now. NOTE: start is only accepted in EST. Default: 30 days ago.
        end: A string in the form of a unix timestamp, YYYY-mm-dd, YY-mm-dd HH:MM:SS. NOTE: end is only accepted in EST. Default: The end of today.
        properties: The stat to apply to the data. Can be one or a comma separated list of max, min. Default: max.
        fields: One or a comma separated list of: return: the number of returning visitors. new: the number of new visitors. people: the number of people on the domain. read: the number of people reading on the domain. domload: the DOM load time. engaged_time_avg: the average enagaged time. write: the number of people writing on the domain. idle: the number of people idle on the domain. internal: the number of people coming from an internal referrer. social: the number of people coming from social services. Default: people.
        
    """
    url = f"https://mashape-community-chartbeat.p.rapidapi.com/historical/traffic/stats/"
    querystring = {'jsonp': jsonp, 'human': human, 'start': start, 'end': end, 'properties': properties, 'fields': fields, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-chartbeat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

