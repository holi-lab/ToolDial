import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feed(to: str, is_from: str, cid: str, force: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of posts with all metrics for the period.
		
		The effectiveness of hashtags, post types and text length
		
		![](https://36627.selcdn.ru/jagajam-static/000000012_e1767a2c-31de-4501-9279-661f7c03e8c1_f.png?time=1666776603)"
    force: true - for priority data analysis and frequent statistics updates
        
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/posts"
    querystring = {'to': to, 'from': is_from, 'cid': cid, }
    if force:
        querystring['force'] = force
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full profile information with statistics and audience demographics.
		Category/Interests, Gender, Age, Location, Followers, Engagement Rate, Fake Followers, Contact email, Quality Score, Verified, Average Interactions, Average Likes, Average Comments, Average Reels Views, Average Reels Likes, Average Reels Comments, Audience genders, Audience Ages, Audience Locations, Lasts Posts, Last Mentions, ...
		
		![](https://36627.selcdn.ru/jagajam-static/000000012_c1e975d2-1d9a-4caa-96b7-b7752981c4bd_f.png?time=1687506046)"
    
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/community"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(perpage: int, iscontactemail: bool=None, isverified: bool=None, minaudiencegenderspercent: str=None, minaudienceagepercent: str=None, minaudiencelocationspercent: str=None, maxinteractions: str=None, mininteractions: str=None, tracktotal: bool=None, minfakefollowers: str=None, maxvideoviews: str=None, minqualityscore: str=None, maxqualityscore: str=None, maxvideocomments: str=None, maxfakefollowers: str=None, maxvideolikes: str=None, minvideolikes: str=None, minvideoviews: str=None, minvideocomments: str=None, maxcomments: str=None, minlikes: str=None, maxlikes: str=None, minaudienceage: str=None, mincomments: str=None, maxaudienceage: str=None, audiencelocations: str=None, audienceages: str=None, maxuserscount: str=None, ages: str=None, genders: str=None, locations: str='united-states', q: str=None, sort: str='-score', tags: str=None, maxviews: str=None, audiencegenders: str=None, socialtypes: str='INST,FB', miner: str=None, minage: str=None, maxage: str=None, minviews: str=None, minuserscount: str=None, maxer: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search in the database of brands and influencers by parameters:
		- Keywords in the title, description, account link
		- Filtering by categories, account types, interests and other tags
		- Demographics of subscribers (country, city, gender, age)
		- Social network
		- Number of subscribers
		- Average Engagement per post
		- Average number of post views
		- Sort by relevance, percentage of selected audience or number of selected audience"
    ages: Account age (for influencer)
0_18
18_21
21_24
24_27
27_30
30_35
35_45
45_100
        genders: Account gender (for influencer)
m - Male
f - Female
        locations: Account location (country or city)
from Tags
        sort: -score - by Relevance
-usersCount - by Followers
-avgViews - by Views
-avgER - by Engagement Rate
-qualityScore - by Quality Score
        tags: from Tags
        socialtypes: INST - Instagram
FB - Facebook
TW - Twitter
YT - YouTube
TT - TikTok
TG - Telegram
        
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/search"
    querystring = {'perPage': perpage, }
    if iscontactemail:
        querystring['isContactEmail'] = iscontactemail
    if isverified:
        querystring['isVerified'] = isverified
    if minaudiencegenderspercent:
        querystring['minAudienceGendersPercent'] = minaudiencegenderspercent
    if minaudienceagepercent:
        querystring['minAudienceAgePercent'] = minaudienceagepercent
    if minaudiencelocationspercent:
        querystring['minAudienceLocationsPercent'] = minaudiencelocationspercent
    if maxinteractions:
        querystring['maxInteractions'] = maxinteractions
    if mininteractions:
        querystring['minInteractions'] = mininteractions
    if tracktotal:
        querystring['trackTotal'] = tracktotal
    if minfakefollowers:
        querystring['minFakeFollowers'] = minfakefollowers
    if maxvideoviews:
        querystring['maxVideoViews'] = maxvideoviews
    if minqualityscore:
        querystring['minQualityScore'] = minqualityscore
    if maxqualityscore:
        querystring['maxQualityScore'] = maxqualityscore
    if maxvideocomments:
        querystring['maxVideoComments'] = maxvideocomments
    if maxfakefollowers:
        querystring['maxFakeFollowers'] = maxfakefollowers
    if maxvideolikes:
        querystring['maxVideoLikes'] = maxvideolikes
    if minvideolikes:
        querystring['minVideoLikes'] = minvideolikes
    if minvideoviews:
        querystring['minVideoViews'] = minvideoviews
    if minvideocomments:
        querystring['minVideoComments'] = minvideocomments
    if maxcomments:
        querystring['maxComments'] = maxcomments
    if minlikes:
        querystring['minLikes'] = minlikes
    if maxlikes:
        querystring['maxLikes'] = maxlikes
    if minaudienceage:
        querystring['minAudienceAge'] = minaudienceage
    if mincomments:
        querystring['minComments'] = mincomments
    if maxaudienceage:
        querystring['maxAudienceAge'] = maxaudienceage
    if audiencelocations:
        querystring['audienceLocations'] = audiencelocations
    if audienceages:
        querystring['audienceAges'] = audienceages
    if maxuserscount:
        querystring['maxUsersCount'] = maxuserscount
    if ages:
        querystring['ages'] = ages
    if genders:
        querystring['genders'] = genders
    if locations:
        querystring['locations'] = locations
    if q:
        querystring['q'] = q
    if sort:
        querystring['sort'] = sort
    if tags:
        querystring['tags'] = tags
    if maxviews:
        querystring['maxViews'] = maxviews
    if audiencegenders:
        querystring['audienceGenders'] = audiencegenders
    if socialtypes:
        querystring['socialTypes'] = socialtypes
    if miner:
        querystring['minER'] = miner
    if minage:
        querystring['minAge'] = minage
    if maxage:
        querystring['maxAge'] = maxage
    if minviews:
        querystring['minViews'] = minviews
    if minuserscount:
        querystring['minUsersCount'] = minuserscount
    if maxer:
        querystring['maxER'] = maxer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_by_id(cid: str, force: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full profile information with statistics and audience demographics.
		Category/Interests, Gender, Age, Location, Followers, Engagement Rate, Fake Followers, Contact email, Quality Score, Verified, Average Interactions, Average Likes, Average Comments, Average Reels Views, Average Reels Likes, Average Reels Comments, Audience genders, Audience Ages, Audience Locations, Lasts Posts, Last Mentions, ...
		
		![](https://36627.selcdn.ru/jagajam-static/000000012_c1e975d2-1d9a-4caa-96b7-b7752981c4bd_f.png?time=1687506046)"
    force: true - for priority data analysis and frequent statistics updates
        
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/community"
    querystring = {'cid': cid, }
    if force:
        querystring['force'] = force
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activity(cid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns data for plotting the activity time graph of account users. Helps to understand when it is better to publish content and make integrations with influencers
		
		![](https://36627.selcdn.ru/jagajam-static/000000012_df890402-1ba3-4da4-855b-84c4f5e43df6_f.png?time=1666777428)"
    
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/statistics/activity"
    querystring = {'cid': cid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrospective(to: str, is_from: str, cid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the history of the number of subscribers, posts, interactions, likes, comments, reposts, engagement for the selected period by day and in total for the period
		
		![](https://36627.selcdn.ru/jagajam-static/000000012_1f14d181-31f7-40ea-b957-fac40f8eee6f_f.png?time=1666779218)"
    
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/statistics/retrospective"
    querystring = {'to': to, 'from': is_from, 'cid': cid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(limit: str, type: str, q: str=None, parent: str='turkey', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all tags with which accounts are marked.
		You can filter the desired tags and nesting."
    type: country, city, category, type, interest
        
    """
    url = f"https://instagram-statistics-api.p.rapidapi.com/rating/tags"
    querystring = {'limit': limit, 'type': type, }
    if q:
        querystring['q'] = q
    if parent:
        querystring['parent'] = parent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

