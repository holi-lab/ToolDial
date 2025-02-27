import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_comments_list_by_shortcode(shortcode: str, minid: str='QVFDZ25VWmhsbUd4LXRfdWh3OTluOGhWbzljdzl0V3NYbWZrcS14SUt0TTd2VGp2dWlSald5cUQyaDZUYmRjbncyVENZbEcwY3pRTGVrUnpwVGRtaC1vZw==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments list via shortcode. you can get more data using minid. The response will include this part:
		"nextminid": "{"servercursor": "QVFEdjNRSEU5SFY4SzR5TE00MV9ISmh6STZfV1dwcnBxbGlPYlhuZTl3T3lvVW4zYVNETXc0YWVHSnJQTHNKR25ncXE1RFdCT2MzTEhTVWpGMTBBNlhIaw==", "isservercursorinverse": true}"
		and the minid will be:
		"QVFEdjNRSEU5SFY4SzR5TE00MV9ISmh6STZfV1dwcnBxbGlPYlhuZTl3T3lvVW4zYVNETXc0YWVHSnJQTHNKR25ncXE1RFdCT2MzTEhTVWpGMTBBNlhIaw==""
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postcommentsshortcode/"
    querystring = {'shortcode': shortcode, }
    if minid:
        querystring['minid'] = minid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments_list_by_media_id(mediaid: int, minid: str='QVFCbjZEUEVfS1ZIZmsyY19ZM3BlS19nWi1KTTFtUnFPRUhaM2dfY1RBcWhFWkwwa1lCTUt4QUMtczVCd3VzRXU4b3ZhMUhnZk5kakNoRkFIYlphNUdGSQ==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments list via media id. you can get more data using minid. The response will include this part:
		"next_min_id": "{"server_cursor": "QVFEdjNRSEU5SFY4SzR5TE00MV9ISmh6STZfV1dwcnBxbGlPYlhuZTl3T3lvVW4zYVNETXc0YWVHSnJQTHNKR25ncXE1RFdCT2MzTEhTVWpGMTBBNlhIaw==", "is_server_cursor_inverse": true}"
		and the minid will be:
		"QVFEdjNRSEU5SFY4SzR5TE00MV9ISmh6STZfV1dwcnBxbGlPYlhuZTl3T3lvVW4zYVNETXc0YWVHSnJQTHNKR25ncXE1RFdCT2MzTEhTVWpGMTBBNlhIaw==""
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postcommentsmediaid/"
    querystring = {'mediaid': mediaid, }
    if minid:
        querystring['minid'] = minid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_app_chrono_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last 1000 likes of a post via shortcode"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikesappchronoshortcode/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_app_chrono_by_media_id(mediaid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last 1000 likes of a post via media id"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikesappchronomediaid/"
    querystring = {'mediaid': mediaid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_list_app_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of people who has liked a post via shortcode and app method"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikesappshortcode/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_list_app_by_media_id(mediaid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of people who has liked a post via media id and app method"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikesappmediaid/"
    querystring = {'mediaid': mediaid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_info_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any post information such as photos, videos, igtvs, reels , . . . when you have the shortcode. Shortcode is usually shown this way in Url: instagram.com/p/{shortcode}"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/mediainfoshortcode/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_list_web_by_media_id(mediaid: int, endcurser: str='QVFDTDJ3bDctMkJyRmp2UnRQVS1xV3ZTZjlSTVViaDROWC0wVnJrb3N6bWs5czR1MWM4SFU5aEoxOHdpS0tNMi1HY2p3ZVpiWVNhME1IUzd1UkNJOFVmNw==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of people who has liked a post via media id and web method"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikeswebmediaid/"
    querystring = {'mediaid': mediaid, }
    if endcurser:
        querystring['endcurser'] = endcurser
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers_list_web_by_shortcode(shortcode: str, endcurser: str='QVFBSnBYdnJWUmRLS1VmRkpOVmZXQ2NBM2FFRlg4Z1c3MzdCR3hydVAxeklRaG9pTm14dElaZ2dJTVZnSy1MSThlbTBqVEF3S0dKekJtX1RwaUh1QXVPVw==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of people who has liked a post via shortcode and web method"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postlikeswebshortcode/"
    querystring = {'shortcode': shortcode, }
    if endcurser:
        querystring['endcurser'] = endcurser
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_info_by_media_id(mediaid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any post information such as photos, videos, igtvs, reels , . . . when you have the id"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/mediainfoid/"
    querystring = {'mediaid': mediaid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_among_followings_by_username(username: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search any query in an Instagram account followings when you have the username, it can be one part or the whole username or full name"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/searchfollowingsusername/"
    querystring = {'username': username, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_among_followings_by_pk(pk: int, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search any query in an Instagram account followings when you have the pk, it can be one part or the whole username or full name"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/searchfollowingspk/"
    querystring = {'pk': pk, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_followings_by_username(username: str, maxid: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lists of any public Instagram account followings via pk"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/followingsusername/"
    querystring = {'username': username, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_followings_by_pk(pk: int, maxid: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lists of any public Instagram account followings via pk"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/followingspk/"
    querystring = {'pk': pk, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_among_followers_by_username(username: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search any query in an Instagram account followers when you have the username, it can be one part or the whole username or full name"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/searchfollowersusername/"
    querystring = {'username': username, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_among_followers_by_pk(pk: int, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search any query in an Instagram account followers when you have the pk, it can be one part or the whole username or full name"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/searchfollowerspk/"
    querystring = {'pk': pk, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_followers_by_username(username: str, maxid: str='QVFEd241ZVBCZk53SUR4TE53OGh3MGEtblI1cmc3dkRJNzZCQmdtemd2TUtZVTFZZWltRWt5YlFYOFZieUVHVVktWGlMc1NSV3JfQkJlYU1FMTd1NzlKeg==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lists of any public Instagram account followers via username"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/followersusername/"
    querystring = {'username': username, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_followers_by_pk(pk: int, maxid: str='QVFEeHk5MXlRWDFGZWtLU1NHb2RRS19DcWExUXFBRGotUmo3ckh6bEZkTWdGOHFTRF9jbWNVZUdLVnJaT0VaM1E5V3FrVnRoNlNrcUFDeG5qVjZENUYtTA==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lists of any public Instagram account followers via pk"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/followerspk/"
    querystring = {'pk': pk, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts_by_username(username: str, maxid: str='2971456511645504270', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all posts contents including photos, videos, reels, igtvs, and etc via username"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postsusername/"
    querystring = {'username': username, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts_by_pk(pk: int, maxid: str='2971456511645504270', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all posts of any Instagram account via pk. including photos, videos, Igtv, reels and etc."
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/postspk/"
    querystring = {'pk': pk, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shortcode_to_media_id(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Media on Instagram has a unique id like 2931558439012794055, but in the Url you will see it this way: https://www.instagram.com/p/Ciu_N9Qpm5d/
		This endpoint will converts Ciu_N9Qpm5d to 2931558439012794055"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/shortcodetomediaid/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_stories_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all stories of any Instagram account via username"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/storiesusername/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_stories_by_pk(pk: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all stories via pk"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/storiespk/"
    querystring = {'pk': pk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_id_to_shortcode(mediaid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Media on Instagram has a unique id like 2931558439012794055, but in the Url you will see it this way: https://www.instagram.com/p/Ciu_N9Qpm5d/
		This endpoint will converts 2931558439012794055 to Ciu_N9Qpm5d"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/mediaidtoshortcode/"
    querystring = {'mediaid': mediaid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of an Instagram account by  username"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/userinfousername/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_by_pk(pk: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information about an Instagram account by pk (the unique id of Instagram account) such as username, profile_pic_url (hd), full_name, media_count, follower_count, following_count and etc"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/userinfopk/"
    querystring = {'pk': pk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed(tag: str, maxid: str='QVFCR01jQ1Z5YVhXRnB3QVRnYndJRjNXQm5aRF9Yb1ktOWRxb3Z3LWg5ZnBXTHl6d05FTnBoSkxOcV9wSVdDQXZHY1dyMGMtMkRpVjUzYTRTalM5NjVGQQ==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It shows the content which is shared on Instagram with a specific hashtag, the first query does not need maxid, but if you want to get the rest of results you need to send it, and it can be taken from previous call as a next_max_id in Json response.
		for example:
		"next_max_id": "QVFBeHItQXBZTHZPdEdNM2RLTmFtODJEZXpkUVIwbktBWDFscFQ5SFIyZTlodWU3bFZ6RXJmaURNMGJGYk1BbFk1WjIzdzlITGJzRjBkM2lNZ2lwZ2s2TA==""
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/tagfeed/"
    querystring = {'tag': tag, }
    if maxid:
        querystring['maxid'] = maxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_info(tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information about a specific hashtag on Instagram"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/taginfo/"
    querystring = {'tag': tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtag(tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search any hashtag on instagram"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/searchtag/"
    querystring = {'tag': tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search any query on Instagram"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/search/"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def username_to_pk_convert(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "it converts any username on instagram to the unique id which is called pk"
    
    """
    url = f"https://instagram-api20.p.rapidapi.com/usernametopk/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

