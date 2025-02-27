import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def imglory(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IMGlory is a membership site that sharing all the MMO courses from big authors around of the world, all in one place
		#imglory #graphickits #imtools
		Address 12st Pariser Platz, Berlin, 10117, Germany
		Hotline 0989.022.861
		Email: support@imglory.co 
		Website https://imglory.co/ 
		https://www.imglory.co/gb-courses/ 
		https://www.imglory.co/graphickits/ 
		https://www.imglory.co/im-tools/ 
		https://www.imglory.co/spy-tools/ 
		https://fb.me/imgloryb/ 
		https://twitter.com/imglory2016 
		https://www.pinterest.com/imglory2010/ 
		https://www.instagram.com/imglory2018/ 
		https://youtube.com/c/imglory/ 
		
		https://answerpail.com/index.php/user/imglory
		https://www.tnrad.org/community/profile/imgloryco/
		http://en.belclimb.be/profile_rout_list.asp?userid=26138
		https://thearticlesdirectory.co.uk/members/imglory/profile/
		https://gitlab.raptorengineering.com/imglory
		http://imglory.idea.informer.com/
		https://www.penname.me/@imgloryco1_gl
		https://kustomcoachwerks.com/Forums/users/imglory/
		https://android.libhunt.com/u/imglory
		https://kwafoo.coe.neu.edu:7788/git/imglory
		https://www.chandigarhcity.com/members/imglory/profile/
		http://www.mathisfunforum.com/profile.php?id=240308
		https://feverclan.com/forums/members/imglory.112228/#about
		https://gitlab.fabcloud.org/imglory
		https://www.voubs.com/user/imglory/225578/fullinfo
		https://rollbol.com/posts/1257537"
    
    """
    url = f"https://imglory.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imglory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

