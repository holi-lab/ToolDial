import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def portrait_for_pet_custom_pet_portrait(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Capture the personality of your royal pet with pet canvas art. Each of Portrait4pet.com renaissance pet portraits are made with love from our talented pet-loving artists. Simply choose a costume from our renaissance royal portraits gallery below, upload your pet’s photo, and we’ll send you artwork to approve the next day. We make the best royal dog portraits and royal cat portraits around.
		#Pet Portrait Royal #People Portrait Royal #Custom video invitation animated #portrait4pet #portrait4petcom
		
		Address :21143 Hawthorne Blvd, Torrance, CA 90503
		Hotline 84795216686
		Gmail: ad.portrait4pet@gmail.com
		
		Website : https://portrait4pet.com/ 
		https://portrait4pet.com/product-category/pet-portrait/ 
		https://portrait4pet.com/product-category/people-portrait/ 
		https://portrait4pet.com/product-category/video-invitation/ 
		https://www.facebook.com/CustomRoyalPortrait 
		https://twitter.com/DigitalsInvite 
		https://www.pinterest.com/Portrait4pet 
		https://www.instagram.com/portrait4petstudio/ 
		https://portrait4pet.tumblr.com/ 
		https://git.qoto.org/portrait4pet
		https://www.gapo.vn/1120847923
		https://www.bbuzzart.com/profile/438447
		https://faithlife.com/account
		http://kiredu.ru/UserProfile/tabid/182/userId/89697/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/62985/Default.aspx
		https://pinshape.com/users/2362936-portrait4pet#designs-tab-open
		https://beermapping.com/account/portrait4pet
		http://gendou.com/user/portrait4pet
		https://sumally.com/portrait4pet
		https://www.furaffinity.net/user/portrait4pet/
		https://www.bitrated.com/portrait4pet
		http://yourlisten.com/portrait4pet
		https://www.plasterersforum.com/members/portrait4pet.79193/
		https://www.jigsawplanet.com/portrait4pet?viewas=2c61a8ad10ff
		https://independent.academia.edu/portrait4pet
		https://thetravelbrief.com/tips/celeken-portrait-4pet
		https://pbase.com/portrait4pet/profile
		https://portrait4petpersonal.simplesite.com/
		https://scm.cms.hu-berlin.de/portrait4pet
		https://yoo.rs/useroverview/186822/portrait4petcom?Ysid=186822
		https://www.pedalroom.com/members/portrait4pet
		https://confengine.com/user/portrait-4pet
		https://www.magcloud.com/user/portrait4pet
		https://forum.cs-cart.com/user/223000-portrait4pet/"
    
    """
    url = f"https://portrait-for-pet-custom-pet-portrait.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "portrait-for-pet-custom-pet-portrait.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

