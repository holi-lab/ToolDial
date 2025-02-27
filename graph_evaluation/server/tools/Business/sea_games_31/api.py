import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sea_games_31(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cổng thông tin chính thức của SEA Games 31 - Việt Nam 2021. Đại hội Thể thao Đông Nam Á 2021. Đại hội Thể thao Đông Nam Á - SEA Games 2021, sẽ diễn ra ở Hà Nội, Việt Nam từ ngày 12 tháng 5 đến ngày 23 tháng 5 năm 2022.
		Website http://seagames.info/ 
		Email:info@seagames.info
		Google map: https://www.google.com/maps?cid=10302187337639672513 
		Địa chỉ : 36 P. Trần Phú, Điện Biên, Ba Đình, Hà Nội Hanoi, Vietnam 100000
		#vietnamxseagames31 #Hà_Nội #Việt_Nam #Lịch_thi_đấu_SEA Games 31
		https://seagames.info/sea-games-31/ 
		https://www.facebook.com/vietnamxseagames31 
		https://sites.google.com/view/vietnamxseagames31 
		https://connect.garmin.com/modern/profile/9fba05fe-267e-4843-9511-74c0fd2d697d
		https://gitlab.haskell.org/seagamesinfo
		https://cartoonmovement.com/cartoonist/18495
		https://www.feedsfloor.com/profile/seagamesinfo
		https://deepai.org/profile/seagamesinfo
		https://www.bark.com/en/gb/company/seagamesinfo/kVPmo/
		https://community.tubebuddy.com/index.php?members/103488/#about
		https://start.me/p/jja0qL/seagamesinfo
		https://marketplace.apartmenttherapy.com/store/d03759a3-53fa-4a94-a31b-dbbba532f142
		https://www.lifeofpix.com/photographers/seagamesinfo/
		https://kuula.co/post/NHp4C
		http://wiki.cs.hse.ru/Участник:Seagamesinfo
		https://bibliocrunch.com/profile/seagamesinfo/
		https://www.dday.it/profilo/seagamesinfo
		https://www.mountainproject.com/user/201342996/seagames-info
		https://www.thingiverse.com/seagamesinfo/designs
		https://www.trepup.com/seagamesinfo-100997554504555
		https://sharree.com/User-seagamesinfo
		https://orcid.org/0000-0002-5432-601X
		https://zindi.africa/users/seagamesinfo
		https://ourstage.com/seagamesinfo
		https://infogram.com/seagamesinfo-1h8n6m35qeovz4x?live
		https://kyteapp.mn.co/members/11333334
		https://www.bhimchat.com/seagamesinfo
		http://pixelhub.me/seagamesinfo"
    
    """
    url = f"https://sea-games-31.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sea-games-31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

