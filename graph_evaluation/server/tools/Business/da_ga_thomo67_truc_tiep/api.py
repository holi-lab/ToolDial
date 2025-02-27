import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def da_ga_thomo67_truc_tiep(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Thomo67 ✓ Xem đá gà trực tiếp Thomo hôm nay mới nhất tại Thomo 678 ✓ Trực tiếp đá gà cựa sắt Thomo Campuchia hôm nay tại bồ gà 999 & 67.
		#đá_gà_trực_tiếp #đá_gà sv388 #đá_gà 68 # #/m/033_l8  /m/033_l8 
		Địa chỉ :568 Đ. Bạch Đằng, Bạch Đằng, Hai Bà Trưng, Hà Nội, Việt Nam
		Hotline 0888.310.444
		Gmail: thomo678net@gmail.com
		Website https://thomo678.net/ 
		https://thomo678.net/da-ga-truc-tiep/ 
		https://thomo678.net/sv388-link-vao-trang-da-ga-dagasv388/ 
		https://thomo678.net/daga68/ 
		https://twitter.com/Thomo678N
		https://artistecard.com/thomo678net
		https://notionpress.com/author/676460
		www.longisland.com/profile/thomo678net
		https://www.noteflight.com/profile/fc3d48ad4d6564cad2c4ef5ad9742abb1b9b9d94
		https://lab.quickbox.io/thomo678net
		www.wishlistr.com/thomo678net
		https://www.hulkshare.com/thomo678net
		https://learn.acloud.guru/profile/thomo678net
		https://hypothes.is/users/thomo678net
		http://www.rohitab.com/discuss/user/584283-thomo678net/
		https://dailygram.com/index.php/profile-434107
		https://www.mtbproject.com/user/201427822/thomo678-net
		https://www.gta5-mods.com/users/thomo678net
		https://mydramalist.com/profile/thomo678net
		https://my.desktopnexus.com/thomo678net/#ProfileComments
		https://worldcosplay.net/member/1065031
		https://kyteapp.mn.co/members/12081656
		https://fundrazr.com/profiles/thomo678-net
		https://www.gaiaonline.com/profiles/thomo678net/45945804/
		https://www.bhimchat.com/thomo678net
		https://pawoo.net/@thomo678net
		https://www.hackathon.io/users/289404
		https://guides.co/p/thomo678-net
		https://www.allmyfaves.com/thomo678net
		https://my.mamul.am/en/profile/4928800/info
		http://www.effecthub.com/user/2347826
		https://kuwestions.248am.com/user/thomo678net
		https://www.mapleprimes.com/users/thomo678net
		https://www.bahamaslocal.com/userprofile/1/142090/thomo678net.html
		https://www.misterpoll.com/forums/1/topics/332657
		https://answerpail.com/index.php/user/thomo678net
		https://anchor.fm/thomo678-net
		https://www.bitsdujour.com/profiles/xUi7Hn
		https://www.techrum.vn/members/thomo678net.210239/#about
		https://www.facer.io/u/thomo678net"
    
    """
    url = f"https://da-ga-thomo67-truc-tiep.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "da-ga-thomo67-truc-tiep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

