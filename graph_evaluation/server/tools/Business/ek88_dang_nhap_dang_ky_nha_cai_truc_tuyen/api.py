import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ek88_dang_nhap_dang_ky_nha_cai(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "EK88 là nhà cái uy tín hàng đầu châu Á, nhà cái EK88 cung cấp tỷ lệ cược tốt nhất, nhiều khuyến mãi, link vào EK88 nhanh không bị chặn mới nhất 2022.
		#EK88 #NhacaiEK88 #CacuocEK88  /m/033_l8  /m/033_l8 #Casino #Hướng_dẫn_cá_cược_EK88 #Đăng_ký_EK88 #Khuyến_mãi_EK88 #link_vào_EK88_2022
		Địa chỉ : 3/39/147 Phố Tân Mai, Tân Mai, Hoàng Mai, Hà Nội, Việt Nam
		Hotline +84886337598
		Website https://ek88.bet/ 
		Gmail: ek88betpro@gmail.com
		https://ek88.bet/huong-dan-ca-cuoc-ek88 
		https://ek88.bet/dang-ky-ek88 
		https://ek88.bet/khuyen-mai-ek88 
		https://en.wikipedia.org/wiki/Online_casino 
		https://www.facebook.com/ek88betpro  
		https://sites.google.com/view/ek88betpro 
		https://twitter.com/ek88betpro 
		https://www.pinterest.com/ek88betpro 
		https://forum.acronis.com/user/407752
		https://www.instapaper.com/p/ek88betpro
		https://pubhtml5.com/homepage/rzkcm
		https://gitlab.tails.boum.org/ek88betpro
		https://sketchfab.com/ek88betpro
		https://muckrack.com/ek88bet-pro/bio
		https://roundme.com/@ek88betpro/about
		https://community.aodyo.com/user/ek88betpro
		https://myanimelist.net/profile/ek88betpro
		https://communities.bentley.com/members/932b2252_2d00_d814_2d00_4e0a_2d00_b1dc_2d00_44abaefbf8f1
		https://pxhere.com/en/photographer-me/3854752
		https://cplusplus.com/user/ek88betpro/
		https://www.picfair.com/users/ek88betpro
		https://replit.com/@ek88betpro
		https://leetcode.com/ek88betpro/
		https://www.intensedebate.com/profiles/ek88betpro
		https://gitlab.kitware.com/ek88betpro
		http://cloudsdeal.xobor.de/u7142_ek-betpro.html
		https://platform.xr4all.eu/ek88betpro
		https://justpaste.it/u/ek88betpro
		https://pantip.com/profile/7111719#topics
		https://www.emoneyspace.com/ek88betpro
		https://www.veoh.com/users/ek88betpro
		https://app.glosbe.com/profile/6949333811654036715
		https://hdvietnam.org/members/ek88betpro.2076447/"
    
    """
    url = f"https://ek88-dang-nhap-dang-ky-nha-cai-truc-tuyen.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ek88-dang-nhap-dang-ky-nha-cai-truc-tuyen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

