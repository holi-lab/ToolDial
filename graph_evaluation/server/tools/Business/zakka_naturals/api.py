import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zakkana_turals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zakka Naturals - Thương hiệu mỹ phẩm thiên nhiên ứng dụng công nghệ tiên tiến. Các sản phẩm của Zakka: serum vitamin c, serum niacinamide, serum b5…
		#zakkanaturalscom #zakkanaturals #serum vitamin c #serum niacinamide #serum b5 #Mỹ_phẩm #Dưỡng_ẩm_da #Làm_dẹp_da #Sữa_rửa_mặt
		Địa chỉ : 12/7 Trương Định, Phường 6, Quận 3, Thành phố Hồ Chí Minh, Việt Nam
		Hotline 0907 050 959 | 02866709966
		Gmail: zakkanaturalscom@gmail.com
		Google map https://www.google.com/maps?cid=1204409815373898565    
		Website https://zakkanaturals.com/ 
		https://zakkanaturals.com/products/vitamin-c-glowfruit-brightening-serum 
		https://zakkanaturals.com/products/niacinamide10-ampoule 
		https://zakkanaturals.com/products/kintsugi-b5-intense-ampoule 
		https://www.facebook.com/zakkanaturals.official 
		https://www.instagram.com/zakka_naturals/ 
		https://www.youtube.com/channel/UCskoM2sK4rIJV2OjSAku7GA 
		https://zakkanaturals.blogspot.com/ 
		https://sites.google.com/view/zakkanaturals/home 
		https://profile.ameba.jp/ameba/zakkanaturals/
		https://community.dynamics.com/members/zakkanaturals
		https://bikepgh.org/message-board/users/zakkanaturals/edit/?updated=true
		https://techbike.vn/members/zakkanaturals.22617/#about
		https://anyflip.com/homepage/nmian#About
		https://tapas.io/zakkanaturalscom
		https://www.historyillinois.org/UserProfile/tabid/3119/userId/90582/Default.aspx
		https://www.imagekind.com/MemberProfile.aspx?MID=4d4e644c-4fb8-44f8-857f-6d09f9d45e76
		https://www.ultimate-guitar.com/u/zakkanaturals
		http://wiki.cs.hse.ru/Участник:Zakkanaturals
		https://www.thingiverse.com/zakkanaturals/designs
		https://stocktwits.com/zakkanaturals
		https://linkhay.com/link/5248725/zakkanaturals
		https://zakkanaturals.cgsociety.org/profile
		https://cults3d.com/en/users/zakkanaturals
		https://www.trepup.com/zakkanaturalscom
		https://app.bountysource.com/people/100761-zakkanaturals
		https://www.goodreads.com/user/show/150350778-zakkana-turals
		https://www.metal-archives.com/users/zakkanaturals
		https://gifyu.com/comzakkanaturals
		https://os.mbed.com/users/zakkanaturals/
		http://photozou.jp/user/top/3313498
		https://gfycat.com/@zakkanaturals
		https://git.project-hobbit.eu/-/snippets/5997
		https://artistecard.com/zakkanaturals
		https://notionpress.com/author/504502
		https://www.longisland.com/profile/zakkanaturals
		https://www.noteflight.com/profile/1e52fc1970a0e29a58a3ce37a51243f025bb071a
		https://lab.quickbox.io/zakkanaturals
		https://www.wishlistr.com/zakkanaturals
		https://learn.acloud.guru/profile/zakkanaturalscom1
		https://hypothes.is/users/zakkanaturals
		http://tupalo.com/en/users/3382398
		http://www.rohitab.com/discuss/user/285982-zakkanaturals/"
    
    """
    url = f"https://zakka-naturals.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zakka-naturals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

