import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lk881(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "LK88 nhà cái cá cược thể thao, cá cược bóng đá, sòng bài, casino...Tham gia đăng ký tài khoản LK88 ngay để có cơ hội trải nghiệm cá cược thể thao, các trò chơi sòng casino đa dạng, hấp dẫn với tỷ lệ thưởng cực cao, đồng thời nhận khuyến mãi lên đến 200%
		#NhacaiLK88 #LinkVaoLK88 #KhuyenmaiLK88 #dangkylk88 #Đăng_ký_LK88 #Lk881com #/m/033_l8  /m/033_l8 #Casino #Ae888vip #Ae888
		Địa chỉ 389 Nguyễn Thị Búp, Tân Chánh Hiệp, Quận 12, Thành phố Hồ Chí Minh
		Hotline 0976187364
		Gmail: LKtrong8h77@gmail.com 
		Website https://lk881.com/ 
		https://lk881.com/dang-ky-lk88/ 
		https://lk881.com/link-vao-lk88/ 
		https://lk881.com/khuyen-mai-lk88/ 
		https://en.wikipedia.org/wiki/Online_casino  
		https://sites.google.com/view/lk881com/nha-cai-lk88 
		https://sites.google.com/view/lk881com 
		https://lk881com.blogspot.com/
		https://cyber.harvard.edu/cyberlaw_winter10/User_talk:192.168.10.74
		https://www.magcloud.com/user/lk881com
		https://forum.cs-cart.com/user/207242-lk881/
		https://www.plurk.com/lk881
		https://www.myminifactory.com/users/lk881
		http://ttlink.com/lk881
		https://forum.acronis.com/user/396298
		https://www.instapaper.com/p/10519633
		https://pubhtml5.com/homepage/hoyog
		https://www.spreaker.com/user/16517741
		https://paper.li/~/publisher/a35ce4e7-b063-4e5c-944c-3b1d4f6a1702
		https://www.zotero.org/lk881
		https://gitlab.pasteur.fr/lk881com
		https://muckrack.com/lk881-com/bio
		https://roundme.com/@lk881/about
		https://www.40billion.com/profile/582488201
		https://kienthucykhoa.edu.vn/members/lk881.23944/
		https://community.aodyo.com/user/lk881com
		https://mmo4me.com/members/lk881.194271/
		https://www.ranker.com/writer/lk881
		https://comicvine.gamespot.com/profile/lk881/
		https://myanimelist.net/profile/lk881
		https://communities.bentley.com/members/efc1f90f_2d00_888d_2d00_4318_2d00_af73_2d00_d217ba8a3513
		https://pxhere.com/en/photographer/3827634
		https://www.cplusplus.com/user/lk881/
		https://www.picfair.com/users/lk881
		https://repo.getmonero.org/lk881
		https://folkd.com/user/lk881c0m
		https://replit.com/@lk881c0m
		https://leetcode.com/lk881com/
		https://www.intensedebate.com/people/lk881
		https://logopond.com/lk881/profile/546477"
    
    """
    url = f"https://lk881.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lk881.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

