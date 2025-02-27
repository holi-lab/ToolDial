import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def five88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Five88 - https://five88.casino  - Trang chủ đăng ký chính thức của nhà cái Five88. Five88 là sân chơi casino, cá cược bóng đá đẳng cấp hàng đầu Việt Nam, nơi người chơi sẽ hoàn toàn hài lòng với những trải nghiệm thú vị đang chờ đón trước mắt.
		#five88 #nhacaifive88 #casino  #/m/033_l8  /m/033_l8 #Casino  #xoso #lode #cacuoc #FIVE88 #dangnhap #linkvaofive882022
		Gmail: five88.casino@gmail.com
		Hotline 0389.575.002
		Địa chỉ 49 Đ. Lê Trọng Tấn La Khê Hà Đông Hà Nội Việt Nam
		https://muckrack.com/five88-casino/bio 
		https://beacons.ai/five88casino
		https://community.dynamics.com/members/five88casino
		https://app.glosbe.com/profile/6939196167221677248
		https://www.furaffinity.net/user/five88casino/
		https://www.picfair.com/users/five88casino
		https://themehunt.com/profile/five88casino
		https://www.zintro.com/profile/zi6e53e00c?ref=
		https://leetcode.com/five88casino/
		https://play.eslgaming.com/player/18222843/
		https://www.elephantjournal.com/profile/five88casino/
		https://mootools.net/forge/profile/five88casino
		https://www.ultimate-guitar.com/u/five88casino
		https://rapidapi.com/user/five88casino
		https://www.360cities.net/profile/five88casino
		https://tapas.io/five88casino1
		https://paroledemamans.com/parents/five88casino
		https://community.tubebuddy.com/index.php?members/104689/
		https://www.lifeofpix.com/photographers/five88casino/
		https://www.intensedebate.com/profiles/five88casino
		https://xipnios.tribe.so/user/five88casino
		https://micro.blog/five88casino
		https://pantip.com/profile/7068733#topics
		https://folkd.com/user/five88casino
		https://www.beatstars.com/five88casino/about"
    
    """
    url = f"https://five88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "five88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

