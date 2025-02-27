import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zbet_dang_nhap_dang_ky_nha_cai_moi_nhat_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ZBET - zbet.casino Nhà cái cá cược trực tuyến uy tín hàng đầu tại châu Mỹ. Nhà cái ZBET88 cung cấp các dịch vụ giải trí trực tuyến đa dạng các sản phẩm: Cá cược thể thao, Đặt cược thể thao ảo, Slot Game, Game 3D Lô đề, xổ số, Keno, Casino onine…
		ZBET thiên đường cá cược trực tuyến hàng đầu châu Mỹ với nhiều năm kinh nghiệm trong lĩnh vực giải trí và sẵn sàng mang đến cho người chơi những giây phút trải nghiệm tuyệt vời nhất.
		#zbet #zbet88 #casino #nhacai #lode #cacuoc #bongda /m/033_l8 /m/033_l8 #Casino #Lô_đề_online #casino_online
		
		Địa chỉ 77 Đ. Trần Não, P. Bình An, Quận 2, Thành phố Hồ Chí Minh 70000
		Hotline 84879195679
		Gmail: zbet.casino@gmail.com
		GG map https://www.google.com/maps?cid=15859541994526576072 
		Website https://zbet.casino/ 
		https://zbet.casino/huong-dan/ 
		https://zbet.casino/lo-de/ 
		https://zbet.casino/casino/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://sites.google.com/view/zbetcasino 
		https://www.beatstars.com/zbetcasino1 
		https://twitter.com/zbetcasino1
		https://os.mbed.com/users/zbetcasino1/
		https://bom.so/profile/zbetcasino
		https://gfycat.com/@zbetcasino1
		https://git.project-hobbit.eu/zbetcasino1
		https://artistecard.com/zbetcasino
		https://notionpress.com/author/634431
		https://www.longisland.com/profile/zbetcasino
		https://forum.codeigniter.com/member.php?action=profile&uid=52852
		https://www.noteflight.com/profile/6b358f29f6b740e067ba43b9cdac33d5b72ae2cb
		https://lab.quickbox.io/zbetcasino
		https://www.wishlistr.com/zbetcasino
		https://www.hulkshare.com/zbetcasino1
		https://learn.acloud.guru/profile/zbetcasino
		https://hypothes.is/users/zbetcasino
		http://www.rohitab.com/discuss/user/575922-zbetcasino/
		https://www.mtbproject.com/user/201419873/zbet-casino
		https://www.gta5-mods.com/users/zbetcasino
		https://mydramalist.com/profile/zbetcasino
		https://my.desktopnexus.com/zbetcasino1/#ProfileComments
		https://worldcosplay.net/member/1061103
		https://kyteapp.mn.co/members/11949564"
    
    """
    url = f"https://zbet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zbet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

