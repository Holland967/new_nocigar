from bs4 import BeautifulSoup
import requests

useragent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
headers = {"User-Agent": useragent}

class Spider:
    def spider(self, url: str) -> BeautifulSoup:
        try:
            response = requests.get(url, headers=headers)
        except:
            print("Error: Could not connect to website.")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
        else:
            print("Error: Website returned status code", response.status_code)
            soup = None
        return soup

    def gzh_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            title = soup.find("h1", {"class": "rich_media_title", "id": "activity-name"}).text.strip()
            try:
                author = soup.find("span", {"class": "rich_media_meta rich_media_meta_text"}).text.strip()
            except:
                author = "Unknown"
            try:
                gzh_name = soup.find("span", {"class": "rich_media_meta rich_media_meta_nickname"}). \
                find("a", {"class": "wx_tap_link js_wx_tap_highlight weui-wa-hotarea"}).text.strip()
            except:
                gzh_name = "Unknown"
            content = soup.find("div", {"class": "rich_media_content", "id": "js_content"}).text.strip()
            article = f"**文章标题**：{title}\n\n**作者**：{author}\n\n**公众号名称**：{gzh_name}\n\n**文章内容**：\n\n{content}"
        else:
            article = None
        return article

    def rmw_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            title = soup.find("div", {"class": "col col-1"}).find("h1").text.strip()
            try:
                subtitle = soup.find("div", {"class": "col col-1"}).find("h4", {"class": "sub"}).text.strip()
            except:
                subtitle = "None"
            try:
                channel_time = soup.find("div", {"class": "channel cf"}).find("div", {"class": "col-1-1"}).text.strip()
                channel_source = soup.find("div", {"class": "channel cf"}).find("a").text.strip()
                channel_content = f"{channel_time}{channel_source}"
            except:
                channel_content = "None"
            content = soup.find("div", {"class": "rm_txt_con cf"}).text.strip()
            article = f"**文章标题**：{title}\n\n**副标题**：{subtitle}\n\n**附加信息**：{channel_content}\n\n**文章内容**：\n\n{content}"
        else:
            article = None
        return article

    def gmw_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            title = soup.find("div", {"class": "g-main"}).find("h1", {"class": "u-title"}).text.strip()
            try:
                source = soup.find("div", {"class": "m_tips"}).find("a").text.strip()
            except:
                source = "None"
            try:
                time = soup.find("div", {"class": "m_tips"}).find("span", {"class": "m-con-time"}).text.strip()
            except:
                time = "None"
            try:
                content = soup.find("div", {"class": "u-mainText"}).text.strip()
            except:
                content = "None"
            article = f"**文章标题**：{title}\n\n**来源**：{source}\n\n**发布时间**：{time}\n\n**文章内容**：\n\n{content}"
        else:
            article = None
        return article

    def cnyt_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            title = soup.find("div", {"class": "article-header"}).find("h1").text.strip()
            try:
                time = soup.find("div", {"class": "byline"}).find("time").text.strip()
            except:
                time = "None"
            try:
                content = soup.find("div", {"class": "article-left"}).text.strip().replace("\n", "").replace("\t", "")
            except:
                content = "None"
            article = f"**文章标题**：{title}\n\n**发布时间**：{time}\n\n**文章内容**：\n\n{content}"
        else:
            article = None
        return article

    def xhw_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            title = soup.find("div", {"class": "head-line clearfix"}).find("h1", {"class": "topFixed"}).\
            find("span", {"class": "title"}).text.strip()
            try:
                year = soup.find("div", {"class": "header-cont clearfix"}).find("span", {"class": "year"}).find("em").text.strip()
                day = soup.find("div", {"class": "header-cont clearfix"}).find("span", {"class": "day"}).text.strip()
                time = soup.find("div", {"class": "header-cont clearfix"}).find("span", {"class": "time"}).text.strip()
                date = f"{year}/{day} {time}"
            except:
                date = "None"
            try:
                source = soup.find("div", {"class": "header-cont clearfix"}).find("div", {"class": "source"}).text.strip()
            except:
                source = "None"
            try:
                content = soup.find("div", {"class": "main clearfix"}).find("span", {"class": "detailContent"}).text.strip()
            except:
                content = "None"
            article = f"**文章标题**：{title}\n\n**发布时间**：{date}\n\n**来源**：{source}\n\n**文章内容**：\n\n{content}"
        else:
            article = None
        return article

    def general_spider(self, url: str) -> str:
        soup = self.spider(url)
        if soup is not None:
            webpage = soup.find("body")
            content = webpage.text.strip().replace("\n", "").replace("\t", "")
        else:
            content = None
        return content

    def check_url(self, url: str) -> str:
        if "mp.weixin.qq.com" in url or "weixin.sogou.com/link?url=" in url:
            return "gzh"
        elif "people.com.cn" in url:
            return "rmw"
        elif "news.cn" in url:
            return "xhw"
        elif "gmw.cn" in url:
            return "gmw"
        elif "cn.nytimes.com" in url:
            return "cnyt"
        else:
            return "general"