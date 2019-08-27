import scrapy
import re
import json

data = {}
all_pages_links = []
class MedicineNetSpider(scrapy.Spider):
    name = "medicine_net_spider"
    start_urls = ['http://www.doctissimo.fr/sante/dictionnaire-medical/initiale-S.htm?page=1']

    def parse(self, response):
        cur_pages_links = response.css("div.multi-columns ul li a::attr(href)").extract()
        if len(cur_pages_links) > 0:
            all_pages_links.extend(cur_pages_links)
            print(response.url)
            cur_url = response.url
            [url_prefix, url_page] = cur_url.split("page=")
            print(url_prefix, url_page)
            next_url_page = int(url_page) + 1
            next_url = url_prefix + 'page=' + str(next_url_page)
            print(next_url)
            yield scrapy.Request(next_url, callback=self.parse)
        else:

            for page in all_pages_links:
                yield scrapy.Request(page, callback=self._parse_term_page)

    def _call_all_links(self):
        for page in all_pages_links:
            yield scrapy.Request(page, callback=self._parse_term_page)

    def _parse_term_page(self, response):
        definition = response.css("div.row.doc-block-definition div div").extract_first()
        term = response.css("div.row.doc-title h1::text").extract_first()
        definition = re.sub('<.*?>', '', definition)


        data[term] = definition



def write_to_json_file():
    with open('data-fr-s.json', 'w') as outfile:
        json.dump(data, outfile)



from scrapy.crawler import CrawlerProcess
if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(MedicineNetSpider)
    process.start()

    print('@@@@@ VALIDATION @@@@@@@@')
    print(len(list(data.keys())))
    print(len(all_pages_links))
    write_to_json_file()
