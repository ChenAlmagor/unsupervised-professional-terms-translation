import scrapy
import re
import json

letter = 'y'
data = {}
class MedicineNetSpider(scrapy.Spider):
    name = "medicine_net_spider"
    start_urls = ['https://www.cun.es/diccionario-medico?letra=' + letter + '&pagina=1']

    def parse(self, response):
        all_page_terms = response.css("div.item-diccionario p:first-child strong::text").extract()
        all_page_definitions = response.css("div.item-diccionario").extract()

        if len(all_page_terms) == 0:
            return

        #process definitions
        for i in range (len(all_page_definitions)):
            all_page_definitions[i] = re.sub('<.*?>', '', all_page_definitions[i])
            all_page_definitions[i] = all_page_definitions[i].strip()\
            .rstrip() \
            .replace('\n', '') \
            .replace('\r', '')
            print(all_page_definitions[i])

            m = re.match('(.*)(\s{4,})(.*)', all_page_definitions[i])

            all_page_definitions[i] = m[3]

        #match term-definition
        print(len(all_page_terms))
        print(len(all_page_definitions))
        for index, term in enumerate(all_page_terms):
            data[term] = all_page_definitions[index]


        cur_url = response.url
        [url_prefix, url_page] = cur_url.split("pagina=")
        next_url_page = int(url_page) + 1
        next_url = url_prefix + 'pagina=' + str(next_url_page)
        yield scrapy.Request(next_url, callback=self.parse)



def write_to_json_file():
    with open('data-es-' + letter +'.json', 'w') as outfile:
        json.dump(data, outfile)


from scrapy.crawler import CrawlerProcess
if __name__ == '__main__':

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(MedicineNetSpider)
    process.start()
    write_to_json_file()
