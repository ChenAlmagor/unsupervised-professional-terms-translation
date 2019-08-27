import scrapy
import re
import json

data = {}
class MedicineNetSpider(scrapy.Spider):
    name = "medicine_net_spider"
    start_urls = ['https://www.medicinenet.com/script/main/alphaidx.asp?p=a_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=b_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=c_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=d_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=e_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=f_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=g_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=h_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=i_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=j_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=k_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=l_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=m_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=n_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=o_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=p_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=q_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=r_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=s_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=t_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=u_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=v_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=w_dict',
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=x_dict'
                  'https://www.medicinenet.com/script/main/alphaidx.asp?p=z_dict'
        'https://www.medicinenet.com/script/main/alphaidx.asp?p=z_dict']

    def parse(self, response):
        all_pages_links = response.css(".AZ_results ul li a::attr(href)").extract()

        print(len(all_pages_links))
        for page in all_pages_links:
            yield scrapy.Request(page, callback=self._parse_term_page)

    def _parse_term_page(self, response):
        whole_definition = response.css("div.apPage p").extract_first()
        print(whole_definition)
        whole_definition = re.sub('<.*?>', '', whole_definition)
        print(whole_definition )
        m = re.match('(.*?):(.*)', whole_definition)
        term = m[1].strip()


        w = response.css("div.apPage p").extract()
        d = ''.join(str(elem) for elem in w)
        d = re.sub('<.*?>', '', d)
        d = d.replace('\n', '') \
            .replace('\r', '')
        extract_d = re.match('(.*?):(.*)', d)
        dush = extract_d[2]\
            .strip() \
            .rstrip() \
            .replace('\n', '') \
            .replace('\r', '')


        data[term] = dush
        yield {
             term: dush
        }

def write_to_json_file():
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)


from scrapy.crawler import CrawlerProcess
if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(MedicineNetSpider)
    process.start()
    write_to_json_file()
