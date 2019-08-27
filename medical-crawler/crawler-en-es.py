import scrapy
import re
import json

dict = []


class MedicineNetSpider(scrapy.Spider):
    name = "medicine_net_spider"
    start_urls = ['https://www.123teachme.com/medical_dictionary/letter/A',
                  'https://www.123teachme.com/medical_dictionary/letter/B',
                  'https://www.123teachme.com/medical_dictionary/letter/C',
                  'https://www.123teachme.com/medical_dictionary/letter/D',
                  'https://www.123teachme.com/medical_dictionary/letter/E',
                  'https://www.123teachme.com/medical_dictionary/letter/F',
                  'https://www.123teachme.com/medical_dictionary/letter/G',
                  'https://www.123teachme.com/medical_dictionary/letter/H',
                  'https://www.123teachme.com/medical_dictionary/letter/I',
                  'https://www.123teachme.com/medical_dictionary/letter/G',
                  'https://www.123teachme.com/medical_dictionary/letter/K',
                  'https://www.123teachme.com/medical_dictionary/letter/L',
                  'https://www.123teachme.com/medical_dictionary/letter/M',
                  'https://www.123teachme.com/medical_dictionary/letter/N',
                  'https://www.123teachme.com/medical_dictionary/letter/O',
                  'https://www.123teachme.com/medical_dictionary/letter/P',
                  'https://www.123teachme.com/medical_dictionary/letter/Q',
                  'https://www.123teachme.com/medical_dictionary/letter/R',
                  'https://www.123teachme.com/medical_dictionary/letter/S',
                  'https://www.123teachme.com/medical_dictionary/letter/T',
                  'https://www.123teachme.com/medical_dictionary/letter/U',
                  'https://www.123teachme.com/medical_dictionary/letter/V',
                  'https://www.123teachme.com/medical_dictionary/letter/W',
                  'https://www.123teachme.com/medical_dictionary/letter/X',
                  'https://www.123teachme.com/medical_dictionary/letter/Y',
                  'https://www.123teachme.com/medical_dictionary/letter/Z'
                  ]


    def parse(self, response):
        all_terms = response.css("u strong").extract()
        all_text = response.css(".content-inner-less::text").extract()
        for i in range(len(all_text)):
            if all_text[i].startswith('\nTechnical (English):'):
                en_prof = all_text[i].split('\nTechnical (English):')[1].strip()
                en_pop = all_text[i + 1].split('\nPopular (English):')[1].strip()
                es_prof = all_text[i + 3].split('\nTechnical:')[1].strip()
                es_pop = all_text[i + 4].split('\nPopular:')[1].strip()

                es_prof_splitted = es_prof.split(',')
                es_pop_splitted = es_pop.split(',')
                for item in es_prof_splitted:
                    if((len(en_prof.split(' ')) < 3) and (len(item .split(' ')) < 3)):
                        en_prof = re.sub('\(.*\)', '', en_prof).strip().replace(' ', '-').replace('/-', '/').lower()
                        es_prof_new = re.sub('\(.*\)', '', item).strip().replace(' ', '-').replace('/-', '/').lower()
                        if (en_prof != '' and es_prof_new !=''):
                            dict.append(en_prof + ' ' + es_prof_new + '\n')

        print(dict)



def write_to_txt_file():
    file1 = open("test-dict-up-to-2-words.txt", "w")

    # \n is placed to indicate EOL (End of Line)
    file1.writelines(dict)
    file1.close()  # to change file access modes


from scrapy.crawler import CrawlerProcess

if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(MedicineNetSpider)
    process.start()
    write_to_txt_file()
