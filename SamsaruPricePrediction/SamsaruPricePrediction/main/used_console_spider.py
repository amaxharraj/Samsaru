import json
from scrapy import Spider
from datetime import datetime

class UsedConsolesSpider(Spider):
    name = "used_consoles_spider"
    allowed_domains = ["asgoodasnew.de"]
    start_urls = ['https://asgoodasnew.de/Konsolen/Xbox/']

    def parse(self, response):
        scrape_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        products = response.css('a.listitem-variants-item')
        for product in products:
            data_variant = product.attrib.get('data-variant')
            if data_variant:
                variant_data = json.loads(data_variant)
                variant = variant_data.get('variant', '').lower()
                if variant in ["neu", "wie neu", "gut", "sehr gut"]:
                    yield {
                        'scrape_date': scrape_date,
                        'title': variant_data.get('name'),
                        'price': variant_data.get('price'),
                        'brand': variant_data.get('brand'),
                        'category': variant_data.get('category'),
                        'variant': variant,
                    }

        next_page = response.css('a.page--next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
