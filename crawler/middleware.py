from collections import defaultdict
import logging

from scrapy.exceptions import IgnoreRequest
import tldextract


class Gather404Middleware(object):
    """ Stop making requests to domains that have enough 404 pages crawled.
    """
    def __init__(self, settings):
        self.domain_status_counts = defaultdict(int)  # (domain, status): count
        self.skipped_domains = set()
        self.min_200 = settings.getint('MIN_200')
        self.min_404 = settings.getint('MIN_404')

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_request(self, request, spider):
        if get_domain(request.url) in self.skipped_domains:
            raise IgnoreRequest

    def process_response(self, request, response, spider):
        domain = get_domain(response.url)
        self.domain_status_counts[domain, response.status] += 1
        if (domain not in self.skipped_domains and
                self.domain_status_counts[domain, 200] >= self.min_200 and
                self.domain_status_counts[domain, 404] >= self.min_404):
            self.skipped_domains.add(domain)
            logging.info('New skipped domain: {} ({} skipped total)'.format(
                domain, len(self.skipped_domains)))
        return response


def get_domain(url):
    return tldextract.extract(url).registered_domain.lower()
