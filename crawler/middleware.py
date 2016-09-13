from collections import defaultdict
import logging

from scrapy.exceptions import IgnoreRequest
import tldextract


class Gather404Middleware(object):
    """ Limit number of requests to each domain.
    """
    def __init__(self, settings):
        self.max_domain_requests = settings.getint('MAX_DOMAIN_REQUESTS')
        self.domain_request_count = defaultdict(int)  # domain: count
        self.skipped_domains = set()

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_request(self, request, spider):
        if get_domain(request.url) in self.skipped_domains:
            raise IgnoreRequest

    def process_response(self, request, response, spider):
        domain = get_domain(response.url)
        self.domain_request_count[domain] += 1
        if (domain not in self.skipped_domains and
                self.domain_request_count[domain] >= self.max_domain_requests):
            self.skipped_domains.add(domain)
            logging.info('New skipped domain: {} '
                         '({} skipped total, crawled {} domains)'.format(
                domain, len(self.skipped_domains), len(self.domain_request_count)))

        return response


def get_domain(url):
    return tldextract.extract(url).registered_domain.lower()
