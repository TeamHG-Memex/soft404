import lxml
from lxml import etree
from lxml.html.clean import Cleaner
import parsel


_clean_html = Cleaner(
    scripts=True,
    javascript=False,  # onclick attributes are fine
    comments=True,
    style=True,
    links=True,
    meta=True,
    page_structure=False,  # <title> may be nice to have
    processing_instructions=True,
    embedded=True,
    frames=True,
    forms=False,  # keep forms
    annoying_tags=False,
    remove_unknown_tags=False,
    safe_attrs_only=False,
).clean_html


def _cleaned_html_tree(html):
    parser = lxml.html.HTMLParser(encoding='utf8')
    tree = lxml.html.fromstring(html.encode('utf8'), parser=parser)
    return _clean_html(tree)


def selector_to_text(sel):
    return sel.xpath('normalize-space()').extract_first('')


def cleaned_selector(html):
    try:
        tree = _cleaned_html_tree(html)
        sel = parsel.Selector(root=tree, type='html')
    except (etree.XMLSyntaxError, etree.ParseError, etree.ParserError):
        # likely plain text
        sel = parsel.Selector(html)
    return sel


def html2text(html):
    """
    Convert html to text.

    >>> html = '<html><style>.div {}</style><body><p>Hello,   world!</body></html>'
    >>> html2text(html)
    'Hello, world!'

    It works with XHTML declared ecodings:
    >>> html = '<?xml version="1.0" encoding="utf-8" ?><html><style>.div {}</style><body>Hello,   world!</p></body></html>'
    >>> html2text(html)
    'Hello, world!'

    >>> html2text("")
    ''
    """
    return selector_to_text(cleaned_selector(html))
