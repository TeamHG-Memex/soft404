import contextlib
import pickle
import struct
import warnings

import langdetect
from langdetect.lang_detect_exception import LangDetectException
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
    except (etree.XMLSyntaxError, etree.ParseError, etree.ParserError,
            UnicodeEncodeError):
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


BLOCK_TAGS = {
    'address', 'article', 'aside', 'audio', 'blockquote', 'canvas', 'dd',
    'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'form',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
    'header', 'hgroup', 'hr', 'noscript', 'ol', 'output', 'p', 'pre',
    'section', 'table', 'tfoot', 'ul', 'video',
    # not really block, but makes sense to include
    'li',
}


def get_text_blocks(tree):
    text_blocks = []
    current = []
    tag = ''
    for node in tree.iter():
        if node.tag in BLOCK_TAGS:
            if current:
                text_blocks.append((tag, ' '.join(current)))
                current = []
            tag = node.tag
        for text in [node.text, node.tail]:
            if text:
                text = text.strip()
                if text:
                    current.append(text)
    if current:
        text_blocks.append((tag, ' '.join(current)))
    return text_blocks


def write_pickle_stream(item, outfile):
    """ Write an item to the file in "pickle stream" format.
    outfile should be open in "wb" mode.
    """
    data = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
    outfile.write(struct.pack('i', len(data)))
    outfile.write(data)


def pickle_stream_reader(infile, indices=None):
    """ A "pickle stream" reader. Yields pairs of (index, item), where index is always
    an index among all items. If indices is given, then only items with specified
    indices are returned, and other items are skipped efficiently.
    """
    if indices is not None:
        indices = set(indices)
    idx = 0
    while True:
        size_data = infile.read(4)
        if not size_data:
            break
        size, = struct.unpack('i', size_data)
        if indices is None or idx in indices:
            item = pickle.loads(infile.read(size))
            yield idx, item
        else:
            infile.seek(infile.tell() + size)
        idx += 1


def get_lang(text):
    try:
        langs = langdetect.detect_langs(text)
    except LangDetectException:
        return ''
    else:
        return langs[0].lang


def batches(lst, size):
    for idx in range(0, len(lst), size):
        yield lst[idx:idx + size]


@contextlib.contextmanager
def ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
