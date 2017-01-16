import re

from html_text.html_text import cleaned_selector, selector_to_text
import numpy as np
from webstruct.html_tokenizer import HtmlTokenizer


BLOCK_TAGS = {
    'address', 'article', 'aside', 'audio', 'blockquote', 'canvas', 'dd',
    'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'form',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
    'header', 'hgroup', 'hr', 'noscript', 'ol', 'output', 'p', 'pre',
    'section', 'table', 'tfoot', 'ul', 'video',
    # not really block, but makes sense to include
    'li', 'body',
}


html_tokenizer = HtmlTokenizer()


def get_text_blocks(tree):
    tokens, _ = html_tokenizer.tokenize_single(tree)
    text_blocks = []
    prev_parent = None
    current = []
    for token in tokens:
        parent = token.parent
        while parent.tag not in BLOCK_TAGS:
            parent = parent.getparent()
        if prev_parent is not None and prev_parent != parent:
            text_blocks.append((prev_parent.tag, ' '.join(current)))
            current = []
        current.append(token.token)
        prev_parent = parent
    if current and prev_parent is not None:
        text_blocks.append((prev_parent.tag, ' '.join(current)))
    return text_blocks


def html_to_item(html):
    sel = cleaned_selector(html)
    text = selector_to_text(sel)
    text_item = {
        'text': text,
        'title': ' '.join(sel.xpath('//title//text()').extract()),
    }
    body = sel.xpath('/html/body')
    text_root = body[0].root if body else sel.root
    text_item['blocks'] = get_text_blocks(text_root)
    return text_item


def item_to_text(item):
    text = [item['text']]
    if item['title']:
        text.extend('__title__{}'.format(w) for w in tokenize(item['title']))
    for tag, block_text in item.get('blocks', []):
        text.extend('__{}__{}'.format(tag, w) for w in tokenize(block_text))
    return ' '.join(text)


token_pattern = r'(?u)\b[_\w][_\w]+\b'


def tokenize(text):
    return re.findall(token_pattern, text, re.U)


def item_numeric_features(item):
    if item.get('blocks'):
        block_lengths = sorted(
            len(tokenize(block)) for _, block in item['blocks'])
    else:
        block_lengths = None
    return [
        len(tokenize(item['text'])),
        len(item['blocks']) if 'blocks' in item else 0,
        np.max(block_lengths) if block_lengths else 0,
        np.median(block_lengths) if block_lengths else 0,
        block_lengths[int(0.8 * len(block_lengths))] if block_lengths else 0,
    ]


class NumericVect(object):
    def get_feature_names(self):
        return ['text_clf', 'n_tokens', 'n_blocks',
                'block_len_max', 'block_len_med', 'block_len_08']
