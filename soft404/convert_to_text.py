#!/usr/bin/env python
import argparse
import multiprocessing

import json_lines

from soft404.utils import (
    cleaned_selector, selector_to_text, get_text_blocks, get_lang,
    write_pickle_stream)


def convert_item(item):
    try:
        sel = cleaned_selector(item.pop('html'))
        text = selector_to_text(sel)
        text_item = {
            'url': item['url'],
            'text': text,
            'title': ' '.join(sel.xpath('/html/head/title//text()').extract()),
            'status': item['status'],
            'lang': get_lang(text)
        }
        body = sel.xpath('/html/body')
        if body:
            text_item['blocks'] = get_text_blocks(body[0].root)
    except Exception:
        return None
    else:
        return text_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Pages in .jl.gz')
    parser.add_argument('outfile', help='Output in .pkls')
    args = parser.parse_args()

    with json_lines.open(args.infile, broken=True) as f:
        with open(args.outfile, 'wb') as outfile:
            n_errors = 0
            with multiprocessing.Pool() as pool:
                for text_item in pool.imap_unordered(
                        convert_item, f, chunksize=500):
                    if text_item is None:
                        n_errors += 1
                    else:
                        write_pickle_stream(text_item, outfile)

    print('Number of errors: {}'.format(n_errors))


if __name__ == '__main__':
    main()
