from soft404.utils import get_text_blocks, html_to_item, cleaned_selector


html = '''\
    <div>
        <h2>one <a>two</a>, <br> three</h2>
        yet more <br/>text
        <h1>another</h1>
        block
    </div>'''


def test_get_text_blocks():
    assert get_text_blocks(cleaned_selector(html).root) == [
        ('h2', 'one two three'),
        ('div', 'yet more text'),
        ('h1', 'another'),
        ('div', 'block')]


def test_get_text_blocks_body_child():
    assert get_text_blocks(
        cleaned_selector('<html><body>foo <br/>bar</body></html>').root) == [
        ('body', 'foo bar'),
        ]


def test_html_to_item():
    assert html_to_item('<title>a page</title> <h1>hi!</h1>') == {
        'blocks': [('h1', 'hi !')],
        'text': 'a page hi!',
        'title': 'a page',
    }
    assert html_to_item(html) == {
        'blocks': [('h2', 'one two three'),
                   ('div', 'yet more text'),
                   ('h1', 'another'),
                   ('div', 'block')],
        'text': 'one two, three yet more text another block',
        'title': '',
    }
