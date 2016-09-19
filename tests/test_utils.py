from soft404.utils import get_text_blocks, cleaned_selector, html2text


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


def test_get_text():
    assert html2text(html) == 'one two, three yet more text another block'

