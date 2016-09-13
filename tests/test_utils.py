from soft404.utils import get_text_blocks, cleaned_selector, html2text


html = '''\
    <div>
        <div>one <a>two</a>, <br> three</div>
        <h1>another</h1>
    </div>'''


def test_get_text_blocks():
    assert get_text_blocks(cleaned_selector(html).root) == [
        ('div', 'one two , three'), ('h1', 'another')]


def test_get_text():
    assert html2text(html) == 'one two, three another'

