import pytest
from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from app.document_processor import process_document, split_text_with_page


def test_process_document(sample_pdf):
    result = process_document(str(sample_pdf))
    assert result['total_chunks'] == 2
    assert [chunk['page'] for chunk in result['chunks']] == [1, 2]
    assert 'renewable energy' in result['chunks'][0]['text']
    assert 'Sao Paulo' in result['chunks'][1]['text']


def test_blank_pdf(tmp_path):
    path = tmp_path / 'blank.pdf'
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.write(path)
    assert process_document(str(path)) == {'chunks': [], 'total_chunks': 0}


def test_corrupt_pdf(tmp_path):
    path = tmp_path / 'broken.pdf'
    path.write_bytes(b'not a pdf')
    with pytest.raises(PdfReadError):
        process_document(str(path))


def test_chunk_overlap():
    chunks = split_text_with_page('abcdefghijklmnop', 7, chunk_size=8, overlap=2)
    assert chunks == [{'text': 'abcdefgh', 'page': 7}, {'text': 'ghijklmn', 'page': 7}, {'text': 'mnop', 'page': 7}]
    assert split_text_with_page('', 1) == []


@pytest.mark.parametrize('size,overlap', [(0, 0), (5, 5), (5, 6), (5, -1)])
def test_invalid_chunk_configuration(size, overlap):
    with pytest.raises(ValueError):
        split_text_with_page('text', 1, chunk_size=size, overlap=overlap)
