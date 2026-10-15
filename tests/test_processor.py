from app.document_processor import process_document


def test_process_document():
    result = process_document("uploads/EnzoSousaDosSantos_agenda4_TI_I.pdf")

    assert "chunks" in result
    assert "total_chunks" in result
    assert result["total_chunks"] > 0
    assert len(result["chunks"]) == result["total_chunks"]

    first_chunk = result["chunks"][0]
    assert "text" in first_chunk
    assert "page" in first_chunk
    assert isinstance(first_chunk["page"], int)
    assert first_chunk["page"] >= 1