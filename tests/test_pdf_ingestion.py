from docling.document_converter import DocumentConverter


def test_pdf_with_table_ingest() -> None:
    converter = DocumentConverter()
    result = converter.convert(
        "https://www.llbean.com/dept_resources/shared/200922_LLBean_Factory_List.pdf?nav=C3taX-518056"
    )
    print(result.document.export_to_markdown())


test_pdf_with_table_ingest()
