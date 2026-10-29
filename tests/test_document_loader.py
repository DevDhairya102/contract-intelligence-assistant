import pytest
from pathlib import Path
from src.document_loader import DocumentLoader

def test_document_loader_loads_all_contracts():
    loader = DocumentLoader(data_dir="data")
    contracts = loader.list_contracts()
    assert len(contracts) == 29, f"Expected 29 contracts, found {len(contracts)}"

def test_document_loader_retrieval_and_slice():
    loader = DocumentLoader(data_dir="data")
    contracts = loader.list_contracts()
    first_c = contracts[0]
    
    text = loader.get_document(first_c)
    assert text is not None
    assert len(text) > 100
    
    # Test slice
    slice_text = loader.get_slice(first_c, 10, 50)
    assert slice_text == text[10:50]

def test_canonical_path_resolution():
    loader = DocumentLoader(data_dir="data")
    sample_file = "ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT.txt"
    
    p1 = loader.get_canonical_path(sample_file)
    p2 = loader.get_canonical_path(f"cuad/{sample_file}")
    p3 = loader.get_canonical_path(f"corpus/cuad/{sample_file}")
    
    assert p1 == p2 == p3 == f"cuad/{sample_file}"
