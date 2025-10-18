from dataclasses import dataclass

@dataclass
class OSIndexSchema:
    index_name: str
    id_field: str
    settings: dict
    mappings: dict
    bulk_chunk_size: int

