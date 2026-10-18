"""Resolve archived source fingerprints after the runtime directory relocation."""
import hashlib


def historical_source(root, relative_path, expected):
    current = root/relative_path
    if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest()==expected:
        return current
    archived = root/'data/training_sources'/(expected+'.py')
    if not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest()!=expected:
        raise ValueError('Missing or changed training source: '+relative_path)
    return archived
