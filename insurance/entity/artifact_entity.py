from dataclasses import dataclass

# generates output and save it in artifact
@dataclass
class DataIngestionArtifact:
    feature_Store_file_path: str
    train_file_path: str
    test_file_path: str
