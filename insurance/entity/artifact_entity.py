from dataclasses import dataclass

# generates output and save it in artifact
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    report_file_path: str
