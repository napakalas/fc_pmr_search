import json
import zipfile
import os
from pmrsearch.setup import METADATA_FILE, SCKAN2PMR, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE, CURRENT_PATH

with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

release_path = f'{CURRENT_PATH}/output/sckan2pmr-releases/release-sckan2pmr-{metadata["pmrindexer_version"]}.zip'

if os.path.exists(release_path):
    os.remove(release_path)

compression = zipfile.ZIP_DEFLATED
zf = zipfile.ZipFile(release_path, mode="w")
for file_path in [METADATA_FILE, SCKAN2PMR, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE]:
    zf.write(file_path, file_path.split('/')[-1], compression)
zf.close()
