import json
import zipfile
from pmrsearch.setup import METADATA_FILE, SCKAN2PMR, CURRENT_PATH

with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

release_name = f'{CURRENT_PATH}/output/sckan2pmr-releases/release-sckan2pmr-{metadata["pmrindexer_version"]}.zip'

compression = zipfile.ZIP_DEFLATED
zf = zipfile.ZipFile(release_name, mode="w")
zf.write(METADATA_FILE, METADATA_FILE.split('/')[-1], compression)
zf.write(SCKAN2PMR, SCKAN2PMR.split('/')[-1], compression)
zf.close()
