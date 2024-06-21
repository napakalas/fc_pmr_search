from pmrsearch.setup import loadJson, EXPOSURES_FILE, SCHEMA_EXPOSURES_FILE, METADATA_FILE, SCHEMA_METADATA_FILE, SCKAN2PMR, SCHEMA_SCKAN2PMR_FILE
from jsonschema import validate, ValidationError

#===============================================================================

def __validate_json(data, schema_file):
    schema = loadJson(schema_file)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        print(e)

if __name__ == '__main__':
    __validate_json(loadJson(EXPOSURES_FILE), SCHEMA_EXPOSURES_FILE)
    __validate_json(loadJson(METADATA_FILE), SCHEMA_METADATA_FILE)
    __validate_json(loadJson(SCKAN2PMR), SCHEMA_SCKAN2PMR_FILE)
#===============================================================================
    