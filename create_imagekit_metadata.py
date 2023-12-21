from imagekitio import ImageKit
from imagekitio.models.CreateCustomMetadataFieldsRequestOptions import CreateCustomMetadataFieldsRequestOptions
from imagekitio.models.CustomMetadataFieldsSchema import CustomMetadataFieldsSchema
from imagekitio.models.CustomMetaDataTypeEnum import CustomMetaDataTypeEnum

imagekit = ImageKit(
        private_key='private_7suIdwvpEqGwxva69nvPXEJw774=',
        public_key='public_MbCiDZEy0QGOxSiwKrwcBbFmzaQ=',
        url_endpoint='https://ik.imagekit.io/endlessprompts'
)

create_custom_metadata_fields_number = imagekit.create_custom_metadata_fields(
    options=CreateCustomMetadataFieldsRequestOptions(
                                                        name="prompt",
                                                        label="prompt",
                                                        schema=CustomMetadataFieldsSchema(
                                                            type=CustomMetaDataTypeEnum.Textarea
                                                        )
                                                     )
)
# Final Result
print(create_custom_metadata_fields_number)

# Raw Response
print(create_custom_metadata_fields_number.response_metadata.raw)

# print the id of created custom metadata fields
print(create_custom_metadata_fields_number.id)

# print the schema's type of created custom metadata fields
print(create_custom_metadata_fields_number.schema.type)