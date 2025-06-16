from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from core.settings import settings
import datetime

class AzureStorageService:
    def __init__(self):
        if not all([
            settings.AZURE_STORAGE_CONNECTION_STRING,
            settings.AZURE_STORAGE_RFP_CONTAINER_NAME
        ]):
            raise ValueError("Required Azure Storage settings are missing")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(settings.AZURE_STORAGE_RFP_CONTAINER_NAME)

    def upload_file(
        self,
        folder_name: str,
        content: bytes | str,
        blob_name: str,
        metadata: dict | None = None
    ) -> str:
        """
        Uploads content to Azure Blob Storage, accepting either bytes or string input.

        Args:
            folder_name (str): Target virtual folder path inside the container.
            content (bytes | str): The file content to upload; string will be encoded to UTF-8.
            blob_name (str): The target blob's name or relative path.
            metadata (dict, optional): Metadata to set on the blob after upload.

        Returns:
            str: The path to the uploaded blob within the container.
        """
        # Normalize the blob path
        blob_path = f"{folder_name.rstrip('/')}/{blob_name.lstrip('/')}"
    
        # Ensure content is bytes
        if isinstance(content, str):
            file_bytes = content.encode('utf-8')
        else:
            file_bytes = content

        # Get a client and upload
        blob_client = self.container_client.get_blob_client(blob_path)
        blob_client.upload_blob(file_bytes, overwrite=True)

        # Set metadata if provided
        if metadata:
            safe_metadata = self.stringify_metadata(metadata)
            blob_client.set_blob_metadata(safe_metadata)

        return blob_path

    def upload_capabilities(
        self,
        folder_name: str,
        content: bytes | str,
        blob_name: str,
    ) -> str:
    
        # Normalize the blob path
        blob_path = f"{folder_name.rstrip('/')}/{blob_name.lstrip('/')}"

        # Ensure content is bytes
        if isinstance(content, str):
            file_bytes = content.encode('utf-8')
        else:
            file_bytes = content

        # Get a client and upload
        blob_client = self.container_client.get_blob_client(blob_path)
        blob_client.upload_blob(file_bytes, overwrite=True)
        return blob_path

    def generate_blob_sas_url(self, blob_name: str) -> str:
        start_time = datetime.datetime.now(datetime.timezone.utc)
        expiry_time = start_time + datetime.timedelta(days=1)
        account_key = self.get_account_key_from_conn_str(settings.AZURE_STORAGE_CONNECTION_STRING)

        sas_token = generate_blob_sas(
            account_name=self.blob_service_client.account_name,
            container_name=self.container_client.container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            start=start_time,
            expiry=expiry_time,
        )

        blob_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_client.container_name}/{blob_name}?{sas_token}"
        return blob_url

    def add_metadata(
            self, 
            folder: str,
            metadata: dict)-> None:
        """
        Add or update metadata for all blobs in the specified virtual folder.

        Args:
            folder (str): The prefix path inside the container to target (e.g., 'invoices/').
            metadata (dict): Key-value pairs to assign as metadata to each blob.
        """
        
        blob_list = self.container_client.list_blobs(name_starts_with=folder)

        for blob in blob_list:
            safe_metadata = self.stringify_metadata(metadata)
            blob_client = self.container_client.get_blob_client(blob.name)
            blob_client.set_blob_metadata(safe_metadata)

    @staticmethod
    def stringify_metadata(metadata: dict) -> dict:
        """Ensure all metadata keys and values are strings."""
        return {str(k): str(v) for k, v in metadata.items()}
    
    @staticmethod
    def get_account_key_from_conn_str(conn_str: str) -> str:
        """Extract the AccountKey from the connection string."""
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                return part.split("=", 1)[1]
        raise ValueError("AccountKey not found in connection string.")