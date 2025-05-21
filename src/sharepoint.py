# src/sharepoint.py
"""Functions for interacting with SharePoint via Microsoft Graph API."""

import streamlit as st
import requests
import time

# --- Constants ---
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# --- Authentication ---

@st.cache_data(ttl=3500) # Cache the token for slightly less than an hour
def get_graph_api_access_token() -> str | None:
    """Retrieves an application access token using Client Credentials Flow."""
    try:
        tenant_id = st.secrets["sharepoint"]["TENANT_ID"]
        client_id = st.secrets["sharepoint"]["CLIENT_ID"]
        client_secret = st.secrets["sharepoint"]["CLIENT_SECRET"]
    except KeyError as e:
        st.error(f"Missing SharePoint credential in secrets.toml: {e}")
        return None

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }

    try:
        token_r = requests.post(token_url, data=token_data, timeout=30)
        token_r.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)
        token = token_r.json().get("access_token")
        if not token:
            st.error("Failed to retrieve access token. Check credentials and permissions.")
            return None
        # st.success("Access token retrieved successfully.") # Optional debug message
        return token
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting access token: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during token retrieval: {e}")
        return None


# --- File Operations ---

def list_msg_files() -> list[dict] | None:
    """Lists .msg files in the configured SharePoint folder."""
    access_token = get_graph_api_access_token()
    if not access_token:
        return None

    try:
        site_id = st.secrets["sharepoint"]["SITE_ID"]
        drive_id = st.secrets["sharepoint"]["DRIVE_ID"]
        folder_path = st.secrets["sharepoint"]["FOLDER_PATH"].strip('/')
    except KeyError as e:
        st.error(f"Missing SharePoint configuration in secrets.toml: {e}")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    # Construct the URL based on whether a folder path is specified
    if folder_path:
        # If path is provided, target the folder within the drive
        list_url = f"{GRAPH_API_ENDPOINT}/sites/{site_id}/drives/{drive_id}/root:/{folder_path}:/children"
    else:
        # If no path, target the root of the drive
        list_url = f"{GRAPH_API_ENDPOINT}/sites/{site_id}/drives/{drive_id}/root/children"

    all_files = []
    params = {
        "$select": "id,name,webUrl,file",
        "$filter": "endswith(name, '.msg')" # Filter for .msg files on the server side
    }

    try:
        while list_url:
            response = requests.get(list_url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            all_files.extend(data.get("value", []))
            list_url = data.get("@odata.nextLink") # Handle pagination
            params = None # Params only needed for the first request

        # Filter again client-side just in case filter failed or wasn't applied (belt and suspenders)
        msg_files = [f for f in all_files if f.get("name", "").lower().endswith(".msg")]
        st.info(f"Found {len(msg_files)} .msg files in SharePoint.")
        return msg_files

    except requests.exceptions.RequestException as e:
        st.error(f"Error listing SharePoint files: {e}")
        if response.status_code == 404:
             st.error(f"Folder not found? Check SITE_ID, DRIVE_ID, and FOLDER_PATH in secrets.")
        elif response.status_code in [401, 403]:
             st.error(f"Permission denied? Check App Registration permissions for MS Graph (Files.ReadWrite.All or similar).")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during file listing: {e}")
        return None


def download_file_content(file_id: str) -> bytes | None:
    """Downloads the content of a specific file by its ID."""
    access_token = get_graph_api_access_token()
    if not access_token:
        return None

    try:
        site_id = st.secrets["sharepoint"]["SITE_ID"]
        drive_id = st.secrets["sharepoint"]["DRIVE_ID"]
    except KeyError as e:
        st.error(f"Missing SharePoint configuration in secrets.toml: {e}")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    # Use the drive item download URL format
    download_url = f"{GRAPH_API_ENDPOINT}/sites/{site_id}/drives/{drive_id}/items/{file_id}/content"

    try:
        response = requests.get(download_url, headers=headers, timeout=120) # Increased timeout for larger files
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file content (ID: {file_id}): {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during file download: {e}")
        return None

