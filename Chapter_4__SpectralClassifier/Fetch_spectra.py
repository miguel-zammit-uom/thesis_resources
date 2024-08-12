import os
import sys
import requests
import cgi
import pyvo
import pyvo.dal
import eso_programmatic as eso
import getpass
from tqdm import tqdm
import pandas as pd


# DISCLAIMER: This code is heavily based on the tutorial provided for programmatic access 
# on the ESO Science Archive Facility
user_credentials = {
    'username': # Enter your ESO Username,
    'password': # Enter your ESO Password
}
dataset_csv_path = # Path to csv including list of all observation ids

class FetchSpectraFiles:

    def __init__(self, tap_url, file_list, file_destination, credentials):

        self.tap_url = tap_url
        self.file_list = file_list
        self.file_destination = file_destination
        self.credentials = credentials

        self.file_url_list = self._file_url_list_generator(file_list)

        self.session, self.tap, self.token = self._login_session(self.tap_url, self.credentials)
        self._fetch_files(file_url_list=self.file_url_list, token=self.token, file_destination=self.file_destination)

    def _get_disposition_filename(self, response):
        """Get the filename from the Content-Disposition in the response's http header"""
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition is None:
            return None
        value, params = cgi.parse_header(content_disposition)
        filename = params["filename"]
        return filename

    def _write_file(self, response, file_destination):
        """Write on disk the retrieved file"""
        if response.status_code == 200:
            # The ESO filename can be found in the response header
            filename = self._get_disposition_filename(response)
            # Let's write on disk the downloaded FITS spectrum using the ESO filename:
            with open(os.path.join(file_destination, filename), 'wb') as f:
                f.write(response.content)
            return filename

    def _file_url_list_generator(self, filelist):
        """Generates access_urls to fetch data"""
        file_url_list = []
        root = 'https://dataportal.eso.org/dataportal_new/file'
        for file in filelist:
            access_url = os.path.join(root, file)
            file_url_list.append(access_url)
        return file_url_list

    def _login_session(self, tap_url, credentials):
        """Start Login Session on ESO Archive Server"""
        username = credentials['username']
        password = credentials['password']

        token = eso.getToken(username, password)
        if token is not None:
            print('token: ' + token)
        else:
            sys.exit(-1)

        session = requests.Session()
        session.headers['Authorization'] = "Bearer " + token

        tap = pyvo.dal.TAPService(tap_url, session=session)

        return session, tap, token

    def _fetch_files(self, file_url_list, token, file_destination):
        for file_url in tqdm(file_url_list):
            headers = None
            if token is not None:
                headers = {"Authorization": "Bearer " + token}
                response = requests.get(file_url, headers=headers)
                filename = self._write_file(response, file_destination=file_destination)
                if not filename:
                    print("Could not get file (status: %d)" % response.status_code)
            else:
                print("Could not authenticate")

data = pd.read_csv(dataset_csv_path)
url_list = data['dp_id'].values

FetchSpectraFiles(tap_url="http://archive.eso.org/tap_obs", file_list=url_list,
                  file_destination='path/to/data/output')
