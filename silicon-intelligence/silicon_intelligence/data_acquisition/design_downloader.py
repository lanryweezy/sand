# silicon_intelligence/data_acquisition/design_downloader.py

import os
import subprocess
import tempfile
from typing import Dict, List
import requests
import zipfile
import tarfile
from pathlib import Path


class DesignDownloader:
    def __init__(self, data_dir: str = "./open_source_designs"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Trusted open-source design repositories
        self.design_sources = {
            'picorv32': {
                'url': 'https://github.com/YosysHQ/picorv32/archive/refs/heads/master.zip',
                'description': 'Minimal RISC-V CPU core',
                'type': 'cpu'
            },
            'sha3': {
                'url': 'https://github.com/kokke/tiny-sha3/archive/refs/heads/master.zip',
                'description': 'SHA-3 hash implementation',
                'type': 'crypto'
            },
            'uart': {
                'url': 'https://github.com/olofk/serv/archive/refs/heads/master.zip',
                'description': 'SERV minimal RISC-V implementation with UART',
                'type': 'peripheral'
            }
        }
    
    def download_design(self, design_name: str) -> str:
        """Download and extract an open source design"""
        if design_name not in self.design_sources:
            raise ValueError(f"Unknown design: {design_name}")
        
        source_info = self.design_sources[design_name]
        url = source_info['url']
        
        print(f"Downloading {design_name} from {url}")
        
        # Create download path
        download_path = os.path.join(self.data_dir, f"{design_name}.zip")
        
        # Download the zip file
        response = requests.get(url)
        with open(download_path, 'wb') as f:
            f.write(response.content)
        
        # Extract to design directory
        extract_path = os.path.join(self.data_dir, design_name)
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Clean up zip file
        os.remove(download_path)
        
        print(f"Downloaded and extracted {design_name} to {extract_path}")
        return extract_path
    
    def find_rtl_files(self, source_path: str) -> List[str]:
        """Find all RTL files in the source directory"""
        rtl_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(('.v', '.sv', '.vh', '.svh', '.vhd', '.vhdl')):
                    rtl_files.append(os.path.join(root, file))
        return rtl_files
    
    def get_design_info(self, design_name: str) -> Dict:
        """Get information about a design"""
        if design_name in self.design_sources:
            return self.design_sources[design_name]
        return {}
    
    def download_all_designs(self) -> Dict[str, str]:
        """Download all available designs"""
        results = {}
        for design_name in self.design_sources:
            try:
                path = self.download_design(design_name)
                results[design_name] = path
                print(f"✓ Downloaded {design_name}")
            except Exception as e:
                print(f"✗ Failed to download {design_name}: {e}")
        
        return results