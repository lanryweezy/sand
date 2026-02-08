# silicon_intelligence/data_integration/open_source_data.py

import os
import subprocess
import tempfile
from typing import Dict, List
import requests
import zipfile
from data.rtl_parser import RTLParser
from core.canonical_silicon_graph import CanonicalSiliconGraph


class OpenSourceDataPipeline:
    def __init__(self, data_dir: str = "./open_source_data"):
        self.data_dir = data_dir
        self.parser = RTLParser()
        os.makedirs(data_dir, exist_ok=True)
        
        self.sources = {
            'picorv32': {
                'url': 'https://github.com/YosysHQ/picorv32/archive/refs/heads/master.zip',
                'description': 'Minimal RISC-V CPU core'
            }
        }
    
    def download_source(self, source_name: str) -> str:
        """Download and extract open source design"""
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        url = self.sources[source_name]['url']
        download_path = os.path.join(self.data_dir, f"{source_name}.zip")
        
        print(f"Downloading {source_name} from {url}")
        
        # Download the zip file
        response = requests.get(url)
        with open(download_path, 'wb') as f:
            f.write(response.content)
        
        # Extract to data directory
        extract_path = os.path.join(self.data_dir, source_name)
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Clean up zip file
        os.remove(download_path)
        
        print(f"Downloaded and extracted {source_name} to {extract_path}")
        return extract_path
    
    def find_rtl_files(self, source_path: str) -> List[str]:
        """Find all RTL files in the source directory"""
        rtl_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(('.v', '.sv', '.vh', '.svh')):
                    rtl_files.append(os.path.join(root, file))
        return rtl_files
    
    def process_design(self, source_name: str) -> Dict:
        """Process an open source design end-to-end"""
        # Download the source
        source_path = self.download_source(source_name)
        
        # Find RTL files
        rtl_files = self.find_rtl_files(source_path)
        
        if not rtl_files:
            print(f"No RTL files found in {source_name}")
            return {}
        
        # Parse the first RTL file (usually the top module)
        main_rtl_file = rtl_files[0]
        print(f"Parsing main RTL file: {main_rtl_file}")
        
        # Build canonical silicon graph from RTL
        try:
            rtl_data = self.parser.parse_verilog(main_rtl_file)
        except Exception as e:
            print(f"Error parsing RTL: {e}")
            return {}
        
        # Create canonical graph
        graph = CanonicalSiliconGraph()
        graph.build_from_rtl(rtl_data)
        
        # Analyze the design
        stats = graph.get_graph_statistics()
        
        result = {
            'source_name': source_name,
            'source_path': source_path,
            'rtl_files_found': len(rtl_files),
            'main_rtl_file': main_rtl_file,
            'graph_stats': stats,
            'rtl_data': rtl_data
        }
        
        print(f"Processed {source_name}: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        return result