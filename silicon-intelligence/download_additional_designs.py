#!/usr/bin/env python3
"""
Download additional open-source designs for the Silicon Intelligence System
"""

import os
import sys
import requests
import zipfile
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

def download_repo(repo_url, dest_dir, repo_name):
    """Download a GitHub repository"""
    print(f"Downloading {repo_name} from {repo_url}...")
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Download the zip file
    zip_url = f"{repo_url}/archive/refs/heads/master.zip"
    response = requests.get(zip_url)
    
    if response.status_code == 200:
        zip_path = os.path.join(dest_dir, f"{repo_name}.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the zip file
        extract_path = os.path.join(dest_dir, repo_name)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Clean up zip file
        os.remove(zip_path)
        
        print(f"Successfully downloaded and extracted {repo_name}")
        return extract_path
    else:
        print(f"Failed to download {repo_name}: HTTP {response.status_code}")
        return None

def main():
    print("Downloading Additional Open-Source Designs")
    print("=" * 60)
    
    # Define open-source designs to download
    designs = [
        {
            'name': 'ibex',
            'url': 'https://github.com/lowRISC/ibex',
            'description': 'Small CPU core from lowRISC'
        },
        {
            'name': 'sha3',
            'url': 'https://github.com/kokke/tiny-sha3',
            'description': 'SHA-3 hash implementation'
        },
        {
            'name': 'serv',
            'url': 'https://github.com/olofk/serv',
            'description': 'Bit-serial RISC-V core'
        },
        {
            'name': 'vexriscv',
            'url': 'https://github.com/SpinalHDL/VexRiscv',
            'description': 'RISC-V CPU written in SpinalHDL'
        }
    ]
    
    # Destination directory
    base_dir = "./open_source_designs_extended"
    os.makedirs(base_dir, exist_ok=True)
    
    # Download each design
    successful_downloads = 0
    for design in designs:
        print(f"\nDownloading {design['name']}: {design['description']}")
        result = download_repo(
            design['url'],
            base_dir,
            design['name']
        )
        
        if result:
            # Count Verilog files in the downloaded design
            verilog_count = 0
            for root, dirs, files in os.walk(result):
                for file in files:
                    if file.endswith(('.v', '.sv')):
                        verilog_count += 1
            
            print(f"  Found {verilog_count} Verilog/SystemVerilog files")
            successful_downloads += 1
        else:
            print(f"  Failed to download {design['name']}")
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"   Total designs attempted: {len(designs)}")
    print(f"   Successful downloads: {successful_downloads}")
    print(f"   Failed downloads: {len(designs) - successful_downloads}")
    print(f"   Designs stored in: {base_dir}")
    print(f"{'='*60}")
    
    # List what was downloaded
    print(f"\nDownloaded designs:")
    for design in designs:
        design_path = os.path.join(base_dir, design['name'])
        if os.path.exists(design_path):
            print(f"   - {design['name']}")
    
    print(f"\nNext steps:")
    print(f"1. Process these designs with the Silicon Intelligence System")
    print(f"2. Use them for training and validation")
    print(f"3. Integrate with EDA tool flows")

if __name__ == "__main__":
    main()