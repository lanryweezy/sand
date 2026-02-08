# silicon_intelligence/data_processing/design_processor.py

import os
from typing import Dict, Any, List
from data_acquisition.design_downloader import DesignDownloader
from data.rtl_parser import RTLParser
from core.canonical_silicon_graph import CanonicalSiliconGraph


class DesignProcessor:
    def __init__(self):
        self.downloader = DesignDownloader()
        self.parser = RTLParser()
    
    def process_design(self, design_name: str) -> Dict[str, Any]:
        """Process an open source design end-to-end"""
        print(f"Processing design: {design_name}")
        
        # Download the design
        source_path = self.downloader.download_design(design_name)
        
        # Find RTL files
        rtl_files = self.downloader.find_rtl_files(source_path)
        
        if not rtl_files:
            print(f"No RTL files found in {design_name}")
            return {}
        
        # Parse the first RTL file (usually the top module)
        main_rtl_file = rtl_files[0]
        print(f"Parsing main RTL file: {os.path.basename(main_rtl_file)}")
        
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
            'design_name': design_name,
            'source_path': source_path,
            'rtl_files_found': len(rtl_files),
            'main_rtl_file': main_rtl_file,
            'graph_stats': stats,
            'rtl_data': rtl_data,
            'success': True
        }
        
        print(f"Processed {design_name}: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        return result
    
    def process_multiple_designs(self, design_names: List[str]) -> List[Dict]:
        """Process multiple designs"""
        results = []
        for design_name in design_names:
            try:
                result = self.process_design(design_name)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {design_name}: {e}")
        
        return results