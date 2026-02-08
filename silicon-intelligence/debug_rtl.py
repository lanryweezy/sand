import sys
import os
import traceback
sys.path.append(os.getcwd())

from data.rtl_parser import RTLParser

try:
    # Parse the picorv32 file that was downloaded
    parser = RTLParser()
    print("About to parse RTL file...")
    rtl_data = parser.parse_verilog('./open_source_designs/picorv32/picorv32-main/picorv32.v')
    print("Successfully parsed RTL data")
    print("Keys:", list(rtl_data.keys()))
    print("Nets count:", len(rtl_data.get('nets', [])))
    if rtl_data.get('nets'):
        print("First net keys:", list(rtl_data['nets'][0].keys()) if len(rtl_data['nets']) > 0 else 'None')
        print("First net:", rtl_data['nets'][0] if len(rtl_data['nets']) > 0 else 'None')
    print("Instances count:", len(rtl_data.get('instances', [])))
    if rtl_data.get('instances'):
        print("First instance:", rtl_data['instances'][0] if len(rtl_data['instances']) > 0 else 'None')
except Exception as e:
    print('Error parsing RTL:', str(e))
    traceback.print_exc()