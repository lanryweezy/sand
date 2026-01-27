"""
Tests for RTL Parser

Tests the parsing of Verilog, SDC, and UPF files.
"""

import pytest
import os
from pathlib import Path
from silicon_intelligence.data.rtl_parser import RTLParser


class TestRTLParserVerilog:
    """Tests for Verilog parsing"""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test"""
        return RTLParser()
    
    @pytest.fixture
    def simple_verilog_path(self):
        """Path to simple test Verilog file"""
        return "tests/fixtures/simple.v"
    
    def test_parse_simple_verilog(self, parser, simple_verilog_path):
        """Test parsing a simple Verilog file"""
        result = parser.parse_verilog(simple_verilog_path)
        
        # Check that result has expected keys
        assert 'instances' in result
        assert 'ports' in result
        assert 'nets' in result
        assert 'modules' in result
        assert 'top_module' in result
        
        # Check that we parsed something
        assert len(result['instances']) > 0
        assert len(result['ports']) > 0
        assert len(result['modules']) > 0
    
    def test_parse_instances(self, parser, simple_verilog_path):
        """Test that instances are parsed correctly"""
        result = parser.parse_verilog(simple_verilog_path)
        
        instances = result['instances']
        assert len(instances) >= 2  # At least counter and adder instances
        
        # Check instance structure
        for instance in instances:
            assert 'name' in instance
            assert 'type' in instance
            assert 'connections' in instance
            assert isinstance(instance['name'], str)
            assert isinstance(instance['type'], str)
    
    def test_parse_ports(self, parser, simple_verilog_path):
        """Test that ports are parsed correctly"""
        result = parser.parse_verilog(simple_verilog_path)
        
        ports = result['ports']
        assert len(ports) > 0
        
        # Check port structure
        for port in ports:
            assert 'name' in port
            assert 'direction' in port
            assert 'width' in port
            assert port['direction'] in ['input', 'output', 'inout']
            assert port['width'] >= 1
    
    def test_parse_nets(self, parser, simple_verilog_path):
        """Test that nets are parsed correctly"""
        result = parser.parse_verilog(simple_verilog_path)
        
        nets = result['nets']
        assert len(nets) > 0
        
        # Check net structure
        for net in nets:
            assert 'name' in net
            assert 'type' in net
            assert 'width' in net
            assert net['type'] in ['wire', 'reg']
            assert net['width'] >= 1
    
    def test_parse_modules(self, parser, simple_verilog_path):
        """Test that modules are parsed correctly"""
        result = parser.parse_verilog(simple_verilog_path)
        
        modules = result['modules']
        assert len(modules) >= 3  # counter, adder, top
        
        # Check that expected modules are present
        module_names = set(modules.keys())
        assert 'counter' in module_names or 'adder' in module_names
    
    def test_identify_top_module(self, parser, simple_verilog_path):
        """Test that top module is identified correctly"""
        result = parser.parse_verilog(simple_verilog_path)
        
        top_module = result['top_module']
        assert top_module is not None
        assert len(top_module) > 0
        assert top_module in result['modules']
    
    def test_parse_nonexistent_file(self, parser):
        """Test that parsing nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            parser.parse_verilog("nonexistent_file.v")
    
    def test_parser_reset(self, parser):
        """Test that parser resets state between parses"""
        parser.parse_verilog("tests/fixtures/simple.v")
        first_instance_count = len(parser.instances)
        
        parser.reset()
        assert len(parser.instances) == 0
        assert len(parser.ports) == 0
        assert len(parser.nets) == 0
        assert len(parser.modules) == 0


class TestRTLParserSDC:
    """Tests for SDC constraint parsing"""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test"""
        return RTLParser()
    
    @pytest.fixture
    def sdc_path(self):
        """Path to SDC test file"""
        return "tests/fixtures/constraints.sdc"
    
    def test_parse_sdc(self, parser, sdc_path):
        """Test parsing SDC file"""
        result = parser.parse_sdc(sdc_path)
        
        # Check that result has expected keys
        assert 'clocks' in result
        assert 'timing_paths' in result
        assert 'input_delays' in result
        assert 'output_delays' in result
        assert 'false_paths' in result
    
    def test_parse_sdc_clocks(self, parser, sdc_path):
        """Test that clocks are parsed correctly"""
        result = parser.parse_sdc(sdc_path)
        
        clocks = result['clocks']
        assert len(clocks) >= 2  # At least 2 clocks in test file
        
        # Check clock structure
        for clock in clocks:
            assert 'name' in clock
            assert 'period' in clock
            assert 'uncertainty' in clock
            assert clock['period'] > 0
            assert clock['uncertainty'] >= 0
    
    def test_parse_sdc_timing_paths(self, parser, sdc_path):
        """Test that timing paths are parsed correctly"""
        result = parser.parse_sdc(sdc_path)
        
        paths = result['timing_paths']
        assert len(paths) > 0
        
        # Check path structure
        for path in paths:
            assert 'from' in path
            assert 'to' in path
            assert 'constraint' in path
            assert path['constraint'] > 0
    
    def test_parse_sdc_input_delays(self, parser, sdc_path):
        """Test that input delays are parsed correctly"""
        result = parser.parse_sdc(sdc_path)
        
        delays = result['input_delays']
        assert len(delays) > 0
        
        # Check delay structure
        for delay in delays:
            assert 'port' in delay
            assert 'clock' in delay
            assert 'delay' in delay
            assert delay['delay'] >= 0
    
    def test_parse_sdc_output_delays(self, parser, sdc_path):
        """Test that output delays are parsed correctly"""
        result = parser.parse_sdc(sdc_path)
        
        delays = result['output_delays']
        assert len(delays) > 0
        
        # Check delay structure
        for delay in delays:
            assert 'port' in delay
            assert 'clock' in delay
            assert 'delay' in delay
            assert delay['delay'] >= 0
    
    def test_parse_sdc_false_paths(self, parser, sdc_path):
        """Test that false paths are parsed correctly"""
        result = parser.parse_sdc(sdc_path)
        
        paths = result['false_paths']
        assert len(paths) > 0
        
        # Check path structure
        for path in paths:
            assert 'from' in path
            assert 'to' in path
    
    def test_parse_nonexistent_sdc(self, parser):
        """Test that parsing nonexistent SDC file returns empty result"""
        result = parser.parse_sdc("nonexistent_file.sdc")
        
        assert result['clocks'] == []
        assert result['timing_paths'] == []
        assert result['input_delays'] == []
        assert result['output_delays'] == []
        assert result['false_paths'] == []


class TestRTLParserUPF:
    """Tests for UPF power constraint parsing"""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test"""
        return RTLParser()
    
    @pytest.fixture
    def upf_path(self):
        """Path to UPF test file"""
        return "tests/fixtures/power.upf"
    
    def test_parse_upf(self, parser, upf_path):
        """Test parsing UPF file"""
        result = parser.parse_upf(upf_path)
        
        # Check that result has expected keys
        assert 'power_domains' in result
        assert 'voltage_domains' in result
        assert 'power_switches' in result
        assert 'isolation_cells' in result
    
    def test_parse_upf_power_domains(self, parser, upf_path):
        """Test that power domains are parsed correctly"""
        result = parser.parse_upf(upf_path)
        
        domains = result['power_domains']
        assert len(domains) > 0
        
        # Check domain structure
        for domain in domains:
            assert 'name' in domain
            assert 'supply' in domain
            assert 'ground' in domain
    
    def test_parse_upf_voltage_domains(self, parser, upf_path):
        """Test that voltage domains are parsed correctly"""
        result = parser.parse_upf(upf_path)
        
        domains = result['voltage_domains']
        assert len(domains) > 0
        
        # Check domain structure
        for domain in domains:
            assert 'name' in domain
            assert 'supply' in domain
            assert 'voltage' in domain
    
    def test_parse_upf_power_switches(self, parser, upf_path):
        """Test that power switches are parsed correctly"""
        result = parser.parse_upf(upf_path)
        
        switches = result['power_switches']
        assert len(switches) > 0
        
        # Check switch structure
        for switch in switches:
            assert 'name' in switch
            assert 'domain' in switch
    
    def test_parse_upf_isolation_cells(self, parser, upf_path):
        """Test that isolation cells are parsed correctly"""
        result = parser.parse_upf(upf_path)
        
        cells = result['isolation_cells']
        assert len(cells) > 0
        
        # Check cell structure
        for cell in cells:
            assert 'domain' in cell
            assert 'power_net' in cell
    
    def test_parse_nonexistent_upf(self, parser):
        """Test that parsing nonexistent UPF file returns empty result"""
        result = parser.parse_upf("nonexistent_file.upf")
        
        assert result['power_domains'] == []
        assert result['voltage_domains'] == []
        assert result['power_switches'] == []
        assert result['isolation_cells'] == []


class TestRTLParserIntegration:
    """Integration tests for RTL parser"""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test"""
        return RTLParser()
    
    def test_build_rtl_data_verilog_only(self, parser):
        """Test building RTL data with only Verilog file"""
        result = parser.build_rtl_data("tests/fixtures/simple.v")
        
        # Check that all expected keys are present
        assert 'instances' in result
        assert 'ports' in result
        assert 'nets' in result
        assert 'modules' in result
        assert 'constraints' in result
        assert 'power_info' in result
        
        # Check that constraints and power_info are empty
        assert result['constraints']['clocks'] == []
        assert result['power_info']['power_domains'] == []
    
    def test_build_rtl_data_with_constraints(self, parser):
        """Test building RTL data with Verilog and SDC files"""
        result = parser.build_rtl_data(
            "tests/fixtures/simple.v",
            sdc_file="tests/fixtures/constraints.sdc"
        )
        
        # Check that constraints are parsed
        assert len(result['constraints']['clocks']) > 0
        assert len(result['constraints']['timing_paths']) > 0
    
    def test_build_rtl_data_with_power(self, parser):
        """Test building RTL data with Verilog and UPF files"""
        result = parser.build_rtl_data(
            "tests/fixtures/simple.v",
            upf_file="tests/fixtures/power.upf"
        )
        
        # Check that power info is parsed
        assert len(result['power_info']['power_domains']) > 0
    
    def test_build_rtl_data_complete(self, parser):
        """Test building complete RTL data with all files"""
        result = parser.build_rtl_data(
            "tests/fixtures/simple.v",
            sdc_file="tests/fixtures/constraints.sdc",
            upf_file="tests/fixtures/power.upf"
        )
        
        # Check that all data is present
        assert len(result['instances']) > 0
        assert len(result['ports']) > 0
        assert len(result['nets']) > 0
        assert len(result['modules']) > 0
        assert len(result['constraints']['clocks']) > 0
        assert len(result['power_info']['power_domains']) > 0


class TestRTLParserEdgeCases:
    """Tests for edge cases and error handling"""
    
    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test"""
        return RTLParser()
    
    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing an empty file"""
        empty_file = tmp_path / "empty.v"
        empty_file.write_text("")
        
        result = parser.parse_verilog(str(empty_file))
        
        # Should return empty but valid structure
        assert result['instances'] == []
        assert result['ports'] == []
        assert result['nets'] == []
        assert result['modules'] == {}
    
    def test_parse_file_with_only_comments(self, parser, tmp_path):
        """Test parsing a file with only comments"""
        comment_file = tmp_path / "comments.v"
        comment_file.write_text("""
        // This is a comment
        /* This is a block comment */
        // Another comment
        """)
        
        result = parser.parse_verilog(str(comment_file))
        
        # Should return empty but valid structure
        assert result['instances'] == []
        assert result['ports'] == []
        assert result['nets'] == []
        assert result['modules'] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
