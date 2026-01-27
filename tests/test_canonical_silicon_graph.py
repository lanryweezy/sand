"""
Tests for CanonicalSiliconGraph

Tests the graph robustness features including deepcopy, validation, serialization, and transactions.
"""

import pytest
import json
import os
import tempfile
import copy
from pathlib import Path
from silicon_intelligence.core.canonical_silicon_graph import (
    CanonicalSiliconGraph, NodeType, EdgeType, NodeAttributes, EdgeAttributes
)


class TestGraphDeepCopy:
    """Tests for graph deepcopy functionality"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing"""
        graph = CanonicalSiliconGraph()
        
        # Add some nodes
        for i in range(5):
            graph.graph.add_node(f"cell_{i}", 
                               node_type=NodeType.CELL,
                               area=1.0 + i,
                               power=0.01 * (i + 1),
                               timing_criticality=0.5)
        
        # Add some edges
        for i in range(4):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION,
                               delay=0.1)
        
        return graph
    
    def test_deepcopy_creates_independent_copy(self, sample_graph):
        """Test that deepcopy creates an independent copy"""
        copied_graph = copy.deepcopy(sample_graph)
        
        # Verify they are different objects
        assert copied_graph is not sample_graph
        assert copied_graph.graph is not sample_graph.graph
        assert copied_graph.metadata is not sample_graph.metadata
    
    def test_deepcopy_preserves_data(self, sample_graph):
        """Test that deepcopy preserves all data"""
        copied_graph = copy.deepcopy(sample_graph)
        
        # Check nodes
        assert copied_graph.graph.number_of_nodes() == sample_graph.graph.number_of_nodes()
        assert set(copied_graph.graph.nodes()) == set(sample_graph.graph.nodes())
        
        # Check edges
        assert copied_graph.graph.number_of_edges() == sample_graph.graph.number_of_edges()
        
        # Check node attributes
        for node in sample_graph.graph.nodes():
            for attr in sample_graph.graph.nodes[node]:
                assert copied_graph.graph.nodes[node][attr] == sample_graph.graph.nodes[node][attr]
    
    def test_deepcopy_modifications_dont_affect_original(self, sample_graph):
        """Test that modifications to copy don't affect original"""
        copied_graph = copy.deepcopy(sample_graph)
        
        # Modify the copy
        copied_graph.graph.add_node("new_cell", node_type=NodeType.CELL, area=5.0)
        copied_graph.graph.nodes["cell_0"]["area"] = 999.0
        
        # Verify original is unchanged
        assert "new_cell" not in sample_graph.graph.nodes()
        assert sample_graph.graph.nodes["cell_0"]["area"] == 1.0
    
    def test_deepcopy_with_large_graph(self):
        """Test deepcopy with a large graph"""
        graph = CanonicalSiliconGraph()
        
        # Create a large graph (1000 nodes)
        for i in range(1000):
            graph.graph.add_node(f"cell_{i}",
                               node_type=NodeType.CELL,
                               area=1.0,
                               power=0.01)
        
        # Add edges
        for i in range(999):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION)
        
        # Deepcopy should work
        copied_graph = copy.deepcopy(graph)
        
        assert copied_graph.graph.number_of_nodes() == 1000
        assert copied_graph.graph.number_of_edges() == 999
    
    def test_copy_method(self, sample_graph):
        """Test shallow copy method"""
        copied_graph = sample_graph.copy()
        
        # Should be different objects
        assert copied_graph is not sample_graph


class TestGraphValidation:
    """Tests for graph consistency validation"""
    
    @pytest.fixture
    def valid_graph(self):
        """Create a valid graph"""
        graph = CanonicalSiliconGraph()
        
        # Add valid nodes
        for i in range(3):
            graph.graph.add_node(f"cell_{i}",
                               node_type=NodeType.CELL,
                               area=1.0,
                               power=0.01,
                               timing_criticality=0.5,
                               estimated_congestion=0.3)
        
        # Add valid edges
        for i in range(2):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION)
        
        return graph
    
    def test_validate_valid_graph(self, valid_graph):
        """Test validation of a valid graph"""
        is_valid, errors = valid_graph.validate_graph_consistency()
        
        assert is_valid
        assert len(errors) == 0
    
    def test_detect_missing_node_attributes(self):
        """Test detection of missing node attributes"""
        graph = CanonicalSiliconGraph()
        
        # Add node without required attributes
        graph.graph.add_node("bad_cell")
        
        is_valid, errors = graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("missing required attribute" in error for error in errors)
    
    def test_detect_invalid_timing_criticality(self):
        """Test detection of invalid timing criticality"""
        graph = CanonicalSiliconGraph()
        
        # Add node with invalid timing criticality
        graph.graph.add_node("cell_0",
                           node_type=NodeType.CELL,
                           timing_criticality=1.5)  # Invalid: > 1.0
        
        is_valid, errors = graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("invalid timing_criticality" in error for error in errors)
    
    def test_detect_invalid_congestion(self):
        """Test detection of invalid congestion"""
        graph = CanonicalSiliconGraph()
        
        # Add node with invalid congestion
        graph.graph.add_node("cell_0",
                           node_type=NodeType.CELL,
                           estimated_congestion=-0.1)  # Invalid: < 0.0
        
        is_valid, errors = graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("invalid estimated_congestion" in error for error in errors)
    
    def test_detect_negative_area(self):
        """Test detection of negative area"""
        graph = CanonicalSiliconGraph()
        
        # Add node with negative area
        graph.graph.add_node("cell_0",
                           node_type=NodeType.CELL,
                           area=-1.0)  # Invalid: negative
        
        is_valid, errors = graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("negative area" in error for error in errors)
    
    def test_detect_negative_power(self):
        """Test detection of negative power"""
        graph = CanonicalSiliconGraph()
        
        # Add node with negative power
        graph.graph.add_node("cell_0",
                           node_type=NodeType.CELL,
                           power=-0.01)  # Invalid: negative
        
        is_valid, errors = graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("negative power" in error for error in errors)
    
    def test_detect_missing_edge_attributes(self, valid_graph):
        """Test detection of missing edge attributes"""
        # Add edge without required attributes
        valid_graph.graph.add_edge("cell_0", "cell_1")
        
        is_valid, errors = valid_graph.validate_graph_consistency()
        
        assert not is_valid
        assert any("missing required attribute" in error for error in errors)
    
    def test_detect_edge_with_missing_endpoint(self):
        """Test detection of edge with missing endpoint"""
        graph = CanonicalSiliconGraph()
        
        # Add node
        graph.graph.add_node("cell_0", node_type=NodeType.CELL)
        
        # Add edge to non-existent node (NetworkX auto-creates the node)
        graph.graph.add_edge("cell_0", "nonexistent", edge_type=EdgeType.CONNECTION)
        
        is_valid, errors = graph.validate_graph_consistency()
        
        # The nonexistent node will be created but won't have required attributes
        assert not is_valid
        assert any("missing required attribute" in error for error in errors)


class TestGraphSerialization:
    """Tests for graph serialization/deserialization"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for serialization testing"""
        graph = CanonicalSiliconGraph()
        
        # Add nodes
        for i in range(3):
            graph.graph.add_node(f"cell_{i}",
                               node_type=NodeType.CELL,
                               area=1.0 + i,
                               power=0.01 * (i + 1),
                               timing_criticality=0.5)
        
        # Add edges
        for i in range(2):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION,
                               delay=0.1)
        
        return graph
    
    def test_serialize_to_json(self, sample_graph):
        """Test serialization to JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            
            # Serialize
            sample_graph.serialize_to_json(filepath)
            
            # Verify file exists
            assert os.path.exists(filepath)
            
            # Verify it's valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'nodes' in data
            assert 'edges' in data
            assert 'metadata' in data
    
    def test_deserialize_from_json(self, sample_graph):
        """Test deserialization from JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            
            # Serialize
            sample_graph.serialize_to_json(filepath)
            
            # Deserialize
            new_graph = CanonicalSiliconGraph()
            new_graph.deserialize_from_json(filepath)
            
            # Verify data is preserved
            assert new_graph.graph.number_of_nodes() == sample_graph.graph.number_of_nodes()
            assert new_graph.graph.number_of_edges() == sample_graph.graph.number_of_edges()
    
    def test_serialization_roundtrip(self, sample_graph):
        """Test that serialization roundtrip preserves all data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "graph.json")
            
            # Serialize
            sample_graph.serialize_to_json(filepath)
            
            # Deserialize
            new_graph = CanonicalSiliconGraph()
            new_graph.deserialize_from_json(filepath)
            
            # Verify nodes
            for node in sample_graph.graph.nodes():
                assert node in new_graph.graph.nodes()
                for attr in sample_graph.graph.nodes[node]:
                    assert new_graph.graph.nodes[node][attr] == sample_graph.graph.nodes[node][attr]
            
            # Verify edges
            for src, dst, key in sample_graph.graph.edges(keys=True):
                assert new_graph.graph.has_edge(src, dst)
    
    def test_serialize_large_graph(self):
        """Test serialization of large graph"""
        graph = CanonicalSiliconGraph()
        
        # Create large graph
        for i in range(1000):
            graph.graph.add_node(f"cell_{i}",
                               node_type=NodeType.CELL,
                               area=1.0,
                               power=0.01)
        
        for i in range(999):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "large_graph.json")
            
            # Should serialize without error
            graph.serialize_to_json(filepath)
            
            # Should deserialize without error
            new_graph = CanonicalSiliconGraph()
            new_graph.deserialize_from_json(filepath)
            
            assert new_graph.graph.number_of_nodes() == 1000
            assert new_graph.graph.number_of_edges() == 999
    
    def test_deserialize_nonexistent_file(self):
        """Test deserialization of nonexistent file"""
        graph = CanonicalSiliconGraph()
        
        with pytest.raises(IOError):
            graph.deserialize_from_json("nonexistent_file.json")
    
    def test_deserialize_invalid_json(self):
        """Test deserialization of invalid JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "invalid.json")
            
            # Write invalid JSON
            with open(filepath, 'w') as f:
                f.write("{ invalid json }")
            
            graph = CanonicalSiliconGraph()
            
            with pytest.raises(IOError):
                graph.deserialize_from_json(filepath)


class TestGraphTransactions:
    """Tests for graph transaction support"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph"""
        graph = CanonicalSiliconGraph()
        
        graph.graph.add_node("cell_0", node_type=NodeType.CELL, area=1.0)
        graph.graph.add_node("cell_1", node_type=NodeType.CELL, area=2.0)
        
        return graph
    
    def test_transaction_success(self, sample_graph):
        """Test successful transaction"""
        with sample_graph.transaction():
            sample_graph.graph.add_node("cell_2", node_type=NodeType.CELL, area=3.0)
            sample_graph.graph.nodes["cell_0"]["area"] = 10.0
        
        # Changes should be applied
        assert "cell_2" in sample_graph.graph.nodes()
        assert sample_graph.graph.nodes["cell_0"]["area"] == 10.0
    
    def test_transaction_rollback_on_error(self, sample_graph):
        """Test transaction rollback on error"""
        original_nodes = set(sample_graph.graph.nodes())
        original_area = sample_graph.graph.nodes["cell_0"]["area"]
        
        try:
            with sample_graph.transaction():
                sample_graph.graph.add_node("cell_2", node_type=NodeType.CELL, area=3.0)
                sample_graph.graph.nodes["cell_0"]["area"] = 10.0
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Changes should be rolled back
        assert set(sample_graph.graph.nodes()) == original_nodes
        assert sample_graph.graph.nodes["cell_0"]["area"] == original_area
        assert "cell_2" not in sample_graph.graph.nodes()
    
    def test_transaction_preserves_metadata(self, sample_graph):
        """Test that transaction preserves metadata"""
        sample_graph.metadata['test_key'] = 'test_value'
        
        try:
            with sample_graph.transaction():
                sample_graph.metadata['test_key'] = 'modified'
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Metadata should be rolled back
        assert sample_graph.metadata['test_key'] == 'test_value'


class TestGraphStatistics:
    """Tests for graph statistics"""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph"""
        graph = CanonicalSiliconGraph()
        
        # Add nodes with different types
        graph.graph.add_node("cell_0", node_type=NodeType.CELL, area=1.0, power=0.01)
        graph.graph.add_node("cell_1", node_type=NodeType.CELL, area=2.0, power=0.02)
        graph.graph.add_node("macro_0", node_type=NodeType.MACRO, area=10.0, power=0.1)
        
        # Add edges
        graph.graph.add_edge("cell_0", "cell_1", edge_type=EdgeType.CONNECTION)
        graph.graph.add_edge("cell_1", "macro_0", edge_type=EdgeType.CONNECTION)
        
        return graph
    
    def test_get_graph_statistics(self, sample_graph):
        """Test getting graph statistics"""
        stats = sample_graph.get_graph_statistics()
        
        assert stats['num_nodes'] == 3
        assert stats['num_edges'] == 2
        assert stats['total_area'] == 13.0  # 1 + 2 + 10
        assert stats['total_power'] == 0.13  # 0.01 + 0.02 + 0.1
        assert 'node_types' in stats
        assert 'edge_types' in stats
    
    def test_statistics_with_empty_graph(self):
        """Test statistics with empty graph"""
        graph = CanonicalSiliconGraph()
        
        stats = graph.get_graph_statistics()
        
        assert stats['num_nodes'] == 0
        assert stats['num_edges'] == 0
        assert stats['total_area'] == 0.0
        assert stats['total_power'] == 0.0


class TestGraphIntegration:
    """Integration tests for graph robustness features"""
    
    def test_deepcopy_and_serialize(self):
        """Test deepcopy followed by serialization"""
        graph = CanonicalSiliconGraph()
        
        # Add data
        for i in range(5):
            graph.graph.add_node(f"cell_{i}",
                               node_type=NodeType.CELL,
                               area=1.0 + i,
                               power=0.01 * (i + 1))
        
        for i in range(4):
            graph.graph.add_edge(f"cell_{i}", f"cell_{i+1}",
                               edge_type=EdgeType.CONNECTION)
        
        # Deepcopy
        copied_graph = copy.deepcopy(graph)
        
        # Serialize both
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.path.join(tmpdir, "original.json")
            copied_path = os.path.join(tmpdir, "copied.json")
            
            graph.serialize_to_json(original_path)
            copied_graph.serialize_to_json(copied_path)
            
            # Deserialize and compare
            graph1 = CanonicalSiliconGraph()
            graph1.deserialize_from_json(original_path)
            
            graph2 = CanonicalSiliconGraph()
            graph2.deserialize_from_json(copied_path)
            
            assert graph1.graph.number_of_nodes() == graph2.graph.number_of_nodes()
            assert graph1.graph.number_of_edges() == graph2.graph.number_of_edges()
    
    def test_transaction_with_validation(self):
        """Test transaction with validation"""
        graph = CanonicalSiliconGraph()
        
        graph.graph.add_node("cell_0", node_type=NodeType.CELL, area=1.0)
        
        # Valid transaction
        with graph.transaction():
            graph.graph.add_node("cell_1", node_type=NodeType.CELL, area=2.0)
            # Add edge to make nodes not orphaned
            graph.graph.add_edge("cell_0", "cell_1", edge_type=EdgeType.CONNECTION)
        
        is_valid, errors = graph.validate_graph_consistency()
        assert is_valid
        
        # Invalid transaction (should rollback)
        try:
            with graph.transaction():
                graph.graph.add_node("bad_cell")  # Missing required attributes
                raise ValueError("Test")
        except ValueError:
            pass
        
        # Should still be valid
        is_valid, errors = graph.validate_graph_consistency()
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
