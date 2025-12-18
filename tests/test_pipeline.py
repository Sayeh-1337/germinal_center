#!/usr/bin/env python
"""Tests for the Germinal Center Analysis Pipeline"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config(self):
        """Test loading configuration from YAML file"""
        from cli.config import load_config
        
        config_path = Path(__file__).parent.parent / "configs" / "dataset1_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert 'dataset_name' in config
            assert config['dataset_name'] == 'dataset1'
    
    def test_get_default_config(self):
        """Test generating default configuration"""
        from cli.config import get_default_config
        
        config = get_default_config("test_dataset")
        assert config['dataset_name'] == "test_dataset"
        assert 'preprocessing' in config
        assert 'segmentation' in config
        assert 'feature_extraction' in config
        assert 'analysis' in config
    
    def test_save_config(self):
        """Test saving configuration to file"""
        from cli.config import save_config, get_default_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config("test")
            output_path = os.path.join(tmpdir, "test_config.yaml")
            save_config(config, output_path)
            assert os.path.exists(output_path)


class TestPreprocess:
    """Test preprocessing functions"""
    
    def test_get_file_list(self):
        """Test file listing function"""
        from cli.commands.preprocess import get_file_list
        
        # Test with non-existent directory
        with pytest.raises(FileNotFoundError):
            get_file_list("/nonexistent/path")
    
    def test_parse_channels(self):
        """Test channel parsing from command line arguments"""
        from cli.main import parse_channels
        
        channels = parse_channels(['dapi:1', 'cd3:2', 'aicda:3'])
        assert channels == {'dapi': 1, 'cd3': 2, 'aicda': 3}
        
        with pytest.raises(ValueError):
            parse_channels(['invalid_format'])


class TestAnalyze:
    """Test analysis functions"""
    
    def test_clean_data(self):
        """Test data cleaning function"""
        import pandas as pd
        import numpy as np
        from cli.commands.analyze import clean_data
        
        # Create sample data
        data = pd.DataFrame({
            'nuc_id': ['1_1', '1_2', '1_3'],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'constant': [1.0, 1.0, 1.0],
            'image': ['img1', 'img1', 'img1']
        })
        
        cleaned = clean_data(data, drop_columns=['image'], index_col='nuc_id')
        
        assert 'constant' not in cleaned.columns  # Constant column removed
        assert 'image' not in cleaned.columns  # Dropped column removed
        assert len(cleaned) == 3
    
    def test_remove_correlated_features(self):
        """Test correlated feature removal"""
        import pandas as pd
        import numpy as np
        from cli.commands.analyze import remove_correlated_features
        
        # Create data with correlated features
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        data['feature3'] = data['feature1'] * 0.99 + np.random.randn(100) * 0.01  # Highly correlated
        
        filtered = remove_correlated_features(data, threshold=0.8)
        
        # Should remove one of the correlated features
        assert len(filtered.columns) < len(data.columns)


class TestCLI:
    """Test CLI functionality"""
    
    def test_create_parser(self):
        """Test argument parser creation"""
        from cli.main import create_parser
        
        parser = create_parser()
        assert parser is not None
    
    def test_init_command(self):
        """Test init command creates config file"""
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_config.yaml")
            
            # This will only work if the package is installed
            # For unit testing, we test the underlying function instead
            from cli.config import get_default_config, save_config
            
            config = get_default_config("test_dataset")
            save_config(config, output_path)
            
            assert os.path.exists(output_path)


def test_imports():
    """Test that all modules can be imported"""
    from cli import main
    from cli import config
    from cli.commands import preprocess
    from cli.commands import segment
    from cli.commands import extract
    from cli.commands import analyze
    from cli.commands import pipeline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

