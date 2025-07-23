import unittest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch
import pandas as pd
from PIL import Image
import io

# Import your modules (adjust import paths as needed)
from field_technician_ai import EquipmentDatabase, KnowledgeBase, ImageAnalyzer

class TestEquipmentDatabase(unittest.TestCase):
    
    def setUp(self):
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        self.test_db.close()
        self.db = EquipmentDatabase(self.test_db.name)
    
    def tearDown(self):
        os.unlink(self.test_db.name)
    
    def test_get_equipment_info(self):
        """Test equipment info retrieval"""
        info = self.db.get_equipment_info(1)
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('type', info)
    
    def test_get_sensor_data(self):
        """Test sensor data retrieval"""
        df = self.db.get_sensor_data(1, 24)
        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            self.assertIn('temperature', df.columns)
            self.assertIn('pressure', df.columns)

class TestKnowledgeBase(unittest.TestCase):
    
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        self.kb = KnowledgeBase("test_api_key")
    
    def test_search_knowledge(self):
        """Test knowledge base search"""
        results = self.kb.search_knowledge("pump troubleshooting")
        self.assertIsInstance(results, list)

class TestImageAnalyzer(unittest.TestCase):
    
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        self.analyzer = ImageAnalyzer("test_api_key")
    
    def test_analyze_image(self):
        """Test image analysis"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        
        with patch.object(self.analyzer.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test analysis result"
            mock_create.return_value = mock_response
            
            result = self.analyzer.analyze_image(img, "test context")
            self.assertEqual(result, "Test analysis result")

if __name__ == '__main__':
    unittest.main()