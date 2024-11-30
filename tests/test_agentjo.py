import unittest
from unittest.mock import MagicMock

from agentjo import strict_json

class TestStrictJson(unittest.TestCase):
    def test_tutorial0(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{'###Sentiment###': 'Positive', '###Adjectives###': ['beautiful', 'sunny'], '###Words###': '7'}"

        # Call the function under test
        res = strict_json(llm=mock_llm, system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'Array of adjectives',
                                    'Words': 'Number of words'})
        self.assertDictEqual(res, {'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7})
if __name__ == '__main__':
    unittest.main()