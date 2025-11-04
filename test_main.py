import unittest
from unittest.mock import patch, Mock
import cv2
import json
from jsonschema import validate
from main import GPTClient, extract_images, process_video, main

class TestMain(unittest.TestCase):

    @patch('main.OpenAI')
    def test_init(self, mock_gpt_client):
        """
        Test initialization of GPTClient.
        """

        mock_api_key = "mock_api_key"
        gpt_client = GPTClient(client_key=mock_api_key)
        mock_gpt_client.assert_called_with(api_key=mock_api_key)

    @patch('main.OpenAI')
    def test_query_model(self, mock_gpt_client):
        """
        Test querying the model.
        """

        mock_response = Mock()
        mock_response.output_text = "This is a mocked response."
        mock_client = Mock()
        mock_client.responses.create.return_value = mock_response

        mock_gpt_client.return_value = mock_client

        mock_api_key = "mock_api_key"
        gpt_client = GPTClient(client_key=mock_api_key)

        result = gpt_client.query_model(
            model="gpt-4o",
            query="This is a test query.",
            inputs=[]
        )

        self.assertEqual(result, "This is a mocked response.")

        # with image input
        result_with_input = gpt_client.query_model(
            model="gpt-4o",
            query="This is a test query.",
            inputs=[{"image_base64": "data:image/png;base64,abc123"}]
        )

        self.assertEqual(result_with_input, "This is a mocked response.")

        # with text input
        result_with_text_input = gpt_client.query_model(
            model="gpt-4o",
            query="This is a test query.",
            inputs=[{"text": "This is some sample text."}]
        )

        self.assertEqual(result_with_text_input, "This is a mocked response.")

    @patch('main.OpenAI')
    def test_query_model_with_exception(self, mock_gpt_client):
        """
        Test querying the model with exception handling.
        """

        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("API error")
        mock_gpt_client.return_value = mock_client

        gpt_client = GPTClient(client_key="mock")
        result = gpt_client.query_model(
            "gpt-4o", 
            "This is a test query."
        )

        self.assertEqual(result, "")

    @patch('main.OpenAI')
    def test_transcribe_audio(self, mock_gpt_client):
        """
        Test audio transcription.
        """

        mock_transcription = Mock()
        mock_transcription.text = "This is a mocked transcription."

        mock_audio = Mock()
        mock_audio.transcriptions.create.return_value = mock_transcription
        
        mock_client = Mock()
        mock_client.audio = mock_audio
        mock_gpt_client.return_value = mock_client

        mock_api_key = "mock_api_key"
        gpt_client = GPTClient(client_key=mock_api_key)
        result = gpt_client.transcribe_audio("mock_audio_file.mp3")

        self.assertEqual(result, "This is a mocked transcription.")

    @patch('main.OpenAI')
    def test_transcribe_audio_with_exception(self, mock_gpt_client):
        """
        Test audio transcription with exception handling.
        """

        mock_client = Mock()
        mock_client.audio.transcriptions.create.side_effect = Exception("Transcription error")
        mock_gpt_client.return_value = mock_client

        gpt_client = GPTClient(client_key="mock")
        result = gpt_client.transcribe_audio("mock_audio_file.mp3")

        self.assertEqual(result, "")

    def test_extract_images(self):
        """
        Test frame extraction from video.
        """

        video = cv2.VideoCapture("./data/AI_Intern_Project.mp4")
        frames = extract_images(video) 

        self.assertIsInstance(frames, list)
        self.assertTrue(all(isinstance(f, str) for f in frames))
        video.release()

    def test_extract_images_with_exception(self):
        """
        Test frame extraction with exception handling.
        """

        mock_video = Mock()
        mock_video.get.return_value = 1
        mock_video.read.return_value = (False, None)

        with self.assertRaises(Exception) as context:
            extract_images(mock_video)

        self.assertEqual(str(context.exception), "No frames extracted from video.")

    def test_process_video(self):
        """
        Test the entire video processing pipeline.
        """

        mock_client = Mock()
        mock_client.transcribe_audio.return_value = "this is a mocked transcription."
        mock_client.query_model.side_effect = [
            "object1, object2",
            "positive",
            {"Q1": "A1", "Q2": "A2"}
        ]

        video_file = "./data/AI_Intern_Project.mp4"
        responses = process_video(mock_client, video_file)

        self.assertIsInstance(responses, dict)
        self.assertIn("transcript", responses)
        self.assertIn("objects", responses)
        self.assertIn("sentiment", responses)
        self.assertIn("QAs", responses)

    def test_process_video_with_exception(self):
        """
        Test video processing with file not found exception.
        """

        with self.assertRaises(Exception) as context:
            process_video(Mock(), "./data/non_existent_file.mp4")

        self.assertEqual(str(context.exception), "No frames extracted from video.")

    def test_main(self):
        """
        Test final output.
        """

        responses = main()

        self.assertIsInstance(responses, dict)
        self.assertTrue(len(responses) == 4)

        # Validate JSON schema
        schema = {
            "type": "object",
            "properties": {
                "transcript": {"type": "string"},
                "objects": {"type": "string"},
                "sentiment": {"type": "string"},
                "QAs": {"type": ["string", "object"]}
            },
            "required": ["transcript", "objects", "sentiment", "QAs"]
        }

        validate(instance=responses, schema=schema)

if __name__ == '__main__':
    unittest.main()