from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
from video_to_mp3 import video_to_mp3
import cv2
import pprint
import json

class GPTClient:
    def __init__(self, client_key):
        """
        Initialize the class.
        """

        self.client = OpenAI(api_key=client_key)

    def query_model(self, model, query, inputs=[]):
        """
        Prompt template.

        Params:
            query (str): task description.
            input (list): additional contents

        Returns:
            Query response
        """

        messages = [{"role": "user", "content": query}]

        for input_item in inputs:
            if isinstance(input_item, dict):
                if "text" in input_item:
                    messages.append({"role": "user", "content": input_item["text"]})
                if "image_base64" in input_item:
                    messages.append({
                        "role": "user", 
                        "content": [
                            {"type": "input_text", "text": query},
                            {"type": "input_image", "image_url": input_item["image_base64"]}
                        ]
                    })
            elif isinstance(input_item, str):
                messages.append({"role": "user", "content": input_item})

        try:
            result = self.client.responses.create(
                model=model,
                input=messages
            )
            return result.output_text
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""

    def transcribe_audio(self, audio_file):
        """
        Convert audio to text.

        Params:
            audio_file (file): Audio file to transcribe

        Returns:
            Text transcript
        """

        try:
            transcription = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=audio_file
            )
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""

        return transcription.text


def extract_images(video, step=100):
    """
    Extract frames from video.

    Params:
        video (cv2.VideoCapture): OpenCV video capture object.

    Returns:
        base64 encoded frames in a list.
    """

    frames = []
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for f in range(n_frames):
        success, frame = video.read()
        if not success:
            break
        if f % step == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    if not frames:
        raise Exception("No frames extracted from video.")

    return frames

def process_video(client, video_file):
    """
    Process video file.

    Params:
        video_file (str): path to the video file.

    Returns:
        Dictionary of responses.
    """

    responses = {}

    #Extract frames
    video = cv2.VideoCapture(video_file)
    vid_frames = extract_images(video)

    #Convert video to audio
    #video_to_mp3.convert(video_file, "./")

    #Transcribe
    audio_file = video_file[:-4] + ".mp3"

    try:
        file = open(audio_file, "rb")
    except Exception as e:
        print(f"Error opening audio file: {e}")
        return responses
    
    txt_transcript = client.transcribe_audio(file)
    responses["transcript"] = txt_transcript
    
    #Detect objects in frames
    query = "These are frames from a video. List objects found in these frames, separated by commas."
    context = [
        {
            "image_base64": f"data:image/jpeg;base64,{frame}"
        }
            for frame in vid_frames
    ]

    objects = client.query_model(model="gpt-4o", query=query, inputs=context)
    responses["objects"] = objects

    #Mode and sentiment
    context = [
        {
            "text": (
                "Classify the overall mode and sentiment of the following video transcript as positive, negative or neutral."
                f'Transcript:\n"""\n{txt_transcript}\n"""'
            )
        }
    ]
    sentiment = client.query_model(model="gpt-4.1-mini", query="", inputs=context)
    responses["sentiment"] = sentiment
    
    #Generate QAs
    context = [
        {
            "text": (
                "Transform the following transcript into 3-4 pairs of questions and answers."
                "Format the answer as a Python dictionary with 'questions' as keys and 'answers' as values."
                "Desired format:{Question:, Answer: }"
                f'Transcript:\n"""\n{txt_transcript}\n"""'
            )
        }
    ]
    qa_list = client.query_model(model="gpt-4.1-mini", query="", inputs=context)
    responses["QAs"] = qa_list

    #Convert responses to json
    responses = json.loads(json.dumps(responses))

    return responses

def main():
    load_dotenv()
    client_key = os.getenv('OPENAI_API_KEY')
    client = GPTClient(client_key=client_key)

    video_file = "./data/AI_Intern_Project.mp4"
    responses = process_video(client, video_file)

    pprint.pprint(responses)

    return responses

if __name__ == "__main__":
    main()