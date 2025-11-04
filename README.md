# Video Processing with OpenAI API

This project uses OpenAI API models like **gpt-4o** and **gpt-4.1-mini** to generate information and get details from a video. 

## Features
- Audio transcription
- Frame analysis using gpt-4o
- Query handling for get information (e.g., mode, sentiment and QA)
- Create JSON output containing details about the video

## Approach
The program operates as follows:

### 1. Video transcription
- Convert video into audio using `video-to-mp3`.
- Uses audio as an input to the **gpt-4o-transcribe** model for text transcription.

### 2. Object detection
- Extract frames from videos using `cv2`. To avoid exceeding rate limits to API call, only every 100th frame is used.
- Uses **gpt-4o** to process and detect objects in the images.
- Query the model using `GptClient` class with custom prompt with instructions and desired output format.

### 3. Mode, sentiment and QA generation
- Use **gpt-4.1-mini** to extract the overall mode, sentiment and generate QAs based on the audio transcript.

4. Output
- Combines all results into a single JSON object:

```json
{
  "transcript": "...",
  "objects": "...",
  "sentiment": "...",
  "mode": "...",
  "QAs": [
    {"Question": "...", "Answer": "..."},
    ...
  ]
}
```

## Use of AI
AI was used in this project to:

- Troubleshoot API query errors 
- Help create comprehensive unit tests
- Add exception handling to handle runtime errors

## Future Improvements
- Process more frames to detect more objects in the video. 
- Include real-time speech analysis to better capture the overall sentiment through vocal nuances such as tone, speed, volume.
- Improve error handling and testing for API calls.

## Dependencies
- `OpenAI API` (gpt-4o, gpt-4.1-mini, gpt-4o-transcribe)
- OpenCV (`cv2`) for frame extraction
- `video_to_mp3` for audio extraction
- `dotenv` for environment variable management
- `jsonschema` for output validation

## To install dependencies
```pip install -r requirements.txt```

## To run the code
```python main.py```

## Environment variable
Store OpenAI API key in the environment variables, for example:
```OPENAI_API_KEY="your_api_key_here"```

## Data
Store video/audio file in the data folder.