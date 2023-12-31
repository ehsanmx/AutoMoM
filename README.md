# AutoMoM (using AI)
Audio Meeting Summarizer and Action Point Generator

## Warning 
This is a Beta version
- Limited to max 30 minutes/200MB (Audio or Video Upload)
- Only Allows 4096 tokens

## Description
This project offers an automated solution for transcribing audio from meetings, summarizing the content, and generating action points. It leverages advanced audio processing techniques and large language models (LLMs) to provide accurate and concise summaries, making it an ideal tool for professionals and teams looking to streamline their meeting documentation process.

### Main Parts
- **Audio Processing**: Handles recording and processing of meeting audio.
- **LLM Operations**: Utilizes LLMs to transcribe, summarize, and generate action points from the processed audio.

## Installation
To set up the project, follow these steps:

1. **Clone the Repository**: Clone the project to your local machine.
   ```
   git clone https://github.com/ehsanmx/AutoMoM.git
   ```

2. **Set Up Python Environment**: It's recommended to use a virtual environment.
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**: Install the required packages using `requirements.txt`.
   ```
   pip install -r requirements.txt
   ```

4. **Additional Setup**:
   - Configure your audio input/output devices.
   - Set up the LLM model path in `llm_operations.py`.

## Usage
Follow these steps to use the application:

1. **Start the Application**: Run the main script.
   ```
   streamlit run src/app.py
   ```
2. **Transcription and Summarization**:
   - The `summarize_transcript` method will transcribe the audio and provide a summary.

3. **Generating Action Points**:
   - The `generate_action_points` method will generate action points based on the meeting's content.

### Examples
- Example 1: Running a full meeting summarization.
- Example 2: Generating action points for a specific section.

## Contributing
We welcome contributions to this project. Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a clear description of the changes.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.

