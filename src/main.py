from src.video_processor import process_video

if __name__ == "__main__":
    input_path = "data/input_video.mp4"
    output_path = "data/output_video.mp4"
    process_video(input_path, output_path, display=True)