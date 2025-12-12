import subprocess
import os

# ffmpeg -i Videos_Nov25_InsideFoggy/176.avi -c:v libx264 -crf 23 -c:a aac -b:a 128k Videos_Nov25_InsideFoggy/176.mp4

def convert_avi_to_mp4(input_file, output_file=None):
    """
    Converts an AVI file to MP4 format using FFmpeg.

    Args:
        input_file (str): The path to the input .avi file.
        output_file (str, optional): The desired path for the output .mp4 file.
                                     If None, uses the same name with a .mp4 extension.
    """
    # 1. Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # 2. Determine output file name
    if output_file is None:
        # Create output file name by replacing the extension
        base, ext = os.path.splitext(input_file)
        if ext.lower() != '.avi':
            print(f"Warning: Input file does not have an .avi extension. Proceeding anyway.")
        output_file = base + ".mp4"

    # 3. The core FFmpeg command:
    # -i: specifies the input file
    # -c:v: specifies the video codec (libx264 is standard for MP4)
    # -c:a: specifies the audio codec (aac is standard for MP4)
    # -b:v: specifies the video bitrate (optional, helps control output size/quality)
    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264", # Video codec for MP4
        "-c:a", "aac",    # Audio codec for MP4
        "-b:v", "2000k",  # Example bitrate (2000 kbit/s), adjust as needed
        "-strict", "experimental", # Needed for older FFmpeg versions with aac
        output_file
    ]

    print(f"Starting conversion: {input_file} -> {output_file}")

    try:
        # Execute the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Conversion successful! Output saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed. FFmpeg output:\n{e.stderr}")
    except FileNotFoundError:
        print("❌ Error: FFmpeg command not found. Make sure FFmpeg is installed and added to your system's PATH.")


# --- Example Usage ---
# Replace 'input.avi' with the actual path to your file
# The output will be 'input.mp4' in the same directory (if output_file is None)
convert_avi_to_mp4("Videos_Nov25_InsideFoggy/176.avi")
# OR
# convert_avi_to_mp4("path/to/your/input.avi", "path/to/your/output.mp4")
