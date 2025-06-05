import ffmpeg

input_file = 'a8bf42d646_floor.mp4'
output_file = 'a8bf42d646_floor.mov'

ffmpeg.input(input_file).output(output_file).run()
