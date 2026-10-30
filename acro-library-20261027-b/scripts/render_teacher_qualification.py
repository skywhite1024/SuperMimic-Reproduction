"""Use the original renderer with an explicit two-thread video encoder limit."""
import imageio.v2 as imageio
import render_library

native_writer = imageio.get_writer
def limited_writer(*args, **kwargs):
    kwargs['ffmpeg_params'] = [*kwargs.get('ffmpeg_params', []), '-threads', '2']
    return native_writer(*args, **kwargs)

if __name__ == '__main__':
    imageio.get_writer = limited_writer
    render_library.main()
