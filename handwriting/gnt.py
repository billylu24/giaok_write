"""Strict reader for CASIA GNT grayscale character records."""
import struct
import numpy as np
from PIL import Image

HEADER=struct.Struct('<I2sHH')

def read_records(stream):
    offset=0
    while True:
        header=stream.read(HEADER.size)
        if not header:return
        if len(header)!=HEADER.size:raise ValueError(f'Truncated GNT header at {offset}')
        size,tag,width,height=HEADER.unpack(header)
        if not width or not height or size!=HEADER.size+width*height:
            raise ValueError(f'Invalid GNT record at {offset}: size={size}, width={width}, height={height}')
        bitmap=stream.read(width*height)
        if len(bitmap)!=width*height:raise ValueError(f'Truncated bitmap at {offset}')
        try:character=tag.decode('gb18030').strip('\x00')
        except UnicodeDecodeError as e:raise ValueError(f'Invalid GNT label {tag!r} at {offset}') from e
        if len(character)!=1:raise ValueError(f'Unexpected GNT label {character!r}')
        yield character,np.frombuffer(bitmap,dtype=np.uint8).reshape(height,width),offset
        offset+=size

def is_hanzi(character):
    return '\u4e00'<=character<='\u9fff'

def normalize_image(image,resolution=96,glyph_extent=80):
    """Preserve aspect ratio, slant and darkness; center in a white canvas."""
    height,width=image.shape
    ratio=glyph_extent/max(width,height)
    new_width=max(1,round(width*ratio));new_height=max(1,round(height*ratio))
    resized=Image.fromarray(image).resize((new_width,new_height),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(resolution,resolution),255)
    canvas.paste(resized,((resolution-new_width)//2,(resolution-new_height)//2))
    return np.asarray(canvas)
