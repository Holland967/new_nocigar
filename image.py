from PIL import Image, ImageOps
import io

def compress_img(img_data):
    img = Image.open(io.BytesIO(img_data))

    exif_data = img.info.get("exif")
    img = ImageOps.exif_transpose(img)
    
    width, height = img.size
    aspect_ratio = width / height
    if width > height:
        target_width = 1024
        target_height = int(target_width / aspect_ratio)
    else:
        target_height = 1024
        target_width = int(target_height * aspect_ratio)
    
    img = img.resize((target_width, target_height), Image.LANCZOS)

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    quality = 100
    output_stream = io.BytesIO()
    while True:
        img.save(output_stream, format="PNG", quality=quality, exif=exif_data)
        if output_stream.tell() <= 1024 * 1024:
            break
        quality -= 5
        output_stream.seek(0)
        output_stream.truncate()

    compressed_img = output_stream.getvalue()
    output_stream.close()
    return compressed_img

def process_img(img_data):
    img = Image.open(io.BytesIO(img_data))
    width, height = img.size
    if width <= 448 and height <= 448:
        compressed_img = img_data
    elif width > 448 and height > 448:
        if len(img_data) > 1024 * 1024:
            compressed_img = compress_img(img_data)
        else:
            if width > 1024 or height > 1024:
                compressed_img = compress_img(img_data)
            else:
                compressed_img = img_data
    return compressed_img
