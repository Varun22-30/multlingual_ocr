from PIL import Image


class ResizeAndPad:
    def __init__(self, size, fill=(255, 255, 255)):
        self.target_height, self.target_width = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        if width == 0 or height == 0:
            return image.resize((self.target_width, self.target_height), Image.BILINEAR)

        scale = min(self.target_width / width, self.target_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (self.target_width, self.target_height), self.fill)

        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
