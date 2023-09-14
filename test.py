from NPerlinNoise import perlinGenerator, Noise
import PIL.Image as Image
import numpy as np

if __name__ == '__main__':
    haze_adder = Noise(frequency=4, octaves=8, persistence=0.5)
    noise_adder = Noise(frequency=16, octaves=8, persistence=0.5)
    img = Image.open('test.png')
    arr = np.array(img)
    haze = (perlinGenerator(haze_adder, (0, 128, img.size[1] - 1), (0, 128, img.size[0]))[0] + 1) / 2
    noise = (perlinGenerator(noise_adder, (0, 128, img.size[1] - 1), (0, 128, img.size[0]))[0]) / 2
    a = arr.max(axis=(0, 1))
    ar2 = ((arr / 2 + noise[..., None] * 128) * haze[..., None] + a * (1 - haze[..., None])).astype(np.uint8)
    im2 = Image.fromarray(ar2)
    im2.show()
