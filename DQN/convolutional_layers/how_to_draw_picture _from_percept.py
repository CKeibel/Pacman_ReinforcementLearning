'''
W - Wall, P - Pacman, G - Ghost, D - Dot, E - Empty, GD - Ghost & Dot

7x7
W W W W W W W
W P E E E E W
W E E E E E W
W E E E E E W
W E E E E E W
W E E E E E W
W W W W W W W

Picture Size: 84 x 84 => 12 x 7 = 84
=> every square should be 12x12 pixels

'''
from PIL import Image, ImageDraw
import numpy as np
percept = [['W' for _ in range(7) ],
            ['W', 'E', 'D', 'D', 'D', 'D', 'W'],
            ['W', 'P', 'W', 'W', 'W', 'D', 'W'],
            ['W', 'D', 'W', 'D', 'D', 'D', 'W'],
            ['W', 'D', 'W', 'D', 'W', 'G', 'W'],
            ['W', 'D', 'D', 'D', 'D', 'E', 'W'],
            ['W' for _ in range(7)]]

im = Image.new("RGB", (140, 140))

draw = ImageDraw.Draw(im)

for x in range(0, len(percept)):
    for y in range(0, len(percept[x])):
        if percept[y][x] == 'W':
            draw.rectangle((x*20, y*20, x*20+20, y*20+20), fill=(0, 0, 139))
        elif percept[y][x] == 'P':
            draw.rectangle((x*20, y*20, x*20+20, y*20+20), fill=(255, 255 ,0))
        elif percept[y][x] == 'G' or percept[x][y] == 'GD':
            draw.rectangle((x*20, y*20, x*20+20, y*20+20), fill=(255, 0, 0))
        elif percept[y][x] == 'D':
            draw.rectangle((x*20, y*20, x*20+20, y*20+20), fill=(255, 255, 255))
        elif percept[y][x] == 'E':
            draw.rectangle((x*20, y*20, x*20+20, y*20+20), fill=(0, 204, 0))


im = im.convert('L')
#im.show()
im.save('pacman_grayscale2.png')
im = np.array(im)
#print(im)
#im = np.reshape(im, (1, 140, 140, 1))
#print(im.shape)
#print(np.min(im))


