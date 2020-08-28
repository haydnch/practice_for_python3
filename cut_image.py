'''
将一张图片填充为正方形后切为n*n张小图
'''
from PIL import Image


# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽的最大值作为新图的边长
    new_image_length = width if width > height else height
    # 生成白底新图片
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将输入的图片居中粘贴到新图
    if width > height:  # 原图宽大于高 填充图片的垂直维度 (x ,y)表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    new_image.save('./cut_images/new_image.png', 'PNG')
    return new_image


# 切图
def cut_image(image, n):
    width, height = image.size
    item_width = int(width / n)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, n):
        for j in range(0, n):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            print(box)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


# 保存图片
def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('./cut_images/' + 'cut_' + str(index) + '.png', 'PNG')
        index += 1


def main():
    file_path = r'C:\Users\apple\Pictures\avatar.PNG'
    image = Image.open(file_path)
    #image.show()
    image = fill_image(image)
    image_list = cut_image(image, 3)
    save_images(image_list)

if __name__ == '__main__':
    main()