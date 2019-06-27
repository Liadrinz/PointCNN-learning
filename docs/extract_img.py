import base64
from bs4 import BeautifulSoup

with open('index3.html', 'r') as f:
    html = ''.join(f.readlines())

name = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]  # 0 for airplane, 1 for chair
real_imgs = []
soup = BeautifulSoup(html, 'lxml')
i = 0
for img in soup.find_all('img'):
    b64img = img.get('src', '').split(',')[-1]
    if i % 2:
        real_imgs.append(base64.b64decode(b64img))
    i += 1

ns = [0, 0]
ptrs = [0, 0]
lyn = 1
while ptrs[0] < 16:
    print(ns[name[ptrs[0]]])
    with open('records3/%s_%d_layer_%d.png' % (['airplane', 'chair'][name[ptrs[0]]], ns[name[ptrs[0]]] + 1, lyn), 'wb') as f:
        f.write(real_imgs[ptrs[1]])
        lyn += 1
        if lyn == 5:
            ns[name[ptrs[0]]] += 1
            lyn = 1
        ptrs[1] += 1
        if ptrs[1] % 4 == 0:
            ptrs[0] += 1
