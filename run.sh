nohup python3 train.py --cuda --threads 8 --dataset VisDrone --backbone ./weights/vgg16_reducedfc.pth --checkpoint  weights/SSD300-VGG16_VisDrone_45000.pth & 

#nohup python3 train.py --cuda --thrends 8 --dataset VisDrone --backbone ./weights/vgg16_reducedfc.pth --checkpoint weights/SSD300-VGG16_0712_45000.pth &
#sudo python3 train.py --voc_root /home/yosunpeng/github/DetectionDataset/VOCdevkit --backbone ./weights/vgg16_reducedfc.pth
