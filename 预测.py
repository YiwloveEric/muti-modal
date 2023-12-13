from PIL import Image
import torch
from torchvision import transforms
from models import VGG  # 确保这与您的项目结构相匹配

# 加载模型
model = VGG('VGG19')
checkpoint = torch.load(r'CK+_VGG19\1\Best_Test_model.pth')

model.eval()

# 图像预处理
image_path = r'D:\study\大三上课程资料\数据采集与融合\多线程图片\610866851-1_b_1.jpg.jpg'
image = Image.open(image_path).convert('RGB')  # 转换为 RGB
transform = transforms.Compose([
    transforms.Resize((44, 44)),  
    transforms.ToTensor(),
])
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)


# 推断
with torch.no_grad():
    output = model(input_tensor)

# 获取预测结果
_, predicted = torch.max(output.data, 1)

# 类别标签
classes = ['生气', '厌恶', '害怕', '开心', '伤心', '惊讶', '蔑视']

# 打印预测结果
print("预测类别:", classes[predicted.item()])
