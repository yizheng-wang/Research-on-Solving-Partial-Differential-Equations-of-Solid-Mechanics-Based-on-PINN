import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from PIL import Image
import sys
sys.path.append("../")
from kan_efficiency import *
import time
# Load and process the image
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.rotate(180)  # Rotate the image by 180 degrees
    image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image horizontally
    image = image.resize((256, 256))  # Ensure image is 256x256 pixels
    image_data = np.array(image) / 255.0  # Normalize to [0.5, 1]
    return image_data

# Set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2024)

name_list = ['picasuo', 'salvator', 'sky']

for name_e in name_list:
    name_painting = name_e
    # Define function u(x) based on image
    image_path = './painting/' + name_painting + '_bw.jpg'
    image_data = load_image(image_path)
    
    def u(xy):
        x, y = xy[:, 0], xy[:, 1]
        x = (x * 255).long()
        y = (y * 255).long()
        u_values = image_data[y, x]
        return torch.tensor(u_values, dtype=torch.float32)
    
    # Generate training data
    N_u = 256
    X_u = np.linspace(0, 1, N_u)
    Y_u = np.linspace(0, 1, N_u)
    X, Y = np.meshgrid(X_u, Y_u)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    U = u(torch.tensor(XY, dtype=torch.float32)).numpy()
    
    # Convert to PyTorch tensors
    XY_tensor = torch.tensor(XY, dtype=torch.float32).cuda()
    U_tensor = torch.tensor(U, dtype=torch.float32).reshape(-1, 1).cuda()
    
    # Define fully connected neural network
    class FCNN(nn.Module):
        def __init__(self):
            super(FCNN, self).__init__()
            self.layer1 = nn.Linear(2, 200)
            self.layer2 = nn.Linear(200, 200)
            self.layer3 = nn.Linear(200, 1)
        
        def forward(self, x):
            x = torch.tanh(self.layer1(x))
            x = torch.tanh(self.layer2(x))
            x = self.layer3(x)
            return x
    
    # Instantiate model, loss function, and optimizer
    
    
    kan_model = KAN([2, 15, 15, 1], base_activation=torch.nn.SiLU, grid_size=100, grid_range=[0, 1.0], spline_order=3).cuda()
    fcnn_model = FCNN().cuda()
    criterion = nn.MSELoss()
    kan_optimizer = optim.Adam(kan_model.parameters(), lr=0.001)
    fcnn_optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)
    
    scheduler_kan = torch.optim.lr_scheduler.MultiStepLR(kan_optimizer, milestones=[1000, 3000, 5000], gamma = 0.1)
    scheduler_mlp = torch.optim.lr_scheduler.MultiStepLR(fcnn_optimizer, milestones=[1000, 3000, 5000], gamma = 0.1)
    
    # Training and evaluation
    num_epochs = 300
    kan_error_list = []
    fcnn_error_list = []
    kan_l2_list = []
    fcnn_l2_list = []
    checkpoint_epochs = list(range(1,num_epochs))
    
    def train_and_evaluate(model1, model2, optimizer1, optimizer2, error_list1, error_list2 , l2_list1, l2_list2):
        start_time = time.time()
        for epoch in range(num_epochs):
            model1.train()
            optimizer1.zero_grad()
            outputs1 = model1(XY_tensor)
            L2_1 = np.linalg.norm(outputs1.data.cpu().numpy() - U_tensor.data.cpu().numpy()) / np.linalg.norm(U_tensor.data.cpu().numpy())
            l2_list1.append(L2_1)
            loss1 = criterion(outputs1, U_tensor)
            loss1.backward()
            error_list1.append(loss1.detach().data.cpu().numpy())
            optimizer1.step()
            scheduler_kan.step()
         
            model2.train()
            optimizer2.zero_grad()
            outputs2 = model2(XY_tensor)
            L2_2 = np.linalg.norm(outputs2.data.cpu().numpy() - U_tensor.data.cpu().numpy()) / np.linalg.norm(U_tensor.data.cpu().numpy())
            l2_list2.append(L2_2)
            loss2 = criterion(outputs2, U_tensor)
            loss2.backward()
            error_list2.append(loss2.detach().data.cpu().numpy())
            optimizer2.step()
            scheduler_mlp.step()
            if (epoch + 1) % 10 == 0:
                end_time = time.time()
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, time: {end_time-start_time:.4f}')
    
            if (epoch + 1) in checkpoint_epochs:
                evaluate_model(model1, model2, epoch + 1)
    
    def evaluate_model(kan_model, fcnn_model, epoch):
        kan_model.eval()
        fcnn_model.eval()
        
        N_u = 256
        X_u = np.linspace(0, 1, N_u)
        Y_u = np.linspace(0, 1, N_u)
        X, Y = np.meshgrid(X_u, Y_u)
        XY = np.vstack([X.ravel(), Y.ravel()]).T
        U = u(torch.tensor(XY, dtype=torch.float32)).reshape(N_u, N_u).numpy()
    
        # Convert to PyTorch tensors
        XY_tensor = torch.tensor(XY, dtype=torch.float32).cuda()
        
        u_pred_kan = kan_model(XY_tensor)
        u_pred_kan = u_pred_kan.data
        u_pred_kan = u_pred_kan.reshape(N_u, N_u).cpu()
        error_kan = torch.abs(u_pred_kan - U)
    
        u_pred_mlp= fcnn_model(XY_tensor)
        u_pred_mlp = u_pred_mlp.data
        u_pred_mlp = u_pred_mlp.reshape(N_u, N_u).cpu()
        error_mlp = torch.abs(u_pred_mlp - U)
        
        
        # mlp
        h1 = plt.contourf(X, Y, u_pred_mlp, levels=100 ,cmap = 'gray')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar(h1).ax.set_title('MLP_pred')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'MLP: epoch={epoch}')
        plt.savefig(f'./results/gif/MLP_pred_{name_painting}_epoch{epoch}.pdf', dpi=300)
        plt.show() 
    
        h1 = plt.contourf(X, Y, error_mlp, levels=100 ,cmap = 'gray')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar(h1).ax.set_title('MLP_error')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'MLP: epoch={epoch}')
        plt.savefig(f'./results/gif/MLP_error_{name_painting}_epoch{epoch}.pdf', dpi=300)
        plt.show() 
    
        # kan
        h1 = plt.contourf(X, Y, u_pred_kan, levels=100 ,cmap = 'gray')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar(h1).ax.set_title('KAN_pred')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'KINN: epoch={epoch}')
        plt.savefig(f'./results/gif/KAN_pred_{name_painting}_epoch{epoch}.pdf', dpi=300)
        plt.show() 
    
        h1 = plt.contourf(X, Y, error_kan, levels=100 ,cmap = 'gray')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.colorbar(h1).ax.set_title('KAN_error')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'KINN: epoch={epoch}')
        plt.savefig(f'./results/gif/KAN_error_{name_painting}_epoch{epoch}.pdf', dpi=300)
        plt.show() 
        
        
    
    # Train models
    train_and_evaluate(kan_model, fcnn_model, kan_optimizer, fcnn_optimizer, kan_error_list, fcnn_error_list, kan_l2_list, fcnn_l2_list)










#%%


import os
from pdf2image import convert_from_path
import imageio

def create_gif(image_folder, gif_name):
    # 获取PNG图像的路径列表
    images = []
    
    # 自定义排序，提取文件名中的epoch数字
    def extract_epoch(file_name):
        # 假设文件名格式为 'MLP_pred_painting_epochXXX.png'
        epoch_number = int(file_name.split('_epoch')[-1].split('.')[0])
        return epoch_number    

    sorted_files = sorted([file for file in os.listdir(image_folder) if file.endswith(".png")], key=extract_epoch)


    for file_name in sorted_files:
        file_path = os.path.join(image_folder, file_name)
        images.append(imageio.imread(file_path))

    # 生成GIF文件
    gif_path = os.path.join('./gif', gif_name)
    imageio.mimsave(gif_path, images, fps=20)  # 设置帧率（1帧每秒）
    print(f"GIF saved at {gif_path}")



def convert_pdf_to_png(name_painting):
    for epoch in list(range(1,num_epochs)):
        pdf_path_mlp = f'results/gif/MLP_pred_{name_painting}_epoch{epoch}.pdf'
        pdf_path_kan = f'results/gif/KAN_pred_{name_painting}_epoch{epoch}.pdf'

        # 将PDF文件转换为PNG
        images_mlp = convert_from_path(pdf_path_mlp)
        images_kan = convert_from_path(pdf_path_kan)

        # 保存转换后的PNG文件
        for i, image in enumerate(images_mlp):
            output_path_mlp = f'gif/{name_painting}/MLP/MLP_pred_{name_painting}_epoch{epoch}.png'
            image.save(output_path_mlp, 'PNG')

        for i, image in enumerate(images_kan):
            output_path_kan = f'gif/{name_painting}/KAN/KAN_pred_{name_painting}_epoch{epoch}.png'
            image.save(output_path_kan, 'PNG')

        print(f"Epoch {epoch}: PNG images saved.")

# 定义文件夹路径



name_list = ['picasuo', 'salvator', 'sky']

for name_painting in name_list:
    output_folder = f'gif/{name_painting}'
    os.makedirs(output_folder, exist_ok=True)    

    # 将PDF转换为PNG
    convert_pdf_to_png(name_painting)
    
    # 创建MLP的GIF
    create_gif(output_folder+'/MLP', f'MLP_pred_{name_painting}_evolution.gif')
    
    # 创建KAN的GIF
    create_gif(output_folder+'/KAN', f'KAN_pred_{name_painting}_evolution.gif')
