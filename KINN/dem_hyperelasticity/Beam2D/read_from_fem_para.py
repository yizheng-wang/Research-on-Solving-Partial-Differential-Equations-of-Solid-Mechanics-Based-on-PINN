import pyvista as pv

# 读取 VTU 文件
vtu_file = "./output/fem/E=1000/beam2d_4x1_fem_v1000000.vtu"
mesh = pv.read(vtu_file)

# 获取坐标点
points = mesh.points

# 获取所有数组数据的名称
array_names = mesh.array_names



# 打印所有数组数据
for name in array_names:
    array_data = mesh.point_data[name]


