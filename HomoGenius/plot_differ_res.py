import matplotlib.pyplot as plt
import numpy as np



train_error_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dic.npy', allow_pickle=True).item()
test_error_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dic.npy', allow_pickle=True).item()
time_dic = np.load('./outdata/res128_data_type3_diff_res/time_dic.npy', allow_pickle=True).item()

train_error_dis_x_1_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_1_dic.npy', allow_pickle=True).item()
train_error_dis_x_2_dic =np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_2_dic.npy', allow_pickle=True).item()
train_error_dis_x_3_dic =np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_3_dic.npy', allow_pickle=True).item()
train_error_dis_x_4_dic =np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_4_dic.npy', allow_pickle=True).item()
train_error_dis_x_5_dic =np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_5_dic.npy', allow_pickle=True).item()
train_error_dis_x_6_dic =np.load('./outdata/res128_data_type3_diff_res/train_error_dis_x_6_dic.npy', allow_pickle=True).item()

train_error_dis_y_1_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_1_dic.npy', allow_pickle=True).item()
train_error_dis_y_2_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_2_dic.npy', allow_pickle=True).item()
train_error_dis_y_3_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_3_dic.npy', allow_pickle=True).item()
train_error_dis_y_4_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_4_dic.npy', allow_pickle=True).item()
train_error_dis_y_5_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_5_dic.npy', allow_pickle=True).item()
train_error_dis_y_6_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_y_6_dic.npy', allow_pickle=True).item()

train_error_dis_z_1_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_1_dic.npy', allow_pickle=True).item()
train_error_dis_z_2_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_2_dic.npy', allow_pickle=True).item()
train_error_dis_z_3_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_3_dic.npy', allow_pickle=True).item()
train_error_dis_z_4_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_4_dic.npy', allow_pickle=True).item()
train_error_dis_z_5_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_5_dic.npy', allow_pickle=True).item()
train_error_dis_z_6_dic = np.load('./outdata/res128_data_type3_diff_res/train_error_dis_z_6_dic.npy', allow_pickle=True).item()

test_error_dis_x_1_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_1_dic.npy', allow_pickle=True).item()
test_error_dis_x_2_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_2_dic.npy', allow_pickle=True).item()
test_error_dis_x_3_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_3_dic.npy', allow_pickle=True).item()
test_error_dis_x_4_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_4_dic.npy', allow_pickle=True).item()
test_error_dis_x_5_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_5_dic.npy', allow_pickle=True).item()
test_error_dis_x_6_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_x_6_dic.npy', allow_pickle=True).item()

test_error_dis_y_1_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_1_dic.npy', allow_pickle=True).item()
test_error_dis_y_2_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_2_dic.npy', allow_pickle=True).item()
test_error_dis_y_3_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_3_dic.npy', allow_pickle=True).item()
test_error_dis_y_4_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_4_dic.npy', allow_pickle=True).item()
test_error_dis_y_5_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_5_dic.npy', allow_pickle=True).item()
test_error_dis_y_6_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_y_6_dic.npy', allow_pickle=True).item()

test_error_dis_z_1_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_1_dic.npy', allow_pickle=True).item()
test_error_dis_z_2_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_2_dic.npy', allow_pickle=True).item()
test_error_dis_z_3_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_3_dic.npy', allow_pickle=True).item()
test_error_dis_z_4_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_4_dic.npy', allow_pickle=True).item()
test_error_dis_z_5_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_5_dic.npy', allow_pickle=True).item()
test_error_dis_z_6_dic = np.load('./outdata/res128_data_type3_diff_res/test_error_dis_z_6_dic.npy', allow_pickle=True).item()
    
    
training_data_resolution = [32, 64, 128]
test_data_resolution = [128]


plt.style.use('seaborn-v0_8')
for test_data_resolution_e in test_data_resolution:
    plt.rcParams['xtick.labelsize'] = 15 # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 15 # y轴刻度字体大小
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_1_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_1_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_1_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")

    plt.title('Dis_x_1', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_1', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_2_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_2_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_2_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_x_2', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_2', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_3_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_3_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_3_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_x_3', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_3', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_4_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_4_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_4_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_x_4', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_4', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_5_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_5_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_5_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_x_5', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_5', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_x_6_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_x_6_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_x_6_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_x_6', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_x_6', dpi=1000)
    plt.show() 
    
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_1_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_1_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_1_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_1', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_1', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_2_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_2_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_2_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_2', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_2', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_3_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_3_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_3_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_3', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_3', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_4_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_4_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_4_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_4', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_4', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_5_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_5_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_5_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_5', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_5', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_y_6_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_y_6_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_y_6_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_y_6', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_y_6', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_1_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_1_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_1_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_1', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_1', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_2_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_2_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_2_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_2', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_2', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_3_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_3_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_3_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_3', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_3', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_4_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_4_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_4_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_4', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_4', dpi=1000)
    plt.show() 
    
for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_5_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_5_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_5_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_5', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_5', dpi=1000)
    plt.show() 

for test_data_resolution_e in test_data_resolution:
    for training_plot_re in training_data_resolution:
        plt.plot(test_error_dis_z_6_dic[training_plot_re][test_data_resolution_e], label = 'Train_res: %i, Test_res %i' % (training_plot_re, test_data_resolution_e))
        final_x = len(test_error_dis_z_6_dic[training_plot_re][test_data_resolution_e])-200
        final_y = test_error_dis_z_6_dic[training_plot_re][test_data_resolution_e][-1]
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02  # 偏移量设置为y轴范围的2%
        plt.text(final_x, final_y+ offset, f'{final_y:.3f}', fontsize=16)  # 显示3位小数
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Relative Error', fontsize=20)
    plt.yscale("log")
    plt.title('Dis_z_6', fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('./Image/Diff_res/L2_loss_train_re_dis_z_6', dpi=1000)
    plt.show() 